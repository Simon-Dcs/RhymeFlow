#!/usr/bin/env python3
"""Compute Phase 0 video metrics using Dense output as the reference.

The script is intentionally independent from svg.utils.metric because that
module constructs LPIPS on import and keeps whole videos on CUDA. Here videos
are decoded pairwise and metrics are accumulated in small batches.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F


MetricRow = Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Phase 0 result root containing dense/ and method subdirs.")
    parser.add_argument("--dense", default=None, help="Dense reference video. Defaults to <root>/dense/<video_name>.")
    parser.add_argument("--video_name", default="1-0.mp4", help="Video filename to compare in each method dir.")
    parser.add_argument("--method", action="append", default=None, help="Method dir name to evaluate. Repeatable.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch_size", type=int, default=1, help="Frame batch size for PSNR/SSIM.")
    parser.add_argument("--lpips_batch_size", type=int, default=1, help="Frame batch size cap for LPIPS.")
    parser.add_argument("--lpips_net", default="vgg", choices=("alex", "vgg", "squeeze"))
    parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS.")
    parser.add_argument("--resize_mismatch", action="store_true", help="Resize prediction frames to Dense resolution.")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit compared frames for quick checks.")
    parser.add_argument("--out_prefix", default=None, help="Output prefix. Defaults to <root>/metrics_vs_dense.")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def discover_methods(root: Path, video_name: str, requested: Optional[List[str]]) -> List[str]:
    if requested:
        return requested

    methods = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name == "dense":
            continue
        if (path / video_name).exists():
            methods.append(path.name)
    return methods


def tensor_from_frames(frames: List[np.ndarray]) -> torch.Tensor:
    arrays = []
    for frame in frames:
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError(f"Expected RGB-like frame, got shape {arr.shape}")
        arrays.append(np.ascontiguousarray(arr[:, :, :3]))
    stacked = np.stack(arrays, axis=0)
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).float().div_(255.0)


def iter_paired_frame_batches(
    ref_path: Path,
    pred_path: Path,
    batch_size: int,
    max_frames: Optional[int],
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    ref_reader = imageio.get_reader(str(ref_path))
    pred_reader = imageio.get_reader(str(pred_path))
    ref_iter = iter(ref_reader)
    pred_iter = iter(pred_reader)
    ref_frames: List[np.ndarray] = []
    pred_frames: List[np.ndarray] = []
    compared = 0

    try:
        while max_frames is None or compared < max_frames:
            try:
                ref_frame = next(ref_iter)
            except StopIteration:
                try:
                    next(pred_iter)
                    raise ValueError(f"Prediction has more frames than Dense: {pred_path}")
                except StopIteration:
                    break

            try:
                pred_frame = next(pred_iter)
            except StopIteration as exc:
                raise ValueError(f"Prediction has fewer frames than Dense: {pred_path}") from exc

            ref_frames.append(ref_frame)
            pred_frames.append(pred_frame)
            compared += 1

            if len(ref_frames) == batch_size:
                yield tensor_from_frames(ref_frames), tensor_from_frames(pred_frames)
                ref_frames = []
                pred_frames = []

        if ref_frames:
            yield tensor_from_frames(ref_frames), tensor_from_frames(pred_frames)
    finally:
        ref_reader.close()
        pred_reader.close()


def ssim_uniform_window(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return one SSIM value per frame using the project's previous 11x11 window."""
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, kernel_size=11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(x * x, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=11, stride=1, padding=5) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def load_lpips_model(device: torch.device, net: str):
    import lpips

    model = lpips.LPIPS(net=net).to(device)
    model.eval()
    return model


def compute_pair_metrics(
    ref_path: Path,
    pred_path: Path,
    device: torch.device,
    batch_size: int,
    lpips_model,
    lpips_batch_size: int,
    resize_mismatch: bool,
    max_frames: Optional[int],
) -> MetricRow:
    frame_count = 0
    pixel_count = 0
    sse = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    lpips_count = 0
    shape: Optional[Tuple[int, int, int]] = None

    with torch.inference_mode():
        for ref_cpu, pred_cpu in iter_paired_frame_batches(ref_path, pred_path, batch_size, max_frames):
            if ref_cpu.shape != pred_cpu.shape:
                if not resize_mismatch:
                    raise ValueError(f"Shape mismatch: {ref_cpu.shape} != {pred_cpu.shape}")
                pred_cpu = F.interpolate(pred_cpu, size=ref_cpu.shape[-2:], mode="bilinear", align_corners=False)

            if shape is None:
                shape = (int(ref_cpu.shape[1]), int(ref_cpu.shape[2]), int(ref_cpu.shape[3]))

            ref = ref_cpu.to(device, non_blocking=True)
            pred = pred_cpu.to(device, non_blocking=True)
            diff = ref - pred
            frame_mse = diff.square().flatten(1).mean(dim=1)

            frame_count += int(ref.shape[0])
            pixel_count += int(diff.numel())
            sse += float(diff.square().sum().item())
            psnr_sum += float((10.0 * torch.log10(1.0 / torch.clamp(frame_mse, min=1.0e-12))).sum().item())
            ssim_sum += float(ssim_uniform_window(ref, pred).sum().item())

            if lpips_model is not None:
                for start in range(0, ref.shape[0], lpips_batch_size):
                    ref_lp = ref[start : start + lpips_batch_size].mul(2.0).sub(1.0)
                    pred_lp = pred[start : start + lpips_batch_size].mul(2.0).sub(1.0)
                    value = lpips_model(ref_lp, pred_lp).flatten()
                    lpips_sum += float(value.sum().item())
                    lpips_count += int(value.numel())

    if frame_count == 0:
        raise ValueError(f"No paired frames compared for {pred_path}")

    mse = sse / pixel_count
    return {
        "frames": frame_count,
        "channels": None if shape is None else shape[0],
        "height": None if shape is None else shape[1],
        "width": None if shape is None else shape[2],
        "mse": mse,
        "psnr_global_db": float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse),
        "psnr_frame_avg_db": psnr_sum / frame_count,
        "ssim": ssim_sum / frame_count,
        "lpips": None if lpips_model is None else lpips_sum / max(1, lpips_count),
    }


def read_summary(method_dir: Path) -> Dict[str, object]:
    path = method_dir / "summary.json"
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def format_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_outputs(rows: List[MetricRow], out_prefix: Path, dense_video: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    fieldnames = [
        "method",
        "video",
        "frames",
        "height",
        "width",
        "mse",
        "psnr_frame_avg_db",
        "psnr_global_db",
        "ssim",
        "lpips",
        "total_time_sec",
        "speedup_vs_dense",
        "peak_allocated_gb",
        "status",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with md_path.open("w") as f:
        f.write("# Phase 0 Metrics vs Dense\n\n")
        f.write(f"Dense reference: `{dense_video}`\n\n")
        f.write(
            "These metrics treat the Dense sample as a pseudo-reference, not as ground truth. "
            "Higher PSNR/SSIM and lower LPIPS mean the method stayed closer to Dense.\n\n"
        )
        f.write(
            "| Method | Frames | PSNR frame avg (dB) | PSNR global (dB) | SSIM | LPIPS | "
            "Total(s) | Speedup | Peak GB |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| {method} | {frames} | {psnr_frame} | {psnr_global} | {ssim} | {lpips} | "
                "{total} | {speedup} | {peak} |\n".format(
                    method=row["method"],
                    frames=row["frames"],
                    psnr_frame=format_float(row["psnr_frame_avg_db"]),
                    psnr_global=format_float(row["psnr_global_db"]),
                    ssim=format_float(row["ssim"], 5),
                    lpips=format_float(row["lpips"], 5),
                    total=format_float(row.get("total_time_sec"), 2),
                    speedup=format_float(row.get("speedup_vs_dense"), 2),
                    peak=format_float(row.get("peak_allocated_gb"), 2),
                )
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(root)

    dense_video = Path(args.dense) if args.dense else root / "dense" / args.video_name
    if not dense_video.exists():
        raise FileNotFoundError(dense_video)

    device = resolve_device(args.device)
    methods = discover_methods(root, args.video_name, args.method)
    if not methods:
        raise RuntimeError(f"No method videos found under {root}")

    lpips_model = None if args.no_lpips else load_lpips_model(device, args.lpips_net)
    out_prefix = Path(args.out_prefix) if args.out_prefix else root / "metrics_vs_dense"

    dense_summary = read_summary(dense_video.parent)
    dense_total = dense_summary.get("total_time_sec")
    rows: List[MetricRow] = [
        {
            "method": "dense",
            "video": str(dense_video),
            "frames": dense_summary.get("num_frames"),
            "height": dense_summary.get("height"),
            "width": dense_summary.get("width"),
            "mse": 0.0,
            "psnr_frame_avg_db": float("inf"),
            "psnr_global_db": float("inf"),
            "ssim": 1.0,
            "lpips": 0.0 if lpips_model is not None else None,
            "total_time_sec": dense_total,
            "speedup_vs_dense": 1.0 if dense_total else None,
            "peak_allocated_gb": dense_summary.get("peak_allocated_gb"),
            "status": dense_summary.get("status", "success"),
        }
    ]

    print(f"Reference: {dense_video}")
    print(f"Device: {device}; methods: {', '.join(methods)}")
    for method in methods:
        method_dir = root / method
        pred_video = method_dir / args.video_name
        if not pred_video.exists():
            print(f"Skip missing method video: {pred_video}", file=sys.stderr)
            continue

        start = time.perf_counter()
        print(f"Computing {method} ...", flush=True)
        metrics = compute_pair_metrics(
            dense_video,
            pred_video,
            device=device,
            batch_size=max(1, args.batch_size),
            lpips_model=lpips_model,
            lpips_batch_size=max(1, args.lpips_batch_size),
            resize_mismatch=args.resize_mismatch,
            max_frames=args.max_frames,
        )
        elapsed = time.perf_counter() - start

        summary = read_summary(method_dir)
        total = summary.get("total_time_sec")
        metrics.update(
            {
                "method": method,
                "video": str(pred_video),
                "metric_time_sec": elapsed,
                "total_time_sec": total,
                "speedup_vs_dense": (dense_total / total) if dense_total and total else None,
                "peak_allocated_gb": summary.get("peak_allocated_gb"),
                "status": summary.get("status", "success"),
            }
        )
        rows.append(metrics)
        print(
            f"{method}: PSNR(avg)={metrics['psnr_frame_avg_db']:.4f} dB, "
            f"SSIM={metrics['ssim']:.5f}, LPIPS={format_float(metrics['lpips'], 5)}, "
            f"time={elapsed:.1f}s"
        )

    write_outputs(rows, out_prefix, dense_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
