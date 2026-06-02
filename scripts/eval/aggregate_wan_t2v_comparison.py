#!/usr/bin/env python3
"""Aggregate Wan2.1 T2V 50-step comparison metrics."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional


METHOD_ORDER = [
    "dense",
    "svg_s03",
    "sap_default_q300_k800_tp092",
    "rhyme_tw10_m2_skip3-5",
    "rhyme_sap_tw8_m3_skip3-5_q350_k1200_tp098_min020_it5",
]

METHOD_LABELS = {
    "dense": "Dense",
    "svg_s03": "SVG s0.3",
    "sap_default_q300_k800_tp092": "SAP q300/k800 tp0.92",
    "rhyme_tw10_m2_skip3-5": "Rhyme Tw10 M2 skip3-5",
    "rhyme_sap_tw8_m3_skip3-5_q350_k1200_tp098_min020_it5": (
        "Rhyme+SAP Tw8 M3 skip3-5 q350/k1200 tp0.98"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--prompt_ids", nargs="*", type=int, default=None)
    parser.add_argument("--out_prefix", default=None)
    return parser.parse_args()


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out


def finite_values(rows: Iterable[Dict[str, object]], key: str) -> List[float]:
    out = []
    for row in rows:
        value = parse_float(row.get(key))
        if value is not None and math.isfinite(value):
            out.append(value)
    return out


def mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def stdev(values: List[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


def discover_prompt_ids(root: Path) -> List[int]:
    ids = []
    for path in root.glob("prompt_*_seed_*"):
        parts = path.name.split("_")
        if len(parts) >= 2:
            try:
                ids.append(int(parts[1]))
            except ValueError:
                pass
    return sorted(set(ids))


def read_prompt_metrics(root: Path, prompt_id: int) -> List[Dict[str, object]]:
    prompt_dirs = sorted(root.glob(f"prompt_{prompt_id}_seed_*"))
    if not prompt_dirs:
        return []
    metrics_path = prompt_dirs[0] / "metrics_vs_dense.csv"
    if not metrics_path.exists():
        return []
    with metrics_path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["prompt_id"] = prompt_id
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    prompt_ids = args.prompt_ids if args.prompt_ids else discover_prompt_ids(root)
    out_prefix = Path(args.out_prefix) if args.out_prefix else root / "wan_t2v_comparison"

    all_rows: List[Dict[str, object]] = []
    for prompt_id in prompt_ids:
        all_rows.extend(read_prompt_metrics(root, prompt_id))

    if not all_rows:
        raise RuntimeError(f"No metrics_vs_dense.csv rows found under {root}")

    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in all_rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    summary_rows: List[Dict[str, object]] = []
    dense_time = mean(finite_values(by_method.get("dense", []), "total_time_sec"))
    for method in METHOD_ORDER:
        rows = by_method.get(method, [])
        if not rows:
            continue
        psnr = finite_values(rows, "psnr_frame_avg_db")
        ssim = finite_values(rows, "ssim")
        lpips = finite_values(rows, "lpips")
        total = finite_values(rows, "total_time_sec")
        speedup = finite_values(rows, "speedup_vs_dense")
        peak = finite_values(rows, "peak_allocated_gb")
        success = sum(1 for row in rows if row.get("status", "success") == "success")
        total_mean = mean(total)
        summary_rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "n": len(rows),
                "success": success,
                "total_time_mean": total_mean,
                "total_time_std": stdev(total),
                "speedup_mean": mean(speedup),
                "speedup_from_mean_dense": None if not dense_time or not total_mean else dense_time / total_mean,
                "speedup_min": min(speedup) if speedup else None,
                "speedup_max": max(speedup) if speedup else None,
                "psnr_mean": mean(psnr),
                "psnr_std": stdev(psnr),
                "psnr_min": min(psnr) if psnr else None,
                "ssim_mean": mean(ssim),
                "lpips_mean": mean(lpips),
                "peak_allocated_mean": mean(peak),
                "peak_allocated_max": max(peak) if peak else None,
            }
        )

    write_csv(
        out_prefix.with_name(out_prefix.name + "_all.csv"),
        all_rows,
        [
            "prompt_id",
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
        ],
    )
    write_csv(
        out_prefix.with_name(out_prefix.name + "_summary.csv"),
        summary_rows,
        [
            "method",
            "label",
            "n",
            "success",
            "total_time_mean",
            "total_time_std",
            "speedup_mean",
            "speedup_from_mean_dense",
            "speedup_min",
            "speedup_max",
            "psnr_mean",
            "psnr_std",
            "psnr_min",
            "ssim_mean",
            "lpips_mean",
            "peak_allocated_mean",
            "peak_allocated_max",
        ],
    )
    with out_prefix.with_name(out_prefix.name + ".json").open("w") as f:
        json.dump({"rows": all_rows, "summary": summary_rows}, f, indent=2)

    md_path = out_prefix.with_name(out_prefix.name + ".md")
    with md_path.open("w") as f:
        f.write("# Wan2.1 T2V 50-Step Comparison\n\n")
        f.write(f"Root: `{root}`\n\n")
        f.write(f"Prompt ids: `{', '.join(str(i) for i in prompt_ids)}`\n\n")
        f.write(
            "Setup: Wan2.1-T2V-1.3B-Diffusers, 720x1280, 81 frames, 50 denoising steps, seed 0. "
            "Metrics use each prompt's Dense output as pseudo-reference.\n\n"
        )
        f.write("| Method | n | Success | Total(s) | Speedup | PSNR | SSIM | LPIPS | Peak GB |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['label']} | {row['n']} | {row['success']} | "
                f"{fmt(row['total_time_mean'], 2)} | {fmt(row['speedup_mean'], 3)} | "
                f"{fmt(row['psnr_mean'])} | {fmt(row['ssim_mean'], 5)} | "
                f"{fmt(row['lpips_mean'], 5)} | {fmt(row['peak_allocated_max'], 2)} |\n"
            )
        f.write("\n## Per-Prompt Rows\n\n")
        f.write("| Prompt | Method | Total(s) | Speedup | PSNR | SSIM | LPIPS | Peak GB | Status |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
        ordered_rows = sorted(
            all_rows,
            key=lambda r: (
                int(r["prompt_id"]),
                METHOD_ORDER.index(str(r["method"])) if str(r["method"]) in METHOD_ORDER else 999,
            ),
        )
        for row in ordered_rows:
            f.write(
                f"| {row['prompt_id']} | {METHOD_LABELS.get(str(row['method']), row['method'])} | "
                f"{fmt(parse_float(row.get('total_time_sec')), 2)} | "
                f"{fmt(parse_float(row.get('speedup_vs_dense')), 3)} | "
                f"{fmt(parse_float(row.get('psnr_frame_avg_db')))} | "
                f"{fmt(parse_float(row.get('ssim')), 5)} | "
                f"{fmt(parse_float(row.get('lpips')), 5)} | "
                f"{fmt(parse_float(row.get('peak_allocated_gb')), 2)} | {row.get('status', '')} |\n"
            )

    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_all.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_summary.csv')}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '.json')}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
