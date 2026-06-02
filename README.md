# RhymeFlow

Training-free acceleration for Wan2.1 text-to-video generation.

This repository is the Wan2.1 T2V-focused release subset of the Sparse-VideoGen
experiments. It contains the Diffusers Wan2.1 inference entrypoint, sparse
attention baselines, RhymeFlow keyframe-preserving generation, and scripts for
Dense-reference speed/quality comparison.

## Scope

Current release scope:

- Model family: Wan2.1 T2V through Hugging Face Diffusers.
- Default model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
- Default benchmark shape: 720x1280, 81 frames, 50 denoising steps, seed 0.
- Methods: Dense, SVG, SAP, RhymeFlow, and RhymeFlow+SAP.

Not included in this release subset: HunyuanVideo, CogVideoX, Cosmos, I2V
scripts, checkpoints, generated videos, and local experiment result folders.

## Installation

Use a CUDA-enabled Python environment. The original experiments were run in the
`SVG` conda environment with PyTorch, Diffusers 0.34.0, FlashInfer, and RAPIDS
cuVS available.

```bash
cd /home/dcs/RhymeFlow
pip install -e .
```

SAP uses KMeans/block-sparse attention and needs FlashInfer plus RAPIDS cuVS
installed for your CUDA version. Dense and RhymeFlow share the same codepath
imports, so keeping the full acceleration environment installed is the safest
way to reproduce the reported numbers.

## Quick Start

Run one prompt and one method:

```bash
GPU_ID=0 PROMPT_ID=3 METHOD=rhyme bash scripts/wan/wan_t2v_case.sh
```

Available `METHOD` values:

| Method | Output dir | Default parameters |
|---|---|---|
| `dense` | `dense` | Full Wan2.1 denoising |
| `svg` | `svg_s03` | `sparsity=0.3`, `num_sampled_rows=64` |
| `sap` | `sap_default_q300_k800_tp092` | `q=300,k=800,top_p=0.92,min_kc_ratio=0.10,iter=50/2` |
| `rhyme` | `rhyme_tw10_m2_skip3-5` | `Tw=10,M=2,skip=3-5,semantic keyframes,scheduler_approx` |
| `rhyme_sap` | `rhyme_sap_tw8_m3_skip3-5_q350_k1200_tp098_min020_it5` | `Tw=8,M=3,skip=3-5` plus `q=350,k=1200,top_p=0.98,min_kc_ratio=0.20,iter=5/2` |

The RhymeFlow defaults preserve explicit keyframes. The open-source scripts do
not enable `--rhyme_no_keyframes`.

## Batch Reproduction

Run the default six unique prompts, excluding the duplicate prompt 6:

```bash
GPU_IDS="0 1 2 3" bash scripts/wan/wan_t2v_batch.sh
```

The default prompt list is:

```text
1 2 3 4 5 7
```

You can override the method set:

```bash
METHODS="dense sap rhyme rhyme_sap" GPU_IDS="0 1 2 3" bash scripts/wan/wan_t2v_batch.sh
```

## Evaluation

After generation, compute PSNR/SSIM/LPIPS against each prompt's Dense output:

```bash
METRIC_GPU=0 bash scripts/wan/wan_t2v_eval.sh
```

Metrics are Dense pseudo-reference metrics, not human ground-truth quality
scores. Higher PSNR/SSIM and lower LPIPS indicate the accelerated sample stayed
closer to the Dense sample for the same prompt and seed.

## Important Defaults

These defaults reflect the current Wan2.1 T2V reproduction state:

- SAP default follows the script-level default used in the experiments:
  `q=300,k=800,top_p=0.92,min_kc_ratio=0.10,iter=50/2`.
- SVG default uses `sparsity=0.3`.
- RhymeFlow default uses the keyframe-preserving balanced prompt-3 grid setting:
  `Tw10/M2/skip3-5`.
- RhymeFlow+SAP default uses the best keyframe-preserving prompt-3 setting found
  against SAP default: `Tw8/M3/skip3-5 + q350/k1200/top_p0.98/min0.20/iter5-2`.

See `docs/WAN_T2V_DEFAULTS.md` and the copied experiment notes under `docs/`
for the measured prompt-3 tradeoffs.

## Repository Layout

```text
wan_t2v_inference.py        Main Wan2.1 T2V inference entrypoint.
dataloader.py               Prompt loading helper.
svg/models/wan/             Wan sparse attention and RhymeFlow implementation.
svg/utils/                  Seed, metric, and keyframe utilities.
svg/kernels/triton/         Optional Triton helper kernels.
scripts/wan/                Reproduction launch/evaluation scripts.
scripts/eval/               Dense-reference metric scripts.
examples/*/prompt.txt       Example prompts.
docs/                       Reproduction notes and default parameter rationale.
```

## License

See `LICENSE.txt`.
