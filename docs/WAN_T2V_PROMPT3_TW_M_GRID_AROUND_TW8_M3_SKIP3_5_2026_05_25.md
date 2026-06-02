# Wan T2V Prompt 3 Tw/M Grid Around Tw8/M3/skip3-5

Date: 2026-05-25

Scope: prompt 3 only, Wan2.1-T2V-1.3B-Diffusers, 720x1280, 81 frames, 50 denoise steps, seed 0. Metrics are computed against the Dense video in `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/dense/3-0.mp4`.

Constraint: preserve explicit Rhyme keyframes. No `--rhyme_no_keyframes` runs are included.

## Grid

Fixed settings:

- method: `RHYME`
- skip: `sss_min_skip=3`, `sss_max_skip=5`
- context: `last_full_nonkey_cpu`
- solver: `scheduler_approx`
- projection: `sigma` / `linear`

Table-1-style local grid around the current prompt-3 setting `Tw8/M3/skip3-5`:

| axis | values |
|---|---|
| `Tw` | 6, 8, 10 |
| `M` | 2, 3, 4, 5 |

Already available and reused before this run:

- `Tw8/M3`
- `Tw8/M5`
- `Tw10/M3`
- `Tw10/M5`

New runs required:

- `Tw6/M2`, `Tw6/M3`, `Tw6/M4`, `Tw6/M5`
- `Tw8/M2`, `Tw8/M4`
- `Tw10/M2`, `Tw10/M4`

## Results

Metric artifacts:

- CSV: `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/prompt3_tw_m_grid_around_tw8_m3_skip3_5.csv`
- JSON: `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/prompt3_tw_m_grid_around_tw8_m3_skip3_5.json`
- Auto markdown: `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/prompt3_tw_m_grid_around_tw8_m3_skip3_5.md`

Timing note: `Tw10/M2` and `Tw10/M4` were first launched together and showed CPU offload contention. I reran both sequentially on one free GPU and the table below uses the sequential summaries.

### Baselines

| Method | Total(s) | Speedup | PSNR(dB) | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|
| Dense | 995.12 | 1.000 | 120.000 | 1.00000 | 0.00000 |
| SAP default q300/k800/tp0.92/min0.10 | 660.18 | 1.507 | 25.345 | 0.83602 | 0.25317 |
| SAP q300/k1000/tp0.90/min0.10 | 639.95 | 1.555 | 25.596 | 0.84146 | 0.24339 |

### Rhyme Tw/M Grid, skip3-5

| Tw | M | Keyframes | Full/Active steps | Total(s) | Speedup | PSNR(dB) | SSIM | LPIPS |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | 2 | [0, 20] | 18/32 | 565.35 | 1.760 | 23.911 | 0.80175 | 0.27108 |
| 6 | 3 | [0, 1, 20] | 18/32 | 591.47 | 1.682 | 23.932 | 0.80166 | 0.27054 |
| 6 | 4 | [0, 1, 9, 20] | 18/32 | 595.26 | 1.672 | 23.979 | 0.80365 | 0.26782 |
| 6 | 5 | [0, 1, 8, 9, 20] | 18/32 | 627.68 | 1.585 | 24.074 | 0.80677 | 0.26560 |
| 8 | 2 | [0, 20] | 20/30 | 598.58 | 1.662 | 27.317 | 0.87267 | 0.21991 |
| 8 | 3 | [0, 1, 20] | 20/30 | 624.02 | 1.595 | 27.332 | 0.87387 | 0.21794 |
| 8 | 4 | [0, 1, 11, 20] | 20/30 | 624.39 | 1.594 | 27.408 | 0.87497 | 0.21641 |
| 8 | 5 | [0, 1, 9, 11, 20] | 20/30 | 657.59 | 1.513 | 27.391 | 0.87536 | 0.21576 |
| 10 | 2 | [0, 20] | 21/29 | 611.28 | 1.628 | 28.136 | 0.88374 | 0.20772 |
| 10 | 3 | [0, 1, 20] | 21/29 | 639.48 | 1.556 | 28.084 | 0.88321 | 0.20720 |
| 10 | 4 | [0, 1, 11, 20] | 21/29 | 640.14 | 1.555 | 28.186 | 0.88548 | 0.20429 |
| 10 | 5 | [0, 1, 10, 11, 20] | 21/29 | 656.12 | 1.517 | 28.114 | 0.88489 | 0.20431 |

### Speedup / PSNR Matrix

Each cell is `speedup / PSNR`.

| Tw \ M | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|
| 6 | 1.760 / 23.911 | 1.682 / 23.932 | 1.672 / 23.979 | 1.585 / 24.074 |
| 8 | 1.662 / 27.317 | 1.595 / 27.332 | 1.594 / 27.408 | 1.513 / 27.391 |
| 10 | 1.628 / 28.136 | 1.556 / 28.084 | 1.555 / 28.186 | 1.517 / 28.114 |

## Observations

- `Tw6` is the fastest band, but the visual-closeness metrics are poor on prompt 3. Even `Tw6/M5` only reaches 24.074 dB PSNR and remains below both SAP baselines.
- Moving from `Tw6` to `Tw8` is the main quality jump: PSNR rises from about 24 dB to about 27.3-27.4 dB while speedup remains 1.51x-1.66x.
- Moving from `Tw8` to `Tw10` gives another about 0.7-0.8 dB PSNR and lower LPIPS, with moderate speed loss. `Tw10/M2` is the strongest speed-quality tradeoff in this grid.
- Increasing `M` does not monotonically improve PSNR on this prompt. It slightly improves SSIM/LPIPS in some rows, but the speed cost is clear. `M2` is especially competitive here because prompt 3 appears to tolerate only endpoint keyframes once `Tw` is high enough.
- All included Rhyme runs have `rhymeflow_no_keyframes=False` in their summaries; no no-keyframe Rhyme shortcut is included.

## Takeaway

For prompt 3 around `Tw8/M3/skip3-5`, the useful settings are:

- Best speed while clearly beating SAP quality: `Tw8/M2/skip3-5`, 598.58s, 1.662x, PSNR 27.317, SSIM 0.87267, LPIPS 0.21991.
- Best balanced setting in this local grid: `Tw10/M2/skip3-5`, 611.28s, 1.628x, PSNR 28.136, SSIM 0.88374, LPIPS 0.20772.
- Best quality in this grid: `Tw10/M4/skip3-5`, 640.14s, 1.555x, PSNR 28.186, SSIM 0.88548, LPIPS 0.20429.

Compared with SAP q300/k1000/tp0.90/min0.10, both `Tw8/M2` and `Tw10/M2` dominate on prompt 3 in speedup and all quality metrics. `Tw10/M4` is essentially tied with SAP-tuned speedup while much closer to Dense.
