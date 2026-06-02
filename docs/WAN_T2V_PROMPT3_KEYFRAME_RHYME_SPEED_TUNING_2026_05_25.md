# Wan T2V Prompt 3 Keyframe-Preserving Rhyme Speed Tuning

Date: 2026-05-25

Scope: prompt 3 only, Wan2.1-T2V-1.3B-Diffusers, 720x1280, 81 frames, 50 denoise steps, seed 0. Metrics are computed against the Dense video in `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/dense/3-0.mp4`.

Important constraint: this round preserves the original Rhyme structure with explicit keyframes. It does not use `--rhyme_no_keyframes`.

## Fixed Baselines

| method | setting | time (s) | speedup | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 50-step | 995.12 | 1.000x | inf | 1.00000 | 0.00000 |
| SAP | q300/k800/top_p0.92/min0.10 | 660.18 | 1.507x | 25.3449 | 0.83602 | 0.25317 |
| SAP | q300/k1000/top_p0.90/min0.10 | 639.95 | 1.555x | 25.5962 | 0.84146 | 0.24339 |
| Rhyme | Tw10/M5/skip2-3 | 754.42 | 1.319x | 28.9836 | 0.89932 | 0.18569 |
| Rhyme | Tw15/M7/skip2-3 | 812.89 | 1.224x | 30.0532 | 0.91620 | 0.16572 |
| Rhyme+SAP | Tw15/M7/skip2-3/q300/k800/top_p0.92/min0.10 | 666.26 | 1.494x | 25.3428 | 0.82929 | 0.26585 |

## Plan

The Table-1 style grid shows quality redundancy: keyframe-enabled Rhyme is much better than SAP on prompt 3, but slower. The first tuning axis is therefore to reduce full/active Rhyme calls without removing keyframes:

- lower `M` from 5/7 to 3 or keep 5 as a conservative anchor;
- lower `Tw` from 10/15 to 5 or 8;
- increase progressive skip from `2-3` to `2-4` or `3-5`.

First wave Rhyme candidates:

| name | Tw | M | skip |
|---|---:|---:|---:|
| `rhyme_kf_tw8_m3_skip2-4` | 8 | 3 | 2-4 |
| `rhyme_kf_tw8_m3_skip3-5` | 8 | 3 | 3-5 |
| `rhyme_kf_tw5_m3_skip3-5` | 5 | 3 | 3-5 |
| `rhyme_kf_tw10_m3_skip3-5` | 10 | 3 | 3-5 |
| `rhyme_kf_tw10_m5_skip3-5` | 10 | 5 | 3-5 |
| `rhyme_kf_tw8_m5_skip3-5` | 8 | 5 | 3-5 |

Selection rule for the next Rhyme+SAP test: prefer a setting that materially improves Rhyme speed over Tw10/M5/skip2-3 while keeping visual metrics clearly better than SAP. If multiple settings satisfy this, choose the fastest one.

## Running Commands

All commands use:

```bash
ROOT_BASE=result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81
PROMPT_ID=3
INFER_STEP=50
HEIGHT=720
WIDTH=1280
NUM_FRAMES=81
SEED=0
RHYME_CONTEXT_MODE=last_full_nonkey_cpu
RHYME_SOLVER=scheduler_approx
```

## Results

Metric artifacts:

- `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/prompt3_keyframe_rhyme_speed_tuning_rhyme.csv`
- `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/prompt3_keyframe_rhyme_sap_tuning_compare.csv`
- `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/prompt3_keyframe_rhyme_sap_refine2_compare.csv`
- `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/prompt3_keyframe_rhyme_sap_iter_tuning_compare.csv`
- `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/prompt3_keyframe_rhyme_sap_boundary_compare.csv`

### Rhyme-Only Keyframe-Preserving Sweep

All rows below use keyframes. No row uses `--rhyme_no_keyframes`.

| method | keyframes | full / active | time (s) | speedup | PSNR | SSIM | LPIPS | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SAP default | n/a | n/a | 660.18 | 1.507x | 25.3449 | 0.83602 | 0.25317 | q300/k800/tp0.92/min0.10 |
| SAP paper-like | n/a | n/a | 639.95 | 1.555x | 25.5962 | 0.84146 | 0.24339 | q300/k1000/tp0.90/min0.10 |
| Rhyme Table1 fastest | [0,1,10,11,20] | 26 / 24 | 754.42 | 1.319x | 28.9836 | 0.89932 | 0.18569 | Tw10/M5/skip2-3 |
| Rhyme | [0,1,20] | 24 / 26 | 688.13 | 1.446x | 28.6771 | 0.90003 | 0.18785 | Tw8/M3/skip2-4 |
| Rhyme | [0,1,20] | 20 / 30 | 624.02 | 1.595x | 27.3316 | 0.87387 | 0.21794 | Tw8/M3/skip3-5 |
| Rhyme | [0,1,20] | 17 / 33 | 573.10 | 1.736x | 23.6918 | 0.77837 | 0.29327 | Tw5/M3/skip3-5, too aggressive |
| Rhyme | [0,1,20] | 21 / 29 | 639.48 | 1.556x | 28.0835 | 0.88321 | 0.20720 | Tw10/M3/skip3-5 |
| Rhyme | [0,1,10,11,20] | 21 / 29 | 656.12 | 1.517x | 28.1137 | 0.88489 | 0.20431 | Tw10/M5/skip3-5 |
| Rhyme | [0,1,9,11,20] | 20 / 30 | 657.59 | 1.513x | 27.3911 | 0.87536 | 0.21576 | Tw8/M5/skip3-5 |

Observation: `Tw8/M3/skip3-5` is the best speed-quality Rhyme setting on prompt 3. It preserves explicit keyframes `[0,1,20]`, improves speed from the Table-1 fastest Rhyme `1.319x` to `1.595x`, and still beats both SAP baselines on PSNR/SSIM/LPIPS.

### Rhyme+SAP Tuning Around Tw8/M3/skip3-5

All rows below keep `Tw8/M3/skip3-5`, keyframes `[0,1,20]`, `last_full_nonkey_cpu`, and `scheduler_approx`.

Selected boundary rows:

| method | SAP setting | time (s) | speedup | PSNR | SSIM | LPIPS | conclusion |
|---|---|---:|---:|---:|---:|---:|---|
| SAP default | q300/k800/tp0.92/min0.10/it50-2 | 660.18 | 1.507x | 25.3449 | 0.83602 | 0.25317 | baseline |
| SAP paper-like | q300/k1000/tp0.90/min0.10/it50-2 | 639.95 | 1.555x | 25.5962 | 0.84146 | 0.24339 | stronger SAP baseline |
| Rhyme+SAP | q300/k800/tp0.92/min0.10/it50-2 | 545.69 | 1.824x | 25.3213 | 0.81983 | 0.28001 | fast, quality below SAP |
| Rhyme+SAP | q350/k1200/tp0.98/min0.20/it5-2 | 656.68 | 1.515x | 25.6789 | 0.84283 | 0.25059 | beats SAP default on speed and all 3 metrics |
| Rhyme+SAP | q400/k1000/tp0.98/min0.20/it5-2 | 663.73 | 1.499x | 25.8321 | 0.84984 | 0.24352 | quality strong, slightly slower than SAP default |
| Rhyme+SAP | q400/k1200/tp0.98/min0.20/it5-2 | 662.11 | 1.503x | 25.8656 | 0.84944 | 0.24355 | nearly crosses speed line; quality strong |
| Rhyme+SAP | q400/k1200/tp0.98/min0.20/it50-2 | 670.17 | 1.485x | 25.9646 | 0.85311 | 0.24010 | beats both SAPs on quality, slower |
| Rhyme+SAP | q450/k1100/tp0.98/min0.20/it5-2 | 678.71 | 1.466x | 25.9869 | 0.84944 | 0.24320 | beats both SAPs on quality, slower |

Interpretation:

- Against the script-level SAP default (`q300/k800/tp0.92/min0.10/it50-2`), `Rhyme+SAP q350/k1200/tp0.98/min0.20/it5-2` is the current best prompt-3 setting: speed `1.515x` vs `1.507x`, PSNR `25.6789` vs `25.3449`, SSIM `0.84283` vs `0.83602`, LPIPS `0.25059` vs `0.25317`.
- Against the stronger paper-like SAP (`q300/k1000/tp0.90/min0.10/it50-2`), I did not find a Rhyme+SAP setting that beats both speed and all quality metrics. The best quality settings beat it on PSNR/SSIM/LPIPS but run at `1.46x-1.50x`, below SAP's `1.555x`.
- Rhyme-only `Tw8/M3/skip3-5` is stronger than Rhyme+SAP on this prompt if we require speed and Dense-reference visual metrics simultaneously: `1.595x`, PSNR `27.3316`, SSIM `0.87387`, LPIPS `0.21794`.

### Recommended Prompt-3 Settings

Rhyme-only:

```bash
GPU_ID=<gpu> PROMPT_ID=3 METHOD=rhyme OUT_NAME=rhyme_kf_tw8_m3_skip3-5 \
INFER_STEP=50 HEIGHT=720 WIDTH=1280 NUM_FRAMES=81 \
ROOT_BASE=result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81 \
TIMEOUT_SEC=10800 RHYME_WARMUP=8 RHYME_KEYFRAMES=3 \
RHYME_MIN_SKIP=3 RHYME_MAX_SKIP=5 \
RHYME_CONTEXT_MODE=last_full_nonkey_cpu RHYME_SOLVER=scheduler_approx \
bash scripts/wan/wan_t2v_10s_baseline_case.sh
```

Rhyme+SAP if comparing to SAP default:

```bash
GPU_ID=<gpu> PROMPT_ID=3 METHOD=rhyme_sap_recommended \
OUT_NAME=rsap_kf_tw8_m3_skip3-5_q350_k1200_tp098_min020_it5 \
INFER_STEP=50 HEIGHT=720 WIDTH=1280 NUM_FRAMES=81 \
ROOT_BASE=result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81 \
TIMEOUT_SEC=10800 RHYME_WARMUP=8 RHYME_KEYFRAMES=3 \
RHYME_MIN_SKIP=3 RHYME_MAX_SKIP=5 \
RHYME_CONTEXT_MODE=last_full_nonkey_cpu RHYME_SOLVER=scheduler_approx \
SAP_QC=350 SAP_KC=1200 SAP_TOP_P=0.98 SAP_MIN_KC_RATIO=0.20 \
SAP_INIT_ITER=5 SAP_STEP_ITER=2 \
bash scripts/wan/wan_t2v_10s_baseline_case.sh
```

Representative videos:

- Rhyme-only best: `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/rhyme_kf_tw8_m3_skip3-5/3-0.mp4`
- Rhyme+SAP default-beating setting: `result/wan/t2v/paper50_baseline_comparison/Step_50-Res_720x1280-Frames_81/prompt_3_seed_0/rsap_kf_tw8_m3_skip3-5_q350_k1200_tp098_min020_it5/3-0.mp4`
