# Wan2.1 T2V Default Parameters

This document records the release defaults used by the Wan2.1 T2V scripts in
this repository.

## Generation Setup

| Field | Default |
|---|---|
| Model | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Resolution | 720x1280 |
| Frames | 81 |
| Steps | 50 |
| Seed | 0 |
| Guidance scale | 5.0 |
| Decode mode | offload transformer before VAE decode, VAE slicing, streaming CPU accumulation |

## Method Defaults

| Method | Parameters |
|---|---|
| Dense | No sparse attention or RhymeFlow step skipping. |
| SVG | `num_sampled_rows=64`, `sparsity=0.3`. |
| SAP | `num_q_centroids=300`, `num_k_centroids=800`, `top_p_kmeans=0.92`, `min_kc_ratio=0.10`, `kmeans_iter_init=50`, `kmeans_iter_step=2`. |
| RhymeFlow | `warmup_steps=10`, `num_keyframes=2`, `sss_schedule=progressive`, `sss_min_skip=3`, `sss_max_skip=5`, `sss_transition_points=0.3,0.7`, `keyframe_strategy=semantic`, `rhyme_context_mode=last_full_nonkey_cpu`, `rhyme_solver=scheduler_approx`, `rhyme_projection_space=sigma`, `rhyme_projection_mode=linear`. |
| RhymeFlow+SAP | RhymeFlow side: `warmup_steps=8`, `num_keyframes=3`, `sss_min_skip=3`, `sss_max_skip=5`, `semantic` keyframes, `last_full_nonkey_cpu`, `scheduler_approx`. SAP side: `q=350`, `k=1200`, `top_p=0.98`, `min_kc_ratio=0.20`, `iter=5/2`. |

## Rationale

The RhymeFlow default is the balanced keyframe-preserving result from the local
prompt-3 grid around `Tw8/M3/skip3-5`. In that grid, `Tw10/M2/skip3-5` reached
611.28 seconds, 1.628x speedup, 28.136 PSNR, 0.88374 SSIM, and 0.20772 LPIPS
against Dense.

The RhymeFlow+SAP default is the strongest verified keyframe-preserving setting
that beats the script-level SAP default on prompt 3 in speed and all three
Dense-reference visual metrics.

All RhymeFlow release defaults keep explicit keyframes. They do not use
`--rhyme_no_keyframes`.
