#!/usr/bin/env bash
set -euo pipefail

PROMPT_IDS="${PROMPT_IDS:-1 2 3 4 5 7}"
INFER_STEP="${INFER_STEP:-50}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-81}"
SEED="${SEED:-0}"
METRIC_GPU="${METRIC_GPU:-0}"
ROOT_BASE="${ROOT_BASE:-result/wan/t2v/Step_${INFER_STEP}-Res_${HEIGHT}x${WIDTH}-Frames_${NUM_FRAMES}}"
METHOD_DIRS="${METHOD_DIRS:-svg_s03 sap_default_q300_k800_tp092 rhyme_tw10_m2_skip3-5 rhyme_sap_tw8_m3_skip3-5_q350_k1200_tp098_min020_it5}"

read -r -a PROMPT_IDS_ARR <<< "${PROMPT_IDS}"
read -r -a METHOD_DIRS_ARR <<< "${METHOD_DIRS}"

for prompt_id in "${PROMPT_IDS_ARR[@]}"; do
    root="${ROOT_BASE}/prompt_${prompt_id}_seed_${SEED}"
    dense="${root}/dense/${prompt_id}-0.mp4"
    if [[ ! -f "${dense}" ]]; then
        echo "Skip metrics for prompt ${prompt_id}: missing ${dense}" >&2
        continue
    fi

    args=(
        --root "${root}"
        --dense "${dense}"
        --video_name "${prompt_id}-0.mp4"
        --device cuda
        --batch_size 1
        --lpips_batch_size 1
    )
    for method_dir in "${METHOD_DIRS_ARR[@]}"; do
        args+=(--method "${method_dir}")
    done

    CUDA_VISIBLE_DEVICES="${METRIC_GPU}" python scripts/eval/compute_video_metrics_vs_dense.py "${args[@]}"
done

python scripts/eval/aggregate_wan_t2v_comparison.py \
    --root "${ROOT_BASE}" \
    --prompt_ids "${PROMPT_IDS_ARR[@]}"
