#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PROMPT_ID="${PROMPT_ID:?PROMPT_ID is required}"
METHOD="${METHOD:?METHOD is required: dense|svg|sap|rhyme|rhyme_sap}"
INFER_STEP="${INFER_STEP:-50}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-81}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
SEED="${SEED:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-10800}"
ROOT_BASE="${ROOT_BASE:-result/wan/t2v/Step_${INFER_STEP}-Res_${HEIGHT}x${WIDTH}-Frames_${NUM_FRAMES}}"

FIRST_TIMES_FP="${FIRST_TIMES_FP:-0.2}"
FIRST_LAYERS_FP="${FIRST_LAYERS_FP:-0.03}"

SVG_SPARSITY="${SVG_SPARSITY:-0.3}"

SAP_QC="${SAP_QC:-300}"
SAP_KC="${SAP_KC:-800}"
SAP_TOP_P="${SAP_TOP_P:-0.92}"
SAP_MIN_KC_RATIO="${SAP_MIN_KC_RATIO:-0.10}"
SAP_INIT_ITER="${SAP_INIT_ITER:-50}"
SAP_STEP_ITER="${SAP_STEP_ITER:-2}"

RHYME_WARMUP="${RHYME_WARMUP:-10}"
RHYME_KEYFRAMES="${RHYME_KEYFRAMES:-2}"
RHYME_SCHEDULE="${RHYME_SCHEDULE:-progressive}"
RHYME_MIN_SKIP="${RHYME_MIN_SKIP:-3}"
RHYME_MAX_SKIP="${RHYME_MAX_SKIP:-5}"
RHYME_TRANSITIONS="${RHYME_TRANSITIONS:-0.3,0.7}"
RHYME_KEYFRAME_STRATEGY="${RHYME_KEYFRAME_STRATEGY:-semantic}"
RHYME_CONTEXT_MODE="${RHYME_CONTEXT_MODE:-last_full_nonkey_cpu}"
RHYME_SOLVER="${RHYME_SOLVER:-scheduler_approx}"

RSAP_RHYME_WARMUP="${RSAP_RHYME_WARMUP:-8}"
RSAP_RHYME_KEYFRAMES="${RSAP_RHYME_KEYFRAMES:-3}"
RSAP_RHYME_MIN_SKIP="${RSAP_RHYME_MIN_SKIP:-3}"
RSAP_RHYME_MAX_SKIP="${RSAP_RHYME_MAX_SKIP:-5}"
RSAP_QC="${RSAP_QC:-350}"
RSAP_KC="${RSAP_KC:-1200}"
RSAP_TOP_P="${RSAP_TOP_P:-0.98}"
RSAP_MIN_KC_RATIO="${RSAP_MIN_KC_RATIO:-0.20}"
RSAP_INIT_ITER="${RSAP_INIT_ITER:-5}"
RSAP_STEP_ITER="${RSAP_STEP_ITER:-2}"

if [[ ! -f "examples/${PROMPT_ID}/prompt.txt" ]]; then
    echo "Prompt ${PROMPT_ID} missing: examples/${PROMPT_ID}/prompt.txt" >&2
    exit 2
fi

PROMPT="$(<"examples/${PROMPT_ID}/prompt.txt")"
ROOT_DIR="${ROOT_BASE}/prompt_${PROMPT_ID}_seed_${SEED}"

DECODE_ARGS=(--decode_after_cache_clear --offload_transformer_before_decode --vae_slicing --vae_stream_cpu --allow_decode_failure)
COMMON_ARGS=(
    --model_id "${MODEL_ID}"
    --prompt "${PROMPT}"
    --prompt_idx 0
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --num_frames "${NUM_FRAMES}"
    --seed "${SEED}"
    --num_inference_steps "${INFER_STEP}"
    --first_times_fp "${FIRST_TIMES_FP}"
    --first_layers_fp "${FIRST_LAYERS_FP}"
)
SVG_ARGS=(--num_sampled_rows 64 --sparsity "${SVG_SPARSITY}")
SAP_ARGS=(
    --num_q_centroids "${SAP_QC}"
    --num_k_centroids "${SAP_KC}"
    --top_p_kmeans "${SAP_TOP_P}"
    --min_kc_ratio "${SAP_MIN_KC_RATIO}"
    --kmeans_iter_init "${SAP_INIT_ITER}"
    --kmeans_iter_step "${SAP_STEP_ITER}"
)
RHYME_ARGS=(
    --warmup_steps "${RHYME_WARMUP}"
    --num_keyframes "${RHYME_KEYFRAMES}"
    --sss_schedule "${RHYME_SCHEDULE}"
    --sss_min_skip "${RHYME_MIN_SKIP}"
    --sss_max_skip "${RHYME_MAX_SKIP}"
    --sss_transition_points "${RHYME_TRANSITIONS}"
    --keyframe_strategy "${RHYME_KEYFRAME_STRATEGY}"
    --rhyme_projection_space sigma
    --rhyme_projection_mode linear
    --rhyme_context_mode "${RHYME_CONTEXT_MODE}"
    --rhyme_solver "${RHYME_SOLVER}"
)
RSAP_ARGS=(
    --num_q_centroids "${RSAP_QC}"
    --num_k_centroids "${RSAP_KC}"
    --top_p_kmeans "${RSAP_TOP_P}"
    --min_kc_ratio "${RSAP_MIN_KC_RATIO}"
    --kmeans_iter_init "${RSAP_INIT_ITER}"
    --kmeans_iter_step "${RSAP_STEP_ITER}"
    --warmup_steps "${RSAP_RHYME_WARMUP}"
    --num_keyframes "${RSAP_RHYME_KEYFRAMES}"
    --sss_schedule "${RHYME_SCHEDULE}"
    --sss_min_skip "${RSAP_RHYME_MIN_SKIP}"
    --sss_max_skip "${RSAP_RHYME_MAX_SKIP}"
    --sss_transition_points "${RHYME_TRANSITIONS}"
    --keyframe_strategy "${RHYME_KEYFRAME_STRATEGY}"
    --rhyme_projection_space sigma
    --rhyme_projection_mode linear
    --rhyme_context_mode "${RHYME_CONTEXT_MODE}"
    --rhyme_solver "${RHYME_SOLVER}"
)

case "${METHOD}" in
    dense)
        PATTERN="dense"
        OUT_NAME="${OUT_NAME:-dense}"
        EXTRA_ARGS=()
        ;;
    svg|svg_default)
        PATTERN="SVG"
        OUT_NAME="${OUT_NAME:-svg_s03}"
        EXTRA_ARGS=("${SVG_ARGS[@]}")
        ;;
    sap|sap_default|sap_recommended)
        PATTERN="SAP"
        OUT_NAME="${OUT_NAME:-sap_default_q300_k800_tp092}"
        EXTRA_ARGS=("${SAP_ARGS[@]}")
        ;;
    rhyme)
        PATTERN="RHYME"
        OUT_NAME="${OUT_NAME:-rhyme_tw10_m2_skip3-5}"
        EXTRA_ARGS=("${RHYME_ARGS[@]}")
        ;;
    rhyme_sap|rhyme_sap_recommended)
        PATTERN="RHYME_SAP"
        OUT_NAME="${OUT_NAME:-rhyme_sap_tw8_m3_skip3-5_q350_k1200_tp098_min020_it5}"
        EXTRA_ARGS=("${RSAP_ARGS[@]}")
        ;;
    *)
        echo "Unknown METHOD: ${METHOD}" >&2
        exit 2
        ;;
esac

OUT_DIR="${ROOT_DIR}/${OUT_NAME}"
mkdir -p "${OUT_DIR}"

if [[ "${SKIP_DONE:-1}" == "1" && -f "${OUT_DIR}/${PROMPT_ID}-0.mp4" && -f "${OUT_DIR}/summary.json" ]]; then
    echo "========== prompt ${PROMPT_ID} ${METHOD} skipped: existing outputs =========="
    exit 0
fi

echo "========== prompt ${PROMPT_ID} ${OUT_NAME} (${PATTERN}) =========="
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s timeout %s python wan_t2v_inference.py' "${GPU_ID}" "${TIMEOUT_SEC}"
    printf ' %q' \
        "${COMMON_ARGS[@]}" \
        --output_file "${OUT_DIR}/${PROMPT_ID}-0.mp4" \
        --logging_file "${OUT_DIR}/${PROMPT_ID}-0.jsonl" \
        --summary_file "${OUT_DIR}/summary.json" \
        --pattern "${PATTERN}" \
        "${DECODE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
    printf '\n'
    exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" timeout "${TIMEOUT_SEC}" python wan_t2v_inference.py \
    "${COMMON_ARGS[@]}" \
    --output_file "${OUT_DIR}/${PROMPT_ID}-0.mp4" \
    --logging_file "${OUT_DIR}/${PROMPT_ID}-0.jsonl" \
    --summary_file "${OUT_DIR}/summary.json" \
    --pattern "${PATTERN}" \
    "${DECODE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"
