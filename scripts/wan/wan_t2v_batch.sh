#!/usr/bin/env bash
set -uo pipefail

# examples/5 and examples/6 are identical in the source checkout; prompt 7 is
# used as the sixth unique prompt by default.
PROMPT_IDS="${PROMPT_IDS:-1 2 3 4 5 7}"
METHODS="${METHODS:-dense svg sap rhyme rhyme_sap}"
GPU_IDS="${GPU_IDS:-0}"
INFER_STEP="${INFER_STEP:-50}"
HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-81}"
ROOT_BASE="${ROOT_BASE:-result/wan/t2v/Step_${INFER_STEP}-Res_${HEIGHT}x${WIDTH}-Frames_${NUM_FRAMES}}"

mkdir -p "${ROOT_BASE}/batch_logs"

read -r -a FREE_GPUS <<< "${GPU_IDS}"
declare -A PID_TO_GPU
declare -A PID_TO_DESC
FAILED=0
RUNNING=0

launch_case() {
    local prompt_id="$1"
    local method="$2"
    local gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    local desc="prompt_${prompt_id}_${method}"
    local log_file="${ROOT_BASE}/batch_logs/${desc}.log"

    echo "launch ${desc} on GPU ${gpu_id}"
    GPU_ID="${gpu_id}" PROMPT_ID="${prompt_id}" METHOD="${method}" \
        INFER_STEP="${INFER_STEP}" HEIGHT="${HEIGHT}" WIDTH="${WIDTH}" NUM_FRAMES="${NUM_FRAMES}" ROOT_BASE="${ROOT_BASE}" \
        bash scripts/wan/wan_t2v_case.sh > "${log_file}" 2>&1 &
    local pid=$!
    PID_TO_GPU["${pid}"]="${gpu_id}"
    PID_TO_DESC["${pid}"]="${desc}"
    RUNNING=$((RUNNING + 1))
}

wait_one() {
    local done_pid
    wait -n -p done_pid
    local rc=$?
    if [[ -z "${done_pid:-}" ]]; then
        return 0
    fi

    local gpu_id="${PID_TO_GPU[${done_pid}]}"
    local desc="${PID_TO_DESC[${done_pid}]}"
    FREE_GPUS+=("${gpu_id}")
    unset "PID_TO_GPU[${done_pid}]"
    unset "PID_TO_DESC[${done_pid}]"
    RUNNING=$((RUNNING - 1))

    if [[ "${rc}" -ne 0 ]]; then
        echo "done ${desc} on GPU ${gpu_id}: failed rc=${rc}"
        FAILED=$((FAILED + 1))
    else
        echo "done ${desc} on GPU ${gpu_id}: ok"
    fi
}

for prompt_id in ${PROMPT_IDS}; do
    for method in ${METHODS}; do
        while [[ "${#FREE_GPUS[@]}" -eq 0 ]]; do
            wait_one
        done
        launch_case "${prompt_id}" "${method}"
    done
done

while [[ "${RUNNING}" -gt 0 ]]; do
    wait_one
done

if [[ "${FAILED}" -ne 0 ]]; then
    echo "Batch finished with ${FAILED} failed case(s)." >&2
    exit 1
fi

echo "Batch finished successfully."
