#!/usr/bin/env bash
# Two-stage no-height-scan Go2W MoE training pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

NUM_ENVS=${NUM_ENVS:-2048}
GPU_IDS=${GPU_IDS:-0,1,2}
EXPERT_ITERS=${EXPERT_ITERS:-4000}
MOE_ITERS=${MOE_ITERS:-3000}
SKIP_EXPERTS=${SKIP_EXPERTS:-0}
PARALLEL=${PARALLEL:-1}
FOREGROUND=${FOREGROUND:-0}
PYTHON=${PYTHON:-python}

LOG_ROOT="logs/rsl_rl"

if [[ "${GO2W_MOE_NOHEIGHT_PIPELINE_INNER:-0}" != "1" && "${FOREGROUND}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found. Re-run with FOREGROUND=1 to run in this shell." >&2
    exit 1
  fi
  TS="$(date +%Y-%m-%d_%H-%M-%S)"
  SESSION="go2w_moe_noheight_${TS}"
  LOG="${SCRIPT_DIR}/logs/go2w_moe_noheight_pipeline_${TS}.log"
  mkdir -p "${SCRIPT_DIR}/logs"
  INNER="cd '${SCRIPT_DIR}' && GO2W_MOE_NOHEIGHT_PIPELINE_INNER=1 NUM_ENVS='${NUM_ENVS}'"
  INNER+=" GPU_IDS='${GPU_IDS}' EXPERT_ITERS='${EXPERT_ITERS}' MOE_ITERS='${MOE_ITERS}'"
  INNER+=" SKIP_EXPERTS='${SKIP_EXPERTS}' PARALLEL='${PARALLEL}' PYTHON='${PYTHON}'"
  INNER+=" bash scripts/train_go2w_moe_no_height.sh 2>&1 | tee '${LOG}'"
  tmux new-session -d -s "${SESSION}" "${INNER}"
  echo "[INFO] Go2W no-height MoE pipeline launched in detached tmux session: ${SESSION}"
  echo "[INFO]   GPUs       : ${GPU_IDS}"
  echo "[INFO]   iters      : experts=${EXPERT_ITERS} moe=${MOE_ITERS}"
  echo "[INFO]   attach     : tmux attach -t ${SESSION}"
  echo "[INFO]   follow log : tail -f ${LOG}"
  echo "[INFO]   stop       : tmux kill-session -t ${SESSION}"
  exit 0
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
FIRST_GPU="${GPU_LIST[0]}"

run_stage_single() {
  local task="$1" iters="$2" gpu="$3"
  export CUDA_VISIBLE_DEVICES="${gpu}"
  echo ">>> Stage: ${task} (${iters} iters, GPU ${gpu})" >&2
  "${PYTHON}" scripts/train.py "${task}" \
    --env.scene.num-envs "${NUM_ENVS}" \
    --agent.max-iterations "${iters}" \
    --agent.run-name "${task}" \
    2>&1 | tee /dev/stderr \
    | grep -oP 'Logging experiment in directory: \K.*' | head -1 | xargs basename
}

run_stage_multi() {
  local task="$1" iters="$2"
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  echo ">>> Stage: ${task} (${iters} iters, GPUs ${GPU_IDS})" >&2
  "${PYTHON}" scripts/train.py "${task}" \
    --gpu-ids all \
    --env.scene.num-envs "${NUM_ENVS}" \
    --agent.max-iterations "${iters}" \
    --agent.run-name "${task}" \
    2>&1 | tee /dev/stderr \
    | grep -oP 'Logging experiment in directory: \K.*' | head -1 | xargs basename
}

latest_checkpoint() {
  local experiment="$1"
  local checkpoint
  checkpoint="$(find "${LOG_ROOT}/${experiment}" -type f -name 'model_*.pt' 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -z "${checkpoint}" ]]; then
    echo "[ERROR] No checkpoint found under ${LOG_ROOT}/${experiment}" >&2
    exit 1
  fi
  printf '%s\n' "${checkpoint}"
}

run_expert() {
  local label="$1" task="$2" experiment="$3" gpu="$4"
  local run_name
  run_name="$(run_stage_single "${task}" "${EXPERT_ITERS}" "${gpu}")"
  echo ">>> Expert ${label} complete: ${LOG_ROOT}/${experiment}/${run_name}" >&2
}

declare -A TASKS=(
  [flat]="Unitree-Go2W-NoHeight-Expert-Flat"
  [rough]="Unitree-Go2W-NoHeight-Expert-Rough"
  [stairs]="Unitree-Go2W-NoHeight-Expert-Stairs"
  [climb]="Unitree-Go2W-NoHeight-Expert-Climb"
)
declare -A EXPS=(
  [flat]="go2w_noheight_expert_flat"
  [rough]="go2w_noheight_expert_rough"
  [stairs]="go2w_noheight_expert_stairs"
  [climb]="go2w_noheight_expert_climb"
)
EXPERT_ORDER=(flat rough stairs climb)

if [[ "${SKIP_EXPERTS}" == "0" ]]; then
  if [[ "${PARALLEL}" == "1" ]]; then
    for ((start = 0; start < ${#EXPERT_ORDER[@]}; start += ${#GPU_LIST[@]})); do
      pids=()
      for ((slot = 0; slot < ${#GPU_LIST[@]} && start + slot < ${#EXPERT_ORDER[@]}; slot++)); do
        label="${EXPERT_ORDER[$((start + slot))]}"
        gpu="${GPU_LIST[$slot]}"
        run_expert "${label}" "${TASKS[$label]}" "${EXPS[$label]}" "${gpu}" &
        pids+=("$!")
      done
      for pid in "${pids[@]}"; do
        wait "${pid}"
      done
    done
  else
    for i in "${!EXPERT_ORDER[@]}"; do
      label="${EXPERT_ORDER[$i]}"
      gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
      run_stage_single "${TASKS[$label]}" "${EXPERT_ITERS}" "${gpu}" >/dev/null
    done
  fi
else
  echo ">>> SKIP_EXPERTS=1, reusing newest checkpoints under ${LOG_ROOT}" >&2
fi

export GO2W_MOE_NOHEIGHT_EXPERT_FLAT
export GO2W_MOE_NOHEIGHT_EXPERT_ROUGH
export GO2W_MOE_NOHEIGHT_EXPERT_STAIRS
export GO2W_MOE_NOHEIGHT_EXPERT_CLIMB
GO2W_MOE_NOHEIGHT_EXPERT_FLAT="$(latest_checkpoint "${EXPS[flat]}")"
GO2W_MOE_NOHEIGHT_EXPERT_ROUGH="$(latest_checkpoint "${EXPS[rough]}")"
GO2W_MOE_NOHEIGHT_EXPERT_STAIRS="$(latest_checkpoint "${EXPS[stairs]}")"
GO2W_MOE_NOHEIGHT_EXPERT_CLIMB="$(latest_checkpoint "${EXPS[climb]}")"

echo ">>> Expert checkpoints:" >&2
echo "  flat  : ${GO2W_MOE_NOHEIGHT_EXPERT_FLAT}" >&2
echo "  rough : ${GO2W_MOE_NOHEIGHT_EXPERT_ROUGH}" >&2
echo "  stairs: ${GO2W_MOE_NOHEIGHT_EXPERT_STAIRS}" >&2
echo "  climb : ${GO2W_MOE_NOHEIGHT_EXPERT_CLIMB}" >&2

if [[ "${GPU_IDS}" == *,* ]]; then
  MOE_RUN="$(run_stage_multi Unitree-Go2W-NoHeight-MoE-Mixed "${MOE_ITERS}")"
else
  MOE_RUN="$(run_stage_single Unitree-Go2W-NoHeight-MoE-Mixed "${MOE_ITERS}" "${FIRST_GPU}")"
fi

echo ""
echo "Go2W no-height MoE pipeline complete."
echo "  moe: ${LOG_ROOT}/go2w_noheight_moe_mixed/${MOE_RUN}"
