#!/usr/bin/env bash
# Two-stage Go2 MoE locomotion training pipeline:
#   1. Pretrain flat / rough / stairs / climb experts.
#   2. Load expert checkpoints into an MoE actor and train the gate on mixed terrain.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

NUM_ENVS=${NUM_ENVS:-2048}
GPU_IDS=${GPU_IDS:-0,1,2,3}
EXPERT_ITERS=${EXPERT_ITERS:-4000}
MOE_ITERS=${MOE_ITERS:-3000}
SKIP_EXPERTS=${SKIP_EXPERTS:-0}
PARALLEL=${PARALLEL:-0}
FOREGROUND=${FOREGROUND:-0}
PYTHON=${PYTHON:-python}

LOG_ROOT="logs/rsl_rl"

if [[ "${MOE_PIPELINE_INNER:-0}" != "1" && "${FOREGROUND}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found. Re-run with FOREGROUND=1 to run in this shell." >&2
    exit 1
  fi
  TS="$(date +%Y-%m-%d_%H-%M-%S)"
  SESSION="go2_moe_${TS}"
  LOG="${SCRIPT_DIR}/logs/moe_pipeline_${TS}.log"
  mkdir -p "${SCRIPT_DIR}/logs"
  INNER="cd '${SCRIPT_DIR}' && MOE_PIPELINE_INNER=1 NUM_ENVS='${NUM_ENVS}'"
  INNER+=" GPU_IDS='${GPU_IDS}' EXPERT_ITERS='${EXPERT_ITERS}' MOE_ITERS='${MOE_ITERS}'"
  INNER+=" SKIP_EXPERTS='${SKIP_EXPERTS}' PARALLEL='${PARALLEL}' PYTHON='${PYTHON}'"
  INNER+=" bash scripts/train_moe.sh 2>&1 | tee '${LOG}'"
  tmux new-session -d -s "${SESSION}" "${INNER}"
  echo "[INFO] MoE pipeline launched in detached tmux session: ${SESSION}"
  echo "[INFO]   GPUs       : ${GPU_IDS}"
  echo "[INFO]   iters      : experts=${EXPERT_ITERS} moe=${MOE_ITERS}"
  echo "[INFO]   attach     : tmux attach -t ${SESSION}"
  echo "[INFO]   follow log : tail -f ${LOG}"
  echo "[INFO]   stop       : tmux kill-session -t ${SESSION}"
  exit 0
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
FIRST_GPU="${GPU_LIST[0]}"

run_stage() {
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
  local label="$1" task="$2" experiment="$3" gpu="$4" out_file="$5"
  local run_name
  run_name="$(run_stage "${task}" "${EXPERT_ITERS}" "${gpu}")"
  printf '%s\n' "${run_name}" > "${out_file}"
  echo ">>> Expert ${label} complete: ${LOG_ROOT}/${experiment}/${run_name}" >&2
}

declare -A TASKS=(
  [flat]="Unitree-Go2-Expert-Flat"
  [rough]="Unitree-Go2-Expert-Rough"
  [stairs]="Unitree-Go2-Expert-Stairs"
  [climb]="Unitree-Go2-Expert-Climb"
)
declare -A EXPS=(
  [flat]="go2_expert_flat"
  [rough]="go2_expert_rough"
  [stairs]="go2_expert_stairs"
  [climb]="go2_expert_climb"
)
EXPERT_ORDER=(flat rough stairs climb)

if [[ "${SKIP_EXPERTS}" == "0" ]]; then
  if [[ "${PARALLEL}" == "1" ]]; then
    tmp_dir="$(mktemp -d)"
    pids=()
    for i in "${!EXPERT_ORDER[@]}"; do
      label="${EXPERT_ORDER[$i]}"
      gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
      run_expert "${label}" "${TASKS[$label]}" "${EXPS[$label]}" "${gpu}" "${tmp_dir}/${label}.run" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  else
    for i in "${!EXPERT_ORDER[@]}"; do
      label="${EXPERT_ORDER[$i]}"
      gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
      run_stage "${TASKS[$label]}" "${EXPERT_ITERS}" "${gpu}" >/dev/null
    done
  fi
else
  echo ">>> SKIP_EXPERTS=1, reusing newest checkpoints under ${LOG_ROOT}" >&2
fi

export GO2_MOE_EXPERT_FLAT
export GO2_MOE_EXPERT_ROUGH
export GO2_MOE_EXPERT_STAIRS
export GO2_MOE_EXPERT_CLIMB
GO2_MOE_EXPERT_FLAT="$(latest_checkpoint "${EXPS[flat]}")"
GO2_MOE_EXPERT_ROUGH="$(latest_checkpoint "${EXPS[rough]}")"
GO2_MOE_EXPERT_STAIRS="$(latest_checkpoint "${EXPS[stairs]}")"
GO2_MOE_EXPERT_CLIMB="$(latest_checkpoint "${EXPS[climb]}")"

echo ">>> Expert checkpoints:" >&2
echo "  flat  : ${GO2_MOE_EXPERT_FLAT}" >&2
echo "  rough : ${GO2_MOE_EXPERT_ROUGH}" >&2
echo "  stairs: ${GO2_MOE_EXPERT_STAIRS}" >&2
echo "  climb : ${GO2_MOE_EXPERT_CLIMB}" >&2

MOE_RUN="$(run_stage Unitree-Go2-MoE-Mixed "${MOE_ITERS}" "${FIRST_GPU}")"

echo ""
echo "MoE pipeline complete."
echo "  moe: ${LOG_ROOT}/go2_moe_mixed/${MOE_RUN}"
