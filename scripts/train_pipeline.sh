#!/usr/bin/env bash
# Staged curriculum training pipeline: flat -> rough -> test.
#
# Each stage resumes the previous stage's checkpoint, so the policy is never
# trained on the extreme test terrain from scratch. All three configs keep the
# `height_scan` observation, so the observation space matches across stages and
# checkpoints load with strict=True.
#
#   stage 1  Unitree-Go2-Flat-Scan   (fresh)        -- multi-GPU
#   stage 2  Unitree-Go2-Rough       (resume st.1)  -- multi-GPU
#   stage 3  Unitree-Go2-Test-Train  (resume st.2)  -- SINGLE GPU (see note)
#
# By default this launches the pipeline in a detached tmux session and returns
# immediately (like scripts/run.sh). It prints how to attach / tail the log / stop it.
#
# NOTE on stage 3: multi-GPU training on the extreme test terrain is currently
# unstable (non-deterministic CUDA faults / hangs in the mjlab/mujoco_warp
# raycast + torchrunx layer -- unrelated to this pipeline). Stage 3 therefore
# runs single-GPU, which is verified reliable. Stages 1-2 use all of GPU_IDS.
# Set TEST_MULTIGPU=1 to force stage 3 multi-GPU anyway (not recommended).
#
# Tunable via environment variables:
#   GPU_IDS        comma-separated GPU ids        (default 0,1,2,3 -> all 4)
#   NUM_ENVS       parallel envs PER GPU          (default 2048; total = NUM_ENVS * #GPUs)
#   FLAT_ITERS     stage 1 training iterations    (default 1500)
#   ROUGH_ITERS    stage 2 training iterations    (default 5000)
#   TEST_ITERS     stage 3 training iterations    (default 10000)
#   TEST_MULTIGPU  set to 1 to run stage 3 multi-GPU too (default 0)
#   FOREGROUND     set to 1 to run in this shell instead of a tmux session
#
# Examples:
#   bash scripts/train_pipeline.sh                       # detached tmux, st.1-2 all GPUs
#   GPU_IDS=1,2,3 bash scripts/train_pipeline.sh         # skip a busy GPU 0
#   GPU_IDS=0 bash scripts/train_pipeline.sh             # whole pipeline single-GPU
#   FOREGROUND=1 FLAT_ITERS=5 ROUGH_ITERS=5 TEST_ITERS=5 NUM_ENVS=64 \
#     bash scripts/train_pipeline.sh                     # quick smoke test, foreground

set -euo pipefail

# Repo root (this script lives in <root>/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NUM_ENVS=${NUM_ENVS:-2048}
GPU_IDS=${GPU_IDS:-0,1,2,3}
FLAT_ITERS=${FLAT_ITERS:-1500}
ROUGH_ITERS=${ROUGH_ITERS:-5000}
TEST_ITERS=${TEST_ITERS:-10000}
TEST_MULTIGPU=${TEST_MULTIGPU:-0}
FOREGROUND=${FOREGROUND:-0}

EXPERIMENT=go2_velocity
LOG_ROOT="logs/rsl_rl/${EXPERIMENT}"

# ---------------------------------------------------------------------------
# Self-launch into a detached tmux session (unless already inside it, or the
# caller asked for FOREGROUND).
# ---------------------------------------------------------------------------
if [[ "${PIPELINE_INNER:-0}" != "1" && "${FOREGROUND}" != "1" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[ERROR] tmux not found. Re-run with FOREGROUND=1 to run in this shell." >&2
    exit 1
  fi
  TS="$(date +%Y-%m-%d_%H-%M-%S)"
  SESSION="go2_pipeline_${TS}"
  LOG="${SCRIPT_DIR}/logs/pipeline_${TS}.log"
  mkdir -p "${SCRIPT_DIR}/logs"
  INNER="cd '${SCRIPT_DIR}' && PIPELINE_INNER=1 NUM_ENVS='${NUM_ENVS}' GPU_IDS='${GPU_IDS}'"
  INNER+=" FLAT_ITERS='${FLAT_ITERS}' ROUGH_ITERS='${ROUGH_ITERS}' TEST_ITERS='${TEST_ITERS}'"
  INNER+=" TEST_MULTIGPU='${TEST_MULTIGPU}' bash scripts/train_pipeline.sh 2>&1 | tee '${LOG}'"
  tmux new-session -d -s "${SESSION}" "${INNER}"
  echo "[INFO] Pipeline launched in detached tmux session: ${SESSION}"
  echo "[INFO]   GPUs       : ${GPU_IDS}  (stage 3 single-GPU unless TEST_MULTIGPU=1)"
  echo "[INFO]   iters      : flat=${FLAT_ITERS} rough=${ROUGH_ITERS} test=${TEST_ITERS}"
  echo "[INFO]   attach     : tmux attach -t ${SESSION}"
  echo "[INFO]   follow log : tail -f ${LOG}"
  echo "[INFO]   stop       : tmux kill-session -t ${SESSION}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Pipeline body (runs inside the tmux session, or directly when FOREGROUND=1).
# ---------------------------------------------------------------------------
FIRST_GPU="${GPU_IDS%%,*}"  # first id in the list, for single-GPU stages

# Run one training stage.
#   $1 = task id   $2 = max iterations   $3 = gpu mode (multi|single)   $4 = resume run name (optional)
# `--gpu-ids` is finicky in tyro (`--gpu-ids 0` is ambiguous), so we follow
# scripts/run.sh: pin devices via CUDA_VISIBLE_DEVICES and pass `--gpu-ids all` only for
# the multi-GPU case (torchrunx); single-GPU relies on train.py's default [0].
# Echoes the run directory name (basename) to stdout for the next stage's resume.
run_stage() {
  local task="$1" iters="$2" mode="$3" resume="${4:-}"
  local gpu_flags=() resume_flags=()
  if [[ "${mode}" == "multi" && "${GPU_IDS}" == *,* ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    gpu_flags=(--gpu-ids all)
  else
    export CUDA_VISIBLE_DEVICES="${FIRST_GPU}"
  fi
  if [[ -n "${resume}" ]]; then
    resume_flags=(--agent.resume True --agent.load-run "${resume}")
  fi
  echo ">>> Stage: ${task} (${iters} iters, ${mode}-GPU${resume:+, resume from ${resume}})" >&2
  python scripts/train.py "${task}" \
    ${gpu_flags[@]+"${gpu_flags[@]}"} \
    --env.scene.num-envs "${NUM_ENVS}" \
    --agent.max-iterations "${iters}" \
    --agent.run-name "${task}" \
    ${resume_flags[@]+"${resume_flags[@]}"} 2>&1 | tee /dev/stderr \
    | grep -oP 'Logging experiment in directory: \K.*' | head -1 | xargs basename
}

TEST_MODE=single
[[ "${TEST_MULTIGPU}" != "0" ]] && TEST_MODE=multi

S1=$(run_stage Unitree-Go2-Flat-Scan  "${FLAT_ITERS}"  multi)
S2=$(run_stage Unitree-Go2-Rough      "${ROUGH_ITERS}" multi          "${S1}")
S3=$(run_stage Unitree-Go2-Test-Train "${TEST_ITERS}"  "${TEST_MODE}" "${S2}")

echo ""
echo "Pipeline complete."
echo "  stage 1 (flat):  ${LOG_ROOT}/${S1}"
echo "  stage 2 (rough): ${LOG_ROOT}/${S2}"
echo "  stage 3 (test):  ${LOG_ROOT}/${S3}"
