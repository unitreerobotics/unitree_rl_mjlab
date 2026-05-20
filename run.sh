#!/usr/bin/env bash
# Runner script: launch multi-GPU mjlab training inside a tmux session.
#
# Usage:
#   ./run.sh <task> [--num_gpus N] [--num_envs N] [--resume EXPERIMENT/LOAD_RUN] [-- EXTRA_ARGS...]
#
# Examples:
#   ./run.sh Unitree-Go2-Flat                    # 4-GPU, default task, 4096 envs
#   ./run.sh Unitree-Go2-Flat --num_gpus 2       # 2-GPU run
#   ./run.sh Unitree-Go2-Flat --num_gpus 1       # single-GPU
#   ./run.sh Unitree-Go2-Flat --num_envs 8192    # override parallel env count
#   ./run.sh go2_velocity --resume logs/rsl_rl/go2_velocity/2026-04-22_18-54-05
#   ./run.sh Unitree-Go2-Flat -- --env.scene.dt=0.02  # pass extra flags after --

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] Python executable not found." >&2
    exit 1
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/scripts/train.py"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] Train script not found at ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
NUM_GPUS=4
NUM_ENVS=4096
TASK=""
RESUME=""
EXTRA_ARGS=()

USAGE="Usage: $0 <task> [--num_gpus N] [--num_envs N] [--resume EXPERIMENT/LOAD_RUN] [-- EXTRA_ARGS...]"

if [[ $# -lt 1 ]]; then
    echo "[ERROR] Task name is required." >&2
    echo "${USAGE}" >&2
    exit 1
fi

TASK="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)   NUM_GPUS="$2";   shift 2 ;;
        --num_envs)   NUM_ENVS="$2";   shift 2 ;;
        --resume)     RESUME="$2";     shift 2 ;;
        --)           shift; EXTRA_ARGS+=("$@"); break ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "${USAGE}" >&2
            exit 1
            ;;
    esac
done

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --num_gpus must be a positive integer: ${NUM_GPUS}" >&2
    exit 1
fi

if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --num_envs must be a positive integer: ${NUM_ENVS}" >&2
    exit 1
fi

# GPU selection. tyro's `list[int] | Literal["all"] | None` signature for
# --gpu-ids is finicky: `--gpu-ids 0` is ambiguous (int vs single-element list)
# and `--gpu-ids 0 1 2 3` is parsed as one value plus three unknown positionals.
# Workaround: set CUDA_VISIBLE_DEVICES and pass `--gpu-ids all` for multi-GPU;
# for single-GPU we rely on train.py's default gpu_ids=[0].
GPU_ENV=()
USE_ALL_GPUS=false
if [[ "${NUM_GPUS}" -gt 1 ]]; then
    VISIBLE=""
    for ((i=0; i<NUM_GPUS; i++)); do
        VISIBLE+="${VISIBLE:+,}$i"
    done
    GPU_ENV=("CUDA_VISIBLE_DEVICES=${VISIBLE}")
    USE_ALL_GPUS=true
fi

# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------
TRAIN_CMD=(
    "${GPU_ENV[@]+"${GPU_ENV[@]}"}"
    "${PYTHON_BIN}"
    "${TRAIN_SCRIPT}"
    "${TASK}"
    "--env.scene.num-envs=${NUM_ENVS}"
)

if [[ "${USE_ALL_GPUS}" == "true" ]]; then
    TRAIN_CMD+=("--gpu-ids" "all")
fi

if [[ -n "${RESUME}" ]]; then
    TRAIN_CMD+=("--agent.resume" "True")
    # Extract experiment_name and load_run from the resume path.
    # Expected format: logs/rsl_rl/<experiment>/<run_id> or just <experiment>/<run_id>
    RESUME_PATH="${SCRIPT_DIR}/${RESUME}"
    if [[ ! -d "${RESUME_PATH}" ]]; then
        echo "[ERROR] Resume directory not found: ${RESUME_PATH}" >&2
        exit 1
    fi
    # Extract experiment name (first component after logs/rsl_rl/)
    EXPERIMENT_NAME="$(echo "${RESUME}" | sed 's|^logs/rsl_rl/||' | cut -d'/' -f1)"
    LOAD_RUN="$(echo "${RESUME}" | sed 's|^logs/rsl_rl/[^/]*/||')"
    TRAIN_CMD+=("--agent.experiment-name" "${EXPERIMENT_NAME}")
    TRAIN_CMD+=("--agent.load-run" "${LOAD_RUN}")
fi

TRAIN_CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

# ---------------------------------------------------------------------------
# tmux session
# ---------------------------------------------------------------------------
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
SESSION="mjlab_train_${TIMESTAMP}_${NUM_GPUS}gpu"

TRAIN_LOG="${SCRIPT_DIR}/logs/train_${TIMESTAMP}_${NUM_GPUS}gpu.log"

printf -v QUOTED_TRAIN_CMD '%q ' "${TRAIN_CMD[@]}"

# Inline awk filter replaces `tee`: mirror every line to the terminal and to a
# log file. The log file starts at TRAIN_LOG (fallback path, used if training
# crashes before announcing its run dir). When train.py prints
# "[INFO] Logging experiment in directory: <path>", the filter mv's the log
# into that dir as train.log and continues writing there, so all run artifacts
# end up in one place. No detached watcher process to lose.
read -r -d '' AWK_FILTER <<'AWK' || true
BEGIN { out = target; n = split(target, parts, "/"); base = parts[n] }
{
  print
  print >> out
  fflush()
  if (!resolved && match($0, /Logging experiment in directory:[[:space:]]*/)) {
    dir = substr($0, RSTART + RLENGTH)
    sub(/[[:space:]]+$/, "", dir)
    if (substr(dir, 1, 1) != "/") dir = script_dir "/" dir
    close(out)
    system("mkdir -p \"" dir "\" && mv \"" out "\" \"" dir "/" base "\"")
    out = dir "/" base
    resolved = 1
  }
}
AWK

printf -v QUOTED_AWK '%q' "${AWK_FILTER}"
FULL_CMD="cd '${SCRIPT_DIR}' && ${QUOTED_TRAIN_CMD} 2>&1 | awk -v target='${TRAIN_LOG}' -v script_dir='${SCRIPT_DIR}' ${QUOTED_AWK}"

echo "[INFO] Starting training in tmux session: ${SESSION}"
echo "[INFO] Python       : ${PYTHON_BIN}"
if [[ "${USE_ALL_GPUS}" == "true" ]]; then
    echo "[INFO] GPUs         : ${NUM_GPUS} (${GPU_ENV[*]} --gpu-ids all)"
else
    echo "[INFO] GPUs         : ${NUM_GPUS} (train.py default gpu_ids=[0])"
fi
echo "[INFO] Num envs     : ${NUM_ENVS}"
echo "[INFO] Task         : ${TASK}"
[[ -n "${RESUME}" ]] && echo "[INFO] Resume from  : ${RESUME}"
echo

tmux new-session -d -s "${SESSION}" bash
tmux send-keys -t "${SESSION}" "${FULL_CMD}" Enter

echo "[INFO] Training launched. Attach with:"
echo "         tmux attach -t ${SESSION}"

# Wait briefly for train.py to announce its run dir so the awk filter can move
# the log into logs/rsl_rl/<experiment>/<run_id>/. We poll for the moved file
# and print its final path so the tail command works directly.
LOG_BASENAME="$(basename "${TRAIN_LOG}")"
RESOLVED_LOG=""
for _ in $(seq 1 60); do
    RESOLVED_LOG="$(find "${SCRIPT_DIR}/logs/rsl_rl" -name "${LOG_BASENAME}" 2>/dev/null | head -1)"
    [[ -n "${RESOLVED_LOG}" ]] && break
    sleep 1
done

echo "[INFO] Or follow the log with:"
if [[ -n "${RESOLVED_LOG}" ]]; then
    echo "         tail -F ${RESOLVED_LOG}"
else
    echo "         tail -F ${TRAIN_LOG}"
    echo "         (run dir not announced yet; tail will keep retrying)"
fi
