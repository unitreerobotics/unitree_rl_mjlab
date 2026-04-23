#!/usr/bin/env bash
# Runner script: launch multi-GPU mjlab training inside a tmux session.
#
# Usage:
#   ./run.sh <task> [--num_gpus N] [--resume PATH] [-- EXTRA_ARGS...]
#
# Examples:
#   ./run.sh Unitree-Go2-Flat                    # 4-GPU, default task
#   ./run.sh Unitree-Go2-Flat --num_gpus 2       # 2-GPU run
#   ./run.sh Unitree-Go2-Flat --num_gpus 1       # single-GPU
#   ./run.sh go2_velocity --resume logs/.../model_1500.pt
#   ./run.sh Unitree-Go2-Flat -- --video          # pass extra flags after --

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
TASK=""
RESUME=""
EXTRA_ARGS=()

if [[ $# -lt 1 ]]; then
    echo "[ERROR] Task name is required." >&2
    echo "Usage: $0 <task> [--num_gpus N] [--resume PATH] [-- EXTRA_ARGS...]" >&2
    exit 1
fi

TASK="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_gpus)   NUM_GPUS="$2";   shift 2 ;;
        --resume)     RESUME="$2";     shift 2 ;;
        --)           shift; EXTRA_ARGS+=("$@"); break ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            echo "Usage: $0 <task> [--num_gpus N] [--resume PATH] [-- EXTRA_ARGS...]" >&2
            exit 1
            ;;
    esac
done

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --num_gpus must be a positive integer: ${NUM_GPUS}" >&2
    exit 1
fi

# Extract experiment_name and run_name from extra args for log directory naming.
EXPERIMENT_NAME=""
RUN_NAME=""
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[i]}" in
        --experiment_name=*)
            EXPERIMENT_NAME="${EXTRA_ARGS[i]#*=}"
            ;;
        --experiment_name)
            if (( i + 1 >= ${#EXTRA_ARGS[@]} )); then
                echo "[ERROR] --experiment_name requires a value" >&2
                exit 1
            fi
            i=$((i + 1))
            EXPERIMENT_NAME="${EXTRA_ARGS[i]}"
            ;;
        --run_name=*)
            RUN_NAME="${EXTRA_ARGS[i]#*=}"
            ;;
        --run_name)
            if (( i + 1 >= ${#EXTRA_ARGS[@]} )); then
                echo "[ERROR] --run_name requires a value" >&2
                exit 1
            fi
            i=$((i + 1))
            RUN_NAME="${EXTRA_ARGS[i]}"
            ;;
    esac
done

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR_NAME="${TIMESTAMP}"
[[ -n "${RUN_NAME}" ]] && RUN_DIR_NAME+="_${RUN_NAME}"

# If experiment_name not provided via extra args, derive from task name.
if [[ -z "${EXPERIMENT_NAME}" ]]; then
    EXPERIMENT_NAME="$(echo "${TASK}" | tr '[:upper:]' '[:lower:]')"
fi

RUN_DIR="${SCRIPT_DIR}/logs/rsl_rl/${EXPERIMENT_NAME}/${RUN_DIR_NAME}"

# ---------------------------------------------------------------------------
# Build training command
# ---------------------------------------------------------------------------
mkdir -p "${RUN_DIR}"

TRAIN_CMD=(
    "${PYTHON_BIN}"
    "${TRAIN_SCRIPT}"
    "${TASK}"
)

[[ -n "${RESUME}" ]] && TRAIN_CMD+=("--resume" "${RESUME}")
TRAIN_CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

# ---------------------------------------------------------------------------
# Log file
# ---------------------------------------------------------------------------
LOG_FILE="${RUN_DIR}/train_${TIMESTAMP}_${NUM_GPUS}gpu.log"

# ---------------------------------------------------------------------------
# tmux session
# ---------------------------------------------------------------------------
SESSION="mjlab_train_${TIMESTAMP}_${NUM_GPUS}gpu"

# Wrap command so output goes to the log file and is visible in tmux pane.
printf -v QUOTED_TRAIN_CMD '%q ' "${TRAIN_CMD[@]}"
FULL_CMD="cd '${SCRIPT_DIR}' && ${QUOTED_TRAIN_CMD} 2>&1 | tee '${LOG_FILE}'"

echo "[INFO] Starting training in tmux session: ${SESSION}"
echo "[INFO] Python       : ${PYTHON_BIN}"
echo "[INFO] GPUs         : ${NUM_GPUS}"
echo "[INFO] Task         : ${TASK}"
echo "[INFO] Experiment   : ${EXPERIMENT_NAME}"
[[ -n "${RUN_NAME}" ]] && echo "[INFO] Run name     : ${RUN_NAME}"
[[ -n "${RESUME}" ]] && echo "[INFO] Resume from  : ${RESUME}"
echo "[INFO] Run dir      : ${RUN_DIR}"
echo "[INFO] Log file     : ${LOG_FILE}"
echo

tmux new-session -d -s "${SESSION}" bash
tmux send-keys -t "${SESSION}" "${FULL_CMD}" Enter

echo "[INFO] Training launched. Attach with:"
echo "         tmux attach -t ${SESSION}"
echo "[INFO] Follow logs with:"
echo "         tail -f ${LOG_FILE}"
