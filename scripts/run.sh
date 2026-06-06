#!/usr/bin/env bash
# Runner script: launch multi-GPU mjlab training inside a tmux session.
#
# Usage:
#   ./scripts/run.sh <task> [--num_gpus N] [--gpus IDS] [--num_envs N] [--resume EXPERIMENT/LOAD_RUN] [-- EXTRA_ARGS...]
#
# Examples:
#   ./scripts/run.sh Unitree-Go2-Flat                    # 4-GPU, default task, 4096 envs
#   ./scripts/run.sh Unitree-Go2-Flat --num_gpus 2       # 2-GPU run (uses GPUs 0,1)
#   ./scripts/run.sh Unitree-Go2-Flat --num_gpus 1       # single-GPU
#   ./scripts/run.sh Unitree-Go2-Flat --gpus 2,3         # pick exact GPUs 2 and 3
#   ./scripts/run.sh Unitree-Go2-Flat --gpus 1           # single run on GPU 1
#   ./scripts/run.sh Unitree-Go2-Flat --num_envs 8192    # override parallel env count
#   ./scripts/run.sh go2_velocity --resume logs/rsl_rl/go2_velocity/2026-04-22_18-54-05
#   ./scripts/run.sh Unitree-Go2-Flat -- --env.scene.dt=0.02  # pass extra flags after --

set -euo pipefail

printf -v MJLAB_ORIGINAL_LAUNCHER_COMMAND '%q ' "$0" "$@"
MJLAB_ORIGINAL_LAUNCHER_COMMAND="${MJLAB_ORIGINAL_LAUNCHER_COMMAND% }"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] Python executable not found." >&2
    exit 1
fi

TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] Train script not found at ${TRAIN_SCRIPT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
NUM_GPUS=4
GPU_IDS=""
NUM_ENVS=4096
TASK=""
RESUME=""
EXTRA_ARGS=()

USAGE="Usage: $0 <task> [--num_gpus N] [--gpus IDS] [--num_envs N] [--resume EXPERIMENT/LOAD_RUN] [-- EXTRA_ARGS...]"

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
        --gpus)       GPU_IDS="$2";    shift 2 ;;
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

if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] --num_envs must be a positive integer: ${NUM_ENVS}" >&2
    exit 1
fi

# GPU selection. tyro's `list[int] | Literal["all"] | None` signature for
# --gpu-ids is finicky: `--gpu-ids 0` is ambiguous (int vs single-element list)
# and `--gpu-ids 0 1 2 3` is parsed as one value plus three unknown positionals.
# Workaround: set CUDA_VISIBLE_DEVICES and pass `--gpu-ids all` for multi-GPU;
# for single-GPU we rely on train.py's default gpu_ids=[0].
#
# --gpus picks the exact physical GPU IDs to expose (e.g. "2,3"); --num_gpus
# just exposes the first N (0..N-1). --gpus takes precedence when both are set.
if [[ -n "${GPU_IDS}" ]]; then
    # Normalize: strip spaces, validate comma-separated non-negative integers.
    VISIBLE="${GPU_IDS// /}"
    if ! [[ "${VISIBLE}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "[ERROR] --gpus must be a comma-separated list of GPU IDs: ${GPU_IDS}" >&2
        exit 1
    fi
    NUM_GPUS=$(awk -F, '{print NF}' <<<"${VISIBLE}")
else
    if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] --num_gpus must be a positive integer: ${NUM_GPUS}" >&2
        exit 1
    fi
    VISIBLE=""
    for ((i=0; i<NUM_GPUS; i++)); do
        VISIBLE+="${VISIBLE:+,}$i"
    done
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    GPU_COUNT="${GPU_COUNT//[[:space:]]/}"
    if [[ -n "${GPU_COUNT}" && "${GPU_COUNT}" =~ ^[0-9]+$ && "${GPU_COUNT}" -gt 0 ]]; then
        declare -A SEEN_GPU_IDS=()
        IFS="," read -r -a REQUESTED_GPU_IDS <<<"${VISIBLE}"
        for gpu_id in "${REQUESTED_GPU_IDS[@]}"; do
            if (( gpu_id >= GPU_COUNT )); then
                echo "[ERROR] Requested GPU ${gpu_id}, but this host only exposes GPU IDs 0..$((GPU_COUNT - 1))." >&2
                echo "[ERROR] Use --gpus with valid physical GPU IDs, e.g. --gpus 2,3 for the last two GPUs on this host." >&2
                exit 1
            fi
            if [[ -n "${SEEN_GPU_IDS[${gpu_id}]:-}" ]]; then
                echo "[ERROR] Duplicate GPU ID requested: ${gpu_id}" >&2
                exit 1
            fi
            SEEN_GPU_IDS["${gpu_id}"]=1
        done
    fi
fi

GPU_ENV=()
USE_ALL_GPUS=false
# Always pin CUDA_VISIBLE_DEVICES so the selected GPUs are honored, even for a
# single explicitly-chosen GPU. train.py's default gpu_ids=[0] then refers to
# the first visible device.
GPU_ENV=(
    "CUDA_VISIBLE_DEVICES=${VISIBLE}"
    "MJLAB_LAUNCHER=scripts/run.sh"
    "MJLAB_LAUNCHER_COMMAND=${MJLAB_ORIGINAL_LAUNCHER_COMMAND}"
    "MJLAB_VISIBLE_GPUS=${VISIBLE}"
    "MJLAB_NUM_GPUS=${NUM_GPUS}"
    "MJLAB_NUM_ENVS=${NUM_ENVS}"
    "MJLAB_RESUME=${RESUME}"
)
if [[ -n "${HEIGHT_SCAN_AE_CHECKPOINT:-}" ]]; then
    GPU_ENV+=("HEIGHT_SCAN_AE_CHECKPOINT=${HEIGHT_SCAN_AE_CHECKPOINT}")
fi
if [[ "${NUM_GPUS}" -gt 1 ]]; then
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
    if [[ "${RESUME}" == /* ]]; then
        RESUME_PATH="${RESUME}"
    else
        RESUME_PATH="${PROJECT_ROOT}/${RESUME}"
    fi
    if [[ ! -d "${RESUME_PATH}" ]]; then
        echo "[ERROR] Resume directory not found: ${RESUME_PATH}" >&2
        exit 1
    fi
    # Extract experiment name and run id after logs/rsl_rl/ when present.
    RESUME_REL="$(echo "${RESUME}" | sed "s|.*/logs/rsl_rl/||")"
    EXPERIMENT_NAME="$(echo "${RESUME_REL}" | cut -d"/" -f1)"
    LOAD_RUN="$(echo "${RESUME_REL}" | sed "s|^[^/]*/||")"
    TRAIN_CMD+=("--agent.experiment-name" "${EXPERIMENT_NAME}")
    TRAIN_CMD+=("--agent.load-run" "${LOAD_RUN}")
fi

TRAIN_CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

# ---------------------------------------------------------------------------
# tmux session
# ---------------------------------------------------------------------------
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
SESSION="mjlab_train_${TIMESTAMP}_${NUM_GPUS}gpu"

SAFE_TASK="$(printf '%s' "${TASK}" | tr -c 'A-Za-z0-9_.-' '_')"
LAUNCHER_LOG_DIR="${PROJECT_ROOT}/logs/launcher/${SAFE_TASK}"
mkdir -p "${LAUNCHER_LOG_DIR}"
TRAIN_LOG="${LAUNCHER_LOG_DIR}/train_${TIMESTAMP}_${NUM_GPUS}gpu.log"

# Fail fast in the caller shell if the Python environment cannot import the
# simulation stack. Without this, tmux exits immediately and only leaves a
# fallback log behind.
"${PYTHON_BIN}" - "${TASK}" <<'PY'
import sys

import mujoco  # noqa: F401
import mujoco_warp  # noqa: F401
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import list_tasks

task = sys.argv[1]
if task not in list_tasks():
    raise SystemExit(f"Task not registered: {task}")
PY

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
FULL_CMD="cd '${PROJECT_ROOT}' && ${QUOTED_TRAIN_CMD} 2>&1 | awk -v target='${TRAIN_LOG}' -v script_dir='${PROJECT_ROOT}' ${QUOTED_AWK}"

echo "[INFO] Starting training in tmux session: ${SESSION}"
echo "[INFO] Python       : ${PYTHON_BIN}"
if [[ "${USE_ALL_GPUS}" == "true" ]]; then
    echo "[INFO] GPUs         : ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${VISIBLE} --gpu-ids all)"
else
    echo "[INFO] GPUs         : ${NUM_GPUS} (CUDA_VISIBLE_DEVICES=${VISIBLE}, train.py default gpu_ids=[0])"
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
    RESOLVED_LOG="$(find "${PROJECT_ROOT}/logs/rsl_rl" -name "${LOG_BASENAME}" 2>/dev/null | head -1)"
    [[ -n "${RESOLVED_LOG}" ]] && break
    sleep 1
done

echo "[INFO] Or follow the log with:"
if [[ -n "${RESOLVED_LOG}" ]]; then
    echo "         tail -F ${RESOLVED_LOG}"
else
    echo "         tail -F ${TRAIN_LOG}"
    echo "         (fallback launcher log; if training starts successfully it will move into logs/rsl_rl/<experiment>/<run>/)"
fi
