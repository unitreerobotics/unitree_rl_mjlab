#!/usr/bin/env bash
# Runner script: launch mjlab policy playback.
#
# Usage:
#   ./play.sh <task> [--checkpoint PATH] [--headless] [-- EXTRA_ARGS...]
#
# Examples:
#   ./play.sh Unitree-Go2-Flat
#   ./play.sh go2_velocity --checkpoint logs/.../model_1500.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[ERROR] Python executable not found." >&2
    exit 1
fi

PLAY_SCRIPT="${SCRIPT_DIR}/scripts/play.py"

if [[ ! -f "${PLAY_SCRIPT}" ]]; then
    echo "[ERROR] Play script not found at ${PLAY_SCRIPT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
TASK=""
CHECKPOINT=""
EXTRA_ARGS=()

if [[ $# -lt 1 ]]; then
    echo "[ERROR] Task name is required." >&2
    echo "Usage: $0 <task> [--checkpoint PATH] [--headless] [-- EXTRA_ARGS...]" >&2
    exit 1
fi

TASK="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --headless)
            EXTRA_ARGS+=("--headless")
            shift
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(find "${SCRIPT_DIR}/logs/rsl_rl" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
fi

if [[ -z "${CHECKPOINT}" ]]; then
    echo "[ERROR] No checkpoint found under ${SCRIPT_DIR}/logs/rsl_rl." >&2
    echo "        Provide one explicitly with --checkpoint PATH." >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint file not found: ${CHECKPOINT}" >&2
    exit 1
fi

CMD=(
    "${PYTHON_BIN}"
    "${PLAY_SCRIPT}"
    "${TASK}"
    "--checkpoint-file=${CHECKPOINT}"
    "--num-envs=1"
)

CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "[INFO] Python      : ${PYTHON_BIN}"
echo "[INFO] Task        : ${TASK}"
echo "[INFO] Checkpoint  : ${CHECKPOINT}"
echo "[INFO] Command     : ${CMD[*]}"
echo

exec "${CMD[@]}"
