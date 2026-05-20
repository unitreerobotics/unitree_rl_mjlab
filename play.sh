#!/usr/bin/env bash
# Runner script: launch mjlab policy playback.
#
# Usage:
#   ./play.sh <task> [--checkpoint PATH] [--video] [--video-length N]
#                    [--video-width W] [--video-height H] [-- EXTRA_ARGS...]
#
# Examples:
#   ./play.sh Unitree-Go2-Flat
#   ./play.sh go2_velocity --checkpoint logs/.../model_1500.pt
#   ./play.sh Unitree-Go2-Flat --video                  # record default-length clip
#   ./play.sh Unitree-Go2-Flat --video --video-length 400
#
# When --video is set, a short MP4 is saved under
#   <checkpoint_dir>/videos/play/rl-video-step-0.mp4
# convenient for checking playback remotely.

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
VIDEO=false
VIDEO_LENGTH=""
VIDEO_WIDTH=""
VIDEO_HEIGHT=""
EXTRA_ARGS=()

if [[ $# -lt 1 ]]; then
    echo "[ERROR] Task name is required." >&2
    echo "Usage: $0 <task> [--checkpoint PATH] [--video] [--video-length N] [-- EXTRA_ARGS...]" >&2
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
        --video)
            VIDEO=true
            shift
            ;;
        --video-length)
            VIDEO_LENGTH="$2"
            shift 2
            ;;
        --video-width)
            VIDEO_WIDTH="$2"
            shift 2
            ;;
        --video-height)
            VIDEO_HEIGHT="$2"
            shift 2
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

if [[ "${VIDEO}" == "true" ]]; then
    CMD+=("--video" "True")
    [[ -n "${VIDEO_LENGTH}" ]] && CMD+=("--video-length=${VIDEO_LENGTH}")
    [[ -n "${VIDEO_WIDTH}"  ]] && CMD+=("--video-width=${VIDEO_WIDTH}")
    [[ -n "${VIDEO_HEIGHT}" ]] && CMD+=("--video-height=${VIDEO_HEIGHT}")
    VIDEO_DIR="$(dirname "${CHECKPOINT}")/videos/play"

    # An X-forwarded DISPLAY (e.g. `host:10.0` over SSH) sets DISPLAY but cannot
    # actually drive GLFW/OpenGL. Treat it like "no local display".
    HAS_LOCAL_DISPLAY=false
    if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
        HAS_LOCAL_DISPLAY=true
    elif [[ -n "${DISPLAY:-}" && -z "${SSH_CONNECTION:-}" && -z "${SSH_CLIENT:-}" ]]; then
        HAS_LOCAL_DISPLAY=true
    fi

    # Headless offscreen rendering requires a GL backend. EGL works on
    # NVIDIA / Mesa hosts without a display server.
    if [[ "${HAS_LOCAL_DISPLAY}" == "false" && -z "${MUJOCO_GL:-}" ]]; then
        export MUJOCO_GL=egl
        echo "[INFO] No local display detected; setting MUJOCO_GL=egl for offscreen render."
    fi

    # Avoid GLFW (native viewer) when DISPLAY is X-forwarded — pick the headless
    # viser viewer unless the user already passed --viewer.
    if [[ "${HAS_LOCAL_DISPLAY}" == "false" ]]; then
        VIEWER_ALREADY_SET=false
        for arg in "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; do
            if [[ "${arg}" == --viewer* ]]; then
                VIEWER_ALREADY_SET=true
                break
            fi
        done
        if [[ "${VIEWER_ALREADY_SET}" == "false" ]]; then
            EXTRA_ARGS+=("--viewer=viser")
            echo "[INFO] Using viser viewer (no local display)."
        fi
    fi
fi

CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "[INFO] Python      : ${PYTHON_BIN}"
echo "[INFO] Task        : ${TASK}"
echo "[INFO] Checkpoint  : ${CHECKPOINT}"
if [[ "${VIDEO}" == "true" ]]; then
    echo "[INFO] Video dir   : ${VIDEO_DIR}"
    [[ -n "${VIDEO_LENGTH}" ]] && echo "[INFO] Video length: ${VIDEO_LENGTH} frames"
fi
echo "[INFO] Command     : ${CMD[*]}"
echo

exec "${CMD[@]}"
