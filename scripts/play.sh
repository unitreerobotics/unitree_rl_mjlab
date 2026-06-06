#!/usr/bin/env bash
# Runner script: launch policy playback.
#
# Usage:
#   ./scripts/play.sh <task> [--simulation mjlab|mujoco|newton|isaacsim]
#                    [--checkpoint PATH] [--isaac-task TASK] [--video]
#                    [--video-attribution] [--video-length N]
#                    [--video-width W] [--video-height H] [-- EXTRA_ARGS...]
#
# Examples:
#   ./scripts/play.sh Unitree-Go2-Flat
#   ./scripts/play.sh Unitree-Go2-Flat --simulation newton --checkpoint logs/.../model_1500.pt
#   ./scripts/play.sh Unitree-Go2-Flat --simulation isaacsim --checkpoint logs/.../model_1500.pt
#   ./scripts/play.sh go2_velocity --checkpoint logs/.../model_1500.pt
#   ./scripts/play.sh Unitree-Go2-Flat --video                  # record default-length clip
#   ./scripts/play.sh Unitree-Go2-Flat --video --video-length 400
#   ./scripts/play.sh Unitree-Go2-Flat --video-attribution     # side-by-side attribution clip
#
# When --video is set for mjlab/mujoco playback, a short MP4 is saved under
#   <checkpoint_dir>/videos/play/rl-video-step-0.mp4
# convenient for checking playback remotely.

set -euo pipefail

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

PLAY_SCRIPT="${SCRIPT_DIR}/play.py"

if [[ ! -f "${PLAY_SCRIPT}" ]]; then
    echo "[ERROR] Play script not found at ${PLAY_SCRIPT}" >&2
    exit 1
fi

usage() {
    cat >&2 <<EOF
Usage: $0 <task> [--simulation mjlab|mujoco|newton|isaacsim] [--checkpoint PATH] [--isaac-task TASK] [--video] [--video-attribution] [--video-length N] [--video-width W] [--video-height H] [-- EXTRA_ARGS...]
EOF
}

require_value() {
    local flag="$1"
    local value="${2:-}"
    if [[ -z "${value}" || "${value}" == --* ]]; then
        echo "[ERROR] ${flag} requires a value." >&2
        usage
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
TASK=""
CHECKPOINT=""
SIMULATION="mjlab"
ISAAC_TASK=""
VIDEO=false
VIDEO_ATTRIBUTION=false
VIDEO_LENGTH=""
VIDEO_WIDTH=""
VIDEO_HEIGHT=""
DRY_RUN=false
EXTRA_ARGS=()

if [[ $# -lt 1 ]]; then
    echo "[ERROR] Task name is required." >&2
    usage
    exit 1
fi

TASK="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --simulation)
            require_value "$1" "${2:-}"
            SIMULATION="$2"
            shift 2
            ;;
        --isaac-task)
            require_value "$1" "${2:-}"
            ISAAC_TASK="$2"
            shift 2
            ;;
        --checkpoint)
            require_value "$1" "${2:-}"
            CHECKPOINT="$2"
            shift 2
            ;;
        --video)
            VIDEO=true
            shift
            ;;
        --video-attribution)
            VIDEO=true
            VIDEO_ATTRIBUTION=true
            shift
            ;;
        --video-length)
            require_value "$1" "${2:-}"
            VIDEO_LENGTH="$2"
            shift 2
            ;;
        --video-width)
            require_value "$1" "${2:-}"
            VIDEO_WIDTH="$2"
            shift 2
            ;;
        --video-height)
            require_value "$1" "${2:-}"
            VIDEO_HEIGHT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
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

case "${SIMULATION}" in
    mjlab|mujoco|newton|isaacsim) ;;
    *)
        echo "[ERROR] Unsupported --simulation: ${SIMULATION}" >&2
        echo "        Expected one of: mjlab, mujoco, newton, isaacsim" >&2
        exit 1
        ;;
esac

LOCAL_SIMULATION=false
if [[ "${SIMULATION}" == "mjlab" || "${SIMULATION}" == "mujoco" ]]; then
    LOCAL_SIMULATION=true
fi

if [[ -z "${CHECKPOINT}" && "${LOCAL_SIMULATION}" == "true" ]]; then
    CHECKPOINT="$(find "${PROJECT_ROOT}/logs/rsl_rl" -type f -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
fi

if [[ -z "${CHECKPOINT}" ]]; then
    if [[ "${LOCAL_SIMULATION}" == "true" ]]; then
        echo "[ERROR] No checkpoint found under ${PROJECT_ROOT}/logs/rsl_rl." >&2
        echo "        Provide one explicitly with --checkpoint PATH." >&2
    else
        echo "[ERROR] --checkpoint is required with --simulation ${SIMULATION}." >&2
        echo "        External Isaac Lab/Newton playback should use an explicit compatible checkpoint." >&2
    fi
    exit 1
fi

if [[ -n "${CHECKPOINT}" && "${CHECKPOINT}" != /* && ! -f "${CHECKPOINT}" && -f "${PROJECT_ROOT}/${CHECKPOINT}" ]]; then
    CHECKPOINT="${PROJECT_ROOT}/${CHECKPOINT}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Checkpoint file not found: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ "${LOCAL_SIMULATION}" == "true" ]]; then
    CMD=(
        "${PYTHON_BIN}"
        "${PLAY_SCRIPT}"
        "${TASK}"
        "--checkpoint-file=${CHECKPOINT}"
        "--num-envs=1"
    )

    if [[ "${VIDEO}" == "true" ]]; then
        CMD+=("--video" "True")
        [[ "${VIDEO_ATTRIBUTION}" == "true" ]] && CMD+=("--video-attribution" "True")
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
    echo "[INFO] Simulation  : ${SIMULATION}"
    echo "[INFO] Task        : ${TASK}"
    echo "[INFO] Checkpoint  : ${CHECKPOINT}"
    if [[ "${VIDEO}" == "true" ]]; then
        echo "[INFO] Video dir   : ${VIDEO_DIR}"
        [[ "${VIDEO_ATTRIBUTION}" == "true" ]] && echo "[INFO] Video mode  : attribution side-by-side"
        [[ -n "${VIDEO_LENGTH}" ]] && echo "[INFO] Video length: ${VIDEO_LENGTH} frames"
    fi
    echo "[INFO] Command     : ${CMD[*]}"
    echo

    if [[ "${DRY_RUN}" == "true" ]]; then
        exit 0
    fi
    exec "${CMD[@]}"
fi

if [[ -z "${ISAACLAB_ROOT:-}" ]]; then
    echo "[ERROR] ISAACLAB_ROOT is required with --simulation ${SIMULATION}." >&2
    echo "        This launcher maps ${SIMULATION} playback through Isaac Lab, not the local mjlab runtime." >&2
    echo "        Set it to the root of an Isaac Lab develop checkout, e.g.:" >&2
    echo "        export ISAACLAB_ROOT=/path/to/IsaacLab" >&2
    echo "" >&2
    echo "        Note: checkpoints trained by this repo's mjlab tasks are not guaranteed" >&2
    echo "        to load in Isaac Lab tasks. For that checkpoint, use:" >&2
    echo "        ./scripts/play.sh ${TASK} --simulation mjlab --checkpoint ${CHECKPOINT}" >&2
    exit 1
fi

ISAACLAB_ROOT="${ISAACLAB_ROOT%/}"
ISAACLAB_SH="${ISAACLAB_ROOT}/isaaclab.sh"
if [[ ! -x "${ISAACLAB_SH}" ]]; then
    echo "[ERROR] Isaac Lab launcher not found or not executable: ${ISAACLAB_SH}" >&2
    echo "        ISAACLAB_ROOT must point at an Isaac Lab checkout that contains isaaclab.sh." >&2
    exit 1
fi

if [[ "${VIDEO_ATTRIBUTION}" == "true" ]]; then
    echo "[ERROR] --video-attribution is only supported by the local mjlab/mujoco playback path." >&2
    exit 1
fi
if [[ -n "${VIDEO_WIDTH}" || -n "${VIDEO_HEIGHT}" ]]; then
    echo "[ERROR] --video-width/--video-height are only mapped for local mjlab/mujoco playback." >&2
    echo "        Pass Isaac Lab-specific rendering flags after -- if needed." >&2
    exit 1
fi

if [[ -z "${ISAAC_TASK}" ]]; then
    TASK_LOWER="${TASK,,}"
    if [[ "${TASK_LOWER}" == *go2* ]]; then
        if [[ "${TASK_LOWER}" == *no*height* ]]; then
            ISAAC_TASK="Isaac-Velocity-Rough-NoHeight-Unitree-Go2-Play-v0"
        elif [[ "${TASK_LOWER}" == *rough* ]]; then
            ISAAC_TASK="Isaac-Velocity-Rough-Unitree-Go2-Play-v0"
        else
            ISAAC_TASK="Isaac-Velocity-Flat-Unitree-Go2-Play-v0"
        fi
    else
        ISAAC_TASK="${TASK}"
    fi
fi

VISUALIZER="newton"
if [[ "${SIMULATION}" == "isaacsim" ]]; then
    VISUALIZER="omniverse"
fi
ISAAC_PLAY_SCRIPT="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rsl_rl/play.py"
SUPPORTS_VISUALIZER=false
if [[ -f "${ISAAC_PLAY_SCRIPT}" ]] && grep -R --include='*.py' -q -- '--visualizer' \
    "${ISAAC_PLAY_SCRIPT}" \
    "${ISAACLAB_ROOT}/source" 2>/dev/null; then
    SUPPORTS_VISUALIZER=true
fi

ISAACLAB_TERM="${TERM:-xterm-256color}"
if [[ "${ISAACLAB_TERM}" == "dumb" ]]; then
    ISAACLAB_TERM="xterm-256color"
fi

CMD=(
    "env"
    "TERM=${ISAACLAB_TERM}"
    "VIRTUAL_ENV="
    "CONDA_PREFIX="
    "${ISAACLAB_SH}"
    "-p"
    "scripts/reinforcement_learning/rsl_rl/play.py"
    "--task"
    "${ISAAC_TASK}"
    "--num_envs"
    "1"
    "--checkpoint"
    "${CHECKPOINT}"
)

if [[ "${SUPPORTS_VISUALIZER}" == "true" ]]; then
    CMD+=("--visualizer" "${VISUALIZER}")
elif [[ "${SIMULATION}" == "newton" ]]; then
    echo "[WARN] This Isaac Lab checkout does not expose --visualizer; launching without Newton visualizer support." >&2
    echo "[WARN] Newton integration requires a Newton-capable Isaac Lab branch/environment." >&2
fi

if [[ "${VIDEO}" == "true" ]]; then
    CMD+=("--video")
    [[ -n "${VIDEO_LENGTH}" ]] && CMD+=("--video_length" "${VIDEO_LENGTH}")
fi

CMD+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

echo "[INFO] Isaac Lab   : ${ISAACLAB_ROOT}"
echo "[INFO] Simulation  : ${SIMULATION}"
echo "[INFO] Visualizer  : ${VISUALIZER}"
echo "[INFO] Source task : ${TASK}"
echo "[INFO] Isaac task  : ${ISAAC_TASK}"
echo "[INFO] Checkpoint  : ${CHECKPOINT}"
echo "[INFO] Command     : ${CMD[*]}"
echo

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

cd "${ISAACLAB_ROOT}"
exec "${CMD[@]}"
