#!/usr/bin/env bash
# Starts the R1 simulation with the perceptive policy in one go: video relay, controller (straight
# into Perceptive, i.e. the highest deploy/robots/r1/config/policy/perceptive/vN), simulator,
# lowstate_shm and heightmap_node.py. Then one attempt over the EUR pallet (perception/tools/
# eval_climb.py); the simulation stays open afterwards until Ctrl+C.
#
#   ./start_perceptive_sim.sh                         # EUR pallet, straight on, 1.5 m, 0.4 m/s
#   ./start_perceptive_sim.sh --block                 # the solid block next to it instead
#   ./start_perceptive_sim.sh --angle 10 --offset 0.2 --distance 1.2 --speed 0.5
#   ./start_perceptive_sim.sh --gamepad               # no attempt: walk with the gamepad
#                                                     # (simulate/config.yaml: joystick_device)
#   Ctrl+C                                            # stops everything
#
# One attempt per start: the robot begins at the origin, so for the next one start again.
# Results: one JSON line per attempt in logs/perceptive_sim/attempts.jsonl; logs of the
# processes in logs/perceptive_sim/<time>/ (both ignored by git).

set -u

TARGET=pallet
GAMEPAD=0
CLIMB_ARGS=()
while (($#)); do
    case "$1" in
        --block) TARGET=block ;;
        --gamepad) GAMEPAD=1 ;;
        --distance|--offset|--angle|--speed)
            [[ $# -ge 2 ]] || { echo "$1 needs a value"; exit 2; }
            CLIMB_ARGS+=("$1=$2"); shift ;;
        *) echo "unknown option $1; usage: $0 [--block] [--gamepad] [--distance m] [--offset m] [--angle deg] [--speed m/s]"
           exit 2 ;;
    esac
    shift
done

MJLAB="$(cd "$(dirname "$0")" && pwd)"
PERCEPTION="$MJLAB/perception"
PYTHON="$PERCEPTION/.venv/bin/python"
MEDIAMTX="${MEDIAMTX:-$HOME/tools/mediamtx/mediamtx}"
LOGS="$MJLAB/logs/perceptive_sim/$(date +%Y-%m-%d_%H-%M-%S)"
RESULTS="$MJLAB/logs/perceptive_sim/attempts.jsonl"

PIDS=()          # started processes, in start order

say()  { printf '\033[1m[start]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[start] %s\033[0m\n' "$*"; STATUS=1; exit 1; }
STATUS=0

# Waits until a log file contains a line, or the process died, or the timeout passed.
wait_for() {  # log pattern seconds pid what
    local log=$1 pattern=$2 secs=$3 pid=$4 what=$5
    for ((i = 0; i < secs * 5; i++)); do
        grep -q "$pattern" "$log" 2>/dev/null && return 0
        kill -0 "$pid" 2>/dev/null || fail "$what ended; see $log"
        sleep 0.2
    done
    fail "$what not ready after ${secs} s; see $log"
}

stop_all() {
    trap - INT TERM EXIT
    echo
    say "stopping ..."
    for ((i = ${#PIDS[@]} - 1; i >= 0; i--)); do
        kill "${PIDS[i]}" 2>/dev/null
    done
    for ((i = 0; i < 50; i++)); do
        alive=0
        for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        ((alive)) || break
        sleep 0.2
    done
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null; done
    wait 2>/dev/null
    say "stopped; logs in $LOGS"
    exit $STATUS
}

# ---- checks ---------------------------------------------------------------------------------
[[ -x "$MEDIAMTX" ]] || fail "mediamtx missing: $MEDIAMTX"
[[ -x "$MJLAB/deploy/robots/r1/build/r1_ctrl" ]] || fail "r1_ctrl not built"
[[ -x "$MJLAB/simulate/build/unitree_mujoco" ]] || fail "unitree_mujoco not built"
[[ -x "$PERCEPTION/cpp/build/lowstate_shm" ]] || fail "lowstate_shm not built (perception/cpp)"
[[ -x "$PYTHON" ]] || fail "perception venv missing: $PYTHON"
[[ -n "${DISPLAY:-}" ]] || fail "no DISPLAY: start this from the desktop"
# The same leftovers as perception/tools/eval_climb_runs.py checks; never an r1_ctrl for the real robot.
pgrep -f "r1_ctrl --network lo" > /dev/null && fail "r1_ctrl is already running; stop it first (pkill -f 'r1_ctrl --network lo')"
pgrep -x unitree_mujoco > /dev/null && fail "unitree_mujoco is already running; stop it first (pkill -x unitree_mujoco)"
pgrep -f "mediamtx mediamtx_sim.yml" > /dev/null && fail "mediamtx is already running; stop it first"
pgrep -f heightmap_node.py > /dev/null && fail "heightmap_node.py is already running; stop it first"
pgrep -x lowstate_shm > /dev/null && fail "lowstate_shm is already running; stop it first (pkill -x lowstate_shm)"

POLICY=$(ls -d "$MJLAB/deploy/robots/r1/config/policy/perceptive"/*/ 2>/dev/null | sort | tail -1)
mkdir -p "$LOGS"
trap stop_all INT TERM EXIT

# ---- 1. video relay (the simulator's head camera publishes into it) --------------------------
say "1/5 mediamtx"
(cd "$MJLAB/simulate" && exec setsid "$MEDIAMTX" mediamtx_sim.yml) > "$LOGS/mediamtx.log" 2>&1 &
PIDS+=($!)
wait_for "$LOGS/mediamtx.log" "WebRTC\] started" 10 $! "mediamtx"

# ---- 2. controller (before the simulator, so the policy holds the robot from the first step) --
if ((GAMEPAD)); then CMD_SOURCE=joystick; else CMD_SOURCE=sport; fi
say "2/5 controller: Perceptive, policy $(basename "$POLICY"), commands from $CMD_SOURCE"
(cd "$MJLAB/deploy/robots/r1/build" && exec setsid ./r1_ctrl --network lo --cmd-source $CMD_SOURCE --arm-sdk \
    --start-state Perceptive) > "$LOGS/controller.log" 2>&1 &
PIDS+=($!)
CTRL_PID=$!
wait_for "$LOGS/controller.log" "Waiting for connection" 10 $CTRL_PID "controller"

# ---- 3. simulator ---------------------------------------------------------------------------
say "3/5 simulator$( ((GAMEPAD)) && echo " with gamepad")"
(cd "$MJLAB/simulate/build" && DISPLAY="$DISPLAY" exec setsid ./unitree_mujoco --joystick $GAMEPAD --stereo 0) \
    > "$LOGS/simulator.log" 2>&1 &
PIDS+=($!)
SIM_PID=$!
wait_for "$LOGS/controller.log" "walking service" 30 $CTRL_PID "controller"
wait_for "$LOGS/simulator.log" "head camera\] publishing" 30 $SIM_PID "simulator camera"

# ---- 4. perception --------------------------------------------------------------------------
say "4/5 lowstate_shm and heightmap_node.py"
(cd "$PERCEPTION" && exec setsid cpp/build/lowstate_shm --iface lo) > "$LOGS/lowstate_shm.log" 2>&1 &
PIDS+=($!)
wait_for "$LOGS/lowstate_shm.log" "r1_lowstate" 10 $! "lowstate_shm"
(cd "$PERCEPTION" && exec setsid "$PYTHON" -u heightmap_node.py --iface lo) > "$LOGS/heightmap.log" 2>&1 &
PIDS+=($!)
wait_for "$LOGS/heightmap.log" "\[heightmap\]" 20 $! "heightmap_node.py"

# Ends the script when one of the parts dies.
watch_parts() {
    for pid in "${PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null || { say "a process ended (pid $pid); see $LOGS"; STATUS=1; stop_all; }
    done
}

# ---- 5. the attempt ---------------------------------------------------------------------------
if ((GAMEPAD)); then
    say "5/5 ready: walk with the gamepad. Ctrl+C stops everything."
else
    say "5/5 attempt over the $( [[ $TARGET == block ]] && echo "solid block" || echo "EUR pallet" ) ${CLIMB_ARGS[*]}"
    (cd "$PERCEPTION" && R1_CLIMB_TARGET=$TARGET exec "$PYTHON" -u tools/eval_climb.py "${CLIMB_ARGS[@]}" \
        --label "$(basename "$POLICY")/$TARGET" --out "$RESULTS") 2>&1 | tee "$LOGS/climb.log"
    echo
    say "result: $(tail -1 "$RESULTS" 2>/dev/null | "$PYTHON" -c 'import json,sys; r=json.loads(sys.stdin.read()); print(r.get("outcome"), {k: r[k] for k in ("distance","offset","angle","speed") if k in r})' 2>/dev/null)"
    say "all results: $RESULTS"
    say "the simulation stays open; Ctrl+C stops everything (start again for the next attempt)"
fi

while true; do
    watch_parts
    sleep 1
done
