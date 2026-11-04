#!/usr/bin/env bash
# Starts the R1 simulation for the Quest in one go: video relay, controller, simulator with the
# head camera, arena-r1-kit's stream.py and the Quest's USB ports. No gamepad: the controller
# goes straight into Velocity and takes its velocity from the Quest's sticks (sport service).
#
#   ./start_quest_sim.sh            # then in the Quest: http://localhost:8080, Enter VR
#   ./start_quest_sim.sh --upper    # also head and right arm (rt/arm_sdk): right A takes over,
#                                   # right grip moves the arm, right B hands back
#   ./start_quest_sim.sh --perceptive  # Perceptive instead of Velocity: the highest
#                                   # deploy/robots/r1/config/policy/perceptive/vN, with lowstate_shm and
#                                   # heightmap_node.py; the robot walks only while the height map is fresh
#   ./start_quest_sim.sh --stereo   # two eyes (6.3 cm apart, assumed), one above the other: depth in the Quest
#                                   # (both options together work too)
#   Ctrl+C                          # stops everything, stream.py first
#
# Logs: logs/quest_sim/<time>/ in this repo (ignored by git).
# Paths: ARENA (default ../arena-r1-kit), MEDIAMTX, FFMPEG_DIR can be set in the environment.

set -u

UPPER=0
STEREO=0
PERCEPTIVE=0
for arg in "$@"; do
    case "$arg" in
        --upper) UPPER=1 ;;
        --stereo) STEREO=1 ;;
        --perceptive) PERCEPTIVE=1 ;;
        *) echo "unknown option $arg; usage: $0 [--upper] [--stereo] [--perceptive]"; exit 2 ;;
    esac
done

# This repo; arena-r1-kit next to it unless ARENA says otherwise. MEDIAMTX and FFMPEG_DIR the same.
MJLAB="$(cd "$(dirname "$0")" && pwd)"
ARENA="${ARENA:-$(dirname "$MJLAB")/arena-r1-kit}"
MEDIAMTX="${MEDIAMTX:-$HOME/tools/mediamtx/mediamtx}"
FFMPEG_DIR="${FFMPEG_DIR:-$HOME/tools/ffmpeg}"
LOGS="$MJLAB/logs/quest_sim/$(date +%Y-%m-%d_%H-%M-%S)"
# CPU priority (nice; higher = yields more): the video for the Quest yields to the robot's own chain.
# Without root nice can only go up, so the controller, simulator, lowstate_shm and heightmap_node
# stay at 0 and the rest is lowered: stream.py (decoding and WebRTC) and mediamtx. Not the
# simulator's ffmpeg: the head camera thread that renders the depth image waits for it, so a
# starved ffmpeg would delay the height map. NICE_VIDEO=0 turns this off.
NICE_VIDEO="${NICE_VIDEO:-10}"

# Each part runs in its own session (setsid): Ctrl+C in the terminal reaches only this script,
# which then stops them in order.
PIDS=()          # started processes, in start order
STREAM_PID=""
TAIL_PID=""
SYSMON_PID=""

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
    # stream.py first: it sends the last zero velocity and says "stopped"; with the Quest's page
    # open that takes up to about 25 s. SIGTERM runs the same cleanup as Ctrl+C (r1/cli.py), and a
    # job started with & ignores SIGINT.
    if [[ -n "$STREAM_PID" ]] && kill -0 "$STREAM_PID" 2>/dev/null; then
        kill -TERM "$STREAM_PID"
        say "waiting for stream.py to say stopped ..."
        for ((i = 0; i < 150; i++)); do kill -0 "$STREAM_PID" 2>/dev/null || break; sleep 0.2; done
        kill -0 "$STREAM_PID" 2>/dev/null && { say "stream.py did not stop in 30 s; ending it"; kill -9 "$STREAM_PID"; }
    fi
    [[ -n "$TAIL_PID" ]] && kill "$TAIL_PID" 2>/dev/null
    [[ -n "$SYSMON_PID" ]] && kill "$SYSMON_PID" 2>/dev/null
    for ((i = ${#PIDS[@]} - 1; i >= 0; i--)); do
        kill "${PIDS[i]}" 2>/dev/null
    done
    wait 2>/dev/null
    say "stopped; logs in $LOGS"
    exit $STATUS
}

# ---- checks ---------------------------------------------------------------------------------
[[ -x "$MEDIAMTX" ]] || fail "mediamtx missing: $MEDIAMTX"
[[ -x "$MJLAB/deploy/robots/r1/build/r1_ctrl" ]] || fail "r1_ctrl not built"
[[ -x "$MJLAB/simulate/build/unitree_mujoco" ]] || fail "unitree_mujoco not built"
[[ -x "$ARENA/.venv/bin/python" ]] || fail "arena-r1-kit venv missing: $ARENA/.venv (its docs/SETUP.md; or set ARENA)"
[[ -n "${DISPLAY:-}" ]] || fail "no DISPLAY: start this from the desktop"
for name in r1_ctrl unitree_mujoco mediamtx; do
    pgrep -x "$name" > /dev/null && fail "$name is already running; stop it first (pkill -x $name)"
done
if ((PERCEPTIVE)); then
    [[ -x "$MJLAB/perception/cpp/build/lowstate_shm" ]] || fail "lowstate_shm not built (perception/cpp)"
    [[ -x "$MJLAB/perception/.venv/bin/python" ]] || fail "perception venv missing: $MJLAB/perception/.venv"
    pgrep -x lowstate_shm > /dev/null && fail "lowstate_shm is already running; stop it first (pkill -x lowstate_shm)"
    pgrep -f heightmap_node.py > /dev/null && fail "heightmap_node.py is already running; stop it first"
fi
for port in 8080 8554 8889; do
    ss -ltnH "sport = :$port" | grep -q . && fail "port $port is in use (an earlier stream.py or mediamtx?); stop it first"
done

mkdir -p "$LOGS"
trap stop_all INT TERM EXIT

# CPU clock, throttling, temperatures, GPU and the busiest processes once a second (sysmon.sh),
# to tell why the simulation sometimes slows down; lowest priority so it disturbs nothing.
(exec setsid nice -n 19 "$MJLAB/sysmon.sh") > "$LOGS/system.log" 2>&1 &
SYSMON_PID=$!

# ---- 1. video relay -------------------------------------------------------------------------
say "1/5 mediamtx (video relay)"
(cd "$MJLAB/simulate" && exec setsid nice -n "$NICE_VIDEO" "$MEDIAMTX" mediamtx_sim.yml) > "$LOGS/mediamtx.log" 2>&1 &
PIDS+=($!)
wait_for "$LOGS/mediamtx.log" "WebRTC\] started" 10 $! "mediamtx"

# ---- 2. controller (before the simulator, so the policy holds the robot from the first step) --
if ((PERCEPTIVE)); then
    START_STATE=Perceptive
    POLICY=$(ls -d "$MJLAB/deploy/robots/r1/config/policy/perceptive"/*/ 2>/dev/null | sort | tail -1)
    say "2/5 controller: Perceptive (policy $(basename "$POLICY")), velocity from the Quest"
else
    START_STATE=Velocity
    say "2/5 controller: velocity from the Quest"
fi
(cd "$MJLAB/deploy/robots/r1/build" && exec setsid ./r1_ctrl --network lo --cmd-source sport --arm-sdk --start-state $START_STATE) \
    > "$LOGS/controller.log" 2>&1 &
PIDS+=($!)
CTRL_PID=$!
wait_for "$LOGS/controller.log" "Waiting for connection" 10 $CTRL_PID "controller"

# ---- 3. simulator ---------------------------------------------------------------------------
say "3/5 simulator with head camera$( ((STEREO)) && echo ", stereo") (no gamepad)"
(cd "$MJLAB/simulate/build" && exec setsid ./unitree_mujoco --joystick 0 --stereo $STEREO) > "$LOGS/simulator.log" 2>&1 &
PIDS+=($!)
SIM_PID=$!
wait_for "$LOGS/controller.log" "walking service" 30 $CTRL_PID "controller"
wait_for "$LOGS/simulator.log" "head camera\] publishing" 30 $SIM_PID "simulator camera"

# ---- 3b. perception (Perceptive only): odometry and height map for the policy ----------------
if ((PERCEPTIVE)); then
    say "3b  lowstate_shm and heightmap_node.py"
    (cd "$MJLAB/perception" && exec setsid cpp/build/lowstate_shm --iface lo) > "$LOGS/lowstate_shm.log" 2>&1 &
    PIDS+=($!)
    wait_for "$LOGS/lowstate_shm.log" "r1_lowstate" 10 $! "lowstate_shm"
    (cd "$MJLAB/perception" && exec setsid .venv/bin/python -u heightmap_node.py --iface lo \
        --timing-log "$LOGS/heightmap_timing.csv") > "$LOGS/heightmap.log" 2>&1 &
    PIDS+=($!)
    wait_for "$LOGS/heightmap.log" "\[heightmap\]" 20 $! "heightmap_node.py"
fi

# ---- 4. arena-r1-kit ------------------------------------------------------------------------
STREAM_ARGS=(--http --webrtc --walk --iface lo --source rtsp://127.0.0.1:8554/robot)
((STEREO)) && STREAM_ARGS+=(--stereo)
if ((UPPER)); then
    # As in arena-r1-kit's docs/DEMO.md. stream.py asks for "follow" before it takes the upper body;
    # in the simulator the script answers.
    STREAM_ARGS+=(--head --arm --teleop-scale 0.7 --arm-slew 40)
    say "4/5 stream.py with head and right arm (answering its \"follow\" prompt)"
else
    say "4/5 stream.py"
fi
(cd "$ARENA" && PATH="$FFMPEG_DIR:$PATH" exec setsid nice -n "$NICE_VIDEO" .venv/bin/python -u stream.py "${STREAM_ARGS[@]}" \
    <<< "follow") > "$LOGS/stream.log" 2>&1 &
STREAM_PID=$!
wait_for "$LOGS/stream.log" "serving on port" 30 $STREAM_PID "stream.py"

# ---- 5. Quest over USB ----------------------------------------------------------------------
# The page's ports reach the Quest through adb reverse. That is lost whenever the Quest reconnects,
# and a Quest plugged in later has none: quest_ports is checked every 2 s while running.
QUEST=""
quest_ports() {
    command -v adb > /dev/null || return
    if ! adb devices 2>/dev/null | grep -qw "device$"; then
        [[ "$QUEST" == on ]] && say "Quest disconnected; plug it in again and reload the page"
        QUEST=off
        return
    fi
    adb reverse --list 2>/dev/null | grep -q "tcp:8080" && adb reverse --list 2>/dev/null | grep -q "tcp:8189" \
        && { QUEST=on; return; }
    if adb reverse tcp:8080 tcp:8080 > /dev/null 2>&1 && adb reverse tcp:8189 tcp:8189 > /dev/null 2>&1; then
        say "Quest connected over USB: open (or reload) http://localhost:8080 in its browser"
        QUEST=on
    fi
}
say "5/5 Quest"
if ! command -v adb > /dev/null; then
    say "adb missing (sudo apt install adb): the Quest is not connected; video on this laptop: http://localhost:8080/flat"
else
    quest_ports
    [[ "$QUEST" == on ]] || say "no Quest on USB yet (adb devices): plug it in and allow USB debugging; this script connects it"
fi

cat <<EOF

  Ready. In the Quest (Wi-Fi on): browser at http://localhost:8080, reload it, wake both
  controllers, Enter VR. Left stick walks, right stick turns, letting go stops, left grip ends VR.
$( ((UPPER)) && printf '%s\n' "  Upper body: right A takes over head and arm (look straight ahead when pressing it), hold" \
  "  right grip and the arm follows your hand, let go and it stays, right B hands back. Walking" \
  "  hands the upper body back first (about 3 s); 1 s after the sticks rest it is taken over again.")
  On this laptop: http://localhost:8080/flat

  Ctrl+C here stops everything. stream.py's output follows; logs in $LOGS

EOF

# Show stream.py (the walk lines) until Ctrl+C or until a part ends.
tail -n +1 -f "$LOGS/stream.log" --pid=$STREAM_PID | grep --line-buffered -v "multicast" &
TAIL_PID=$!
TICK=0
while true; do
    for pid in "${PIDS[@]}" "$STREAM_PID"; do
        kill -0 "$pid" 2>/dev/null || { say "a process ended (pid $pid); see $LOGS"; stop_all; }
    done
    ((TICK++ % 2 == 0)) && quest_ports
    sleep 1
done
