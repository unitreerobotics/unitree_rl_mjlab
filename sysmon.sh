#!/usr/bin/env bash
# Logs the laptop's state once a second, to find out why the simulation sometimes slows down 6-10x
# (2026-11-04: every part at once, the height map's own CPU time included, i.e. slower CPU rather
# than waiting for one). One line per second:
#
#   time  cpu MHz avg/min/max  throttle events package/core since the last line  temperatures (°C)
#   GPU °C, MHz, %, W, throttle reasons  load  top 3 processes by CPU over that second
#
#   ./sysmon.sh > system.log          # until Ctrl+C or killed; start_quest_sim.sh starts it itself
#
# Throttle counters (thermal_throttle/*_throttle_count) count the CPU's thermal throttle events;
# a rise means the CPU was clocked down because it got too hot. GPU throttle reasons: nvidia-smi's
# clocks_event_reasons.active bit mask (0x1 idle, 0x4 power cap, 0x20/0x40 thermal slowdown, ...).

set -u
INTERVAL="${INTERVAL:-1}"

sum_counts() {  # file name under thermal_throttle; summed over CPUs (package: per package, cpu0's)
    local f=$1 total=0 v
    for v in /sys/devices/system/cpu/cpu*/thermal_throttle/"$f"; do
        [[ -r "$v" ]] && total=$((total + $(<"$v")))
    done
    echo "$total"
}

temps() {
    local z out=""
    for z in /sys/class/thermal/thermal_zone*; do
        local t; t=$(<"$z/type")
        case "$t" in x86_pkg_temp|SEN*|acpitz) ;; *) continue ;; esac
        local v; v=$(<"$z/temp")
        ((v > 0)) && out+="$t=$((v / 1000)) "
    done
    echo "${out% }"
}

# GPU: one nvidia-smi that keeps running (spawning it every second costs ~50 ms CPU each time).
GPU_FILE=$(mktemp)
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=temperature.gpu,clocks.gr,utilization.gpu,power.draw,clocks_event_reasons.active \
        --format=csv,noheader,nounits -lms $((INTERVAL * 1000)) > "$GPU_FILE" 2>/dev/null &
    GPU_PID=$!
else
    GPU_PID=""
fi
trap '[[ -n "$GPU_PID" ]] && kill "$GPU_PID" 2>/dev/null; rm -f "$GPU_FILE"; exit 0' INT TERM EXIT

echo "# sysmon: $(nproc) CPUs, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'), profile $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo -)"
echo "# time      cpuMHz avg/min/max  throttle pkg/core  temps  | gpu °C,MHz,%,W,reasons | load1 | top CPU% (over the second)"
prev_pkg=$(sum_counts package_throttle_count)
prev_core=$(sum_counts core_throttle_count)
while true; do
    # top's second sample covers the last INTERVAL seconds; it also paces the loop.
    top_lines=$(top -b -n 2 -d "$INTERVAL" -o %CPU -w 200 2>/dev/null | awk '/^top -/{n++} n==2 && /^ *[0-9]+ /' | head -3 \
        | awk '{printf "%s(%s) %s%%  ", $12, $1, $9}')
    mhz=$(awk -F: '/cpu MHz/{v=$2+0; s+=v; n++; if(min==""||v<min)min=v; if(v>max)max=v} END{printf "%d/%d/%d", s/n, min, max}' /proc/cpuinfo)
    pkg=$(sum_counts package_throttle_count); core=$(sum_counts core_throttle_count)
    gpu=$(tail -1 "$GPU_FILE" 2>/dev/null | tr -d ' ')
    printf '%s  %s  %d/%d  %s | %s | %s | %s\n' "$(date +%H:%M:%S)" "$mhz" $((pkg - prev_pkg)) $((core - prev_core)) \
        "$(temps)" "${gpu:--}" "$(cut -d' ' -f1 /proc/loadavg)" "$top_lines"
    prev_pkg=$pkg; prev_core=$core
done
