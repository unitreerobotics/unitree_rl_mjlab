#!/usr/bin/env bash
# Run batch policy evaluations on the stairs corridor terrain, 100 runs each.
#
# Mirrors scripts/run_eval.sh but evaluates only the `stairs_corridor` terrain.
# Reads checkpoints from a CSV (one path per row under a `checkpoint` header),
# defaulting to tmp/checkpoints.csv.
#
# Examples:
#   scripts/run_stairs_eval.sh --gpus 0
#   scripts/run_stairs_eval.sh --gpus 1,2,3 --num-runs 100
#   scripts/run_stairs_eval.sh --checkpoints-csv tmp/checkpoints.csv --gpus 0,1
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_stairs_eval.sh --gpus GPU[,GPU...] [options]

Required:
  --gpus GPU[,GPU...]      Comma-separated GPU IDs used round-robin, e.g. 1,2,3.

Options:
  --checkpoints-csv PATH   CSV with a 'checkpoint' header and one checkpoint path
                           per row. Default: tmp/checkpoints.csv.
  --num-runs N, --runs N   Parallel evaluation runs per checkpoint. Default: 100.
  --output-dir DIR         Output root. Default: logs/data/eval/YYYYMMDD_stairs_${NUM_RUNS}runs.
  --video-run N            Run index to record video for. Default: 0. Use -1 to disable.
  --skip-aggregate         Skip combined_summary.csv generation.
  -h, --help               Show this help.

Outputs:
  <output-dir>/<label>/stairs_corridor/summary.csv   per-checkpoint metrics
  <output-dir>/combined_summary.csv                  aggregated across checkpoints
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

trim_spaces() {
  local value=$1
  value=${value#${value%%[![:space:]]*}}
  value=${value%${value##*[![:space:]]}}
  printf '%s' "$value"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

TERRAIN=stairs_corridor
NUM_RUNS=100
VIDEO_RUN=0
CHECKPOINTS_CSV="tmp/checkpoints.csv"
GPUS_CSV=""
OUT_BASE=""
SKIP_AGGREGATE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints-csv)
      [[ $# -ge 2 ]] || die "--checkpoints-csv requires a path."
      CHECKPOINTS_CSV=$2
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || die "--gpus requires a comma-separated list."
      GPUS_CSV=$2
      shift 2
      ;;
    --num-runs|--runs)
      [[ $# -ge 2 ]] || die "$1 requires a positive integer."
      NUM_RUNS=$2
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || die "--output-dir requires a path."
      OUT_BASE=$2
      shift 2
      ;;
    --video-run)
      [[ $# -ge 2 ]] || die "--video-run requires an integer."
      VIDEO_RUN=$2
      shift 2
      ;;
    --skip-aggregate)
      SKIP_AGGREGATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$GPUS_CSV" ]] || die "--gpus is required, e.g. --gpus 0 or --gpus 1,2,3."
is_positive_int "$NUM_RUNS" || die "--num-runs must be a positive integer; got '$NUM_RUNS'."
[[ "$VIDEO_RUN" =~ ^-?[0-9]+$ ]] || die "--video-run must be an integer; got '$VIDEO_RUN'."
[[ -f "$CHECKPOINTS_CSV" ]] || die "Checkpoint CSV not found: $CHECKPOINTS_CSV"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
for i in "${!GPUS[@]}"; do
  GPUS[$i]=$(trim_spaces "${GPUS[$i]}")
  [[ -n "${GPUS[$i]}" ]] || die "--gpus contains an empty entry: '$GPUS_CSV'"
  [[ "${GPUS[$i]}" =~ ^[0-9]+$ ]] || die "GPU IDs must be integers; got '${GPUS[$i]}'."
done
NUM_GPUS=${#GPUS[@]}
[[ "$NUM_GPUS" -gt 0 ]] || die "--gpus must include at least one GPU ID."

if [[ -z "$OUT_BASE" ]]; then
  OUT_BASE="logs/data/eval/$(date +%Y%m%d)_stairs_${NUM_RUNS}runs"
fi
LOG_DIR="$OUT_BASE/_logs"
mkdir -p "$LOG_DIR"

mapfile -t CKPTS < <(tail -n +2 "$CHECKPOINTS_CSV" | sed '/^[[:space:]]*$/d')
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  die "No checkpoints found in $CHECKPOINTS_CSV (add .pt paths under the 'checkpoint' header)."
fi

echo "[INFO] checkpoints_csv=$CHECKPOINTS_CSV"
echo "[INFO] terrain=$TERRAIN"
echo "[INFO] output_dir=$OUT_BASE"
echo "[INFO] gpus=${GPUS[*]}"
echo "[INFO] ${#CKPTS[@]} checkpoint(s), ${NUM_RUNS} runs each."

failed=0
job=0
for ckpt in "${CKPTS[@]}"; do
  encoder=$(basename "$(dirname "$(dirname "$ckpt")")")
  timestamp=$(basename "$(dirname "$ckpt")")
  label="${encoder}_${timestamp}"
  gpu=${GPUS[$((job % NUM_GPUS))]}
  out="$OUT_BASE/$label/$TERRAIN"
  log="$LOG_DIR/${label}__${TERRAIN}.log"
  mkdir -p "$out"
  echo "[LAUNCH] gpu=$gpu $label / $TERRAIN -> $out"
  (
    PYTHONUNBUFFERED=1 uv run python tools/evaluate_policy.py \
      --checkpoint "$ckpt" \
      --eval-terrain "$TERRAIN" \
      --num-runs "$NUM_RUNS" \
      --video-run "$VIDEO_RUN" \
      --gpus "$gpu" \
      --output-dir "$out" \
      > "$log" 2>&1
    rc=$?
    echo "[DONE rc=$rc] $label / $TERRAIN"
    exit "$rc"
  ) &
  job=$((job + 1))
  while [[ "$(jobs -r | wc -l)" -ge "$NUM_GPUS" ]]; do
    if ! wait -n; then
      failed=1
    fi
  done
done

while [[ "$(jobs -r | wc -l)" -gt 0 ]]; do
  if ! wait -n; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  die "One or more evaluations failed. Check logs under $LOG_DIR."
fi

echo "[ALL DONE] $job evaluation(s) finished. Outputs under $OUT_BASE/<label>/$TERRAIN/"

if [[ "$SKIP_AGGREGATE" -eq 0 ]]; then
  echo "[AGGREGATE] Building combined summary..."
  uv run python tmp/aggregate_eval.py --base "$OUT_BASE" --out "$OUT_BASE/combined_summary.csv"
  echo "[AGGREGATE] Wrote $OUT_BASE/combined_summary.csv"
else
  echo "[AGGREGATE] Skipped."
fi
