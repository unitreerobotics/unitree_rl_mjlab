#!/usr/bin/env bash
# Run batch policy evaluations across the standard velocity eval terrains.
#
# Example:
#   scripts/run_eval.sh --num-runs 10 --gpus 1,2,3 --checkpoints-csv tmp/checkpoints.csv
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_eval.sh --checkpoints-csv PATH --gpus GPU[,GPU...] [options]

Required:
  --checkpoints-csv PATH   CSV with a 'checkpoint' header and one checkpoint path per row.
  --gpus GPU[,GPU...]      Comma-separated GPU IDs used round-robin, e.g. 1,2,3,4.

Options:
  --num-runs N, --runs N   Number of parallel evaluation runs per checkpoint/terrain.
                           Default: 100.
  --output-dir DIR         Output root. Default: logs/data/eval/YYYYMMDD_eval_${NUM_RUNS}runs.
  --video-run N            Run index to record video for. Default: 0. Use -1 to disable.
  --skip-artifacts         Skip couple.mp4/result.jpg/result_annotated.jpg generation.
  -h, --help               Show this help.

Outputs:
  <output-dir>/combined_summary.csv
  <output-dir>/couple.mp4
  <output-dir>/result.jpg
  <output-dir>/result_annotated.jpg
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

TERRAINS=(rough_curriculum_corridor perlin_noise_corridor random_spread_boxes_corridor)
NUM_RUNS=100
VIDEO_RUN=0
CHECKPOINTS_CSV=""
GPUS_CSV=""
OUT_BASE=""
SKIP_ARTIFACTS=0

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
    --skip-artifacts)
      SKIP_ARTIFACTS=1
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

[[ -n "$CHECKPOINTS_CSV" ]] || die "--checkpoints-csv is required, e.g. --checkpoints-csv tmp/checkpoints.csv."
[[ -n "$GPUS_CSV" ]] || die "--gpus is required, e.g. --gpus 1,2,3,4."
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
  OUT_BASE="logs/data/eval/$(date +%Y%m%d)_eval_${NUM_RUNS}runs"
fi
LOG_DIR="$OUT_BASE/_logs"
mkdir -p "$LOG_DIR"

mapfile -t CKPTS < <(tail -n +2 "$CHECKPOINTS_CSV" | sed '/^[[:space:]]*$/d')
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  die "No checkpoints found in $CHECKPOINTS_CSV (add .pt paths under the 'checkpoint' header)."
fi

echo "[INFO] checkpoints_csv=$CHECKPOINTS_CSV"
echo "[INFO] output_dir=$OUT_BASE"
echo "[INFO] gpus=${GPUS[*]}"
echo "[INFO] ${#CKPTS[@]} checkpoint(s) x ${#TERRAINS[@]} terrain(s) = $(( ${#CKPTS[@]} * ${#TERRAINS[@]} )) jobs, ${NUM_RUNS} runs each."

failed=0
job=0
for ckpt in "${CKPTS[@]}"; do
  encoder=$(basename "$(dirname "$(dirname "$ckpt")")")
  timestamp=$(basename "$(dirname "$ckpt")")
  label="${encoder}_${timestamp}"
  for terrain in "${TERRAINS[@]}"; do
    gpu=${GPUS[$((job % NUM_GPUS))]}
    out="$OUT_BASE/$label/$terrain"
    log="$LOG_DIR/${label}__${terrain}.log"
    mkdir -p "$out"
    echo "[LAUNCH] gpu=$gpu $label / $terrain -> $out"
    (
      PYTHONUNBUFFERED=1 uv run python tools/evaluate_policy.py \
        --checkpoint "$ckpt" \
        --eval-terrain "$terrain" \
        --num-runs "$NUM_RUNS" \
        --video-run "$VIDEO_RUN" \
        --gpus "$gpu" \
        --output-dir "$out" \
        > "$log" 2>&1
      rc=$?
      echo "[DONE rc=$rc] $label / $terrain"
      exit "$rc"
    ) &
    job=$((job + 1))
    while [[ "$(jobs -r | wc -l)" -ge "$NUM_GPUS" ]]; do
      if ! wait -n; then
        failed=1
      fi
    done
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

echo "[ALL DONE] $job evaluations finished. Videos under $OUT_BASE/<label>/<terrain>/run_000/"

echo "[AGGREGATE] Building combined summary..."
uv run python tmp/aggregate_eval.py --base "$OUT_BASE" --out "$OUT_BASE/combined_summary.csv"
echo "[AGGREGATE] Wrote $OUT_BASE/combined_summary.csv"

if [[ "$SKIP_ARTIFACTS" -eq 0 ]]; then
  echo "[ARTIFACTS] Building grid video and annotated result image..."
  uv run python scripts/make_eval_grid.py --base "$OUT_BASE"
  echo "[ARTIFACTS] Wrote $OUT_BASE/couple.mp4 and $OUT_BASE/result_annotated.jpg"
else
  echo "[ARTIFACTS] Skipped."
fi
