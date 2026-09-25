#!/bin/bash
# Evaluate the gap-0.58 runs (450 epochs, seed 0) on the validation split.
# One worker per GPU listed in EVAL58_GPUS.
#
#   EVAL58_GPUS="0" PY=/path/to/ml-env/bin/python ./eval_final58.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
GPUS=(${EVAL58_GPUS:-0})
JOBS=jobs_eval58.queue
LOCK=jobs_eval58.lock

if [ ! -f "$JOBS" ]; then
  cat > "$JOBS" <<'Q'
final58_lgo_gino_lr3e-3 splits_cfd_gap58 gap58val
final58_lgo_ginot_lr1e-3 splits_cfd_gap58 gap58val
final58_lgo_lr1e-3 splits_cfd_gap58 gap58val
final58_deeponet_lr3e-4 splits_cfd_gap58 gap58val
final58_ginot_lr1e-4 splits_cfd_gap58 gap58val
final58_gino_lr3e-3 splits_cfd_gap58 gap58val
final58_transolver_lr1e-3 splits_cfd_gap58 gap58val
final30_lgo_gino_lr3e-3 splits_cfd_gap gapval
Q
fi

take () { flock 9; local j; j=$(head -1 "$JOBS"); [ -z "$j" ] && return 1; sed -i 1d "$JOBS"; echo "$j"; }

worker () {
  local gpu=$1
  while true; do
    local job; job=$(exec 9>>"$LOCK"; take) || break
    [ -z "$job" ] && break
    local run split tag; read -r run split tag <<< "$job"
    local out="results/${run}_${tag}.json"
    [ -f "$out" ] && { echo "[gpu$gpu] skip $run (done)"; continue; }
    [ -f "runs/${run}/best.pth" ] || { echo "[gpu$gpu] SKIP $run: no best.pth"; continue; }
    local freegb; freegb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    [ "$freegb" -lt 10 ] && { echo "[gpu$gpu] ABORT $run: ${freegb}G disk"; break; }
    echo "[gpu$gpu] START $run -> $split/val $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$gpu LGO_KNN_WORKERS=4 $PY -u evaluate.py \
      --run "runs/${run}" --split "${split}/val" --out "$out" \
      > "logs/eval_final58/${run}.log" 2>&1
    echo "[gpu$gpu] DONE  $run rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty"
}

mkdir -p logs/eval_final58 results
touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== EVAL FINAL58 COMPLETE $(date) ==="
