#!/bin/bash
# Evaluate the seed-replication runs. Queue lines: "<run> <split_dir> <tag>".
# The queue file is consumed as it runs, so pass a copy.
#
#   cp queues/eval_seeds.txt jobs_eval_seeds.q
#   EVAL_QUEUE=jobs_eval_seeds.q EVAL_GPUS="0" PY=/path/to/ml-env/bin/python ./eval_seeds.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
GPUS=(${EVAL_GPUS:-0})
JOBS=${EVAL_QUEUE:?set EVAL_QUEUE}
LOCK="${JOBS%.q}.lock"

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
    CUDA_VISIBLE_DEVICES=$gpu LGO_KNN_WORKERS=${EVAL_KNN_WORKERS:-4} $PY -u evaluate.py \
      --run "runs/${run}" --split "${split}/val" --out "$out" \
      > "logs/eval_seeds/${run}.log" 2>&1
    echo "[gpu$gpu] DONE  $run rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty"
}

mkdir -p logs/eval_seeds results
touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== EVAL SEEDS COMPLETE ($JOBS) $(date) ==="
