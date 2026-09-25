#!/bin/bash
# Evaluate the seed-0 runs on the held-out test splits.
# Queue lines: "<run> <split_path> <tag>". The queue file is consumed as it
# runs, so pass a copy.
#
#   cp queues/eval_test.txt jobs_test.q
#   TEST_QUEUE=jobs_test.q TEST_GPUS="0" PY=/path/to/ml-env/bin/python ./eval_test.sh
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/ginotEnv/bin/python}"
[ -x "$PY" ] || { echo "set PY to the project interpreter (conda env: ginotEnv)" >&2; exit 1; }
GPUS=(${TEST_GPUS:-0})
JOBS=${TEST_QUEUE:?set TEST_QUEUE}
LOCK="${JOBS%.q}.lock"
MIN_GPU_MB=${TEST_MIN_GPU_MB:-20000}

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
    [ -d "$split" ] || { echo "[gpu$gpu] SKIP $run: no split $split"; continue; }
    local gpumb; gpumb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    if [ "$gpumb" -lt "$MIN_GPU_MB" ]; then
      echo "[gpu$gpu] WAIT $run: only ${gpumb}MiB free, requeueing"
      (exec 9>>"$LOCK"; flock 9; printf '%s\n' "$job" >> "$JOBS"); sleep 240; continue
    fi
    echo "[gpu$gpu] START $run -> $split $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$gpu LGO_KNN_WORKERS=2 $PY -u evaluate.py \
      --run "runs/${run}" --split "$split" --out "$out" \
      > "logs/eval_test/${run}_${tag}.log" 2>&1
    echo "[gpu$gpu] DONE  $run rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty"
}

mkdir -p logs/eval_test results
touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== TEST PASS COMPLETE ($JOBS) $(date) ==="
