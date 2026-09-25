#!/bin/bash
# Physics diagnostics (velocity gradients, wall compliance) for the seed-0
# runs on both splits. Queue lines: "<grad|walls> <run> <split_dir>".
# The queue file is consumed as it runs, so pass a copy.
#
#   cp queues/physics.txt jobs_physics.q
#   PHYS_QUEUE=jobs_physics.q PHYS_GPUS="0" PY=/path/to/ml-env/bin/python ./physics_sweep.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
GPUS=(${PHYS_GPUS:-0})
JOBS=${PHYS_QUEUE:?set PHYS_QUEUE}
LOCK="${JOBS%.q}.lock"
MIN_GPU_MB=${PHYS_MIN_GPU_MB:-8000}
KNN=${PHYS_KNN_WORKERS:-2}

take () { flock 9; local j; j=$(head -1 "$JOBS"); [ -z "$j" ] && return 1; sed -i 1d "$JOBS"; echo "$j"; }

worker () {
  local gpu=$1
  while true; do
    local job; job=$(exec 9>>"$LOCK"; take) || break
    [ -z "$job" ] && break
    local test run split; read -r test run split <<< "$job"
    local out script
    case "$test" in
      grad)  script=physics_gradients.py; out="results/physics_grad_${run}.json" ;;
      walls) script=physics_walls.py;     out="results/physics_walls_${run}.json" ;;
      *)     echo "[gpu$gpu] BAD TEST '$test' in '$job'"; continue ;;
    esac
    [ -f "$out" ] && { echo "[gpu$gpu] skip ${test} ${run} (done)"; continue; }
    [ -f "runs/${run}/best.pth" ] || { echo "[gpu$gpu] SKIP ${run}: no best.pth"; continue; }
    local gpumb; gpumb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    if [ "$gpumb" -lt "$MIN_GPU_MB" ]; then
      # take() has already consumed this line: put it back and wait.
      echo "[gpu$gpu] WAIT ${test} ${run}: only ${gpumb}MiB free, requeueing"
      (exec 9>>"$LOCK"; flock 9; printf '%s\n' "$job" >> "$JOBS")
      sleep 300
      continue
    fi
    echo "[gpu$gpu] START ${test} ${run} -> ${split}/val $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$gpu LGO_KNN_WORKERS=$KNN $PY -u "$script" \
      --run "runs/${run}" --split "${split}/val" --out "$out" \
      > "logs/physics/${test}_${run}.log" 2>&1
    echo "[gpu$gpu] DONE  ${test} ${run} rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty"
}

mkdir -p logs/physics results
touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== PHYSICS SWEEP COMPLETE ($JOBS) $(date) ==="
