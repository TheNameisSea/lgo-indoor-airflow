#!/bin/bash
# Learning-rate sweep: six arms x rates {3e-3, 1e-3, 3e-4, 1e-4}, gap-0.30
# split, 150 epochs, seed 0. One worker per GPU listed in GPUS.
#
#   GPUS="0" PY=/path/to/ml-env/bin/python ./tune_sweep.sh
#
# Passes the retired far-field shells (--k_solid_far / --far_voxels) for
# the LGO arms, as the original runs did. Do not copy those flags into a
# new run.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
EP=150
GPUS=(${GPUS:-0})
JOBS=jobs.queue
LOCK=jobs.lock

# ---- build the job list (cost-ordered: slowest first, so the tail packs) ----
if [ ! -f "$JOBS" ]; then
  : > "$JOBS"
  for lr in 3e-3 1e-3 3e-4 1e-4; do
    for arm in lgo lgo_ginot gino transolver deeponet ginot; do
      echo "$arm $lr" >> "$JOBS"
    done
  done
fi

take_job () {                      # atomically pop one line
  flock 9
  local j; j=$(head -1 "$JOBS")
  [ -z "$j" ] && return 1
  sed -i 1d "$JOBS"
  echo "$j"
}

worker () {
  local gpu=$1
  while true; do
    local job; job=$(exec 9>>"$LOCK"; take_job) || break
    [ -z "$job" ] && break
    local arm lr; read -r arm lr <<< "$job"
    local name="tune_${arm}_lr${lr}"
    if [ -f "runs/${name}/best.pth" ]; then
      echo "[gpu$gpu] skip ${name} (done)"; continue
    fi
    local cfg='{}' extra=() workers=2
    case "$arm" in
      transolver) cfg='{"n_pc_tokens":15000,"eval_chunk":5120}' ;;
      gino)       cfg='{"n_pc_tokens":15000,"grid_res":32}' ;;
      lgo)        extra=(--k_solid_far 32 --far_voxels 0.4 0.8 1.6); workers=4 ;;
      lgo_ginot)  extra=(--global_query local --k_solid_far 32 --far_voxels 0.4 0.8 1.6); workers=4 ;;
    esac
    echo "[gpu$gpu] START ${name} $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$gpu LGO_KNN_WORKERS=$workers $PY -u train.py \
      --model "$arm" --cfg "$cfg" \
      --train_dir splits_cfd_gap/train --val_dir splits_cfd_gap/val \
      --out_dir "runs/${name}" --lr "$lr" --epochs $EP --seed 0 \
      --val_every 25 --log_every 25 "${extra[@]}" \
      > "logs/tune/${name}.log" 2>&1
    echo "[gpu$gpu] DONE  ${name} rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty, worker exiting"
}

touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== SWEEP COMPLETE $(date) ==="
