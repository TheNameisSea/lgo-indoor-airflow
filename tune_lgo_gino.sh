#!/bin/bash
# Learning-rate sweep for LGO-GINO on the gap-0.30 split, 150 epochs, seed 0,
# rates {3e-3, 1e-3, 3e-4, 1e-4}. One worker per GPU listed in TUNE_GPUS.
#
#   TUNE_GPUS="0" PY=/path/to/ml-env/bin/python ./tune_lgo_gino.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
EP=150
GPUS=(${TUNE_GPUS:-0})
JOBS=jobs_tune_gino.queue
LOCK=jobs_tune_gino.lock
MIN_FREE_GB=10
# neuralop's native_neighbor_search builds a full torch.cdist(queries, data).
# Refuse a GPU that cannot hold it.
MIN_GPU_MB=20000

if [ ! -f "$JOBS" ]; then
  cat > "$JOBS" <<'Q'
lgo_gino 3e-3
lgo_gino 1e-3
lgo_gino 3e-4
lgo_gino 1e-4
Q
fi

take () { flock 9; local j; j=$(head -1 "$JOBS"); [ -z "$j" ] && return 1; sed -i 1d "$JOBS"; echo "$j"; }

worker () {
  local gpu=$1
  while true; do
    local job; job=$(exec 9>>"$LOCK"; take) || break
    [ -z "$job" ] && break
    local arm lr; read -r arm lr <<< "$job"
    local name="tune_${arm}_lr${lr}"
    if [ -f "runs/${name}/best.pth" ]; then echo "[gpu$gpu] skip ${name}"; continue; fi
    local freegb; freegb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    if [ "$freegb" -lt "$MIN_FREE_GB" ]; then
      echo "[gpu$gpu] ABORT ${name}: only ${freegb}G disk"; break; fi
    local gpumb; gpumb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    if [ "$gpumb" -lt "$MIN_GPU_MB" ]; then
      echo "[gpu$gpu] ABORT ${name}: only ${gpumb}MiB GPU free (<${MIN_GPU_MB})"; break; fi
    echo "[gpu$gpu] START ${name} $(date +%H:%M) (${freegb}G disk, ${gpumb}MiB gpu)"
    CUDA_VISIBLE_DEVICES=$gpu LGO_KNN_WORKERS=4 $PY -u train.py \
      --model "$arm" --cfg '{"n_pc_tokens":15000,"grid_res":32}' \
      --blind_return_velocity \
      --train_dir splits_cfd_gap/train --val_dir splits_cfd_gap/val \
      --out_dir "runs/${name}" --lr "$lr" --epochs $EP --seed 0 \
      --val_every 25 --log_every 25 \
      > "logs/tune_gino/${name}.log" 2>&1
    echo "[gpu$gpu] DONE  ${name} rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty"
}

mkdir -p logs/tune_gino
touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== LR TUNE COMPLETE $(date) ==="
