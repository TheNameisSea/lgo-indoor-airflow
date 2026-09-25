#!/bin/bash
# Seed replication (seeds 1 and 2), 450 epochs, per-arm flags as in
# sweep_final58.sh. Queue lines: "<g30|g58> <arm> <lr> <seed>".
# The queue file is consumed as it runs, so pass a copy.
#
#   cp queues/seeds.txt jobs_seeds.q
#   SEED_QUEUE=jobs_seeds.q SEED_GPUS="0" PY=/path/to/ml-env/bin/python ./sweep_seeds.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
EP=450
GPUS=(${SEED_GPUS:-0})
JOBS=${SEED_QUEUE:?set SEED_QUEUE}
LOCK="${JOBS%.q}.lock"
MIN_FREE_GB=15
MIN_GPU_MB=20000          # gino/lgo_gino: neuralop cdist OOMs below this

take () { flock 9; local j; j=$(head -1 "$JOBS"); [ -z "$j" ] && return 1; sed -i 1d "$JOBS"; echo "$j"; }

worker () {
  local gpu=$1
  while true; do
    local job; job=$(exec 9>>"$LOCK"; take) || break
    [ -z "$job" ] && break
    local split arm lr seed; read -r split arm lr seed <<< "$job"
    local dir name
    if [ "$split" = "g58" ]; then dir=splits_cfd_gap58; name="final58_${arm}_lr${lr}_s${seed}"
    else                          dir=splits_cfd_gap;   name="final30_${arm}_lr${lr}_s${seed}"; fi
    [ -f "runs/${name}/best.pth" ] && { echo "[gpu$gpu] skip ${name}"; continue; }
    local freegb; freegb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    [ "$freegb" -lt "$MIN_FREE_GB" ] && { echo "[gpu$gpu] ABORT ${name}: ${freegb}G disk"; break; }
    local gpumb; gpumb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")
    [ "$gpumb" -lt "$MIN_GPU_MB" ] && { echo "[gpu$gpu] ABORT ${name}: ${gpumb}MiB gpu"; break; }
    local cfg='{}'; local extra=(); local workers=8
    case "$arm" in
      transolver)        cfg='{"n_pc_tokens":15000,"eval_chunk":5120}'; workers=2 ;;
      gino|lgo_gino)     cfg='{"n_pc_tokens":15000,"grid_res":32}';     workers=4 ;;
      deeponet|ginot)    workers=2 ;;
      lgo_ginot)         extra=(--global_query local) ;;
    esac
    echo "[gpu$gpu] START ${name} [${split}] $(date +%H:%M) (${freegb}G, ${gpumb}MiB)"
    CUDA_VISIBLE_DEVICES=$gpu LGO_KNN_WORKERS=$workers $PY -u train.py \
      --model "$arm" --cfg "$cfg" --blind_return_velocity \
      --train_dir "$dir/train" --val_dir "$dir/val" \
      --out_dir "runs/${name}" --lr "$lr" --epochs $EP --seed "$seed" \
      --val_every 25 --log_every 25 "${extra[@]}" \
      > "logs/seeds/${name}.log" 2>&1
    echo "[gpu$gpu] DONE  ${name} rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty"
}

mkdir -p logs/seeds
touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== SEED SWEEP COMPLETE ($JOBS) $(date) ==="
