#!/bin/bash
# Train every arm on the gap-0.58 split (and LGO-GINO on gap-0.30),
# 450 epochs, seed 0. One worker per GPU listed in F58_GPUS.
#
#   F58_GPUS="0" PY=/path/to/ml-env/bin/python ./sweep_final58.sh
#
# The per-arm flags here are the reference that sweep_seeds.sh copies.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
EP=450
GPUS=(${F58_GPUS:-0})
JOBS=jobs_final58.queue
LOCK=jobs_final58.lock
MIN_FREE_GB=15
MIN_GPU_MB=20000          # gino/lgo_gino: neuralop cdist OOMs below this

if [ ! -f "$JOBS" ]; then
  # slowest first; "split arm lr"
  cat > "$JOBS" <<'Q'
g58 gino 3e-3
g58 lgo_gino 3e-3
g30 lgo_gino 3e-3
g58 transolver 1e-3
g58 ginot 1e-4
g58 lgo_ginot 1e-3
g58 deeponet 3e-4
g58 lgo 1e-3
Q
fi

take () { flock 9; local j; j=$(head -1 "$JOBS"); [ -z "$j" ] && return 1; sed -i 1d "$JOBS"; echo "$j"; }

worker () {
  local gpu=$1
  while true; do
    local job; job=$(exec 9>>"$LOCK"; take) || break
    [ -z "$job" ] && break
    local split arm lr; read -r split arm lr <<< "$job"
    local dir name
    if [ "$split" = "g58" ]; then dir=splits_cfd_gap58; name="final58_${arm}_lr${lr}"
    else                          dir=splits_cfd_gap;   name="final30_${arm}_lr${lr}"; fi
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
      --out_dir "runs/${name}" --lr "$lr" --epochs $EP --seed 0 \
      --val_every 25 --log_every 25 "${extra[@]}" \
      > "logs/final58/${name}.log" 2>&1
    echo "[gpu$gpu] DONE  ${name} rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty"
}

mkdir -p logs/final58
touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== SWEEP 2 + LGO-GINO COMPLETE $(date) ==="
