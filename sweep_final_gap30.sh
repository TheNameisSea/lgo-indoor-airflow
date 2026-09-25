#!/bin/bash
# Train every arm on the gap-0.30 split, 450 epochs, seed 0.
# One worker per GPU listed in GPUS.
#
#   GPUS="0" PY=/path/to/ml-env/bin/python ./sweep_final_gap30.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
EP=450
GPUS=(${GPUS:-0})
JOBS=jobs_final30.queue
LOCK=jobs_final30.lock

if [ ! -f "$JOBS" ]; then
  cat > "$JOBS" <<'Q'
gino 3e-3
transolver 1e-3
ginot 1e-4
lgo_ginot 1e-3
deeponet 3e-4
lgo 1e-3
Q
fi

take () { flock 9; local j; j=$(head -1 "$JOBS"); [ -z "$j" ] && return 1; sed -i 1d "$JOBS"; echo "$j"; }

worker () {
  local gpu=$1
  while true; do
    local job; job=$(exec 9>>"$LOCK"; take) || break
    [ -z "$job" ] && break
    local arm lr; read -r arm lr <<< "$job"
    local name="final30_${arm}_lr${lr}"
    if [ -f "runs/${name}/best.pth" ]; then echo "[gpu$gpu] skip ${name}"; continue; fi
    local cfg='{}'; local extra=(); local workers=8
    case "$arm" in
      transolver) cfg='{"n_pc_tokens":15000,"eval_chunk":5120}'; workers=2 ;;
      gino)       cfg='{"n_pc_tokens":15000,"grid_res":32}';     workers=2 ;;
      deeponet|ginot) workers=2 ;;
      lgo_ginot)  extra=(--global_query local) ;;
    esac
    echo "[gpu$gpu] START ${name} $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES=$gpu LGO_KNN_WORKERS=$workers $PY -u train.py \
      --model "$arm" --cfg "$cfg" --blind_return_velocity \
      --train_dir splits_cfd_gap/train --val_dir splits_cfd_gap/val \
      --out_dir "runs/${name}" --lr "$lr" --epochs $EP --seed 0 \
      --val_every 25 --log_every 25 "${extra[@]}" \
      > "logs/tune/${name}.log" 2>&1
    echo "[gpu$gpu] DONE  ${name} rc=$? $(date +%H:%M)"
  done
  echo "[gpu$gpu] queue empty"
}

mkdir -p logs/tune
touch "$LOCK"
for g in "${GPUS[@]}"; do worker "$g" & done
wait
echo "=== SWEEP 1 (gap-0.30, 450 ep, leak-free, no shells) COMPLETE $(date) ==="
