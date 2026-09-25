#!/bin/bash
# K ablation: LGO-GDON on the gap-0.30 split, 150 epochs, seed 0, one
# neighbour-count setting changed per run ("base" changes none). One GPU,
# sequential. Parameter count does not depend on K.
#
#   GPU=0 PY=/path/to/ml-env/bin/python ./sweep_k_gap30.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
GPU=${GPU:-0}
EP=150
JOBS=jobs_ksweep.queue
if [ ! -f "$JOBS" ]; then
  cat > "$JOBS" <<'Q'
base
sup0    --k_supply 0
ret0    --k_return 0
leak0   --k_leak 0
sup8    --k_supply 8
sup128  --k_supply 128
sol128  --k_solid 128
Q
fi
mkdir -p logs/ksweep
while :; do
  job=$(head -1 "$JOBS"); [ -z "$job" ] && break
  sed -i 1d "$JOBS"
  tag=$(echo "$job" | awk '{print $1}'); flags=$(echo "$job" | awk '{$1=""; print}')
  name="ksweep_${tag}"
  [ -f "runs/${name}/best.pth" ] && { echo "skip ${name}"; continue; }
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  [ "$free" -lt 10 ] && { echo "ABORT ${name}: ${free}G free"; break; }
  echo "[gpu$GPU] START ${name} ${flags} $(date +%H:%M) (${free}G free)"
  CUDA_VISIBLE_DEVICES=$GPU LGO_KNN_WORKERS=8 $PY -u train.py \
    --model lgo --cfg '{}' --blind_return_velocity \
    --train_dir splits_cfd_gap/train --val_dir splits_cfd_gap/val \
    --out_dir "runs/${name}" --lr 1e-3 --epochs $EP --seed 0 \
    --val_every 25 --log_every 25 $flags \
    > "logs/ksweep/${name}.log" 2>&1
  echo "[gpu$GPU] DONE  ${name} rc=$? $(date +%H:%M)"
done
echo "=== K SWEEP COMPLETE $(date) ==="
