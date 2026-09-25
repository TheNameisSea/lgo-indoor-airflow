#!/bin/bash
# Evaluate the gap-0.30 runs (450 epochs, seed 0) on the validation split.
#
#   GPU=0 PY=/path/to/ml-env/bin/python ./eval_final30.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
GPU=${GPU:-0}
mkdir -p logs/eval_final30 results
for name in final30_gino_lr3e-3 final30_ginot_lr1e-4 \
            final30_transolver_lr1e-3 final30_deeponet_lr3e-4; do
  out="results/${name}_gapval.json"
  [ -f "$out" ] && { echo "skip $name (done)"; continue; }
  [ -f "runs/${name}/best.pth" ] || { echo "SKIP $name: no best.pth"; continue; }
  echo "=== eval $name on gpu$GPU $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=$GPU LGO_KNN_WORKERS=4 $PY -u evaluate.py \
    --run "runs/${name}" --split splits_cfd_gap/val --out "$out" \
    > "logs/eval_final30/${name}.log" 2>&1
  echo "  rc=$? -> $out"
done
echo "=== EVAL final30 COMPLETE $(date) ==="
