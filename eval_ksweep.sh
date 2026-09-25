#!/bin/bash
# Evaluate the K-ablation runs on the gap-0.30 validation split.
#
#   GPU=0 PY=/path/to/ml-env/bin/python ./eval_ksweep.sh
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
GPU=${GPU:-0}
mkdir -p logs/eval_ksweep results
for name in ksweep_base ksweep_sup0 ksweep_ret0 ksweep_leak0 ksweep_sup8 ksweep_sup128 ksweep_sol128; do
  out="results/${name}_gapval.json"
  [ -f "$out" ] && { echo "skip $name (done)"; continue; }
  [ -f "runs/${name}/best.pth" ] || { echo "SKIP $name: no best.pth"; continue; }
  free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  [ "$free" -lt 10 ] && { echo "ABORT $name: ${free}G free"; break; }
  echo "=== eval $name on gpu$GPU $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=$GPU LGO_KNN_WORKERS=4 $PY -u evaluate.py \
    --run "runs/${name}" --split splits_cfd_gap/val --out "$out" \
    > "logs/eval_ksweep/${name}.log" 2>&1
  echo "  rc=$? -> $out $(date +%H:%M)"
done
echo "=== EVAL KSWEEP COMPLETE $(date) ==="
