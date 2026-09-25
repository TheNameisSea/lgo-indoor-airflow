#!/bin/bash
# Equal-budget comparison at 150 epochs, all arms on the gap-0.30 split.
#
#   PY=/path/to/ml-env/bin/python ./run_baselines_short.sh
#
# Passes the retired far-field shells (--k_solid_far / --far_voxels) for
# the LGO arms, as the original runs did. Do not copy those flags into a
# new run.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
EP=150
run () {  # name model lr cfg extra...
  local name=$1 model=$2 lr=$3 cfg=$4; shift 4
  [ -f "runs/${name}/best.pth" ] && { echo "skip ${name} (done)"; return; }
  echo "=== ${name} : ${model} lr=${lr} cfg=${cfg} ==="
  CUDA_VISIBLE_DEVICES=3 LGO_KNN_WORKERS=2 $PY -u train.py \
    --model "${model}" --cfg "${cfg}" \
    --train_dir splits_cfd_gap/train --val_dir splits_cfd_gap/val \
    --out_dir "runs/${name}" --lr "${lr}" --epochs $EP --seed 0 \
    --val_every 25 --log_every 25 "$@" > "logs/${name}.log" 2>&1
  echo "  ${name} rc=$?"
}
run short_transolver transolver 3e-4 '{"n_pc_tokens":15000,"eval_chunk":5120}'
run short_gno        gno        3e-4 '{"radius":0.06}'
run short_deeponet   deeponet   3e-4 '{}'
run short_gino       gino       3e-4 '{"n_pc_tokens":15000,"grid_res":32}'
# `lgo_ginot`, not `lgo`: `lgo` names the Geom-DeepONet hybrid.
run gap143_short_s0  lgo_ginot  3e-3 '{}' --global_query local --k_solid_far 32 --far_voxels 0.4 0.8 1.6
echo "=== phase 1 complete ==="
