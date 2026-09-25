#!/bin/bash
# ShapeNet-Car training runs. One GPU, sequential; a cell whose best.pth exists
# is skipped, so the script can be re-run after an interruption.
#
#   GPU=0 PY=/path/to/ml-env/bin/python ./run_shapenet.sh              # every cell
#   GPU=0 PY=/path/to/ml-env/bin/python ./run_shapenet.sh NAME [NAME...] # selected cells
#
# Needs the ShapeNet split trees -- see REPRODUCING.md step 2.
cd "$(dirname "$0")" || exit 1
PY=${PY:-python}
GPU=${GPU:-0}
F0=splits_shapenet_matched_csv      # published fold 0
F1=splits_shapenet_fold1_csv        # published fold 1

# name                         model           cfg                                    tree  lr
CELLS=$(cat <<'EOF'
sn_gate_gino_s0                gino            {}                                     F0    1e-3
sn_fold0_lgo_gino_s0           lgo_gino        {}                                     F0    1e-3
sn_fold0_gino_sdf_s0           gino            {"latent_sdf":true}                    F0    1e-3
sn_fold0_lgo_gino_sdf_s0       lgo_gino        {"latent_sdf":true}                    F0    1e-3
sn_fold0_transolver_s0         transolver      {"mlp_ratio":2}                        F0    1e-3
sn_fold0_transolver_sdf_s0     transolver      {"mlp_ratio":2,"point_sdf":true}       F0    1e-3
sn_fold0_lgo_transolver_s0     lgo_transolver  {"mlp_ratio":2}                        F0    1e-3
sn_f1_lgo_gino_lr3e-3          lgo_gino        {}                                     F1    3e-3
sn_f1_lgo_gino_lr1e-3          lgo_gino        {}                                     F1    1e-3
sn_f1_lgo_gino_lr3e-4          lgo_gino        {}                                     F1    3e-4
sn_f1_lgo_gino_lr1e-4          lgo_gino        {}                                     F1    1e-4
sn_f1_lgo_gino_sdf_lr3e-3      lgo_gino        {"latent_sdf":true}                    F1    3e-3
EOF
)

mkdir -p logs/shapenet
echo "$CELLS" | while read -r name model cfg tree lr; do
  [ -z "$name" ] && continue
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name"; then continue; fi
  if [ -f "runs/$name/best.pth" ]; then echo "[shapenet] skip $name (done)"; continue; fi
  dir=$([ "$tree" = F1 ] && echo "$F1" || echo "$F0")
  echo "[shapenet] $name start $(date)"
  CUDA_VISIBLE_DEVICES=$GPU LGO_KNN_WORKERS=2 $PY -u train.py --model "$model" \
    --cfg "$cfg" \
    --strat_wall_dist 0.05 --n_case_pool 75000 --lambda_t 0 \
    --train_dir "$dir/train" --val_dir "$dir/val" \
    --out_dir "runs/$name" --lr "$lr" --epochs 200 --seed 0 \
    --val_every 20 --log_every 20 \
    > "logs/shapenet/$name.log" 2>&1 < /dev/null
  echo "[shapenet] $name done rc=$? $(date)"
done
