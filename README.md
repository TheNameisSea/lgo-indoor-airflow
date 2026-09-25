# LGO — a local–global neural operator for indoor airflow

Predicts velocity (u, v, w) and temperature T at arbitrary query points in a 3D
room from a boundary point cloud and its boundary conditions — no signed
distance field, no volume mesh, no hand-extracted vent anchors.

The core component is a **local branch**: each query gathers nearby boundary
points stratified by patch type (supply / return / solid / leak), sees them only
as offsets in metres, and cross-attends over them. It attaches to a host decoder
with a zero-initialised output projection, so each hybrid is bit-identical to its
host at initialisation.

| arm | host | flag |
|---|---|---|
| LGO-GDON | Geom-DeepONet | `--model lgo_gdon` |
| LGO-GINOT | GINOT cross-attention trunk | `--model lgo_ginot` |
| LGO-GINO | GINO graph-kernel + FNO | `--model lgo_gino` |
| LGO-TRANSOLVER | Transolver trunk | `--model lgo_transolver` |

The hosts alone are `--model deeponet`, `ginot`, `gino`, `transolver`.

## Install

```bash
# ML environment (Python 3.12). Install torch for your CUDA version first.
pip install -r requirements.txt

# CFD environment — only needed to build splits or regenerate the dataset.
conda env create -f environment-cfd.yml
```

## Data

No data is committed. Download it and place it as described in
[REPRODUCING.md §1](REPRODUCING.md#1-put-the-data-in-place):

| data | source | goes in |
|---|---|---|
| indoor CFD dataset | Zenodo [10.5281/zenodo.22955315](https://doi.org/10.5281/zenodo.22955315) | `cfd/dataset/` |
| trained checkpoints (optional) | released after the review period | `runs/` |
| ShapeNet-Car | Umetani release + Zenodo 13737721 | `external/shapenet_car/` |

Every directory that expects data also carries a `DOWNLOAD.txt`.

## Quick start

```bash
PY=/path/to/ml-env/bin/python
CPY=/path/to/cfd-env/bin/python

$PY -m tests.test_knn_cache && $PY -m tests.test_local_branch && $PY -m tests.test_lgo

$CPY data/splits_cfd_gap.py --gap 0.30 --extend --materialize     # after placing the data

CUDA_VISIBLE_DEVICES=0 LGO_KNN_WORKERS=8 $PY -u train.py \
  --model lgo_ginot --global_query local --blind_return_velocity \
  --train_dir splits_cfd_gap/train --val_dir splits_cfd_gap/val \
  --out_dir runs/NAME --lr 1e-3 --epochs 450 --seed 0

$PY evaluate.py --run runs/NAME --split splits_cfd_gap/val --out results/NAME_val.json
```

## Documentation

- [REPRODUCING.md](REPRODUCING.md) — data placement and the commands behind every reported result
- [USAGE.md](USAGE.md) — environments, flags, evaluation, tests, troubleshooting
- [cfd/README.md](cfd/README.md) — the CFD dataset-generation pipeline

## Citation

```bibtex
@inproceedings{anonymous2027separation,
  title     = {Separation-Controlled Layout Generalisation for Three-Dimensional
               Indoor Airflow Prediction with a Dual-Stream Neural Operator},
  author    = {Anonymous Authors},
  booktitle = {Submitted to the International Conference on Learning Representations},
  year      = {2027},
  note      = {Under review}
}
```
