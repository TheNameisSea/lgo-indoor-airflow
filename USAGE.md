# Using this repo

A reference for environments, commands and flags. For the step-by-step path to
the paper's numbers, see [REPRODUCING.md](REPRODUCING.md).

## 1. Arms

One local branch, four hosts. Each hybrid and its host are bit-identical at
initialisation, so every pair below is an exact ablation.

| arm | host | hybrid flag | host flag |
|---|---|---|---|
| LGO-GDON | Geom-DeepONet | `--model lgo_gdon` (alias `lgo`) | `--model deeponet` |
| LGO-GINOT | GINOT cross-attention trunk | `--model lgo_ginot` | `--model ginot` |
| LGO-GINO | GINO graph-kernel + FNO | `--model lgo_gino` | `--model gino` |
| LGO-TRANSOLVER | Transolver trunk | `--model lgo_transolver` | `--model transolver` |

LGO-TRANSOLVER injects the branch at the trunk input rather than after the
stack, because Transolver's output head sits inside its final block.

## 2. Environments

Two conda environments, not interchangeable:

| | contains | used for |
|---|---|---|
| ML env | torch, CUDA | training, evaluation, tests |
| CFD env | OpenFOAM v2412, no torch | `cfd/` pipeline, split construction |

```bash
PY=/path/to/ml-env/bin/python
CPY=/path/to/cfd-env/bin/python
```

- Call the interpreter directly with `-u`. `conda run` buffers stdout, so a
  working job looks identical to a hung one.
- `cfd/sweep.py` needs OpenFOAM on `PATH`, not only the interpreter:
  `PATH=/path/to/cfd-env/bin:$PATH WM_PROJECT_DIR=/path/to/cfd-env $CPY cfd/sweep.py ...`

| variable | effect |
|---|---|
| `CUDA_VISIBLE_DEVICES` | which GPU; always set it |
| `LGO_KNN_WORKERS` | KD-tree query threads (default: all cores). Set 2–8, especially with several jobs running |
| `JOBS` | `cfd/export_all.sh` only: cases exported in parallel (default 3) |

## 3. Repo map

```
train.py                  training entry point, all arms
evaluate.py               indoor metric suite -> results/*.json
evaluate_shapenet.py      ShapeNet-Car metric suite (banded by wall distance)
evaluate_nulls.py         zero-parameter baselines, indoor
evaluate_nulls_shapenet.py
physics_gradients.py      gradient and divergence, model vs CFD
physics_walls.py          wall compliance and temperature bounds
physics_timing.py         inference cost
visualize.py              prediction vs truth slices for a run
visualize_case.py         a dataset case alone, no model
compare_runs.py           training curves from history.json, no GPU
thermo_zone.py            zone-aware checkpoint selection (--select_on)
shapenet_drag.py          drag coefficient from surface fields (CFD env)
make_*.py                 figure builders

model/                    local_branch.py (the branch), boundary_lookup.py (the gather),
                          lgo.py, lgo_ginot.py, lgo_gino.py, lgo_transolver.py (hybrids)
baselines/models/         deeponet, gino, gno, transolver, transolver_sdf (hosts)
data/                     knn_cache.py, split builders, committed split JSONs, ShapeNet loaders
cfd/                      OpenFOAM dataset-generation pipeline
legacy/                   vendored trainer and dataset code
tests/                    guards; run them after any change

*.sh                      training and evaluation drivers, see REPRODUCING.md
run_shapenet.sh           every ShapeNet-Car training run
queues/                   work lists for the queue-driven drivers (copy before use)
```

Data directories (`cfd/dataset/`, `external/`, `splits_*/`, `runs/`,
`results/`, `logs/`) are not committed; each holds a `DOWNLOAD.txt`.

`legacy/` and `baselines/models/` are wrapped, never edited, so each host flag
keeps producing exactly the model its file defines. Add behaviour by subclassing
or wrapping.

## 4. Training

```bash
CUDA_VISIBLE_DEVICES=0 LGO_KNN_WORKERS=8 $PY -u train.py \
  --model lgo_ginot --global_query local --blind_return_velocity \
  --train_dir splits_cfd_gap/train --val_dir splits_cfd_gap/val \
  --out_dir runs/NAME --lr 1e-3 --epochs 450 --seed 0
```

Learning rates:

| arm | lr | | arm | lr |
|---|---|---|---|---|
| `lgo_gdon` | 1e-3 | | `deeponet` | 3e-4 |
| `lgo_ginot` | 1e-3 | | `ginot` | 1e-4 |
| `lgo_gino` | 3e-3 | | `gino` | 3e-3 |
| `lgo_transolver` | 1e-3 | | `transolver` | 1e-3 |

Per-arm settings:

- **`lgo_ginot`** needs `--global_query local`. The startup banner must read
  `global=True`. Do not use `--model lgo_ginot_local`: it switches the global
  path off, and `--global_query` then has no effect.
- **`lgo_gdon`** takes its decoder settings in `--cfg`, e.g.
  `--cfg '{"inject":"seed"}'`. `inject` changes behaviour without changing
  parameter names, so it must travel in `--cfg` to be recorded in `args.json`.
- **ShapeNet-Car** runs add `--strat_wall_dist 0.05 --n_case_pool 75000 --lambda_t 0`.
  The two go together: `--strat_wall_dist` with a smaller pool biases the
  sampling partition.

Main flags:

| flag | default | |
|---|---|---|
| `--blind_return_velocity` | off | pass it for indoor runs |
| `--k_supply` / `--k_return` / `--k_solid` / `--k_leak` | 32 / 16 / 32 / 8 | neighbours per group |
| `--k_solid_far`, `--far_voxels` | 0 / – | leave unset |
| `--n_rbf` / `--rbf_max` | 12 / 4.0 | radial basis on neighbour distance |
| `--soft_knn` | 0 | smooth neighbour window (extra neighbours, zero new parameters) |
| `--local_query` | `learned` | |
| `--cfg` | `{}` | host-specific settings, JSON |
| `--epochs` / `--lr` / `--seed` | 2000 / 1e-3 / 0 | |
| `--vel_floor` | 0.2 | floor in the relative velocity loss |
| `--accum_cases` | 1 | cases per optimiser step |
| `--select_on` | `joint` | checkpoint criterion; see `thermo_zone.py` |
| `--val_every` / `--log_every` / `--ckpt_every` | 25 / 25 / 500 | |

Rules:

- **Do not pass `--stats_path`.** Without it, normalisation is computed from the
  run's own training set and saved in the run directory.
- **Do not early-stop.** The schedule is OneCycle: the learning rate rises for
  the first 30% of epochs, and the final quality comes from the anneal. To
  shorten a run, relaunch with a smaller `--epochs`.
- **Budgets are in optimiser updates.** With `--accum_cases 1`, updates =
  epochs × training cases; compare runs at equal updates.
- **One job per GPU.** A job still loading its dataset holds no GPU memory, so a
  free-memory check cannot detect it.
- Startup (loading cases, building KD-trees) takes several minutes before the
  first epoch.

Each run writes `runs/NAME/{args.json, history.json, best.pth, last.pth, thermo_stats.pt}`.

## 5. Evaluation

```bash
$PY evaluate.py --run runs/NAME --split splits_cfd_gap/val --out results/NAME_val.json
$PY evaluate_nulls.py --train_dir splits_cfd_gap/train --split splits_cfd_gap/val \
    --out results/nulls_gap_val.json
$PY evaluate_shapenet.py --run runs/NAME --split splits_shapenet_matched_csv/val \
    --out results/NAME_shapenet.json
```

- Always pass `--split`; the default path does not exist.
- The model is rebuilt from the run's own `args.json`, including
  `--blind_return_velocity`. Never edit an old `args.json`.
- ShapeNet-Car runs must be scored with `evaluate_shapenet.py`. `evaluate.py`
  bands by distance to supply vents and would report no zones on a car.
- Results carry per-case arrays (`per_case`, `per_case_zones`,
  `per_case_comfort`); compare models with paired tests over them.

## 6. Physics and visualisation

```bash
$PY physics_gradients.py --run runs/NAME --split splits_cfd_gap/val --out results/physics_gradients_NAME.json
$PY physics_walls.py     --run runs/NAME --split splits_cfd_gap/val --out results/physics_walls_NAME.json
$PY visualize.py --run runs/NAME
$PY visualize_case.py --case splits_cfd_gap/val/<case>
$PY compare_runs.py
```

Gradient statistics are reported as a model/CFD ratio computed with one
estimator on both fields at the same points.

## 7. Tests

```bash
$PY -m tests.test_knn_cache        # the stratified gather
$PY -m tests.test_local_branch     # the branch: equivariance, zero init
$PY -m tests.test_lgo              # LGO-GDON, including inject handling
$PY -m tests.test_lgo_gino
$PY -m tests.test_lgo_transolver   # [1] branch-off is bit-identical to the host
$PY -m tests.test_splits_shapenet
$PY -m tests.test_sweep_tiers      # CFD layout generator
```

Run `test_lgo_transolver` after any change to `model/lgo_transolver.py` or
`baselines/models/transolver.py`: the hybrid restates the host's forward pass,
so the two can drift without an error.

Known failures: `tests/test_lgo_ginot.py` (tests a configuration the model no
longer has) and `tests/test_query_gradient.py` (finite differences are not valid
on the hard-gather field). `tests/test_real_case.py` needs the dataset.

## 8. Troubleshooting

| symptom | cause |
|---|---|
| `lgo_ginot` banner shows `global=False` | `--model lgo_ginot_local` was used |
| an old checkpoint evaluates badly | it is being rebuilt with different flags; rebuild from its own `args.json` |
| job prints nothing for minutes | normal startup; use `-u` and check `nvidia-smi` |
| a sweep cell never ran | the queue line was consumed by a worker that then failed its resource check; re-add the line |
| `cfd/sweep.py` fails on `blockMesh` | OpenFOAM not on `PATH` |
| ShapeNet evaluation reports no zones | scored with `evaluate.py`; use `evaluate_shapenet.py` |
| `evaluate.py` cannot find the split | pass `--split` explicitly |
