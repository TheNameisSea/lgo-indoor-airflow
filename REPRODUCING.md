# Reproducing the results

The sequence from a clean checkout to the paper's numbers. Flags and
environments are covered in [USAGE.md](USAGE.md).

Every command assumes two interpreters, set once per shell, and is run from the
repository root:

```bash
PY=/path/to/ml-env/bin/python        # torch + CUDA
CPY=/path/to/cfd-env/bin/python      # OpenFOAM tooling, no torch
```

---

## 1. Put the data in place

### Indoor CFD dataset

Download it from Zenodo [10.5281/zenodo.22955315](https://doi.org/10.5281/zenodo.22955315)
and extract it so that each case is a directory directly under `cfd/dataset/`:

The 4.25 GB archive is stored as 41 parts of 100 MB. Download the parts and
`SHA256SUMS`, join the parts, check the joined archive, then extract it:

```bash
mkdir -p zenodo_download && cd zenodo_download
Z=https://zenodo.org/records/22955315/files
wget -c "$Z/SHA256SUMS"
for i in $(seq -w 0 40); do wget -c "$Z/indoor_cfd_dataset.tar.gz.part-0$i"; done
cat indoor_cfd_dataset.tar.gz.part-* > indoor_cfd_dataset.tar.gz
sha256sum -c SHA256SUMS                    # must print: indoor_cfd_dataset.tar.gz: OK
cd .. && tar -xzf zenodo_download/indoor_cfd_dataset.tar.gz -C cfd/dataset/
```

`wget -c` resumes an interrupted part, so the loop can simply be re-run.

The result:

```
cfd/dataset/
├── B000/
│   ├── Fluid_data.csv      interior points (the training targets)
│   ├── New_HVAC.csv        all six vent panels
│   ├── Leak.csv
│   └── ...                 18 wall and furniture patches — 21 CSVs per case
├── B001/
└── ...                     193 case directories
```

Check it:

```bash
ls -d cfd/dataset/*/ | wc -l          # 193
for d in cfd/dataset/*/; do [ "$(ls "$d" | wc -l)" = 21 ] || echo "incomplete: $d"; done
```

### Trained checkpoints (optional)

The trained checkpoints are released after the review period. Place them in
`runs/`, one directory per run:

```
runs/<run_name>/args.json          the flags the run was trained with
runs/<run_name>/best.pth           the checkpoint
runs/<run_name>/thermo_stats.pt    its normalisation statistics (evaluation needs them)
runs/<run_name>/history.json       its training and validation log
```

A checkpoint is rebuilt from its own `args.json` and normalised with its own
`thermo_stats.pt`, so keep the four files together.

### ShapeNet-Car

Two downloads, both into `external/shapenet_car/`:

```bash
./fetch_shapenet_raw.sh                 # raw Umetani release, 2.03 GB -> external/shapenet_car/mlcfd_data.zip
cd external/shapenet_car
unzip -q mlcfd_data.zip 'mlcfd_data/training_data/*'
mkdir -p raw && for f in mlcfd_data/training_data/param*.tar.gz; do tar -xzf "$f" -C raw; done
curl -L -o processed-car-pressure-data.zip \
  https://zenodo.org/records/13737721/files/processed-car-pressure-data.zip
unzip -q processed-car-pressure-data.zip -x '__MACOSX/*'
cd ../..
```

The raw release carries the volume velocity field; the Zenodo record supplies
the case identifiers the committed splits use. The result:

```
external/shapenet_car/
├── raw/param0/ ... raw/param8/                     one directory per car
└── processed-car-pressure-data/data/press_*.npy
```

---

## 2. Build the split trees

The split assignments are committed under `data/`. These commands build symlink
trees from them; do not pass `--freeze`, which would overwrite the committed
assignment.

```bash
$CPY data/splits_cfd.py       --materialize
$CPY data/splits_cfd_gap.py   --gap 0.30 --extend --materialize
$CPY data/splits_cfd_gap58.py --materialize
```

ShapeNet-Car:

```bash
$PY data/shapenet_export.py                                     # -> splits_shapenet_csv/
$PY data/shapenet_export.py --manifest data/splits_shapenet_fold0.json \
    --out splits_shapenet_matched_csv                            # published fold 0
$PY data/shapenet_fold_repartition.py --fold_id 1               # published fold 1
```

---

## 3. Run the tests

```bash
$PY -m tests.test_knn_cache && $PY -m tests.test_local_branch && $PY -m tests.test_lgo
$PY -m tests.test_lgo_gino && $PY -m tests.test_lgo_transolver
```

`tests/test_lgo_ginot.py` is known to fail: it tests a configuration the model no
longer has.

---

## 4. Train

| arm | lr | | arm | lr |
|---|---|---|---|---|
| `lgo_gdon` | 1e-3 | | `deeponet` | 3e-4 |
| `lgo_ginot` | 1e-3 | | `ginot` | 1e-4 |
| `lgo_gino` | 3e-3 | | `gino` | 3e-3 |
| `lgo_transolver` | 1e-3 | | `transolver` | 1e-3 |

The drivers carry the per-arm configuration; `sweep_final58.sh` is the reference
the others copy from. Drivers that take a queue consume it as they run, so give
them a copy of the file in `queues/`. Each queue driver runs one worker per GPU
listed in its `*_GPUS` variable.

```bash
GPUS="0"     ./sweep_final_gap30.sh                          # gap-0.30, all arms, 450 epochs, seed 0
F58_GPUS="0" ./sweep_final58.sh                              # gap-0.58, all arms, 450 epochs, seed 0

cp queues/seeds.txt jobs_seeds.q
SEED_QUEUE=jobs_seeds.q SEED_GPUS="0" ./sweep_seeds.sh       # seeds 1 and 2, both splits

GPU=0        ./sweep_k_gap30.sh                              # K ablation (150 epochs)
GPUS="0"     ./tune_sweep.sh                                 # learning-rate sweep, six arms
TUNE_GPUS="0" ./tune_lgo_gino.sh                             # learning-rate sweep, LGO-GINO
./run_baselines_short.sh                                     # 150-epoch equal-budget comparison
```

Run one job per GPU, and launch long jobs detached:

```bash
setsid nohup ./sweep_final58.sh > logs/sweep_final58.log 2>&1 < /dev/null & disown
```

---

## 5. Evaluate

```bash
GPU=0 ./eval_final30.sh                                      # gap-0.30, seed 0, validation
EVAL58_GPUS="0" ./eval_final58.sh                            # gap-0.58, seed 0, validation
GPU=0 ./eval_ksweep.sh                                       # K ablation

cp queues/eval_seeds.txt jobs_eval_seeds.q
EVAL_QUEUE=jobs_eval_seeds.q EVAL_GPUS="0" ./eval_seeds.sh   # seeds 1 and 2

cp queues/eval_test.txt jobs_test.q
TEST_QUEUE=jobs_test.q TEST_GPUS="0" ./eval_test.sh          # held-out test sets, seed 0
```

Zero-parameter baselines, on each split and set:

```bash
$PY evaluate_nulls.py --train_dir splits_cfd_gap/train   --split splits_cfd_gap/val    --out results/nulls_gap_val.json
$PY evaluate_nulls.py --train_dir splits_cfd_gap/train   --split splits_cfd_gap/test   --out results/nulls_gap_test.json
$PY evaluate_nulls.py --train_dir splits_cfd_gap58/train --split splits_cfd_gap58/val  --out results/nulls_gap58_val.json
$PY evaluate_nulls.py --train_dir splits_cfd_gap58/train --split splits_cfd_gap58/test --out results/nulls_gap58_test.json
```

A single run: `$PY evaluate.py --run runs/NAME --split splits_cfd_gap/val --out results/NAME_val.json`.
Always pass `--split`.

---

## 6. Physics diagnostics

```bash
cp queues/physics.txt jobs_physics.q
PHYS_QUEUE=jobs_physics.q PHYS_GPUS="0" ./physics_sweep.sh   # gradients + wall compliance, seed 0
```

---

## 7. ShapeNet-Car

```bash
GPU=0 ./run_shapenet.sh                     # every cell, sequentially
GPU=0 ./run_shapenet.sh sn_gate_gino_s0 sn_fold0_lgo_gino_s0   # or selected cells
```

The cells, all 200 epochs, seed 0:

| cells | what |
|---|---|
| `sn_gate_gino_s0`, `sn_fold0_lgo_gino_s0` | `gino` / `lgo_gino` pair, fold 0, lr 1e-3 |
| `sn_fold0_gino_sdf_s0`, `sn_fold0_lgo_gino_sdf_s0` | the same pair with the distance channel |
| `sn_fold0_transolver_s0`, `sn_fold0_transolver_sdf_s0`, `sn_fold0_lgo_transolver_s0` | Transolver, with and without the distance channel and the branch |
| `sn_f1_lgo_gino_lr{3e-3,1e-3,3e-4,1e-4}`, `sn_f1_lgo_gino_sdf_lr3e-3` | learning-rate tune on fold 1 |

The reported volume relative L2 is the final-epoch validation value:

```bash
$PY -c "import json,sys; print(json.load(open(sys.argv[1]))[-1]['val']['vel_relL2'])" runs/NAME/history.json
```

Banded metrics and zero-parameter baselines:

```bash
$PY evaluate_shapenet.py --run runs/NAME --split splits_shapenet_matched_csv/val --out results/NAME_shapenet.json
$PY evaluate_nulls_shapenet.py --split published --field velocity --banded
$PY evaluate_nulls_shapenet.py --split gap --holdout test --field velocity --banded
```

Use `evaluate_shapenet.py` for these runs, not `evaluate.py`.

---

## 8. Regenerating the CFD dataset instead of downloading it

Needs OpenFOAM v2412 on `PATH`.

```bash
$CPY cfd/step_to_stl.py                             # STL patches from step/H105.step
$CPY -m tests.test_sweep_tiers                      # layout generator guards
PATH=/path/to/cfd-env/bin:$PATH WM_PROJECT_DIR=/path/to/cfd-env \
  $CPY cfd/sweep.py --jobs 1                        # solve every layout
$CPY cfd/sanity.py --all cfd/run                    # physical checks
./cfd/export_all.sh                                 # -> cfd/dataset/
```

Then continue from step 2. [cfd/README.md](cfd/README.md) describes the pipeline.
