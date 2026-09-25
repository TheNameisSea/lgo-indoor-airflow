# CFD pipeline — generating the indoor dataset

Generates the indoor dataset with OpenFOAM: one room, six ceiling vents, and the
vent layout as the only variable between cases. Only needed if you want to
regenerate the data rather than download it.

Run everything from the repository root, inside the CFD environment:

```bash
conda env create -f environment-cfd.yml     # OpenFOAM v2412, gmsh, vtk
conda activate cfdEnv
CPY=python
```

## Pipeline

| step | script | output |
|---|---|---|
| 1. CAD to patches | `step_to_stl.py` | `cfd/geometry/*.stl` from `step/H105.step` |
| 2. layouts, meshes, solves | `sweep.py` | one solved case per layout under `cfd/run/` |
| 3. checks | `sanity.py` | pass/fail per case |
| 4. export | `export_all.sh` | 21 CSVs per case under `cfd/dataset/` |

```bash
$CPY cfd/step_to_stl.py                      # 1
$CPY -m tests.test_sweep_tiers               # layout-generator guards
$CPY cfd/sweep.py --list                     # the layouts that will be solved
$CPY cfd/sweep.py --only D015 D001 --jobs 1  # try two cases first
$CPY cfd/sweep.py --jobs 1                   # 2: every layout
$CPY cfd/sanity.py --all cfd/run             # 3
./cfd/export_all.sh                          # 4 (resumable; JOBS=N exports in parallel)
```

`sweep.py` and `export_all.sh` are resumable: finished cases are skipped. A case
that reaches the final iteration but fails `sanity.py` is not exported.

The layouts are generated from a fixed seed inside `sweep.py`, so the same code
produces the same case names and parameters. Changing the design-box constants in
`sweep.py` changes every downstream layout; `tests/test_sweep_tiers.py` guards
against that.

## Scripts

| script | role |
|---|---|
| `step_to_stl.py` | converts the STEP file to named STL patches (`--step`, `--out`) |
| `make_layout.py` | places the six vent panels for one layout (`--spread`, `--row`, `--dx`, `--dy`, `--offsets`) |
| `make_case.py` | builds one `buoyantSimpleFoam` case: mesh, fields, controls |
| `vent_bc.py` | writes the supply boundary condition from `vent_profile.json` |
| `solve.py` | runs one case, ramping the supply profile in |
| `sweep.py` | all of the above, for every layout (`--jobs`, `--procs`, `--only`, `--list`) |
| `sanity.py` | per-case physical checks (below) |
| `export_ml.py` | one solved case to the loader's CSV layout (`export_all.sh` calls it) |

Fixed inputs, committed: `case_spec.json` (flow rate, temperatures, wall
conditions), `vent_profile.json` (the fitted supply-diffuser profile) and
`geometry/patches.json`.

`fit_vent_profile.py`, `sensitivity.py` and `validate.py` derived and checked
those inputs against the original reference simulations, which are not
distributed; they take the reference data directory as a required argument.

## Sanity checks

| check | catches |
|---|---|
| `mass_balance` | solver/boundary-condition inconsistency |
| `supply_flux` | a broken or mis-centred vent profile |
| `leak_flux` | a wrong leak boundary condition |
| `return_flux` = supply − leak | outlets not absorbing the inflow |
| `speed_mean`, `speed_max` | implausible room-mean or peak speed |
| `T_range`, `limiters`, `bounding`, `finite` | a runaway or unhealthy solve |

## Resources

`sweep.py` runs one case at a time on 8 MPI ranks by default (`--procs 8`). On
the machine used for the dataset, running several cases at once was slower than
one, because MPI ranks busy-wait. Set `--jobs` and `--procs` to fit your cores.
