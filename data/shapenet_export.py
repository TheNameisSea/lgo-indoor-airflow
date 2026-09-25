#!/usr/bin/env python
"""Export ShapeNet-Car onto the 21-CSV loader contract — S7.

WHY EXPORT CSVs RATHER THAN SUBCLASS THE DATASET. `train.py` deliberately does
not reimplement `train_thermo.main()`, so that dataset construction, the loss,
the optimiser, `args.json` and checkpoint selection stay identical across every
arm BY CONSTRUCTION. Writing ShapeNet-Car into the format that pipeline already
reads keeps that property: nothing about the training path changes, and the
ablation pair stays exact. The cost is ~1 GB of CSV, which is cheap against the
2 GB source and against a second, drifting dataset implementation.

THE MAPPING, and every judgement call in it:

    volume (29,498 pts, velocity)  ->  Fluid_data.csv      the interior/target
    surface (3,682 pts)            ->  Car_surface.csv     the boundary/input

`_load_raw_room` classifies by filename: `fluid_data` is interior, `new_hvac` is
vents, `leak` is the leak group, and **anything else falls to the wall branch,
which zeroes every target column**. `Car_surface.csv` therefore lands in the
solid group with its values blinded, which is what we want -- see BLINDING.

**BLINDING: surface pressure is an OUTPUT and must never reach the encoder.**
This is `--blind_return_velocity` one dataset removed. Return-vent velocity was
solver output that reached the encoder on 46% of the point cloud; surface
pressure here is solver output on 100% of the boundary. The wall branch zeroes
it for us, so the export writes the pressure column as 0.0 and keeps the real
values nowhere the model can see them. **If you ever want the surface-pressure
task instead, that is a different export and a different claim** -- do not get
there by un-blinding this one.

**ONE BOUNDARY GROUP.** A car has a single boundary type, so the stratified
gather (supply / return / solid / leak) collapses to `("solid",)`. That is
supported -- `LocalBranch` takes `groups` and sizes its one-hot from it -- but
it means this benchmark exercises the **relative-coordinate gather** and NOT the
stratification. Do not
report it as if it exercised the whole design.

**NO TEMPERATURE, NO INTERIOR PRESSURE.** Neither exists in this dataset. Both
columns are written as 0.0 and must be switched off in the loss
(`--lambda_t 0 --lambda_p 0`); pressure is already unsupervised by default.
`ThermoDataset._fix_temperature_stats` guards a zero std to 1.0, so a constant
channel is safe, but a nonzero `--lambda_t` would train against a constant and
quietly waste capacity.

**⚠ THE INTERIOR STRATIFICATION THRESHOLD IS AN INDOOR CONSTANT AND IT DOES NOT
TRANSFER.** `_process_single_room` splits interior points into "stream" and
"background" by `(vel_mag >= p80) | (vel_mag >= 0.15)`. That 0.15 m/s is an
indoor-air constant. Here the freestream is 20 m/s and the measured mean speed
is 17.2 m/s, so **essentially every point satisfies it**, the background pool
empties, and supervision becomes uniform over the volume. Given that 79% of
volume points sit in the outer band carrying only 8.4% of the case-specific
signal, uniform sampling spends most of the
budget on freestream. This is a REAL limitation of running this dataset through
the indoor pipeline unchanged. It is recorded rather than silently patched,
because changing the parent's threshold would change it for the indoor runs too
-- a silent cross-dataset change. Decide it deliberately before training.

    python data/shapenet_export.py --limit 2      # smoke test
    python data/shapenet_export.py                # all 611, into splits_shapenet_csv/
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                     # so this runs as a script too

from data.shapenet_raw import mapping, surface, volume   # noqa: E402
from data.splits_shapenet import MANIFEST, discover      # noqa: E402

OUT = os.path.join(ROOT, "splits_shapenet_csv")

COLS = ["X (m)", "Y (m)", "Z (m)",
        "Velocity[i] (m/s)", "Velocity[j] (m/s)", "Velocity[k] (m/s)",
        "Pressure (Pa)", "Temperature (K)"]
HEADER = ",".join(COLS)


def _write(path, xyz, uvw):
    """One CSV in the loader's column contract. Pressure and T are always 0."""
    n = len(xyz)
    block = np.zeros((n, 8), dtype=np.float64)
    block[:, 0:3] = xyz
    if uvw is not None:
        block[:, 3:6] = uvw
    # %.6g keeps ~6 significant figures: the coordinates span ~6 units and the
    # velocities ~25 m/s, so this is far finer than either field's own accuracy
    # while roughly halving the file against full repr.
    np.savetxt(path, block, delimiter=",", header=HEADER, comments="", fmt="%.6g")


def export_case(nid, dest, m):
    os.makedirs(dest, exist_ok=True)
    vx, vv = volume(nid, m)
    _write(os.path.join(dest, "Fluid_data.csv"), vx, vv)
    sx, _sp = surface(nid, m)
    # _sp is DELIBERATELY DISCARDED -- see BLINDING in the module docstring.
    _write(os.path.join(dest, "Car_surface.csv"), sx, None)
    return len(vx), len(sx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--manifest", default=None,
                    help="split manifest to export (default: the gap split). "
                         "Pass data/splits_shapenet_fold<N>.json for a "
                         "published-fold tree -- it covers all 889 cases and "
                         "so needs mapping_all(), which this selects "
                         "automatically.")
    ap.add_argument("--limit", type=int, default=None,
                    help="export only the first N cases (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="re-write cases that already have both CSVs")
    a = ap.parse_args()

    man = a.manifest or MANIFEST
    if not os.path.exists(man):
        raise SystemExit(f"[export] no split manifest at {man}. Run "
                         f"data/splits_shapenet.py --gap 0.045 --freeze "
                         f"(or --matched_regime --freeze)")
    assignment = json.load(open(man))

    # A manifest carrying `u###` ids is the 889-case matched-regime tree, which
    # the 798-entry mapping cannot resolve. Select on the CONTENT rather than on
    # a flag, so passing --manifest alone is always sufficient and the two can
    # never be mismatched by forgetting a second argument.
    unlisted = [c for c in assignment if c.startswith("u")]
    if unlisted:
        from data.shapenet_raw import mapping_all
        m = mapping_all()
        cases = sorted(assignment)
        print(f"[export] matched-regime manifest: {len(assignment)} cases, "
              f"{len(unlisted)} of them not in the Zenodo manifest)")
    else:
        m = mapping()
        cases = discover()

    todo = [c for c in cases if c in assignment]
    if a.limit:
        todo = todo[:a.limit]
    print(f"[export] {len(todo)} cases -> {a.out}")

    counts = {k: 0 for k in sorted(set(assignment.values()))}
    skipped = 0
    for i, nid in enumerate(todo):
        split = assignment[nid]
        dest = os.path.join(a.out, split, nid)
        done = all(os.path.exists(os.path.join(dest, f))
                   for f in ("Fluid_data.csv", "Car_surface.csv"))
        if done and not a.force:
            skipped += 1
            counts[split] += 1
            continue
        nv, ns = export_case(nid, dest, m)
        counts[split] += 1
        if i % 50 == 0:
            print(f"  [{i + 1}/{len(todo)}] {nid} -> {split}  "
                  f"volume {nv} surface {ns}", flush=True)
    print(f"[export] done: " + "  ".join(f"{k}={v}" for k, v in counts.items())
          + (f"  ({skipped} already present)" if skipped else ""))


if __name__ == "__main__":
    main()
