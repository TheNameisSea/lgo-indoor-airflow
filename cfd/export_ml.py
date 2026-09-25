#!/usr/bin/env python
"""Export a solved OpenFOAM case to the CSV layout the ML loader reads.

The loader (`legacy/pi_ginot/dataset.py`) globs `**/*.csv` under a case folder
and classifies each file by its **lowercased basename**:

    "fluid_data" in name  -> interior points (the query/target cloud)
    "leak"       in name  -> leak opening
    "new_hvac"   in name  -> the vents, supply and return together
    anything else         -> a solid wall patch, named by the file

Required columns are `X (m) Y (m) Z (m)`, `Velocity[i|j|k] (m/s)`,
`Pressure (Pa)` and, for the thermo variant, `Temperature (K)`. A missing
velocity/pressure column is filled with 0.0 rather than dropped, so wall files
need only geometry -- but we write the real values anyway, since we have them.

Three details of the contract that are easy to get wrong:

  * **`New_HVAC.csv` has no patch column.** The loader recovers supply/return
    geometrically, by clustering and labelling each cluster by the sign of its
    MEAN w (the "A5 rule"). So the six vents must be written into ONE file, and
    their velocities must be the real solved ones -- the classification depends
    on them.
  * **Temperature exists only on `Fluid_data.csv` and `New_HVAC.csv`.** The
    thermo loader relies on that to compute interior-only statistics and to
    blind the return-vent temperature, which is the room's mixed mean and would
    otherwise hand the model most of the answer. Do not add a temperature column
    to wall files.
  * **Pressure.** The loader subtracts each case's interior mean pressure at
    load time, so an absolute offset does not matter; `p_rgh` is exported as the
    pressure, since that is the field the solver actually solves.

Interior points are cell centres; boundary points are face centres with the
patch's own boundary values.

Run:
    python cfd/export_ml.py --case cfd/run/base --out cfd/dataset/base
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_case import patch_face_centres
from validate import read_field

COLS = ["Temperature (K)", "Velocity: Magnitude (m/s)", "Velocity[i] (m/s)",
        "Velocity[j] (m/s)", "Velocity[k] (m/s)", "Pressure (Pa)",
        "X (m)", "Y (m)", "Z (m)"]
VENT_FILE = "New_HVAC.csv"
LEAK_FILE = "Leak.csv"
FLUID_FILE = "Fluid_data.csv"


def latest(case):
    t = [d for d in os.listdir(case)
         if re.fullmatch(r"\d+", d) and os.path.isdir(os.path.join(case, d))
         and d != "0"]
    if not t:
        raise SystemExit(f"{case}: no converged time directory")
    return max(t, key=float)


def boundary_values(path, patch, n_faces, width=1):
    """Values on one patch: nonuniform list, uniform fill, or calculated."""
    txt = open(path).read()
    body = txt[txt.index("boundaryField"):]
    m = re.search(r"^    " + re.escape(patch) + r"\n    \{(.*?)\n    \}",
                  body, re.S | re.M)
    if not m:
        return np.zeros((n_faces, width)) if width > 1 else np.zeros(n_faces)
    blk = m.group(1)
    kind = "vector" if width == 3 else "scalar"
    nu = re.search(r"nonuniform List<" + kind + r">\s*\n?\s*(\d+)\s*\n?\(\n(.*?)\n\)",
                   blk, re.S)
    if nu:
        raw = nu.group(2).replace("(", " ").replace(")", " ")
        a = np.fromstring(raw, sep=" ")
        return a.reshape(-1, 3) if width == 3 else a
    un = re.search(r"uniform\s+\(?\s*([-\d.eE+\s]+?)\s*\)?\s*;", blk)
    if un:
        v = np.fromstring(un.group(1), sep=" ")
        return (np.tile(v, (n_faces, 1)) if width == 3
                else np.full(n_faces, float(v[0])))
    return np.zeros((n_faces, width)) if width > 1 else np.zeros(n_faces)


def frame(xyz, U, p, T=None):
    d = {
        "Velocity[i] (m/s)": U[:, 0], "Velocity[j] (m/s)": U[:, 1],
        "Velocity[k] (m/s)": U[:, 2],
        "Velocity: Magnitude (m/s)": np.linalg.norm(U, axis=1),
        "Pressure (Pa)": p,
        "X (m)": xyz[:, 0], "Y (m)": xyz[:, 1], "Z (m)": xyz[:, 2],
    }
    if T is not None:
        d["Temperature (K)"] = T
    cols = [c for c in COLS if c in d]
    return pd.DataFrame(d)[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", default=None)
    args = ap.parse_args()

    t = args.time or latest(args.case)
    d = os.path.join(args.case, t)
    os.makedirs(args.out, exist_ok=True)

    cpath = next(q for q in (os.path.join(d, "C"), os.path.join(args.case, "0", "C"))
                 if os.path.exists(q))
    C = read_field(cpath)
    U = read_field(os.path.join(d, "U"))
    T = read_field(os.path.join(d, "T"))
    P = read_field(os.path.join(d, "p_rgh"))
    if P.ndim == 0 or len(P) == 1:
        P = np.full(len(C), float(np.ravel(P)[0]))

    frame(C, U, P, T).to_csv(os.path.join(args.out, FLUID_FILE), index=False)
    print(f"{FLUID_FILE:16} {len(C):8,} interior points")

    centres = patch_face_centres(args.case)
    vents, written = [], []
    for patch, xyz in sorted(centres.items()):
        n = len(xyz)
        if n == 0:
            continue
        Ub = boundary_values(os.path.join(d, "U"), patch, n, 3)
        Tb = boundary_values(os.path.join(d, "T"), patch, n)
        Pb = boundary_values(os.path.join(d, "p_rgh"), patch, n)
        if patch.startswith("HVAC_"):
            # ONE file for all six: the loader has no patch column and recovers
            # supply vs return by clustering and mean-w sign.
            vents.append(frame(xyz, Ub, Pb, Tb))
        elif patch == "Leak":
            frame(xyz, Ub, Pb).to_csv(os.path.join(args.out, LEAK_FILE),
                                      index=False)
            written.append((LEAK_FILE, n))
        else:
            # No temperature column on walls: the thermo loader depends on only
            # the fluid and vent files carrying one.
            frame(xyz, Ub, Pb).to_csv(os.path.join(args.out, f"{patch}.csv"),
                                      index=False)
            written.append((f"{patch}.csv", n))

    if vents:
        v = pd.concat(vents, ignore_index=True)
        v.to_csv(os.path.join(args.out, VENT_FILE), index=False)
        w = v["Velocity[k] (m/s)"].to_numpy()
        print(f"{VENT_FILE:16} {len(v):8,} vent points   "
              f"w<0 {100*(w < 0).mean():.0f}%  mean w {w.mean():+.4f}")
    for name, n in written:
        print(f"{name:16} {n:8,}")
    print(f"\nwrote {len(written) + 2} CSVs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
