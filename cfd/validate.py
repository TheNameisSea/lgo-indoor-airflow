#!/usr/bin/env python
"""The gate: compare a generated case against the STAR-CCM+ case it reproduces.

Until this passes, nothing generated here is trustworthy training data.

The STEP geometry (`H105`) turns out to be an exact match for **Case_05**: its
supply centres are (1.96, 1.55), (4.46, 1.55), (6.96, 1.55), identical to
Case_05's to 2 decimal places, at low fan and 286.15 K supply -- our operating
point. Case_14 is the same layout at high fan. So the comparison is like for
like, not an approximate stand-in.

What this can and cannot show
-----------------------------
This is NOT a reproduction test, and a large disagreement is not automatically a
bug. `cfd/README.md` records that the thermal wall BCs are *chosen* rather than
recovered (no surface CSV carries temperature), and the solver differs
(buoyantSimpleFoam/kEpsilon vs STAR-CCM+). Two consequences:

  * **Temperature is the weakest comparison.** Our wall temperatures came from
    near-surface fluid in Case_07, which is a different layout, and a near-wall
    fluid value is not a surface value. Expect an offset.
  * **Velocity is the meaningful one.** The supply BC is measured from this very
    dataset and the flow rates are matched, so the circulation pattern is a
    fair test of whether the momentum-method diffuser reproduces the real jet.

Comparisons are made AT THE CFD POINTS, never on an interpolated grid:
interpolating truth onto a grid flattens steep gradients.

Our domain stops at the ceiling (z = 3.20) because the plenum and vanes were
replaced by a boundary condition, so the 25.5% of reference points above the
ceiling have no counterpart here and are excluded, not counted as error.

Run:
    conda activate cfdEnv
    python cfd/validate.py --case cfd/run/base --time 5500 --ref Case_05
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CEILING_Z = 3.20
ASHRAE_LO, ASHRAE_HI = 0.10, 1.80      # standing occupied zone


def read_field(path):
    """internalField of an OpenFOAM ascii volScalar/volVectorField."""
    txt = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*\n?(\d+)\s*\n\((.*?)\n\)\s*;",
                  txt, re.S)
    if not m:
        u = re.search(r"internalField\s+uniform\s+\(?([-\d.eE+ ]+)\)?\s*;", txt)
        if u:
            v = np.fromstring(u.group(1), sep=" ")
            return v
        raise SystemExit(f"cannot parse internalField in {path}")
    kind, n, body = m.group(1), int(m.group(2)), m.group(3)
    if kind == "vector":
        arr = np.fromstring(body.replace("(", " ").replace(")", " "), sep=" ")
        return arr.reshape(n, 3)
    return np.fromstring(body, sep=" ")


def taylor(pred, truth):
    """Correlation, amplitude ratio, bias, and the skill score they define."""
    p, t = pred.ravel(), truth.ravel()
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    sp, st = p.std(), t.std()
    if sp < 1e-12 or st < 1e-12:
        return dict(rho=np.nan, s=np.nan, bias=np.nan, rmse=np.nan, r2=np.nan)
    rho = float(np.corrcoef(p, t)[0, 1])
    s = float(sp / st)
    bias = float((p.mean() - t.mean()) / st)
    return dict(rho=rho, s=s, bias=bias,
                rmse=float(np.sqrt(((p - t) ** 2).mean())),
                r2=float(rho ** 2 - (rho - s) ** 2 - bias ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cfd/run/base")
    ap.add_argument("--time", default=None, help="default: latest time written")
    ap.add_argument("--ref", default="Case_05")
    ap.add_argument("--root", required=True,
                    help="directory of the reference STAR-CCM+ cases (not distributed)")
    args = ap.parse_args()

    if args.time is None:
        args.time = max((d for d in os.listdir(args.case)
                         if re.fullmatch(r"\d+", d)
                         and os.path.isdir(os.path.join(args.case, d))), key=float)
    t = os.path.join(args.case, args.time)
    # C depends only on the mesh, so 0/C is valid at every time.
    cpath = next(q for q in (os.path.join(t, "C"), os.path.join(args.case, "0", "C"))
                 if os.path.exists(q))
    C = read_field(cpath)
    U = read_field(os.path.join(t, "U"))
    T = read_field(os.path.join(t, "T"))
    print(f"ours: {len(C)} cells at t={args.time}")

    ref = pd.read_csv(os.path.join(args.root, args.ref, "Fluid_data.csv"))
    P = np.c_[ref["X (m)"], ref["Y (m)"], ref["Z (m)"]]
    Ur = np.c_[[ref[f"Velocity[{c}] (m/s)"] for c in "ijk"]].T
    Tr = ref["Temperature (K)"].values
    print(f"{args.ref}: {len(P)} points")

    below = P[:, 2] < CEILING_Z
    print(f"  below the ceiling: {below.sum()} ({100*below.mean():.1f}%) -- the "
          f"rest is plenum, which our domain does not have")
    P, Ur, Tr = P[below], Ur[below], Tr[below]

    # Nearest cell. Our cells are 12.5-50 mm and the reference is dense, so this
    # is a fair sample of our field at their locations; points further than a
    # cell diagonal from any cell centre are outside our domain and dropped.
    tree = cKDTree(C)
    d, idx = tree.query(P, workers=-1)
    keep = d < 0.10
    print(f"  matched within 0.10 m: {keep.sum()} ({100*keep.mean():.1f}%), "
          f"median distance {np.median(d[keep])*1000:.0f} mm")
    P, Ur, Tr, idx = P[keep], Ur[keep], Tr[keep], idx[keep]
    Up, Tp = U[idx], T[idx]

    def block(name, mask=None):
        m = slice(None) if mask is None else mask
        n = len(P[m])
        if n < 100:
            print(f"\n{name}: only {n} points, skipped")
            return
        sp = np.linalg.norm(Up[m], axis=1)
        st = np.linalg.norm(Ur[m], axis=1)
        print(f"\n{name}  ({n} points)")
        print(f"  {'field':7} {'ours':>9} {'ref':>9} {'rho':>7} {'s':>6} "
              f"{'bias':>7} {'rmse':>8} {'R2':>7}")
        for lbl, a, b in (("|U|", sp, st), ("u", Up[m][:, 0], Ur[m][:, 0]),
                          ("v", Up[m][:, 1], Ur[m][:, 1]),
                          ("w", Up[m][:, 2], Ur[m][:, 2]),
                          ("T", Tp[m], Tr[m])):
            r = taylor(a, b)
            print(f"  {lbl:7} {a.mean():9.4f} {b.mean():9.4f} {r['rho']:7.3f} "
                  f"{r['s']:6.3f} {r['bias']:7.3f} {r['rmse']:8.4f} {r['r2']:7.3f}")

    block("WHOLE ROOM below the ceiling")
    band = (P[:, 2] >= ASHRAE_LO) & (P[:, 2] <= ASHRAE_HI)
    block("ASHRAE occupied band (0.10-1.80 m)", band)
    jet = P[:, 2] > 3.0
    block("Ceiling jet (z > 3.0 m)", jet)

    print("\nbulk circulation below the ceiling (the quantity the ML model "
          "cannot infer):")
    print(f"  ours  u {Up[:,0].mean():+.4f}  v {Up[:,1].mean():+.4f}  "
          f"w {Up[:,2].mean():+.4f}  |U| {np.linalg.norm(Up,axis=1).mean():.4f}")
    print(f"  ref   u {Ur[:,0].mean():+.4f}  v {Ur[:,1].mean():+.4f}  "
          f"w {Ur[:,2].mean():+.4f}  |U| {np.linalg.norm(Ur,axis=1).mean():.4f}")
    print(f"  T     ours {Tp.mean():.2f} K   ref {Tr.mean():.2f} K   "
          f"offset {Tp.mean()-Tr.mean():+.2f} K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
