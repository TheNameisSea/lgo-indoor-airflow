#!/usr/bin/env python
"""The gate that matters: does moving a vent change OUR field the way it
changes the reference field?

We are not reproducing STAR-CCM+ (`cfd/README.md`), so an absolute match was
never the target and a systematic bias shared by every generated case does not
harm an ML model learning a layout -> field mapping. What *would* harm it is
wrong layout sensitivity: if the jet's decay length is wrong, the region each
vent influences is wrong, and that is exactly the signal the dataset exists to
teach.

So compare **deltas**, not fields:

    d_ours = U(our layout B) - U(our layout A)
    d_ref  = U(ref layout B) - U(ref layout A)

and ask whether d_ours tracks d_ref. A model trained on our cases learns
d_ours; if that matches d_ref in direction and magnitude, the dataset carries
the right physics regardless of any common-mode offset.

Everything is evaluated at layout A's reference points, below the ceiling. The
two reference clouds are different meshes, so layout B's reference is gathered
by nearest neighbour, as are both of our solutions -- all four fields land on
one common set of points before any subtraction.

Run:
    python cfd/sensitivity.py \
        --case-a cfd/run/base --ref-a Case_05 \
        --case-b cfd/run/L01  --ref-b Case_01
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate import read_field

CEILING_Z = 3.20


def latest(case):
    return max((d for d in os.listdir(case)
                if re.fullmatch(r"\d+", d) and os.path.isdir(os.path.join(case, d))),
               key=float)


def load_ours(case, time=None):
    t = time or latest(case)
    d = os.path.join(case, t)
    cpath = next(q for q in (os.path.join(d, "C"), os.path.join(case, "0", "C"))
                 if os.path.exists(q))
    return read_field(cpath), read_field(os.path.join(d, "U")), \
        read_field(os.path.join(d, "T")), t


def load_ref(root, name):
    r = pd.read_csv(os.path.join(root, name, "Fluid_data.csv"))
    P = np.c_[r["X (m)"], r["Y (m)"], r["Z (m)"]]
    U = np.c_[[r[f"Velocity[{c}] (m/s)"] for c in "ijk"]].T
    return P, U, r["Temperature (K)"].values


def agreement(a, b, label):
    """How well does `a` (ours) track `b` (reference)?"""
    sa, sb = a.std(), b.std()
    rho = float(np.corrcoef(a, b)[0, 1]) if sa > 0 and sb > 0 else float("nan")
    s = sa / sb if sb > 0 else float("nan")
    bias = (a.mean() - b.mean()) / sb if sb > 0 else float("nan")
    r2 = rho ** 2 - (rho - s) ** 2 - bias ** 2
    print(f"  {label:6} {b.mean():+9.4f} {a.mean():+9.4f} {sb:8.4f} {sa:8.4f} "
          f"{rho:7.3f} {s:6.3f} {r2:7.3f}")
    return dict(rho=rho, s=s, bias=bias, r2=r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-a", default="cfd/run/base")
    ap.add_argument("--case-b", default="cfd/run/L01")
    ap.add_argument("--ref-a", default="Case_05")
    ap.add_argument("--ref-b", default="Case_01")
    ap.add_argument("--root", required=True,
                    help="directory of the reference STAR-CCM+ cases (not distributed)")
    ap.add_argument("--max-dist", type=float, default=0.10)
    ap.add_argument("--time-a", default=None)
    ap.add_argument("--time-b", default=None)
    args = ap.parse_args()

    Ca, Ua, Ta, ta = load_ours(args.case_a, args.time_a)
    Cb, Ub, Tb, tb = load_ours(args.case_b, args.time_b)
    print(f"ours A: {args.case_a} @ t={ta}   ours B: {args.case_b} @ t={tb}")

    Pa, URa, TRa = load_ref(args.root, args.ref_a)
    Pb, URb, TRb = load_ref(args.root, args.ref_b)
    keep = Pa[:, 2] < CEILING_Z
    Pa, URa, TRa = Pa[keep], URa[keep], TRa[keep]
    print(f"ref  A: {args.ref_a} ({keep.sum():,} pts below ceiling)   "
          f"ref B: {args.ref_b}")

    # Everything onto layout A's points.
    d_rb, i_rb = cKDTree(Pb).query(Pa, workers=-1)
    d_oa, i_oa = cKDTree(Ca).query(Pa, workers=-1)
    d_ob, i_ob = cKDTree(Cb).query(Pa, workers=-1)
    ok = (d_rb <= args.max_dist) & (d_oa <= args.max_dist) & (d_ob <= args.max_dist)
    print(f"  {ok.sum():,} points matched in all four fields "
          f"({100*ok.mean():.1f}%)")

    P = Pa[ok]
    dU_ref = URb[i_rb[ok]] - URa[ok]
    dU_our = Ub[i_ob[ok]] - Ua[i_oa[ok]]
    dT_ref = TRb[i_rb[ok]] - TRa[ok]
    dT_our = Tb[i_ob[ok]] - Ta[i_oa[ok]]

    print(f"\nLAYOUT SENSITIVITY  {args.ref_a} -> {args.ref_b}")
    print(f"  how large is the change at all? "
          f"ref |dU| {np.linalg.norm(dU_ref, axis=1).mean():.4f} m/s, "
          f"ours {np.linalg.norm(dU_our, axis=1).mean():.4f}")

    def block(mask, title):
        if mask.sum() < 500:
            print(f"\n{title}: too few points"); return
        print(f"\n{title}  ({mask.sum():,} points)")
        print(f"  {'delta':6} {'ref mean':>9} {'our mean':>9} {'ref sd':>8} "
              f"{'our sd':>8} {'rho':>7} {'s':>6} {'R2':>7}")
        for j, nm in enumerate(("du", "dv", "dw")):
            agreement(dU_our[mask, j], dU_ref[mask, j], nm)
        agreement(dT_our[mask], dT_ref[mask], "dT")

    z = P[:, 2]
    block(np.ones(len(P), bool), "whole room (below ceiling)")
    block((z >= 0.10) & (z <= 1.80), "ASHRAE occupied band")
    block(z > 3.0, "ceiling jet (z > 3.0 m)")

    # Does the change even show up where the vents moved?
    print("\nchange by height band (mean |dU|, m/s)")
    print(f"  {'band':12} {'ref':>8} {'ours':>8} {'ratio':>7}")
    for lo, hi in ((0, .5), (.5, 1.0), (1.0, 1.8), (1.8, 2.6), (2.6, 3.05), (3.05, 3.2)):
        m = (z >= lo) & (z < hi)
        if m.sum() < 200:
            continue
        r = np.linalg.norm(dU_ref[m], axis=1).mean()
        o = np.linalg.norm(dU_our[m], axis=1).mean()
        print(f"  {lo:4.2f}-{hi:4.2f}    {r:8.4f} {o:8.4f} {o/max(r,1e-9):7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
