#!/usr/bin/env python
"""Audit what actually reaches the model.

    "Audit inputs at inference. Remove solution-derived return velocity,
     interior temperature/pressure, and truth-derived masks from input
     features. Prevent target leakage."

This script does the AUDIT only. It changes no feature, retrains nothing and
alters no checkpoint: it loads one case exactly as `evaluate.py` does, reads the
tensor the geometry encoder is handed, and writes what is measurably in it.

    $CPY export_feature_schema.py [--case splits_cfd_gap/val/B030]

WHY A MEASUREMENT AND NOT A CODE READING. The blinding is applied in two places
-- `pi_ginot.dataset` zeroes pressure per patch class, and `thermo.dataset`
then re-blinds temperature everywhere except supply vents -- and the column
layout is easy to misread (`pc_full` is xyz(3) + one-hot(5) + targets(5), so a
slice that looks right can be off by three). Reading max|value| per channel per
class settles what survives, rather than what the code appears to intend.
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

OUT = os.path.join(ROOT, "exports", "feature_schema.csv")

# How each patch class's velocity/temperature ORIGINATES in the CFD, which is
# what decides whether feeding it to the model is legitimate.
PROVENANCE = {
    "supply": ("prescribed",
               "fixed inlet BC: 0.571 m/s down, 287.7 K, identical in every "
               "case. Known before the solve, so it is an input by right."),
    "return": ("solution-derived",
               "pressure-outlet. Velocity here is SOLVED, not prescribed -- it "
               "is an output of the same CFD run the model is asked to "
               "reproduce. This is the leak."),
    "wall": ("geometry-implied",
             "no-slip. Zero velocity follows from the patch type, not from the "
             "solution; a geometry-only pipeline could supply it."),
    "leak": ("prescribed",
             "door undercut, prescribed at the measured 0.0009 m3/s and held "
             "layout-independent."),
}
CLS_NAMES = {0: "leak", 1: "wall", 2: "supply", 3: "return", 4: "unused_class_4"}
CHANNELS = ["u", "v", "w", "p_rgh", "T"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="splits_cfd_gap/val/B030")
    ap.add_argument("--stats", default="runs/gap65_gqlocal_s0/thermo_stats.pt")
    args = ap.parse_args()

    from thermo.dataset import PC_CLS, ThermoDataset
    ds = ThermoDataset(case_dirs=[args.case], n_case_pool=200_000,
                       n_boundary_pool=250_000, n_pc=15000, pc_mode="full",
                       saved_stats=args.stats, stats_path="/tmp/_schema.pt")
    pc = ds.cases[0]["pc_full"]
    oh = pc[:, PC_CLS].argmax(1)
    n_total = len(pc)

    rows = []
    for cls in sorted(set(int(v) for v in oh.unique())):
        m = oh == cls
        name = CLS_NAMES.get(cls, f"class_{cls}")
        prov, note = PROVENANCE.get(name, ("unknown", ""))
        for c, ch in enumerate(CHANNELS):
            mx = float(pc[m][:, 8 + c].abs().max())
            carried = mx > 0.0
            if not carried:
                cls_final, why = "blinded", "zeroed before the encoder sees it"
            elif prov == "solution-derived":
                cls_final, why = "LEAK: solution-derived input", note
            elif prov == "geometry-implied":
                cls_final, why = "geometry-implied", note
            else:
                cls_final, why = "prescribed", note
            rows.append(dict(
                patch_class=name, channel=ch, n_points=int(m.sum()),
                share_of_point_cloud=f"{float(m.float().mean()):.4f}",
                carried_to_model=int(carried),
                max_abs_normalised=f"{mx:.4f}",
                cfd_provenance=prov, classification=cls_final, note=why))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    leak = [r for r in rows if r["classification"].startswith("LEAK")]
    share = sum(float(r["share_of_point_cloud"]) for r in leak) / max(len(leak), 1)
    print(f"\nwrote {OUT}  ({len(rows)} rows, case {args.case})")
    print(f"point cloud fed to the geometry encoder: {n_total} points")
    print(f"\nLEAKING channels: {', '.join(r['patch_class']+'.'+r['channel'] for r in leak)}")
    print(f"carried on {share:.1%} of the input point cloud")
    print("\nTargets the model predicts: u, v, w, p_rgh, T at interior points.")
    print("Return-vent u/v/w are the SAME QUANTITY from the SAME solve, so this")
    print("is target leakage in the strict sense, not merely a strong feature.")


if __name__ == "__main__":
    main()
