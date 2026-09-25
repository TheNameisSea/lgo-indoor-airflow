#!/usr/bin/env python
"""Banded evaluation for ShapeNet-Car — the companion to `evaluate.py`.

WHY A SEPARATE ENTRY POINT. `evaluate.py` splits the field by **distance to the
nearest supply vent**. A car has no supply vents, so `zone_metrics` returns `{}`
and `evaluate.py` runs to completion producing NO zone breakdown at all -- it
fails silently, which is the worst way to fail. This module supplies the band
definition that ShapeNet-Car actually needs: **distance to the car surface**.

IT REUSES `evaluate.build_from_args` AND `evaluate.predict` RATHER THAN
REIMPLEMENTING THEM. Rebuilding a model from its own `args.json` is the part
with the teeth in it -- `--model lgo` meant two different architectures either
side of 2026-09-11 and dispatch goes through `is_legacy_ginot_run()`. A second
copy of that logic is how historical runs become unreproducible. Only the
banding is new here.

WHY BANDING. Most volume points are freestream, so a single global number is
dominated by it and can hide what the model does near the car. Report per band.

The bands come from `data.shapenet_raw.BANDS`, which is also what
`evaluate_nulls_shapenet.py` uses. One definition, deliberately: if the model
path and the null path banded differently, every model-vs-null comparison on
this benchmark would be void.

PER-CASE ARRAYS. `per_case_bands` holds one row per held-out case per band;
compare runs with a paired test over them, not by their means.

    python evaluate_shapenet.py --run runs/NAME \\
        --split splits_shapenet_csv/val --out results/NAME_shapenet_val.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

from data.shapenet_raw import BANDS, N_VOLUME, band_masks

# Volume points closer than this to the surface are the ones the published
# loader excludes (exactly 994 per case). Chosen three orders of magnitude below
# the nearest genuine exterior point (1.3e-3), measured across cases.
SHARED_EPS = 1e-4          # noqa: E402
from evaluate import build_from_args, predict            # noqa: E402
from thermo.metrics_joint import joint_metrics           # noqa: E402


def velocity_metrics(true, pred):
    """`joint_metrics` with temperature off -- this dataset has none.

    RELATIVE L2 IS NOT RECOMPUTED HERE. `joint_metrics` already returns
    `vel_relL2`, and it is bit-identical to a local implementation (verified).
    A second copy of a shared metric cannot disagree today and will disagree
    eventually.

    `vel_relL2` is the key to lead with on this benchmark. R2 flatters it badly
    -- a constant velocity field already scores R2 near 1 here.
    """
    return joint_metrics(true, pred, has_temperature=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True,
                    help="e.g. splits_shapenet_csv/val -- ALWAYS pass this")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--exterior_only", action="store_true",
                    help="score the volume over the EXTERIOR points only, "
                         "excluding the 994 per case that coincide exactly with "
                         "a surface point. This is the published metric domain "
                         "(28,504 + 3,682 = 32,186). Our default includes them, which can only make "
                         "the number worse -- they sit on the wall, in the "
                         "highest-error band. Use this for any number printed "
                         "beside a published one.")
    ap.add_argument("--n_case_pool", type=int, default=1_500_000,
                    help="large enough that no interior point is dropped; "
                         "ShapeNet-Car has 29,498 per case")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from thermo.dataset import ThermoDataset

    a = json.load(open(os.path.join(args.run, "args.json")))
    if not a.get("stats_path"):
        local = os.path.join(args.run, "thermo_stats.pt")
        if not os.path.exists(local):
            raise SystemExit(f"{args.run}: args.json has no stats_path and "
                             f"{local} does not exist -- cannot rebuild the "
                             f"normalisation this model was trained with.")
        a["stats_path"] = local

    folders = sorted(glob.glob(os.path.join(args.split, "*/")))
    if not folders:
        raise SystemExit(f"no case directories under {args.split}")
    names = [os.path.basename(f.rstrip("/")) for f in folders]
    ds = ThermoDataset(case_dirs=[f.rstrip("/") for f in folders],
                       saved_stats=a["stats_path"], stats_path=a["stats_path"],
                       n_case_pool=args.n_case_pool, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")

    import evaluate as _ev
    _ev._CKPT[0] = args.ckpt
    model, a = build_from_args(args.run, ds, device)
    if getattr(model, "use_local", False) or getattr(model, "use_hbc", False):
        model.register_cases(ds, verbose=False,
                             normals_k=a.get("normals_k", 16),
                             with_normals=not a.get("no_normals", False))
    model.eval()

    # ⚠ RETENTION GUARD. The parent pools interior points BEFORE we band them,
    # capping 'stream' at int(n_case_pool * 0.4) and keeping whatever background
    # remains. On this dataset the speed rule calls 97.6% of points 'stream', so
    # an undersized pool drops stream points while keeping ALL background -- and
    # background is the SLOW points, which are the near-wall ones. The near band
    # then comes out over-represented and every banded number is quietly biased.
    # Measured: n_case_pool=40000 gives a 6.9% near band against a true ~5.0%.
    # This cannot be caught by looking at the output, so it is checked here.
    n_kept = len(ds.cases[0]["pool_stream_xyz"]) + len(ds.cases[0]["pool_bg_xyz"])
    n_have = N_VOLUME
    if n_kept < n_have:
        raise SystemExit(
            f"[shapenet] REFUSING TO BAND: only {n_kept:,} of {n_have:,} interior "
            f"points survived pooling at --n_case_pool {args.n_case_pool:,}. The "
            f"stream cap is 0.4*n_case_pool, so the retained set is biased toward "
            f"the near-wall band and every banded metric below would be wrong.\n"
            f"  Use --n_case_pool {int(np.ceil(n_have / 0.4 / 1000) * 1000):,} or "
            f"more (default {1_500_000:,} is safe).")
    print(f"[shapenet] retention OK: {n_kept:,}/{n_have:,} interior points kept")

    shared_dropped = []
    per_case, per_case_bands = {}, {b: {} for b, _, _ in BANDS}
    band_frac = {b: [] for b, _, _ in BANDS}
    with torch.no_grad():
        for name, room in zip(names, ds.cases):
            xyz_m, true, pred = predict(model, room, ds, device, chunk=args.chunk)
            per_case[name] = velocity_metrics(true, pred)
            # recomputed below if --exterior_only trims the point set

            # The boundary cloud for THIS case, denormalised to match xyz_m.
            wall = (room["pool_wall_xyz"] * ds.coord_scale + ds.coord_min).numpy()
            masks, dist = band_masks(xyz_m, wall)

            if args.exterior_only:
                # The shared points are EXACTLY coincident in the raw data, so
                # after the float32 normalise/denormalise round-trip they sit at
                # ~1e-6 while the nearest genuine exterior point is at 1.3e-3 --
                # three orders of margin, measured. A distance test is therefore
                # exact here AND robust, where tuple-matching floats would not be.
                ext = dist > SHARED_EPS
                n_shared = int((~ext).sum())
                true, pred = true[ext], pred[ext]
                xyz_m = xyz_m[ext]
                masks, dist = band_masks(xyz_m, wall)
                per_case[name] = velocity_metrics(true, pred)
                shared_dropped.append(n_shared)
            for b, mask in masks.items():
                if mask.sum() < 100:
                    continue
                z = velocity_metrics(true[mask], pred[mask])
                z["n_points"] = int(mask.sum())
                per_case_bands[b][name] = z
                band_frac[b].append(float(mask.mean()))
            print(f"  {name}: R2 {per_case[name]['vel_r2']:+.4f}  "
                  f"relL2 {per_case[name]['vel_relL2'] * 100:.2f}%  "
                  + "  ".join(f"{b} {per_case_bands[b][name]['vel_r2']:+.3f}"
                              for b, _, _ in BANDS if name in per_case_bands[b]),
                  flush=True)

    def agg(rows):
        if not rows:
            return {}
        keys = [k for k in next(iter(rows.values())) if k != "n_points"]
        return {k: float(np.mean([r[k] for r in rows.values()])) for k in keys}

    if shared_dropped:
        u = sorted(set(shared_dropped))
        print(f"[shapenet] --exterior_only: dropped {u if len(u) < 4 else u[:3]} "
              f"surface-coincident points per case (published domain: 28,504 "
              f"exterior + 3,682 surface = 32,186)")

    res = {"run": args.run, "split": args.split, "ckpt": args.ckpt,
           "exterior_only": bool(args.exterior_only),
           "shared_dropped_per_case": shared_dropped,
           "n_cases": len(names), "cases": names,
           "global": agg(per_case),
           "bands": {b: agg(per_case_bands[b]) for b, _, _ in BANDS},
           "band_point_frac": {b: float(np.mean(v)) if v else 0.0
                               for b, v in band_frac.items()},
           "per_case": per_case,
           "per_case_bands": per_case_bands}

    print(f"\n{'band':>8}{'pts':>9}{'vel_r2':>10}{'vel_rho':>10}"
          f"{'vel_s':>9}{'relL2':>9}")
    g = res["global"]
    print(f"{'GLOBAL':>8}{'100%':>9}{g['vel_r2']:>10.4f}{g['vel_rho']:>10.4f}"
          f"{g.get('vel_s', float('nan')):>9.3f}{g['vel_relL2'] * 100:>8.2f}%")
    for b, _, _ in BANDS:
        z = res["bands"][b]
        if not z:
            continue
        print(f"{b:>8}{res['band_point_frac'][b] * 100:>8.1f}%"
              f"{z['vel_r2']:>10.4f}{z['vel_rho']:>10.4f}"
              f"{z.get('vel_s', float('nan')):>9.3f}{z['vel_relL2'] * 100:>8.2f}%")
    print("\n[shapenet] the NEAR band is the one that carries the claim; the "
          "outer band is ~79% of points and ~8% of the signal.")
    print("[shapenet] compare against the nulls from evaluate_nulls_shapenet.py, "
          "never against zero.")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[shapenet] -> {args.out}")


if __name__ == "__main__":
    main()
