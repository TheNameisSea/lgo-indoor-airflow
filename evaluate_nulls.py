#!/usr/bin/env python
"""Zero-parameter nulls: the bar every learned model must clear.

These were flagged and never run:

  * `mean`      -- predict the TRAINING-SET MEAN field at every query point.
  * `retrieval` -- copy the field of the training layout whose HVAC cloud is
                   closest, nearest-neighbour interpolated onto the val points.

They matter more here than any extra deep baseline. The paper diagnoses the
old model as *emitting the training-set mean circulation* -- 0.0099 m/s from the
training mean against 0.0755 m/s from the truth. `mean` IS that failure mode with
zero parameters, so it is the number that says whether the generated dataset
fixed anything. And with 67 training layouts that deliberately TRANSLATE the vent
array (unlike the old 9, which shared one centroid at x = 4.46 m), `retrieval`
becomes a genuinely strong competitor -- "copy the most similar training room" is
the first thing a referee asks about.

Scored by importing `evaluate.py`'s own functions, so a null number is directly
comparable to a model number: same point sets, same zone splits, same ASHRAE
setbacks, same `joint_metrics`. Only `predict()` is replaced. The ||grad u||
measurement is skipped -- it needs a model.

    CPY=/path/to/ml-env/bin/python
    $CPY evaluate_nulls.py --train_dir splits_cfd/train --split splits_cfd/val \
         --out results/nulls_val.json
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

import evaluate as ev
from thermo.metrics_joint import fmt, joint_metrics, summarise

WORKERS = int(os.environ.get("LGO_KNN_WORKERS", "-1"))


def interior(room, ds):
    """The exact points and physical targets `evaluate.predict` is scored on."""
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    tgt = torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], 0)
    xyz_m = (xyz_n * ds.coord_scale + ds.coord_min).numpy().astype(np.float64)
    true = (tgt * ds.target_std + ds.target_mean).numpy()
    return xyz_m, true


def hvac_cloud(room, ds):
    """Supply + return points in METRES -- the layout descriptor.

    The vents are the only thing that varies across this dataset, so the HVAC
    cloud IS the layout. Using the raw cloud rather than clustered panel
    centroids keeps this free of a cluster threshold and of any assumption about
    how many vents a case has -- tier C jitters them independently.
    """
    xyz_n = torch.cat([room["pool_in_xyz"], room["pool_out_xyz"]], 0)
    return (xyz_n * ds.coord_scale + ds.coord_min).numpy().astype(np.float64)


def chamfer(a, b):
    """Symmetric mean nearest-neighbour distance [m] between two clouds."""
    da, _ = cKDTree(b).query(a, k=1, workers=WORKERS)
    db, _ = cKDTree(a).query(b, k=1, workers=WORKERS)
    return 0.5 * (float(da.mean()) + float(db.mean()))


def train_mean(train_ds):
    """Per-case channel mean, then mean over cases.

    Averaged per case first so a case with more mesh points does not dominate --
    the metrics are per-case-then-averaged too, so this matches how the model is
    scored.
    """
    per_case = [interior(r, train_ds)[1].mean(0) for r in train_ds.cases]
    return np.mean(per_case, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", default="splits_cfd/train")
    ap.add_argument("--split", default="splits_cfd/val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stats", default=None,
                    help="reuse a run's thermo_stats.pt; any stats give identical "
                         "numbers (everything is denormalised before scoring), "
                         "this only skips recomputing them")
    ap.add_argument("--nulls", nargs="+", default=["mean", "retrieval"],
                    choices=("mean", "retrieval"))
    ap.add_argument("--donor_floor", type=float, nargs="*", default=None,
                    metavar="M",
                    help="Re-run the retrieval null with the donor restricted to "
                         "training layouts at least M metres away, in the PANEL "
                         "metric that data/splits_cfd_gap.py uses. Answers 'what "
                         "does retrieval score when the nearest AVAILABLE donor "
                         "is M away' from already-solved cases, which is the "
                         "falsification test for widening the design box: the "
                         "0.58 m target is the vent panel width and is inherited, "
                         "never measured. Several values may be given; the "
                         "expensive field loading is shared across them.")
    ap.add_argument("--setback", type=float, default=ev.SETBACK_EXTERIOR_M)
    ap.add_argument("--setback_internal", type=float, default=ev.SETBACK_INTERNAL_M)
    ap.add_argument("--setback_furniture", type=float, default=ev.SETBACK_FURNITURE_M)
    args = ap.parse_args()

    from thermo.dataset import ThermoDataset
    ds_kw = dict(n_case_pool=1_500_000, n_boundary_pool=250_000, n_pc=15000,
                 pc_mode="full")

    train_folders = sorted(glob.glob(os.path.join(args.train_dir, "*/")))
    val_folders = sorted(glob.glob(os.path.join(args.split, "*/")))
    train_names = [os.path.basename(f.rstrip("/")) for f in train_folders]
    val_names = [os.path.basename(f.rstrip("/")) for f in val_folders]

    print(f"[nulls] {len(train_folders)} train / {len(val_folders)} val cases")
    # The dataset writes `stats_path` itself when saved_stats is None (and the
    # thermo subclass re-saves the temperature-corrected version over it), so let
    # it do that rather than reconstructing the blob format here.
    stats = args.stats or os.path.join(ROOT, "results", "nulls_train_stats.pt")
    os.makedirs(os.path.dirname(stats), exist_ok=True)
    train_ds = ThermoDataset(case_dirs=[f.rstrip("/") for f in train_folders],
                             saved_stats=args.stats, stats_path=stats, **ds_kw)
    # Val is normalised with TRAIN statistics, exactly as train_thermo does.
    val_ds = ThermoDataset(case_dirs=[f.rstrip("/") for f in val_folders],
                           saved_stats=stats, stats_path=stats, **ds_kw)

    # Zone splits need the supply KD-tree per case, built the way evaluate.py
    # builds it (normals off -- only the supply tree is used here).
    from data.knn_cache import BoundaryIndex
    groups = ev.make_groups(None)
    case_index = [BoundaryIndex(r, val_ds.coord_min, val_ds.coord_scale,
                                val_ds.target_mean, val_ds.target_std,
                                with_normals=False, groups=groups, far_voxels=None)
                  for r in val_ds.cases]

    mu = train_mean(train_ds) if "mean" in args.nulls else None
    if mu is not None:
        print(f"[nulls] training-set mean  u={mu[0]:+.4f} v={mu[1]:+.4f} "
              f"w={mu[2]:+.4f} m/s  T={mu[4]:.3f} K")

    train_hvac = train_fields = None
    if "retrieval" in args.nulls:
        train_hvac = [hvac_cloud(r, train_ds) for r in train_ds.cases]
        train_fields = [interior(r, train_ds) for r in train_ds.cases]

    # Panel-metric distances, ONLY when a donor floor is asked for. The
    # retrieval null selects on the chamfer of the raw HVAC cloud, but the gap
    # split and the tier-D separation guarantee are both stated in the panel
    # metric, and the two differ by roughly 3x. A floor applied in the wrong
    # metric would test the
    # wrong distance.
    panel_D = None
    if args.donor_floor:
        from data.splits_cfd_gap import layout_distance, panels
        tp = [panels(f.rstrip("/")) for f in train_folders]
        vp = [panels(f.rstrip("/")) for f in val_folders]
        panel_D = np.array([[layout_distance(v, t) for t in tp] for v in vp])
        print(f"[nulls] panel distances: val x train {panel_D.shape}, "
              f"min {panel_D.min():.3f} max {panel_D.max():.3f} m")

    jobs = [(n, None) for n in args.nulls]
    if args.donor_floor and "retrieval" in args.nulls:
        jobs += [("retrieval", f) for f in args.donor_floor]

    results = {}
    for null, floor in jobs:
        key = null if floor is None else f"retrieval_floor_{floor:.2f}"
        raw, uni, zones, comfort, donors, scored = [], [], [], [], [], []
        print(f"\n===== null: {key} =====")
        for ci, (room, name) in enumerate(zip(val_ds.cases, val_names)):
            t0 = time.time()
            xyz, true = interior(room, val_ds)

            if null == "mean":
                pred = np.broadcast_to(mu, (len(xyz), len(mu))).copy()
                donor = None
            else:
                q = hvac_cloud(room, val_ds)
                d = np.array([chamfer(q, h) for h in train_hvac])
                allowed = np.ones(len(d), bool)
                if floor is not None:
                    allowed = panel_D[ci] >= floor
                    if not allowed.any():
                        # No donor is far enough. Skipping would silently score
                        # this null on an easier subset of cases than the model.
                        print(f"  {name}: NO donor >= {floor:.2f} m -- skipped")
                        continue
                j = int(np.flatnonzero(allowed)[np.argmin(d[allowed])])
                order = np.argsort(np.where(allowed, d, np.inf))
                donor = {"case": train_names[j], "chamfer_m": float(d[j]),
                         "panel_m": float(panel_D[ci][j]) if panel_D is not None
                         else None, "donor_floor_m": floor,
                         "n_allowed": int(allowed.sum()),
                         "runner_up": train_names[int(order[1])],
                         "runner_up_chamfer_m": float(d[order[1]])}
                dxyz, dtrue = train_fields[j]
                _, nn = cKDTree(dxyz).query(xyz, k=1, workers=WORKERS)
                pred = dtrue[nn]
            donors.append(donor)
            scored.append(name)

            m = joint_metrics(true, pred)
            u = joint_metrics(true[(c := ev.uniform_cells(xyz))], pred[c])
            raw.append(m)
            uni.append(u)
            zones.append(ev.zone_metrics(case_index[ci], xyz, true, pred))
            comfort.append(ev.comfort_metrics(
                val_folders[ci].rstrip("/"), xyz, true, pred, args.setback,
                args.setback_internal, args.setback_furniture))

            tag = (f" <- {donor['case']} (chamfer {donor['chamfer_m']:.3f} m"
                   + (f", panel {donor['panel_m']:.3f} m, "
                      f"{donor['n_allowed']} donors allowed)"
                      if donor.get("panel_m") is not None else ")")) if donor else ""
            print(f"  {name}  {len(xyz):>9,} pts  {time.time()-t0:5.1f}s{tag}")
            print(f"    raw:     {fmt(m)}")
            print(f"    uniform: {fmt(u)}")
            for zn, zm in comfort[-1].items():
                print(f"    {zn:16} v_rho={zm['vel_rho']:+.3f} "
                      f"v_R2={zm['vel_r2']:+.3f} T_mae={zm['temp_mae']:.3f}K")

        res = {"run": f"null_{key}", "split": args.split,
               "donor_floor_m": floor, "n_skipped": len(val_names) - len(scored),
               "train_dir": args.train_dir, "cases": scored,
               "per_case": raw, "per_case_uniform": uni,
               "per_case_zones": zones, "per_case_comfort": comfort,
               "mean": summarise(raw), "mean_uniform": summarise(uni)}
        if null == "mean":
            res["train_mean_physical"] = mu.tolist()
        else:
            res["per_case_donor"] = donors
        for zn, _, _ in ev.ZONES:
            zs = [z[zn] for z in zones if zn in z]
            if zs:
                res.setdefault("mean_zones", {})[zn] = summarise(zs)
        for zn, _, _ in ev.ASHRAE_ZONES:
            cs = [c[zn] for c in comfort if zn in c]
            if cs:
                res.setdefault("mean_comfort", {})[zn] = summarise(cs)

        print(f"\n--- {key}: MEAN over {len(scored)} cases ---")
        print(f"  raw:     {fmt(res['mean'])}")
        print(f"  uniform: {fmt(res['mean_uniform'])}")
        for zn, zm in res.get("mean_zones", {}).items():
            print(f"  {zn:11} v_rho={zm['vel_rho']:+.3f} v_R2={zm['vel_r2']:+.3f} "
                  f"score={zm['score']:.4f}")
        for zn, zm in res.get("mean_comfort", {}).items():
            print(f"  {zn:16} v_rho={zm['vel_rho']:+.3f} v_R2={zm['vel_r2']:+.3f} "
                  f"T_mae={zm['temp_mae']:.3f}K score={zm['score']:.4f}")
        results[key] = res

    out = args.out or os.path.join(ROOT, "results", "nulls_val.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
