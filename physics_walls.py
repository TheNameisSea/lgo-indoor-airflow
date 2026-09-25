#!/usr/bin/env python
"""Wall compliance + the temperature-boundedness
freebie, for one run.

    $PY physics_walls.py --run runs/<run> --split splits_cfd_gap/val \
        --out results/walls_<run>.json

WALL COMPLIANCE. Nothing in any reported arm imposes no-slip: `hbc=False`
everywhere and the soft BC penalties are zero. So this is an HONEST test of
whether the geometry stream learned the wall, not a check that a constraint was
applied. Each arm is queried AT THE SOLID BOUNDARY POINTS and reports, kept
separate because they are physically different failures and a model may commit
one without the other:

    penetration = u . n           flow THROUGH the wall
    slip        = ||u - (u.n)n||  flow ALONG it

The reference at the surface is EXACTLY ZERO -- the CFD imposes no-slip there --
so the model's absolute numbers are the error. In the near-wall fluid band the
CFD is not zero, so both fields are reported and the comparison is model vs
truth. The band exists because a surface-only number can pass while the approach
to it is wrong (S3.5.2).

NORMALS ARE NOT ESTIMATED HERE. They come from the neighbour-index build
(`surface_normals` in data/knn_cache.py, PCA + orientation toward the fluid),
the same ones the local branch consumes -- and the index is built with FIXED
parameters for every arm, so branch-less arms are measured against identical
geometry rather than against whatever they happened to register.

POINT-WEIGHTED RMS, AND THAT IS THE CONVENTION. Area weighting needs face areas
this export does not carry, and the boundary cloud is a surface SAMPLING, not a
mesh, so point counts are not integration weights. Reporting the weighting is
part of reporting the number.

TEMPERATURE RANGE. This reports whether the model invents temperatures the
solver never produced in that case -- an overshoot / extrapolation check against
the CFD's own interior range, per case.

There is no boundary temperature range to check against in this export: the wall
and furniture patch CSVs carry no `Temperature (K)` column at all (only
`New_HVAC.csv` and `Fluid_data.csv` do), so `pool_wall_tgt` and `pool_leak_tgt`
are 100% zeros -- a missing-data sentinel, not a boundary condition. Reading
those zeros as a bound produces a fabricated violation, which is why the bound
here is the SOLVER's own range and the supply range is carried alongside it as
reference only.

  * `supply_*`        the supply-air range, reference only
  * `cfd_*`/`pred_*`  the ranges compared
  * `*_vs_cfd`        the reported quantity: overshoot beyond the solver
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

import evaluate as ev                                    # noqa: E402
from data.knn_cache import (BASE_GROUPS, BoundaryIndex,  # noqa: E402
                            T_IDX, _POOL)

CHUNK = 16384


def rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")


@torch.no_grad()
def query_velocity(model, lat, xyz_m, ds, device):
    """Model velocity [m/s] at arbitrary points given in METRES."""
    cmin = ds.coord_min.numpy().astype(np.float64)
    cscale = ds.coord_scale.numpy().astype(np.float64)
    xn = torch.from_numpy(((np.asarray(xyz_m, np.float64) - cmin) / cscale)).float()
    tm, ts = ds.target_mean.to(device), ds.target_std.to(device)
    out = torch.cat([
        model.decode_query(lat, xn[s:s + CHUNK].unsqueeze(0).to(device)).squeeze(0)
        for s in range(0, len(xn), CHUNK)], 0)
    return (out * ts[:5] + tm[:5]).cpu().numpy()


def wall_stats(vel, nrm):
    """Speed / penetration / slip, point-weighted. `nrm` is unit, fluid-facing."""
    speed = np.linalg.norm(vel, axis=1)
    pen = np.einsum("ij,ij->i", vel, nrm)                # signed: >0 leaves wall
    slip = np.linalg.norm(vel - pen[:, None] * nrm, axis=1)
    return {
        "n_points": int(len(vel)),
        "speed_rms": rms(speed),
        "speed_p95": float(np.percentile(speed, 95)) if len(speed) else float("nan"),
        "speed_max": float(speed.max()) if len(speed) else float("nan"),
        "penetration_rms": rms(pen),
        "penetration_absmean": float(np.abs(pen).mean()) if len(pen) else float("nan"),
        "penetration_max": float(np.abs(pen).max()) if len(pen) else float("nan"),
        "slip_rms": rms(slip),
        "slip_max": float(slip.max()) if len(slip) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="splits_cfd_gap/val")
    ap.add_argument("--out", default="results/physics_walls.json")
    ap.add_argument("--n_wall", type=int, default=20000,
                    help="solid boundary points sampled per case")
    ap.add_argument("--band_m", type=float, default=0.05,
                    help="near-wall fluid band thickness [m]")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = json.load(open(os.path.join(args.run, "args.json")))
    local = os.path.join(args.run, "thermo_stats.pt")
    if os.path.exists(local):
        a["stats_path"] = local

    from thermo.dataset import ThermoDataset
    folders = sorted(glob.glob(os.path.join(args.split, "*/")))
    names = [os.path.basename(f.rstrip("/")) for f in folders]
    ds = ThermoDataset(case_dirs=[f.rstrip("/") for f in folders],
                       saved_stats=a["stats_path"], stats_path=a["stats_path"],
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    model, a = ev.build_from_args(args.run, ds, device)
    if getattr(model, "use_local", False) or getattr(model, "use_hbc", False):
        model.register_cases(ds, verbose=False,
                             normals_k=a.get("normals_k", 16),
                             with_normals=not a.get("no_normals", False))

    tm = ds.target_mean.numpy().astype(np.float64)
    ts = ds.target_std.numpy().astype(np.float64)
    rng = np.random.default_rng(args.seed)
    per_case = []

    for room, name in zip(ds.cases, names):
        # Geometry index with FIXED parameters -- identical for every arm.
        bi = BoundaryIndex(room, ds.coord_min, ds.coord_scale,
                           ds.target_mean, ds.target_std,
                           normals_k=16, with_normals=True, groups=BASE_GROUPS)
        solid, feat = bi.xyz["solid"], bi.feat["solid"]
        if len(solid) == 0:
            print(f"  {name:<20} SKIP: no solid points", flush=True)
            continue
        sel = rng.choice(len(solid), size=min(args.n_wall, len(solid)), replace=False)
        sxyz, snrm = solid[sel], feat[sel, 0:3].astype(np.float64)
        nn = np.linalg.norm(snrm, axis=1, keepdims=True)
        good = nn[:, 0] > 1e-6                    # PCA can return a degenerate normal
        sxyz, snrm = sxyz[good], snrm[good] / nn[good]

        lat = model.encode_geometry(room["pc_full"].unsqueeze(0).to(device))

        # --- on the surface: CFD reference is exactly zero (no-slip) ---
        surf = wall_stats(query_velocity(model, lat, sxyz, ds, device)[:, 0:3], snrm)

        # --- near-wall fluid band: CFD is not zero, so report both ---
        xyz_m, true, pred = ev.predict(model, room, ds, device)
        d = bi.wall_distance(xyz_m.astype(np.float64))
        band = (d > 0.0) & (d <= args.band_m)
        rec = {"case": name, "surface": surf,
               "band_m": args.band_m, "n_band": int(band.sum())}
        if band.sum() >= 100:
            _, bidx = bi.tree["solid"].query(
                np.ascontiguousarray(xyz_m[band], dtype=np.float64), k=1)
            bn = feat[bidx, 0:3].astype(np.float64)
            bnn = np.linalg.norm(bn, axis=1, keepdims=True)
            ok = bnn[:, 0] > 1e-6
            bn = bn[ok] / bnn[ok]
            rec["band_model"] = wall_stats(pred[band][ok, 0:3].astype(np.float64), bn)
            rec["band_cfd"] = wall_stats(true[band][ok, 0:3].astype(np.float64), bn)

        # --- temperature: overshoot vs the solver; max principle is blocked ---
        # _POOL is knn_cache's own mapping -- a second copy of the group->pool
        # table is exactly the kind of duplication that has already cost this
        # project a silent wall leak. Only `supply` carries a real temperature:
        # `solid` and `leak` are 100% zeros because the CSVs have no T column.
        st = room.get(f"{_POOL['supply']}_tgt")
        sT = (st.numpy()[:, T_IDX].astype(np.float64) * ts[T_IDX] + tm[T_IDX]
              if st is not None and len(st) else np.array([np.nan]))
        Tp, Tt = pred[:, T_IDX].astype(np.float64), true[:, T_IDX].astype(np.float64)
        lo, hi = float(Tt.min()), float(Tt.max())      # the SOLVER's own range
        rec["temperature"] = {
            "supply_min_K": float(np.min(sT)), "supply_max_K": float(np.max(sT)),
            "cfd_min_K": lo, "cfd_max_K": hi,
            "pred_min_K": float(Tp.min()), "pred_max_K": float(Tp.max()),
            "frac_below_cfd": float((Tp < lo).mean()),
            "frac_above_cfd": float((Tp > hi).mean()),
            "frac_outside_cfd": float(((Tp < lo) | (Tp > hi)).mean()),
            "max_undershoot_vs_cfd_K": float(max(0.0, lo - Tp.min())),
            "max_overshoot_vs_cfd_K": float(max(0.0, Tp.max() - hi)),
        }
        per_case.append(rec)
        t = rec["temperature"]
        print(f"  {name:<20} wall: speed_rms {surf['speed_rms']:.4f}  "
              f"pen_rms {surf['penetration_rms']:.4f}  slip_rms {surf['slip_rms']:.4f}"
              f" | T outside CFD {t['frac_outside_cfd']*100:5.2f}%  "
              f"over {t['max_overshoot_vs_cfd_K']:.3f}K "
              f"under {t['max_undershoot_vs_cfd_K']:.3f}K", flush=True)

    out = {"run": args.run, "split": args.split, "n_wall": args.n_wall,
           "band_m": args.band_m, "weighting": "point-weighted RMS (no face areas)",
           "per_case": per_case, "mean": {}}
    def avg(path):
        vals = []
        for c in per_case:
            node = c
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
                if node is None:
                    break
            if isinstance(node, (int, float)) and np.isfinite(node):
                vals.append(float(node))
        return float(np.mean(vals)) if vals else float("nan")
    for grp in ("surface", "band_model", "band_cfd"):
        keys = ("speed_rms", "speed_p95", "speed_max", "penetration_rms",
                "penetration_absmean", "penetration_max", "slip_rms", "slip_max")
        out["mean"][grp] = {k: avg([grp, k]) for k in keys}
    out["mean"]["temperature"] = {k: avg(["temperature", k]) for k in
                                  ("supply_min_K", "supply_max_K",
                                   "cfd_min_K", "cfd_max_K",
                                   "pred_min_K", "pred_max_K",
                                   "frac_outside_cfd", "frac_below_cfd",
                                   "frac_above_cfd", "max_overshoot_vs_cfd_K",
                                   "max_undershoot_vs_cfd_K")}
    out["mean"]["temperature"]["note"] = (
        "Reference range is the CFD's own interior, per case. Wall/furniture "
        "patch CSVs carry no Temperature column, so the export has no boundary "
        "temperature range; supply_* is reference only.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    m = out["mean"]
    print(f"\n=== MEAN over {len(per_case)} cases -- {out['weighting']} ===")
    print(f"  surface     speed_rms {m['surface']['speed_rms']:.4f}  "
          f"pen_rms {m['surface']['penetration_rms']:.4f}  "
          f"slip_rms {m['surface']['slip_rms']:.4f}   (CFD reference: exactly 0)")
    print(f"  band model  speed_rms {m['band_model']['speed_rms']:.4f}  "
          f"pen_rms {m['band_model']['penetration_rms']:.4f}  "
          f"slip_rms {m['band_model']['slip_rms']:.4f}")
    print(f"  band CFD    speed_rms {m['band_cfd']['speed_rms']:.4f}  "
          f"pen_rms {m['band_cfd']['penetration_rms']:.4f}  "
          f"slip_rms {m['band_cfd']['slip_rms']:.4f}")
    t = m["temperature"]
    print(f"  temperature CFD [{t['cfd_min_K']:.2f}, {t['cfd_max_K']:.2f}] K   "
          f"pred [{t['pred_min_K']:.2f}, {t['pred_max_K']:.2f}]   "
          f"outside {t['frac_outside_cfd']*100:.3f}%   "
          f"over {t['max_overshoot_vs_cfd_K']:.3f} K  "
          f"under {t['max_undershoot_vs_cfd_K']:.3f} K")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
