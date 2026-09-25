#!/usr/bin/env python
"""Score an LGO run on every interior CFD point, through the prior protocol.

Metrics come from `legacy/thermo/metrics_joint.py` unmodified, so an LGO number
and a GINOT+SE+HBC number are produced by the same code. Three views are always
reported together, because the prior project established that any one of them
alone is misleading:

  raw       every interior point. 39% of them cluster near the diffusers, so
            this is mostly a near-jet score.
  uniform   one point per 5 cm cube -- removes the mesh-density bias. This is
            the honest headline; it is where the prior project's models fell
            from 0.95 to 0.44 velocity rho.
  zones     by distance to the nearest SUPPLY vent: near-jet <1 m, transition
            1-2 m, far-field >2 m. The near-jet zone holds 93% of the field
            variance in 27% of the points, so a global number hides exactly the
            region a ventilation engineer cares about.

Optionally (--grad_points) it also reports ||grad u||_F against the CFD's own.
The prior 2.52-vs-0.84 comparison is RETRACTED (it is a mean against a
median, across two datasets and two estimators). The models' velocity
gradients are 3x too weak, no loss fixed it, and it is the sharpest statement of
what "the occupied zone is unsolved" means. LGO's whole hypothesis is that
seeing real boundary geometry at real offsets gives a mechanism for sharper
gradients, so this number is a first-class result, not a diagnostic.

    python evaluate.py --run runs/lgo_hbc_s0
    python evaluate.py --run runs/lgo_hbc_s0 --split splits/test   # ONCE, at the end
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

from eval_thermo import uniform_cells
from data.knn_cache import make_groups
from thermo.metrics_joint import fmt, joint_metrics, summarise

# RENAMED 2026-09-16: the third band was called "occupied", one word away from
# the ASHRAE 55 occupied zone below and a different region entirely. This band is
# >2 m from a supply vent; its median distance to the nearest SOLID is 0.01 m, so
# it is overwhelmingly wall/floor/furniture boundary layer and 15% of it sits
# above 2.5 m. No comfort claim can rest on it. `far_field` says what it is.
#
# COMPATIBILITY: result JSONs written before this date carry the key "occupied"
# under `mean_zones`/`per_case[*].zones`. Anything reading historical results must
# accept both -- see LEGACY_ZONE_ALIASES.
ZONES = (("near_jet", 0.0, 1.0), ("transition", 1.0, 2.0), ("far_field", 2.0, np.inf))
LEGACY_ZONE_ALIASES = {"occupied": "far_field"}

# ASHRAE 55 occupied zone -- a HEIGHT band with setbacks from surfaces. A
# different thing from the distance-to-vent bands above, and the only region a
# thermal-comfort claim can rest on. The distance band `far_field` (named
# "occupied" in older result files) is dominated by wall, floor and furniture
# boundary layers.
#
#   seated 0.10-1.35 m      standing 0.10-1.80 m
#
# THE SETBACK IS PER SURFACE CLASS. An earlier version applied one 0.3 m
# distance to every solid, which excluded the air just above every desk --
# exactly where a seated person's head is -- and left only 2% of the interior.
# Furniture still needs a SMALL setback, or the metric scores the boundary layer
# on a desktop; it just must not need the same clearance as a wall.
SETBACK_EXTERIOR_M = 1.0     # outside walls, windows, doors, fixed HVAC
SETBACK_INTERNAL_M = 0.3     # internal walls and partitions
SETBACK_FURNITURE_M = 0.1    # tables, cabinets, sinks, screens -- boundary layer only
EXTERIOR_SURFACES = ("wall_01", "wall_02", "wall_03", "wall_04", "wall_door",
                     "door", "window")
HVAC_SURFACES = ("new_hvac", "wall_hvac", "diffuser", "inlet_ac", "outlet_air",
                 "wall_ac")
INTERNAL_SURFACES = ("column", "board")
# Floor and ceiling impose no setback of their own: the height band is their
# rule, and ASHRAE's 0.10 m lower bound IS the floor clearance. Everything else
# (table, cabinet, sink, tv, leak) is furniture and takes SETBACK_FURNITURE_M.
NO_SETBACK_SURFACES = ("floor", "ceiling")
ASHRAE_ZONES = (("ashrae_seated", 0.10, 1.35), ("ashrae_standing", 0.10, 1.80))


from model.lgo import LGO_FAMILY, is_legacy_ginot_run  # noqa: E402
from model.lgo_gino import LGO_GINO_FAMILY             # noqa: E402
from model.lgo_transolver import LGO_TRANSOLVER_FAMILY   # noqa: E402

# Every name that has EVER meant the GINOT-trunk architecture. The bare `lgo*`
# names are here because they meant it until 2026-09-11 and twelve committed
# runs still carry them; `is_legacy_ginot_run()` is what separates those from a
# post-rename `--model lgo`, which means the DeepONet architecture instead.
GINOT_LGO_FAMILY = ("lgo_ginot", "lgo_ginot_hbc",
                    "lgo_ginot_local", "lgo_ginot_local_hbc",
                    "lgo", "lgo_hbc", "lgo_local", "lgo_local_hbc")
# Which checkpoint build_from_args loads. best.pth is selected on the GLOBAL
# joint score, which is ~94% near-jet by variance -- so it is worth being able
# to ask what a different epoch does in the occupied zone.
_CKPT = ["best.pth"]


def _voxels_of(a):
    if not a.get("k_solid_far"):
        return None
    return a.get("far_voxels") or [a.get("far_voxel", 0.5)]


def _groups_of(a):
    return make_groups(_voxels_of(a))


def build_from_args(run, ds, device):
    """Rebuild the model described by a run's own args.json, and load best.pth.

    Every arm is assembled the same way `train.py` assembles it, so a checkpoint
    is always loaded by the code that produced it. Non-LGO arms fall through to
    the prior project's own loader, which keeps their numbers reproducible here
    without a second implementation of them.

    THE MODEL NAME ALONE IS NOT ENOUGH. `--model lgo` meant the GINOT-trunk
    architecture before 2026-09-11 and the DeepONet one after, so dispatch goes
    through `is_legacy_ginot_run()`, which keys on the ABSENCE of `lgo_arch` in
    args.json. Get this wrong and `gap143`, `gap65` and both gap-0.58 arms are
    rebuilt as a different model -- `load_state_dict` is strict so it would
    raise rather than silently mis-score, but the result is the same: every
    historical number becomes unreproducible.
    """
    a = json.load(open(os.path.join(run, "args.json")))
    legacy_ginot = is_legacy_ginot_run(a)

    if a["model"] in LGO_FAMILY and not legacy_ginot:
        # MUST be routed before the legacy loader. This model is a DeepONet plus
        # a local branch, and legacy_eval.load_model keys on the baseline name
        # it contains -- it would build a bare `deeponet` and then either refuse
        # the checkpoint or, if anything ever loosened strict loading, silently
        # evaluate the baseline while reporting this model's name.
        from model.lgo import build_lgo
        model = build_lgo(a, ds, device, verbose=False)
        model.load_state_dict(
            torch.load(os.path.join(run, _CKPT[0]), map_location=device))
        model.eval()
        return model, a

    if a["model"] in LGO_GINO_FAMILY:
        # Same reason as the LGO_FAMILY branch above: legacy_eval.load_model
        # keys on the baseline name this arm contains and would build a bare
        # `gino`, then either refuse the checkpoint or silently score the
        # baseline under this model's name.
        from model.lgo_gino import build_lgo_gino
        model = build_lgo_gino(a, ds, device, verbose=False)
        model.load_state_dict(
            torch.load(os.path.join(run, _CKPT[0]), map_location=device))
        model.eval()
        return model, a

    if a["model"] in LGO_TRANSOLVER_FAMILY:
        # Same reason again: the arm's name contains the baseline's, so the
        # legacy loader would build a bare `transolver` and score it under this
        # model's name.
        from model.lgo_transolver import build_lgo_transolver
        model = build_lgo_transolver(a, ds, device, verbose=False)
        model.load_state_dict(
            torch.load(os.path.join(run, _CKPT[0]), map_location=device))
        model.eval()
        return model, a

    if not (legacy_ginot or a["model"] in GINOT_LGO_FAMILY):
        import eval_thermo as legacy_eval
        model, _ = legacy_eval.load_model(run, ds, device)
        return model, a

    import argparse as _ap

    import train_thermo
    from model.lgo_ginot import LGOGinot

    trunk = train_thermo.build(_ap.Namespace(**{**a, "model": "ginot"}), ds, device)
    model = LGOGinot(
        trunk, ds.coord_min, ds.coord_scale, ds.target_mean, ds.target_std,
        d_model=a["embed_dim"], n_heads=a.get("local_heads", 8),
        k=dict({"supply": a["k_supply"], "return": a["k_return"],
                "solid": a["k_solid"], "leak": a["k_leak"]},
               **{g: a["k_solid_far"] for g in _groups_of(a) if g.startswith("solid_far")}),
        # The group set is baked into the trained encoder's one-hot width, so it
        # must be reconstructed from the run's own args, not assumed.
        groups=_groups_of(a), far_voxels=_voxels_of(a),
        n_local_layers=a["n_local_layers"], n_rbf=a["n_rbf"],
        local_hidden=a["local_hidden"], rbf_max=a["rbf_max"],
        local_query=a.get("local_query", "learned"),
        # MUST come from args.json. `global_query` changes what feeds the room
        # latent, but not the parameter NAMES, so omitting it loads the
        # checkpoint without complaint and then evaluates a different
        # architecture than the one that was trained -- a Fourier-query model
        # wearing a local-query model's weights.
        global_query=a.get("global_query", "local"),
        use_local=True, use_global="local" not in a["model"],
        use_hbc=a["model"].endswith("hbc"), ell_wall=a["ell_wall"],
        chunk_knn=a.get("chunk_knn", 200_000)).to(device)
    model.load_state_dict(torch.load(os.path.join(run, _CKPT[0]), map_location=device))
    model.eval()
    return model, a


@torch.no_grad()
def predict(model, room, ds, device, chunk=16384):
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    tgt = torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], 0)
    tm, ts = ds.target_mean.to(device), ds.target_std.to(device)
    lat = model.encode_geometry(room["pc_full"].unsqueeze(0).to(device))
    out = torch.cat([
        model.decode_query(lat, xyz_n[s:s + chunk].unsqueeze(0).to(device)).squeeze(0)
        for s in range(0, len(xyz_n), chunk)], 0)
    xyz_m = (xyz_n * ds.coord_scale + ds.coord_min).numpy()
    return xyz_m, (tgt.to(device) * ts + tm).cpu().numpy(), (out * ts[:5] + tm[:5]).cpu().numpy()


def zone_metrics(idx, xyz_m, true, pred):
    """Split by distance to the nearest supply vent.

    Takes the boundary index directly rather than reading `model._active`: the
    zones are defined by where the vents are, which is a property of the CASE,
    not of the model. Reading it off the model meant a bare `ginot` Trunk (which
    has no per-case state) crashed here, so the baseline could not be given the
    per-region breakdown that the whole comparison is supposed to be read by.
    """
    if idx is None or idx.count["supply"] == 0:
        return {}
    d, _ = idx.tree["supply"].query(np.ascontiguousarray(xyz_m, np.float64), k=1, workers=-1)
    out = {}
    for name, lo, hi in ZONES:
        m = (d >= lo) & (d < hi)
        if m.sum() < 100:
            continue
        z = joint_metrics(true[m], pred[m])
        z["n_points"] = int(m.sum())
        z["frac_points"] = float(m.mean())
        z["frac_variance"] = float(true[m, 0:3].var(0).sum() * m.sum()
                                   / (true[:, 0:3].var(0).sum() * len(true)))
        out[name] = z
    return out


_COMFORT_CACHE = {}


def comfort_setback_trees(case_dir):
    """KD-trees of the setback surfaces, one per class (exterior/internal/furniture)."""
    if case_dir in _COMFORT_CACHE:
        return _COMFORT_CACHE[case_dir]
    import glob as _glob

    import pandas as pd
    from scipy.spatial import cKDTree
    buckets = {"ext": [], "int": [], "furn": []}
    for f in _glob.glob(os.path.join(case_dir, "*.csv")):
        stem = os.path.basename(f).lower().replace(".csv", "")
        if stem == "fluid_data" or stem in NO_SETBACK_SURFACES:
            continue
        if stem in EXTERIOR_SURFACES or stem in HVAC_SURFACES:
            key = "ext"
        elif stem in INTERNAL_SURFACES:
            key = "int"
        else:
            key = "furn"
        buckets[key].append(pd.read_csv(f, usecols=["X (m)", "Y (m)", "Z (m)"]).to_numpy())
    trees = {k: (cKDTree(np.vstack(v)) if v else None) for k, v in buckets.items()}
    _COMFORT_CACHE[case_dir] = trees
    return trees


def comfort_metrics(case_dir, xyz_m, true, pred, d_ext=SETBACK_EXTERIOR_M,
                    d_int=SETBACK_INTERNAL_M, d_furn=SETBACK_FURNITURE_M):
    """ASHRAE 55 occupied zone: height band, set back per surface class."""
    out = {}
    trees = comfort_setback_trees(case_dir)
    ok = np.ones(len(xyz_m), bool)
    for key, dist in (("ext", d_ext), ("int", d_int), ("furn", d_furn)):
        t = trees.get(key)
        if t is not None and dist > 0:
            ok &= t.query(xyz_m, k=1, workers=-1)[0] >= dist
    z = xyz_m[:, 2]
    for name, lo, hi in ASHRAE_ZONES:
        m = (z >= lo) & (z <= hi) & ok
        if m.sum() < 100:
            continue
        c = joint_metrics(true[m], pred[m])
        c["n_points"] = int(m.sum())
        c["frac_points"] = float(m.mean())
        # Flat scalars: summarise() averages every value in this dict, and a
        # nested dict is not summable.
        c["setback_exterior_m"] = float(d_ext)
        c["setback_internal_m"] = float(d_int)
        c["setback_furniture_m"] = float(d_furn)
        out[name] = c
    return out


def grad_frobenius(model, room, ds, device, n=20000, seed=0, grad_h=0.01):
    """Median ||grad u||_F [1/s] of the prediction, by central differences.

    Compared against the CFD's 2.52 and the prior baseline's 0.84. Restricted to
    the live interior: with a hard-BC model, 32% of occupied-zone points sit in
    the clamped shell where u == 0, and including them makes any derivative
    statistic meaningless (the prior project's +0.281 -> +0.003 trap).
    """
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    sel = torch.randperm(len(xyz_n), generator=torch.Generator().manual_seed(seed))[:n]
    x = xyz_n[sel].to(device)
    active = getattr(model, "_active", None)
    if getattr(model, "use_hbc", False) and active is not None:
        xm = (x.cpu() * ds.coord_scale + ds.coord_min).numpy().astype(np.float64)
        keep = active.wall_distance(xm) > 0.05
        x = x[torch.as_tensor(keep, device=device)]
    if len(x) == 0:
        return {}
    lat = model.encode_geometry(room["pc_full"].unsqueeze(0).to(device))
    scale = ds.coord_scale.to(device)          # normalized coords -> metres
    sig = ds.target_std[0:3].to(device)

    @torch.no_grad()
    def vel(q):
        out = torch.cat([model.decode_query(lat, q[s:s + 16384].unsqueeze(0)).squeeze(0)
                         for s in range(0, len(q), 16384)], 0)
        return out[:, 0:3] * sig

    # CENTRAL DIFFERENCES, NOT AUTOGRAD. The local branch gathers its neighbours
    # through scipy, which means `_metres` does .detach().cpu().numpy() and the
    # autograd path from the prediction back to the query coordinate is SEVERED.
    # Autograd therefore reports only the global path's derivative -- it is blind
    # to the local branch's spatial variation -- and on the local-only arm, where
    # there is no global path at all, it raises outright ("differentiated Tensor
    # ... not used in the graph"). That is how this was found. Finite differences
    # measure the field the model actually produces, identically for every arm.
    #
    # `h` is a real length and the number depends on it: the prior project's own
    # K-sweep of the CFD gradient spans 3.38 (small neighbourhood) to 1.19
    # (large), with 2.52 at K=60. Report h alongside the value.
    h_m = float(grad_h)
    rows = []
    for j in range(3):
        e = torch.zeros(3, device=device)
        e[j] = h_m / scale[j]                   # metres -> normalized units
        rows.append((vel(x + e) - vel(x - e)) / (2.0 * h_m))
    J = torch.stack(rows, -1)                   # (N, 3 out, 3 in)
    fro = J.flatten(1).norm(dim=1)
    q = torch.quantile(fro.float(), torch.tensor([0.5, 0.9, 0.99], device=device))
    return {"grad_fro_median": float(q[0]), "grad_fro_p90": float(q[1]),
            "grad_fro_p99": float(q[2]), "grad_fro_mean": float(fro.mean()),
            "grad_h_m": h_m, "n_grad_points": int(len(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default=os.path.join(ROOT, "splits", "val"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--ckpt", default="best.pth",
                    help="checkpoint file inside the run dir")
    ap.add_argument("--blind_return_velocity", action="store_true",
                    help="Leak probe: zero solved return-vent u/v/w in the "
                         "encoder input, keeping vent positions and class. "
                         "Inference-only, so the result is an UPPER BOUND on "
                         "the leak's contribution, not the leak-free score.")
    ap.add_argument("--setback", type=float, default=SETBACK_EXTERIOR_M,
                    help="setback [m] from outside walls, windows, doors and "
                         "fixed HVAC equipment")
    ap.add_argument("--setback_internal", type=float, default=SETBACK_INTERNAL_M,
                    help="setback [m] from internal walls and partitions")
    ap.add_argument("--setback_furniture", type=float, default=SETBACK_FURNITURE_M,
                    help="small setback [m] from furniture, enough to skip its "
                         "boundary layer without excluding the air above a desk")
    ap.add_argument("--grad_points", type=int, default=20000,
                    help="0 disables the ||grad u|| measurement")
    args = ap.parse_args()

    _CKPT[0] = args.ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from thermo.dataset import ThermoDataset

    a = json.load(open(os.path.join(args.run, "args.json")))
    # A run that recomputed its own normalisation (the CV folds) records
    # stats_path=null, because train_thermo defaults the path inside main()
    # rather than through argparse. The file it wrote lives in the run dir.
    if not a.get("stats_path"):
        local = os.path.join(args.run, "thermo_stats.pt")
        if not os.path.exists(local):
            raise SystemExit(f"{args.run}: args.json has no stats_path and "
                             f"{local} does not exist -- cannot rebuild the "
                             f"normalisation this model was trained with.")
        a["stats_path"] = local
    folders = sorted(glob.glob(os.path.join(args.split, "*/")))
    names = [os.path.basename(f.rstrip("/")) for f in folders]
    ds = ThermoDataset(case_dirs=[f.rstrip("/") for f in folders],
                       saved_stats=a["stats_path"], stats_path=a["stats_path"],
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")

    # A run TRAINED with return velocity blinded must be SCORED that way, or the
    # evaluation is a train/test mismatch. args.json is the record, so honour it
    # even when the flag is not passed here.
    if a.get("blind_return_velocity") and not args.blind_return_velocity:
        print("[probe] args.json records blind_return_velocity=True; "
              "applying it so evaluation matches training")
    if args.blind_return_velocity or a.get("blind_return_velocity"):
        # Leak probe. Return vents are pressure OUTLETS, so their velocity is an
        # output of the same solve the model is asked to reproduce, yet it is fed
        # to the geometry encoder on ~46% of the point cloud. Zero it here to ask
        # whether the model leans on it. Positions and the CLS_OUTLET one-hot are
        # untouched, so the layout signal -- where the vents are, which is the
        # only thing that varies in this dataset -- is fully preserved.
        #
        # This is INFERENCE-ONLY, so the model still saw those values in
        # training: any drop is an upper bound on the leak's contribution and
        # partly train/test mismatch. It cannot be read as the leak-free score;
        # only retraining with the same blinding gives that.
        from thermo.dataset import PC_CLS
        n_z = n_tot = 0
        for case in ds.cases:
            pc = case.get("pc_full")
            if pc is None or pc.shape[1] < 11:
                continue
            m = pc[:, PC_CLS].argmax(dim=-1) == 3          # CLS_OUTLET
            pc[m, 8:11] = 0.0                              # u, v, w
            n_z += int(m.sum()); n_tot += len(pc)
        print(f"[probe] zeroed return-vent u/v/w on {n_z:,} of {n_tot:,} "
              f"point-cloud points ({n_z / max(n_tot, 1):.1%})")

    model, a = build_from_args(args.run, ds, device)
    if getattr(model, "use_local", False) or getattr(model, "use_hbc", False):
        model.register_cases(ds, verbose=False,
                             normals_k=a.get("normals_k", 16),
                             with_normals=not a.get("no_normals", False))

    # Zone splits need vent positions for EVERY arm, including the bare `ginot`
    # trunk that carries no per-case state. Build them from the dataset (normals
    # off: only the supply KD-tree is used here, and normals are the slow part).
    from data.knn_cache import BoundaryIndex
    case_index = [BoundaryIndex(r, ds.coord_min, ds.coord_scale, ds.target_mean,
                                ds.target_std, with_normals=False,
                                groups=_groups_of(a), far_voxels=_voxels_of(a))
                  for r in ds.cases]

    raw, uni, zones, comfort, grads = [], [], [], [], []
    for ci, (room, name) in enumerate(zip(ds.cases, names)):
        t0 = time.time()
        xyz, true, pred = predict(model, room, ds, device, args.chunk)
        dt = time.time() - t0
        m, u = joint_metrics(true, pred), joint_metrics(true[(c := uniform_cells(xyz))],
                                                        pred[c])
        raw.append(m)
        uni.append(u)
        zones.append(zone_metrics(case_index[ci], xyz, true, pred))
        comfort.append(comfort_metrics(folders[ci].rstrip("/"), xyz, true, pred,
                                       args.setback, args.setback_internal,
                                       args.setback_furniture))
        print(f"  {name}  {len(xyz):>9,} pts  {dt:5.1f}s ({len(xyz)/dt/1e6:.2f} M/s)")
        print(f"    raw:     {fmt(m)}")
        print(f"    uniform: {fmt(u)}")
        for zn, zm in zones[-1].items():
            print(f"    {zn:11} n={zm['n_points']:>8,} ({zm['frac_points']:.0%} pts, "
                  f"{zm['frac_variance']:.0%} var)  v_rho={zm['vel_rho']:+.3f} "
                  f"v_R2={zm['vel_r2']:+.3f} S={zm['score']:.3f}")
        for zn, zm in comfort[-1].items():
            print(f"    {zn:16} n={zm['n_points']:>8,} ({zm['frac_points']:.0%} pts)"
                  f"  v_rho={zm['vel_rho']:+.3f} v_R2={zm['vel_r2']:+.3f} "
                  f"T_mae={zm['temp_mae']:.3f}K")
        if args.grad_points:
            g = grad_frobenius(model, room, ds, device, args.grad_points)
            grads.append(g)
            if g:
                print(f"    ||grad u||_F median {g['grad_fro_median']:.3f} 1/s "
                      f"(h={g['grad_h_m']:.2f} m central diff; NOT comparable "
                      f"to the prior 2.52, which is a mean from another dataset "
                      f"and estimator -- see physics_gradients.py)")

    res = {"run": os.path.basename(args.run.rstrip("/")), "split": args.split,
           "cases": names, "per_case": raw, "per_case_uniform": uni,
           "per_case_zones": zones, "per_case_comfort": comfort,
           "per_case_grad": grads,
           "mean": summarise(raw), "mean_uniform": summarise(uni)}
    for zn, _, _ in ZONES:
        zs = [z[zn] for z in zones if zn in z]
        if zs:
            res.setdefault("mean_zones", {})[zn] = summarise(zs)
    for zn, _, _ in ASHRAE_ZONES:
        cs = [c[zn] for c in comfort if zn in c]
        if cs:
            res.setdefault("mean_comfort", {})[zn] = summarise(cs)
    if grads and any(grads):
        res["mean_grad"] = summarise([g for g in grads if g])

    print(f"\n--- MEAN over {len(names)} cases ---")
    print(f"  raw:     {fmt(res['mean'])}")
    print(f"  uniform: {fmt(res['mean_uniform'])}")
    for zn, zm in res.get("mean_zones", {}).items():
        print(f"  {zn:11} v_rho={zm['vel_rho']:+.3f} v_R2={zm['vel_r2']:+.3f} "
              f"score={zm['score']:.4f}")
    if res.get("mean_comfort"):
        print(f"  -- ASHRAE 55 occupied zone (setback {args.setback:.1f} m walls/"
              f"HVAC, {args.setback_internal:.1f} m internal, "
              f"{args.setback_furniture:.1f} m furniture) --")
        for zn, zm in res["mean_comfort"].items():
            print(f"  {zn:16} ({zm['frac_points']:.0%} pts) v_rho={zm['vel_rho']:+.3f} "
                  f"v_R2={zm['vel_r2']:+.3f} v_mae={zm['vel_mae']:.3f}m/s "
                  f"T_mae={zm['temp_mae']:.3f}K score={zm['score']:.4f}")
    if "mean_grad" in res:
        print(f"  ||grad u||_F median {res['mean_grad']['grad_fro_median']:.3f} 1/s")

    out = args.out or os.path.join(args.run, f"eval_{os.path.basename(args.split.rstrip('/'))}.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
