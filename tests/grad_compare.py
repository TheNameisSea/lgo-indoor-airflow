"""Model vs CFD velocity gradient, measured by ONE estimator on BOTH.

Every earlier gradient number in this project was measured two different ways on
the two sides and is not a comparison:

  * autograd on the model  -- and the local branch gathers neighbours through
    scipy, so the autograd path back to the query is severed. It reported the
    global path only, and raised outright on the local-only arm.
  * central differences at h = 1 cm on the model, against a KNN least-squares
    fit at K = 60 (radius ~ 19 cm) on the CFD. Those measure different length
    scales, and the CFD's own value moves with the scale.

Here the model is evaluated AT the CFD points and both fields go through the
same weighted least-squares fit over the same neighbourhoods, so the ratio is
meaningful. Live interior only (> 0.05 m from a wall): with a hard-BC model 32%
of occupied-zone points sit in the clamped shell where u == 0 and any derivative
statistic over them is trivially zero.

    python -m tests.grad_compare --run runs/lgo_lr3e-3_s0
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))


def jacobian_lsq(xyz, vel, sel, K, tree):
    """Weighted LSQ fit of J at `sel`, from the K nearest points."""
    _, idx = tree.query(xyz[sel], k=K + 1, workers=-1)
    dx = xyz[idx[:, 1:]] - xyz[sel][:, None, :]
    du = vel[idx[:, 1:]] - vel[sel][:, None, :]
    w = 1.0 / np.maximum(np.linalg.norm(dx, axis=2, keepdims=True), 1e-6)
    A, B = dx * w, du * w
    AtA = np.einsum("nki,nkj->nij", A, A)
    AtB = np.einsum("nki,nkj->nij", A, B)
    AtA += np.eye(3) * 1e-9 * np.trace(AtA, axis1=1, axis2=2)[:, None, None]
    J = np.linalg.solve(AtA, AtB)
    return np.linalg.norm(J.reshape(len(J), -1), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--case", default="splits/val/Case_07")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--min_wall", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()

    from thermo.dataset import ThermoDataset
    from evaluate import build_from_args
    from data.knn_cache import BoundaryIndex

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = json.load(open(os.path.join(args.run, "args.json")))
    ds = ThermoDataset(case_dirs=[os.path.join(ROOT, args.case)],
                       saved_stats=a["stats_path"], stats_path=a["stats_path"],
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    room = ds.cases[0]
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    true = (torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], 0)
            * ds.target_std + ds.target_mean).numpy()[:, 0:3].astype(np.float64)
    xyz = (xyz_n * ds.coord_scale + ds.coord_min).numpy().astype(np.float64)

    model, _ = build_from_args(args.run, ds, dev)
    if hasattr(model, "register_cases"):
        try:
            model.register_cases(ds, verbose=False)
        except TypeError:
            model.register_cases(ds)
    model.eval()
    with torch.no_grad():
        lat = model.encode_geometry(room["pc_full"].unsqueeze(0).to(dev))
        out = torch.cat([
            model.decode_query(lat, xyz_n[s:s + args.chunk].unsqueeze(0).to(dev)).squeeze(0)
            for s in range(0, len(xyz_n), args.chunk)], 0)
    pred = (out.cpu() * ds.target_std[:5] + ds.target_mean[:5]).numpy()[:, 0:3].astype(np.float64)

    bi = BoundaryIndex(room, ds.coord_min, ds.coord_scale, ds.target_mean,
                       ds.target_std, with_normals=False)
    live = bi.wall_distance(xyz) > args.min_wall
    xyz, true, pred = xyz[live], true[live], pred[live]
    sel = np.random.default_rng(0).choice(len(xyz), min(args.n, len(xyz)), replace=False)
    tree = cKDTree(xyz)

    ft = jacobian_lsq(xyz, true, sel, args.K, tree)
    fp = jacobian_lsq(xyz, pred, sel, args.K, tree)
    st = np.linalg.norm(true, axis=1)
    sp = np.linalg.norm(pred, axis=1)

    name = os.path.basename(args.run.rstrip("/"))
    print(f"\n  {name}  |  {os.path.basename(args.case)}  |  K={args.K}, "
          f"{len(sel):,} live points\n")
    print(f"  {'':10} {'median':>9} {'mean':>9} {'p90':>9} {'p99':>9}")
    print(f"  {'CFD':10} {np.median(ft):9.3f} {ft.mean():9.3f} "
          f"{np.percentile(ft,90):9.3f} {np.percentile(ft,99):9.3f}")
    print(f"  {'model':10} {np.median(fp):9.3f} {fp.mean():9.3f} "
          f"{np.percentile(fp,90):9.3f} {np.percentile(fp,99):9.3f}")
    print(f"  {'ratio':10} {np.median(fp)/np.median(ft):9.2f} "
          f"{fp.mean()/ft.mean():9.2f}   (1.0 = right roughness; "
          f">1 too rough, <1 too smooth)")
    print(f"\n  speed  CFD median {np.median(st):.4f} m/s   model {np.median(sp):.4f} m/s")


if __name__ == "__main__":
    main()
