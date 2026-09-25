"""What is ||grad u||_F of the CFD itself, at a stated length scale?

The prior project's "CFD 2.52" is one point on a curve, not a constant: their own
K-sweep of the same quantity gives 3.38 / 2.52 / 1.90 / 1.42 / 1.19 as the
neighbourhood grows. So "the model is at 6 and the CFD is at 2.52" is not a
comparison until both are measured the same way.

This measures the ground truth by weighted least squares on the CFD point cloud
itself -- fit u(x + d) ~ u(x) + J d over the K nearest interior points -- and
reports it at several K, so a model number measured by central differences at
step h can be read against the truth at a comparable scale. Nothing is written
to the dataset.

    python -m tests.cfd_gradient --case splits/val/Case_07
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="splits/val/Case_07")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--K", type=int, nargs="+", default=[10, 20, 30, 60, 120])
    ap.add_argument("--min_wall", type=float, default=0.05)
    args = ap.parse_args()

    from thermo.dataset import ThermoDataset
    from data.knn_cache import BoundaryIndex

    stats = os.path.join(ROOT, "stats", "thermo_stats.pt")
    ds = ThermoDataset(case_dirs=[os.path.join(ROOT, args.case)], saved_stats=stats,
                       stats_path=stats, n_case_pool=1_500_000,
                       n_boundary_pool=250_000, n_pc=15000, pc_mode="full")
    room = ds.cases[0]
    xyz = (torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
           * ds.coord_scale + ds.coord_min).numpy().astype(np.float64)
    vel = (torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], 0)
           * ds.target_std + ds.target_mean).numpy()[:, 0:3].astype(np.float64)

    # Same live-interior filter the model-side measurement uses, so the two are
    # taken over comparable point sets.
    bi = BoundaryIndex(room, ds.coord_min, ds.coord_scale, ds.target_mean,
                       ds.target_std, with_normals=False)
    live = bi.wall_distance(xyz) > args.min_wall
    xyz, vel = xyz[live], vel[live]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(xyz), min(args.n, len(xyz)), replace=False)

    tree = cKDTree(xyz)
    print(f"\n  {os.path.basename(args.case)}: {len(xyz):,} live interior points "
          f"(> {args.min_wall} m from a wall), sampling {len(sel):,}\n")
    print(f"  {'K':>4} {'radius':>9} {'median':>8} {'MEAN':>8} {'p90':>8} {'p99':>8}   "
          f"||grad u||_F of the CFD [1/s]")
    for K in args.K:
        d, idx = tree.query(xyz[sel], k=K + 1, workers=-1)
        nb, nv = xyz[idx[:, 1:]], vel[idx[:, 1:]]
        dx = nb - xyz[sel][:, None, :]                       # (n, K, 3)
        du = nv - vel[sel][:, None, :]                       # (n, K, 3)
        # Weighted least squares for J: minimise |J dx - du|, weight ~ 1/r.
        w = 1.0 / np.maximum(np.linalg.norm(dx, axis=2, keepdims=True), 1e-6)
        A, B = dx * w, du * w
        AtA = np.einsum("nki,nkj->nij", A, A)
        AtB = np.einsum("nki,nkj->nij", A, B)
        AtA += np.eye(3) * 1e-9 * np.trace(AtA, axis1=1, axis2=2)[:, None, None]
        J = np.linalg.solve(AtA, AtB)                        # (n, 3 in, 3 out)
        fro = np.linalg.norm(J.reshape(len(J), -1), axis=1)
        q = np.percentile(fro, [50, 90, 99])
        print(f"  {K:>4} {np.median(d[:, -1]):9.3f} {q[0]:8.3f} {fro.mean():8.3f} {q[1]:8.3f} {q[2]:8.3f}")
    print(f"\n  The prior project quotes 2.52 for the CFD at K=60; 'radius' is the "
          f"median distance\n  to the Kth neighbour, i.e. the length scale each row "
          f"actually measures.")


if __name__ == "__main__":
    main()
