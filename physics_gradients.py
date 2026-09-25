#!/usr/bin/env python
"""Matched-operator gradient and divergence, model against CFD.

    "gradient/vorticity discrepancy using the same operator and smoothing scale
     for CFD and prediction"
    "Run the same reconstruction on CFD cell velocities. The two reference
     residuals expose interpolation error versus the solver's actual flux
     balance."

WHY THIS EXISTS. `evaluate.py:grad_frobenius` measures the MODEL's gradient by
central differences at h = 0.01 m, and its output used to be printed beside
"CFD 2.52". That comparison is void three times over: 2.52 is a MEAN (ours is
1.63) where we quote a median (~0.5), it comes from the prior project's
STAR-CCM+ data rather than this dataset, and it was produced by a K=60 KNN
estimator whose own sweep spans 3.38 -> 1.19. The paper already records
that every gradient claim built on it was retracted.

The only defensible comparison applies ONE estimator to BOTH fields at the SAME
points. That is what this does:

  * query the model at the CFD's own cell centres, so both fields live on the
    identical point set and no interpolation is applied to either;
  * fit a local weighted least-squares linear model over the K nearest cell
    centres, giving the full 3x3 Jacobian for the CFD field and for the
    prediction;
  * report ||grad u||_F and div(u) for both, and the RATIO model/CFD, which is
    the robust quantity -- absolute values move with K, the ratio much less;
  * sweep K, so the operator dependence is visible rather than hidden.

The CFD's own div(u) under the same estimator is the REFERENCE FLOOR. It is not
zero: it measures what a local linear fit to cell-centre velocities returns on a
boundary-refined mesh, and the model can only be judged against it, never
against zero. This is the alternative to a face-flux balance, which is blocked
until face
areas and cell volumes are exported.

    $PY physics_gradients.py --run <run_dir> --split splits_cfd_gap/val
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

import evaluate as ev  # noqa: E402

WORKERS = int(os.environ.get("LGO_KNN_WORKERS", "-1"))


def local_jacobian(xyz, vel, idx, k, weighted=True):
    """(N,3,3) velocity Jacobian by local weighted least squares.

    For each sample point, fit `u(x) ~ u0 + J (x - x0)` over its k nearest cell
    centres. Identical code path for CFD and prediction -- that is the whole
    point of the exercise.
    """
    tree = cKDTree(xyz)
    d, nb = tree.query(xyz[idx], k=k, workers=WORKERS)      # (N,k)
    dX = xyz[nb] - xyz[idx][:, None, :]                     # (N,k,3)
    dU = vel[nb] - vel[idx][:, None, :]                     # (N,k,3)
    if weighted:
        # Gaussian in units of the neighbourhood's own scale, so the weighting
        # does not import a length that varies with local mesh density.
        h = np.maximum(d[:, -1:], 1e-9)
        w = np.exp(-(d / h) ** 2)[:, :, None]
    else:
        w = np.ones_like(dX[:, :, :1])
    # Solve min_J || W^.5 (dU - dX J^T) ||  ->  J^T = (dX^T W dX)^-1 dX^T W dU
    A = np.einsum("nki,nkj->nij", dX * w, dX)               # (N,3,3)
    B = np.einsum("nki,nkj->nij", dX * w, dU)               # (N,3,3)
    A = A + 1e-12 * np.eye(3)[None]
    JT = np.linalg.solve(A, B)                              # (N,3,3): dU/dx
    return np.swapaxes(JT, 1, 2)                            # (N,3,3) J[i,j]=du_i/dx_j


def stats(J):
    fro = np.linalg.norm(J.reshape(len(J), -1), axis=1)
    div = np.abs(J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2])
    q = lambda a: [float(np.median(a)), float(a.mean()),
                   float(np.percentile(a, 90)), float(np.percentile(a, 99))]
    return {"fro_median": q(fro)[0], "fro_mean": q(fro)[1],
            "fro_p90": q(fro)[2], "fro_p99": q(fro)[3],
            "div_median": q(div)[0], "div_mean": q(div)[1],
            "div_p90": q(div)[2], "div_p99": q(div)[3]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="splits_cfd_gap/val")
    ap.add_argument("--out", default="results/physics_gradients.json")
    ap.add_argument("--k", type=int, nargs="+", default=[20, 60, 120])
    ap.add_argument("--n_points", type=int, default=20000)
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

    per_case, rng = [], np.random.default_rng(args.seed)
    for ci, (room, name) in enumerate(zip(ds.cases, names)):
        t0 = time.time()
        xyz, true, pred = ev.predict(model, room, ds, device)
        xyz = xyz.astype(np.float64)
        idx = rng.choice(len(xyz), size=min(args.n_points, len(xyz)),
                         replace=False)
        rec = {"case": name, "n_points": int(len(idx))}
        for k in args.k:
            Jc = local_jacobian(xyz, true[:, 0:3].astype(np.float64), idx, k)
            Jm = local_jacobian(xyz, pred[:, 0:3].astype(np.float64), idx, k)
            sc, sm = stats(Jc), stats(Jm)
            rec[f"K{k}"] = {"cfd": sc, "model": sm,
                            "fro_ratio": sm["fro_median"] / max(sc["fro_median"], 1e-12),
                            "div_ratio": sm["div_median"] / max(sc["div_median"], 1e-12)}
        per_case.append(rec)
        k0 = args.k[0]
        print(f"  {name:<20} {time.time()-t0:5.1f}s  "
              + "  ".join(f"K{k}: |grad| cfd {rec[f'K{k}']['cfd']['fro_median']:.3f} "
                          f"model {rec[f'K{k}']['model']['fro_median']:.3f} "
                          f"({rec[f'K{k}']['fro_ratio']:.2f}x)" for k in args.k),
              flush=True)

    out = {"run": args.run, "split": args.split, "k_values": args.k,
           "n_points_per_case": args.n_points, "per_case": per_case, "mean": {}}
    for k in args.k:
        m = {}
        for side in ("cfd", "model"):
            for key in per_case[0][f"K{k}"][side]:
                m[f"{side}_{key}"] = float(np.mean(
                    [c[f"K{k}"][side][key] for c in per_case]))
        for key in ("fro_ratio", "div_ratio"):
            m[key] = float(np.mean([c[f"K{k}"][key] for c in per_case]))
        out["mean"][f"K{k}"] = m

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== MEAN over {len(per_case)} cases, one estimator on both fields ===")
    print(f"{'K':>5}{'CFD |grad| med':>16}{'model |grad| med':>18}{'ratio':>8}"
          f"{'CFD |div| med':>15}{'model |div| med':>17}{'ratio':>8}")
    for k in args.k:
        m = out["mean"][f"K{k}"]
        print(f"{k:>5}{m['cfd_fro_median']:>16.3f}{m['model_fro_median']:>18.3f}"
              f"{m['fro_ratio']:>8.2f}{m['cfd_div_median']:>15.4f}"
              f"{m['model_div_median']:>17.4f}{m['div_ratio']:>8.2f}")
    print(f"\nsaved {args.out}")
    print("The CFD's own div is the REFERENCE FLOOR, not zero: it is what a local "
          "linear fit\nto cell-centre velocities returns on this mesh. Judge the "
          "model against it.")


if __name__ == "__main__":
    main()
