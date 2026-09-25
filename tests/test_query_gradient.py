"""Does d(prediction)/d(query) actually exist, and is it RIGHT?

The local branch gathers its neighbours through a scipy KD-tree, and the
offsets `dxyz = x_boundary - x_query` were computed in numpy. That severs the
autograd path from the prediction back to the query coordinate: the offsets
arrive as constants, so the derivative through the local branch is identically
zero. Autograd then reports only the global path -- and on a local-only arm it
raises outright. Every earlier gradient claim in this project was retracted for
that reason.

`LGO.grad_wrt_query = True` rebuilds those offsets in torch from the live query
tensor, restoring the path. This test is the receipt. It compares the autograd
Jacobian against CENTRAL DIFFERENCES on the same points, because a silently
wrong gradient is invisible in a loss curve and would make anything built on it
-- a curl head, an equivariance measurement -- worthless.

Neighbour SELECTION stays non-differentiable and that is correct: it is
piecewise constant in the query. Points where the k-th neighbour changes are a
measure-zero set on which the true derivative is undefined; a few sampled
queries may land near one, so the test reports a ROBUST statistic (median
relative error) rather than requiring every point to agree.

    python -m tests.test_query_gradient
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

RUN = os.path.join(ROOT, "runs", "gap143_gqlocal_s0")
CASE = os.path.join(ROOT, "splits_cfd_gap", "val", "B030")
N_Q, H_M = 24, 1e-3          # queries sampled; central-difference step [m]


def build():
    import evaluate as ev
    from thermo.dataset import ThermoDataset
    ds = ThermoDataset(case_dirs=[CASE],
                       saved_stats=os.path.join(RUN, "thermo_stats.pt"),
                       stats_path="/tmp/_qgrad_stats.pt",
                       n_case_pool=200_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, a = ev.build_from_args(RUN, ds, dev)
    # The local branch and the no-slip mask need the FULL boundary pools, not
    # the encoder's 15k subsample, so the per-case index must be registered
    # before any decode. evaluate.py does the same.
    if getattr(model, "use_local", False) or getattr(model, "use_hbc", False):
        model.register_cases(ds, verbose=False,
                             normals_k=a.get("normals_k", 16),
                             with_normals=not a.get("no_normals", False))
    model.eval()
    return model, ds, dev


def jacobian_autograd(model, lat, q):
    """(N, 3, 3) d(u_i)/d(x_j) in physical units, by autograd."""
    q = q.clone().requires_grad_(True)
    out = model.decode_query(lat, q.unsqueeze(0)).squeeze(0)[:, 0:3]
    rows = []
    for i in range(3):
        # allow_unused is REQUIRED for the severed baseline: with the offsets
        # coming from numpy the query is not in the graph at all, and autograd
        # raises rather than returning zero. That raise is itself the evidence
        # the path was broken, so the test records it instead of crashing on it.
        g, = torch.autograd.grad(out[:, i].sum(), q, retain_graph=(i < 2),
                                 create_graph=False, allow_unused=True)
        rows.append(torch.zeros_like(q) if g is None else g)
    return torch.stack(rows, 1)          # (N, 3 out, 3 in), normalised coords


def jacobian_fd(model, lat, q, scale, h_m):
    """Same Jacobian by central differences, stepping in METRES."""
    rows = []
    with torch.no_grad():
        for j in range(3):
            e = torch.zeros(3, device=q.device, dtype=q.dtype)
            e[j] = h_m / scale[j]                       # metres -> normalised
            up = model.decode_query(lat, (q + e).unsqueeze(0)).squeeze(0)[:, 0:3]
            dn = model.decode_query(lat, (q - e).unsqueeze(0)).squeeze(0)[:, 0:3]
            rows.append((up - dn) / (2.0 * (h_m / scale[j])))
    return torch.stack(rows, 2)          # (N, 3 out, 3 in)


def main():
    model, ds, dev = build()
    room = ds.cases[0]
    scale = ds.coord_scale.to(dev)
    lat = model.encode_geometry(room["pc_full"].unsqueeze(0).to(dev))
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    sel = torch.randperm(len(xyz_n), generator=torch.Generator().manual_seed(0))[:N_Q]
    q = xyz_n[sel].to(dev)

    print(f"  {N_Q} queries, central-difference step {H_M} m\n")

    model.grad_wrt_query = False
    off = jacobian_autograd(model, lat, q)
    print(f"  grad_wrt_query = False -> |J|_max {off.abs().max():.3e}   "
          f"(severed: the query is not in the graph at all)")

    model.grad_wrt_query = True
    ja = jacobian_autograd(model, lat, q)
    jf = jacobian_fd(model, lat, q, scale, H_M)
    assert ja.abs().max() > 0, "autograd still returns zero -- offsets not rebuilt"

    num = (ja - jf).flatten(1).norm(dim=1)
    den = jf.flatten(1).norm(dim=1).clamp_min(1e-12)
    rel = (num / den).cpu().numpy()
    med, p90 = float(np.median(rel)), float(np.percentile(rel, 90))
    print(f"  grad_wrt_query = True  -> |J|_max {ja.abs().max():.3e}")
    print(f"  autograd vs central differences, relative error per query:")
    print(f"     median {med:.4f}   p90 {p90:.4f}   max {rel.max():.4f}")

    assert med < 0.05, (f"autograd disagrees with finite differences: median "
                        f"relative error {med:.4f}. The restored gradient is "
                        f"WRONG -- do not build a curl head on it.")
    print(f"\n  PASS: the query-coordinate gradient exists and matches finite "
          f"differences\n        (median {med:.2%} relative error; outliers are "
          f"queries near a neighbour-set change)")


if __name__ == "__main__":
    main()
