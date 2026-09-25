#!/usr/bin/env python
"""Hybrid vs its branch-less host vs CFD truth, five heights, one field.

    $PY make_pair_figure.py --hybrid_run runs/final58_lgo_gino_lr3e-3 \
        --host_run runs/final58_gino_lr3e-3 --case splits_cfd_gap58/val/D061

Three rows -- LGO+host / host alone / CFD truth -- across five heights, written
as SEPARATE figures for speed and temperature. The pair is an exact ablation:
the two runs are bit-identical at initialisation and differ only by the local
branch, so the top two rows isolate what the branch does to the field.

ONE COLOUR SCALE PER FIGURE, TAKEN FROM THE TRUTH ROW. Neither model can look
better by predicting a different range, and the two model rows are directly
comparable to each other as well as to the truth.

The panels are built by `visualize.build_panels`, the same code `visualize.py`
uses -- including its obstacle test, which must be done in 3D at the exact slice
height (the comments there record three attempts that produced well-formed and
wrong pictures). NO METRIC IS COMPUTED HERE: interpolating truth across a slab
flattens steep gradients, which is the prior project's trap T10. Numbers come
from `evaluate.py`, at the CFD points.

REPRODUCIBILITY. The dataset subsamples the boundary cloud, so two renders of
the same case differ unless the seed is fixed -- measured at ~18% of pixels.
`--seed` (default 0) pins it, and the seed is written into the figure caption.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

from visualize import HEIGHTS, build_panels, display_grid  # noqa: E402

FIELDS = {
    "vel":  dict(idx=None, cmap="turbo",     label="speed [m/s]",   floor_zero=True),
    "temp": dict(idx=4,    cmap="coolwarm",  label="temperature [K]", floor_zero=False),
}


def load_run(run, case, device, seed):
    """Prediction at every interior CFD point of `case`, for one run."""
    from thermo.dataset import ThermoDataset
    from evaluate import build_from_args
    a = json.load(open(os.path.join(run, "args.json")))
    if not a.get("stats_path"):
        local = os.path.join(run, "thermo_stats.pt")
        if not os.path.exists(local):
            raise SystemExit(f"{run}: no stats_path and no {local}")
        a["stats_path"] = local
    np.random.seed(seed)
    torch.manual_seed(seed)
    ds = ThermoDataset(case_dirs=[os.path.join(ROOT, case)],
                       saved_stats=a["stats_path"], stats_path=a["stats_path"],
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    room = ds.cases[0]
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    true = (torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], 0)
            * ds.target_std + ds.target_mean).numpy()
    xyz = (xyz_n * ds.coord_scale + ds.coord_min).numpy()
    model, _ = build_from_args(run, ds, device)
    if hasattr(model, "register_cases"):
        try:
            model.register_cases(ds, verbose=False)
        except TypeError:
            model.register_cases(ds)
    model.eval()
    with torch.no_grad():
        lat = model.encode_geometry(room["pc_full"].unsqueeze(0).to(device))
        out = torch.cat([model.decode_query(lat, xyz_n[s:s + 16384].unsqueeze(0).to(device)).squeeze(0)
                         for s in range(0, len(xyz_n), 16384)], 0)
    pred = (out.cpu() * ds.target_std[:5] + ds.target_mean[:5]).numpy()
    return xyz, true, pred


def scalar(arr, field):
    return np.linalg.norm(arr[:, 0:3], axis=1) if field == "vel" else arr[:, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid_run", required=True)
    ap.add_argument("--host_run", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--fields", nargs="+", default=["vel", "temp"])
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--tag", default="")
    ap.add_argument("--heights", type=float, nargs="+", default=HEIGHTS)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--slab", type=float, default=0.15)
    ap.add_argument("--levels", type=int, default=32)
    ap.add_argument("--obstacle_r", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raster_dpi", type=int, default=400,
                    help="resolution of the rasterised contour fill inside the "
                         "PDF; text and axes remain vector")
    ap.add_argument("--hybrid_label", default=None)
    ap.add_argument("--host_label", default=None)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_name = os.path.basename(args.case.rstrip("/"))
    hy_name = args.hybrid_label or os.path.basename(args.hybrid_run.rstrip("/"))
    ho_name = args.host_label or os.path.basename(args.host_run.rstrip("/"))

    print(f"[{case_name}] hybrid {hy_name} ...")
    xyz, true, pred_h = load_run(args.hybrid_run, args.case, dev, args.seed)
    print(f"[{case_name}] host   {ho_name} ...")
    # The host is re-loaded on the SAME case with the SAME seed, so both models
    # see an identical boundary-cloud sample and the rows differ only by model.
    xyz2, true2, pred_b = load_run(args.host_run, args.case, dev, args.seed)
    if xyz.shape != xyz2.shape or not np.allclose(xyz, xyz2):
        raise SystemExit("the two runs disagree on the case point set -- refusing "
                         "to draw rows that are not at identical locations")

    X, Y, extent = display_grid(xyz, args.n)
    os.makedirs(args.out_dir, exist_ok=True)

    for field in args.fields:
        spec = FIELDS[field]
        panels = build_panels(
            xyz, {"hyb": scalar(pred_h, field), "host": scalar(pred_b, field),
                  "truth": scalar(true, field)},
            args.heights, X, Y, slab=args.slab, obstacle_r=args.obstacle_r)

        # scale from the TRUTH row only
        v = np.concatenate([panels[z]["truth"][np.isfinite(panels[z]["truth"])].ravel()
                            for z in args.heights if panels[z]])
        lo = 0.0 if spec["floor_zero"] else float(np.percentile(v, 1))
        hi = float(np.percentile(v, 99))

        ncol = len(args.heights)
        fig, axes = plt.subplots(3, ncol, figsize=(3.0 * ncol, 6.6), squeeze=False,
                                 constrained_layout=True)
        rows = [("hyb", f"{hy_name}\n(host + local branch)"),
                ("host", f"{ho_name}\n(host alone)"),
                ("truth", "CFD TRUTH")]
        levels = np.linspace(lo, hi, args.levels)
        im = None
        for r, (key, label) in enumerate(rows):
            for c, z in enumerate(args.heights):
                ax = axes[r][c]
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                if panels[z] is None:
                    ax.set_facecolor("0.9")
                else:
                    ax.imshow(np.where(panels[z]["obstacle"], 1.0, np.nan),
                              origin="lower", extent=extent, cmap="gray_r",
                              vmin=0, vmax=1.6, interpolation="nearest", zorder=3)
                    im = ax.contourf(X, Y, np.clip(panels[z][key], lo, hi),
                                     levels=levels, cmap=spec["cmap"], extend="both")
                    try:
                        im.set_edgecolor("face")
                    except AttributeError:
                        for coll in im.collections:
                            coll.set_edgecolor("face")
                    # RASTERIZE THE FILL, KEEP EVERYTHING ELSE VECTOR. 32 levels
                    # over 15 panels is ~10-16 MB of vector paths per PDF and
                    # 137 MB for the set -- larger than anything else in the
                    # repo. Titles, axes and the colourbar stay vector, so the
                    # figure is still publication-quality; only the field data
                    # is a raster, at `raster_dpi`.
                    im.set_rasterized(True)
                if r == 0:
                    ax.set_title(f"z = {z:.2f} m", fontsize=11)
            axes[r][0].set_ylabel(label, fontsize=9)
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.020, pad=0.012,
                     label=spec["label"])
        fig.suptitle(
            f"{case_name} (held-out layout)   |   {field} at all {len(xyz):,} interior "
            f"CFD points   |   colour scale from the truth row   |   seed {args.seed}",
            fontsize=12)
        tag = f"_{args.tag}" if args.tag else ""
        out = os.path.join(args.out_dir, f"pair_{field}_{ho_name}_{case_name}{tag}.pdf")
        fig.savefig(out, bbox_inches="tight", dpi=args.raster_dpi)
        fig.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out} (+ .png)")


if __name__ == "__main__":
    main()
