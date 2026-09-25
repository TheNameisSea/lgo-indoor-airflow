#!/usr/bin/env python
"""Slices of a run's prediction against CFD truth, at the real CFD points.

The model is evaluated at EVERY interior point of the case (~1.16 M), not on a
synthetic grid. Truth and prediction therefore live at identical locations, and
each display slab is built from the same point set by the same interpolation --
so any smoothing artefact hits both rows equally and cannot flatter the model.

Layout: 5 heights across, four rows --

    velocity prediction / velocity truth / temperature prediction / temperature truth

ONE COLOUR SCALE PER FIELD, shared by every panel of that field in the figure,
so panels are comparable across heights as well as between prediction and truth.
The scale is taken from the TRUTH, so a model cannot look better by predicting a
different range.

    0.60 m   ankle / seated lower body
    1.35 m   seated head height
    1.70 m   standing head height
    2.80 m   upper room
    3.10 m   right under the FCUs, where the supply jets enter

TRUTH IS INTERPOLATED ONLY BECAUSE THIS IS A PICTURE. The prior project's trap
T10: interpolating truth across a slab flattens steep gradients and moved a
reported R2 by 1.4. No metric is computed here -- evaluate.py scores at the CFD
points.

    python visualize.py --run runs/lgo_gqlocal_lr3e-3_s0
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
from scipy.interpolate import griddata
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

HEIGHTS = [0.60, 1.35, 1.70, 2.80, 3.10]



def display_grid(xyz, n):
    """Common (X, Y) display grid for every panel, in metres."""
    lo, hi = xyz.min(0), xyz.max(0)
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], max(8, int(n * (hi[1] - lo[1]) / (hi[0] - lo[0]))))
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return X, Y, (xs[0], xs[-1], ys[0], ys[-1])


def build_panels(xyz, fields, heights, X, Y, slab=0.15, blank=None,
                 obstacle_r=None, obstacle_smooth=1, verbose=True):
    """Interpolated slabs for each height, with obstacles found in 3D.

    `fields` is {key: per-point array}; the return is {z: {key: 2-D array,
    "obstacle": bool array}} with None where a slab holds too few points.

    SHARED BY `visualize.py` AND `make_pair_figure.py` ON PURPOSE. A second copy
    of the obstacle test is exactly the duplication this project has already
    paid for elsewhere, and the comments below are the record of three attempts
    that produced well-formed, wrong pictures.
    """
    flat = np.column_stack([X.ravel(), Y.ravel()])
    tree3d = cKDTree(xyz)          # every interior point, in 3D
    blank_r = blank if blank is not None else slab
    cell = max(X[0, 1] - X[0, 0], Y[1, 0] - Y[0, 0])
    panels = {}
    for z in heights:
        m = np.abs(xyz[:, 2] - z) < slab
        if m.sum() < 50:
            panels[z] = None
            continue
        src = xyz[m][:, :2]
        d_fluid, _ = cKDTree(src).query(flat, k=1, workers=-1)
        d_fluid = d_fluid.reshape(X.shape)

        # OBSTACLES ARE FOUND IN 3D, AT THE EXACT HEIGHT OF THE SLICE.
        # griddata triangulates the fluid points, and a Delaunay triangle happily
        # spans a table or a cabinet -- so a thick slab does not merely leave a
        # gap there, it fills the gap with an interpolation drawn straight
        # through solid furniture. But the test cannot be done on the slab's 2D
        # projection: two earlier attempts failed that way. Comparing "is solid
        # nearer than fluid" found 0% obstacle everywhere, because the CFD mesh
        # is boundary-refined and fluid hugs every surface. Testing proximity to
        # projected solid geometry then called 98% of the z=3.10 plane obstacle,
        # because the ceiling at 3.20 m lies inside that slab and its shadow
        # covers the whole room -- while a query at 3.10 m is in open air below.
        #
        # Air is the thing that is actually observed, so ask for it directly: a
        # cell is solid where there is NO interior CFD point near (x, y, z) in
        # three dimensions.
        d3, _ = tree3d.query(np.column_stack([flat, np.full(len(flat), z)]),
                             k=1, workers=-1)
        obstacle = d3.reshape(X.shape) > (obstacle_r if obstacle_r is not None else 0.10)
        # Consolidate into patches. The raw test speckles: the mesh threads
        # points between desk legs and through small gaps, so a solid desk comes
        # out as a cloud of blobs. Closing joins them into the object they
        # belong to, and opening drops isolated specks that are really local
        # mesh sparsity rather than geometry.
        if obstacle_smooth:
            st = np.ones((3, 3), bool)
            obstacle = ndimage.binary_closing(obstacle, st, iterations=obstacle_smooth)
            obstacle = ndimage.binary_opening(obstacle, st, iterations=1)
            # NO binary_fill_holes here. It fills any enclosed region, and the
            # desks form a U -- so it swallowed the air inside the U and took the
            # z=1.70 m panel from 25% obstacle to 59%, hiding real field. Closing
            # joins the speckle of a single object; filling invents objects.
        nodata = d_fluid > max(blank_r, 2.0 * cell)
        masked = obstacle | nodata

        g = lambda v: np.where(masked, np.nan, griddata(src, v[m], (X, Y), method="linear"))
        panels[z] = {k: g(v) for k, v in fields.items()}
        panels[z]["obstacle"] = obstacle
        if verbose:
            print(f"  z={z:.2f} m: {m.sum():,} CFD points in slab, "
                  f"{obstacle.mean():.0%} of the plane is obstacle")
    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--case", default="splits/val/Case_07")
    ap.add_argument("--out", default=None)
    ap.add_argument("--heights", type=float, nargs="+", default=HEIGHTS)
    ap.add_argument("--n", type=int, default=300, help="display grid resolution")
    ap.add_argument("--slab", type=float, default=0.15,
                    help="HALF-thickness [m] of the CFD slab behind each panel, "
                         "so the default is a 0.30 m slice. Thicker gives a "
                         "denser point set and a cleaner contour, at the cost of "
                         "averaging over more z: a thin slab through the "
                         "furniture band (1.35/1.70 m) is mostly holes, because "
                         "chairs and tables carve the CFD points away.")
    ap.add_argument("--levels", type=int, default=32)
    ap.add_argument("--blank", type=float, default=None,
                    help="blank the panel where no CFD point lies within this "
                         "distance [m]; defaults to the slab half-thickness, so "
                         "it widens with the slice instead of punching holes a "
                         "thicker slab has already filled")
    ap.add_argument("--obstacle_r", type=float, default=None,
                    help="a grid cell is solid when NO interior CFD point lies "
                         "within this radius [m] of (x,y,z) in 3D; default "
                         "0.10 m. Obstacles are patched and excluded from the "
                         "interpolation, because griddata triangulates the fluid "
                         "points and will otherwise draw contours straight "
                         "through a table.")
    ap.add_argument("--obstacle_smooth", type=int, default=1,
                    help="morphological closing iterations used to turn the "
                         "raw solid test into coherent patches; 0 disables")
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()

    from thermo.dataset import ThermoDataset
    from evaluate import build_from_args

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = json.load(open(os.path.join(args.run, "args.json")))
    # A run that recomputed its own normalisation records stats_path=null, because
    # train_thermo defaults the path inside main() rather than through argparse.
    # The file it wrote lives in the run dir. Same resolution as evaluate.py:319 --
    # without it ThermoDataset is handed None and torch.save raises deep in
    # _build_dataset with no hint that the run is the thing to look at.
    if not a.get("stats_path"):
        local = os.path.join(args.run, "thermo_stats.pt")
        if not os.path.exists(local):
            raise SystemExit(f"{args.run}: args.json has no stats_path and "
                             f"{local} does not exist -- cannot rebuild the "
                             f"normalisation this model was trained with.")
        a["stats_path"] = local
    ds = ThermoDataset(case_dirs=[os.path.join(ROOT, args.case)],
                       saved_stats=a["stats_path"], stats_path=a["stats_path"],
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    room = ds.cases[0]
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    true = (torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], 0)
            * ds.target_std + ds.target_mean).numpy()
    xyz = (xyz_n * ds.coord_scale + ds.coord_min).numpy()

    model, _ = build_from_args(args.run, ds, dev)
    if hasattr(model, "register_cases"):
        try:
            model.register_cases(ds, verbose=False)
        except TypeError:
            model.register_cases(ds)
    model.eval()

    print(f"predicting at all {len(xyz_n):,} interior CFD points ...")
    with torch.no_grad():
        lat = model.encode_geometry(room["pc_full"].unsqueeze(0).to(dev))
        out = torch.cat([
            model.decode_query(lat, xyz_n[s:s + args.chunk].unsqueeze(0).to(dev)).squeeze(0)
            for s in range(0, len(xyz_n), args.chunk)], 0)
    pred = (out.cpu() * ds.target_std[:5] + ds.target_mean[:5]).numpy()

    speed_t, speed_p = np.linalg.norm(true[:, 0:3], axis=1), np.linalg.norm(pred[:, 0:3], axis=1)
    temp_t, temp_p = true[:, 4], pred[:, 4]

    X, Y, extent = display_grid(xyz, args.n)

    panels = build_panels(xyz, {"vt": speed_t, "vp": speed_p,
                                "tt": temp_t, "tp": temp_p},
                          args.heights, X, Y, slab=args.slab, blank=args.blank,
                          obstacle_r=args.obstacle_r,
                          obstacle_smooth=args.obstacle_smooth)

    def scale(keys, floor_zero):
        v = np.concatenate([panels[z][k][np.isfinite(panels[z][k])].ravel()
                            for z in args.heights if panels[z] for k in keys])
        return (0.0 if floor_zero else float(np.percentile(v, 1)),
                float(np.percentile(v, 99)))

    # One scale per field, from the truth panels only.
    v_lo, v_hi = scale(["vt"], True)
    t_lo, t_hi = scale(["tt"], False)

    ncol = len(args.heights)
    fig, axes = plt.subplots(4, ncol, figsize=(3.0 * ncol, 8.6), squeeze=False,
                             constrained_layout=True)
    rows = [("vp", "turbo", v_lo, v_hi, "PREDICTION\nspeed"),
            ("vt", "turbo", v_lo, v_hi, "CFD TRUTH\nspeed"),
            ("tp", "coolwarm", t_lo, t_hi, "PREDICTION\ntemperature"),
            ("tt", "coolwarm", t_lo, t_hi, "CFD TRUTH\ntemperature")]

    ims = {}
    for r, (key, cmap, vlo, vhi, label) in enumerate(rows):
        levels = np.linspace(vlo, vhi, args.levels)
        for c, z in enumerate(args.heights):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if panels[z] is None:
                ax.set_facecolor("0.9")
            else:
                # Draw the obstacles first, as a solid patch, so what the slice
                # cuts through is explicit instead of being an unexplained hole.
                ax.imshow(np.where(panels[z]["obstacle"], 1.0, np.nan),
                          origin="lower", extent=extent, cmap="gray_r",
                          vmin=0, vmax=1.6, interpolation="nearest", zorder=3)
                im = ax.contourf(X, Y, np.clip(panels[z][key], vlo, vhi),
                                 levels=levels, cmap=cmap, extend="both")
                try:
                    im.set_edgecolor("face")
                except AttributeError:
                    for coll in im.collections:
                        coll.set_edgecolor("face")
                ims[cmap] = im
            if r == 0:
                ax.set_title(f"z = {z:.2f} m", fontsize=11)
        axes[r][0].set_ylabel(label, fontsize=10)

    # One colourbar per field, spanning that field's two rows.
    fig.colorbar(ims["turbo"], ax=axes[0:2, :].ravel().tolist(),
                 fraction=0.020, pad=0.012, label="speed [m/s]")
    fig.colorbar(ims["coolwarm"], ax=axes[2:4, :].ravel().tolist(),
                 fraction=0.020, pad=0.012, label="temperature [K]")

    name = os.path.basename(args.run.rstrip("/"))
    fig.suptitle(f"{name}   |   {os.path.basename(args.case)} (held-out layout)   |   "
                 f"evaluated at all {len(xyz_n):,} interior CFD points   |   "
                 f"one colour scale per field", fontsize=12)
    out = args.out or f"results/slices_{name}_all.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()