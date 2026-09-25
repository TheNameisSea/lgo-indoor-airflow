#!/usr/bin/env python
"""Slices of ONE case's CFD data -- no model, no run directory.

`visualize.py` needs a trained run because it plots prediction against truth.
This is the data-only counterpart: what a single exported case actually
contains, for eyeballing shape and sanity before or during training.

Two figures:

  <case>_slices.png    speed / vertical velocity w / temperature, across heights.
                       Supply vents in red, returns in blue, on every panel.
  <case>_section.png   a vertical XZ section through the supply row, where the
                       downward jets and the thermal plume are actually visible.

Loaded through `ThermoDataset` rather than by reading the CSVs directly, so the
supply/return split uses the project's own patch-clustering rule (`pi_ginot`
correction A5 -- classify a PATCH by its net flux, not a point by the sign of its
own w) instead of a second implementation that could drift from it.

Obstacle detection is `visualize.py`'s, unchanged: found in 3D at the exact
height of the slice, closed and opened but NEVER hole-filled.

    PY=/path/to/ml-env/bin/python
    $PY visualize_case.py --case splits_cfd/val/s121_r230_dxp000
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

HEIGHTS = [0.10, 0.60, 1.35, 1.70, 2.80, 3.10]


def build_panels(xyz, fields, heights, n, slab, obstacle_r, smooth, blank):
    """{z: {name: 2D array}} plus the shared grid. Ported from visualize.py."""
    lo, hi = xyz.min(0), xyz.max(0)
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], max(8, int(n * (hi[1] - lo[1]) / (hi[0] - lo[0]))))
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    flat = np.column_stack([X.ravel(), Y.ravel()])
    extent = (xs[0], xs[-1], ys[0], ys[-1])
    cell = max(xs[1] - xs[0], ys[1] - ys[0])
    tree3d = cKDTree(xyz)

    panels = {}
    for z in heights:
        m = np.abs(xyz[:, 2] - z) < slab
        if m.sum() < 50:
            panels[z] = None
            print(f"  z={z:.2f} m: only {m.sum()} points in slab -- skipped")
            continue
        src = xyz[m][:, :2]
        d_fluid, _ = cKDTree(src).query(flat, k=1, workers=-1)
        d_fluid = d_fluid.reshape(X.shape)

        # Obstacles in 3D at the exact height -- see visualize.py for why the two
        # cheaper tests (2D projection, "is solid nearer than fluid") both fail.
        d3, _ = tree3d.query(np.column_stack([flat, np.full(len(flat), z)]),
                             k=1, workers=-1)
        obstacle = d3.reshape(X.shape) > obstacle_r
        if smooth:
            st = np.ones((3, 3), bool)
            obstacle = ndimage.binary_closing(obstacle, st, iterations=smooth)
            obstacle = ndimage.binary_opening(obstacle, st, iterations=1)
            # NO binary_fill_holes -- it swallows the air inside the U-shaped desks.
        masked = obstacle | (d_fluid > max(blank, 2.0 * cell))

        panels[z] = {k: np.where(masked, np.nan,
                                 griddata(src, v[m], (X, Y), method="linear"))
                     for k, v in fields.items()}
        # Grey out everything that is NOT fluid, not just what the 3D test calls
        # obstacle. A thin desk leg fails that test -- the nearest interior point
        # is within the radius, through the leg -- but still has no fluid in the
        # slab, so it lands in `nodata`. Drawing only `obstacle` grey left those
        # as unexplained white holes.
        panels[z]["masked"] = masked
        print(f"  z={z:.2f} m: {m.sum():>8,} points in slab, "
              f"{obstacle.mean():.0%} obstacle, {masked.mean():.0%} non-fluid")
    return panels, X, Y, extent


def patch_boxes(xyz, link_m=0.05):
    """Axis-aligned (x, y) bounding box per vent panel.

    Drawn as an outline rather than as scattered points: a vent carries ~2,300
    points, and at any visible marker size they merge into an opaque block that
    on a diverging red/blue colourmap is indistinguishable from field data. The
    first version of this figure showed exactly that and read as strong +w over
    every supply vent, which is the opposite of the truth (measured -0.571 m/s).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    if len(xyz) < 2:
        return []
    lab = fcluster(linkage(xyz, method="single"), t=link_m, criterion="distance")
    out = []
    for k in np.unique(lab):
        c = xyz[lab == k]
        out.append((c[:, 0].min(), c[:, 1].min(),
                    np.ptp(c[:, 0]), np.ptp(c[:, 1]), c[:, 2].mean()))
    return out


def draw_vents(ax, sup_boxes, ret_boxes):
    """Supply solid, return dashed, black on a white halo so it survives any map."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle
    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    for boxes, ls in ((sup_boxes, "-"), (ret_boxes, (0, (3, 2)))):
        for x0, y0, dx, dy, _ in boxes:
            r = Rectangle((x0, y0), dx, dy, fill=False, ec="black", lw=1.2,
                          ls=ls, zorder=5)
            r.set_path_effects(halo)
            ax.add_patch(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="one case directory")
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "results", "figs"))
    ap.add_argument("--heights", type=float, nargs="+", default=HEIGHTS)
    ap.add_argument("--n", type=int, default=260, help="grid columns")
    ap.add_argument("--slab", type=float, default=0.05, help="half-thickness [m]")
    ap.add_argument("--obstacle_r", type=float, default=0.10)
    ap.add_argument("--obstacle_smooth", type=int, default=1)
    ap.add_argument("--blank", type=float, default=None)
    ap.add_argument("--levels", type=int, default=24)
    args = ap.parse_args()

    case = args.case.rstrip("/")
    name = os.path.basename(case)
    os.makedirs(args.out_dir, exist_ok=True)

    from thermo.dataset import ThermoDataset
    stats = os.path.join(args.out_dir, f"_{name}_stats.pt")
    ds = ThermoDataset(case_dirs=[case], saved_stats=None, stats_path=stats,
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    room = ds.cases[0]

    def metres(key):
        return (room[key] * ds.coord_scale + ds.coord_min).numpy()

    xyz_n = np.concatenate([room["pool_stream_xyz"].numpy(),
                            room["pool_bg_xyz"].numpy()], 0)
    tgt_n = np.concatenate([room["pool_stream_tgt"].numpy(),
                            room["pool_bg_tgt"].numpy()], 0)
    xyz = xyz_n * ds.coord_scale.numpy() + ds.coord_min.numpy()
    tgt = tgt_n * ds.target_std.numpy() + ds.target_mean.numpy()   # [u,v,w,p,T]

    supply, ret = metres("pool_in_xyz"), metres("pool_out_xyz")
    solid = metres("pool_wall_xyz")

    u, v, w, T = tgt[:, 0], tgt[:, 1], tgt[:, 2], tgt[:, 4]
    speed = np.linalg.norm(tgt[:, 0:3], axis=1)

    print(f"\n=== {name} ===")
    print(f"  interior points   {len(xyz):>10,}   "
          f"targets {tgt.shape} = [u, v, w, p, T]")
    print(f"  boundary cloud    {len(solid):>10,} solid, {len(supply):,} supply, "
          f"{len(ret):,} return")
    print(f"  room extent       x {xyz[:,0].min():.2f}-{xyz[:,0].max():.2f} m, "
          f"y {xyz[:,1].min():.2f}-{xyz[:,1].max():.2f} m, "
          f"z {xyz[:,2].min():.2f}-{xyz[:,2].max():.2f} m")
    print(f"  speed             mean {speed.mean():.4f}, median "
          f"{np.median(speed):.4f}, p99 {np.percentile(speed, 99):.3f} m/s")
    print(f"  bulk circulation  u {u.mean():+.4f}  v {v.mean():+.4f}  "
          f"w {w.mean():+.4f} m/s")
    print(f"  temperature       {T.min():.2f} - {T.max():.2f} K, mean "
          f"{T.mean():.3f}, std {T.std():.3f} K")
    print(f"  supply vents      z {supply[:,2].min():.2f}-{supply[:,2].max():.2f} m, "
          f"x {supply[:,0].min():.2f}-{supply[:,0].max():.2f} m, "
          f"y {supply[:,1].min():.2f}-{supply[:,1].max():.2f} m")
    print(f"\nbuilding {len(args.heights)} horizontal slices ...")

    sup_boxes, ret_boxes = patch_boxes(supply), patch_boxes(ret)
    print(f"  vent panels       {len(sup_boxes)} supply, {len(ret_boxes)} return")
    for x0, y0, dx, dy, z0 in sorted(sup_boxes):
        print(f"    supply  x {x0:.2f}+{dx:.2f}  y {y0:.2f}+{dy:.2f}  z {z0:.2f}")
    for x0, y0, dx, dy, z0 in sorted(ret_boxes):
        print(f"    return  x {x0:.2f}+{dx:.2f}  y {y0:.2f}+{dy:.2f}  z {z0:.2f}")

    fields = {"speed": speed, "w": w, "T": T}
    panels, X, Y, extent = build_panels(
        xyz, fields, args.heights, args.n, args.slab, args.obstacle_r,
        args.obstacle_smooth, args.blank if args.blank is not None else args.slab)

    def scale(key, floor_zero=False, symmetric=False):
        vals = np.concatenate([panels[z][key][np.isfinite(panels[z][key])].ravel()
                               for z in args.heights if panels[z]])
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        if symmetric:
            a = max(abs(lo), abs(hi))
            return -a, a
        return (0.0 if floor_zero else float(lo)), float(hi)

    rows = [("speed", "turbo", *scale("speed", floor_zero=True), "speed |U|  [m/s]"),
            ("w", "RdBu_r", *scale("w", symmetric=True), "vertical w  [m/s]\n(blue = down)"),
            ("T", "coolwarm", *scale("T"), "temperature  [K]")]

    ncol = len(args.heights)
    fig, axes = plt.subplots(3, ncol, figsize=(2.9 * ncol, 7.4), squeeze=False,
                             constrained_layout=True)
    ims = {}
    for r, (key, cmap, vlo, vhi, label) in enumerate(rows):
        levels = np.linspace(vlo, vhi, args.levels)
        for c, z in enumerate(args.heights):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            if panels[z] is None:
                ax.set_facecolor("0.9")
            else:
                ax.imshow(np.where(panels[z]["masked"], 1.0, np.nan),
                          origin="lower", extent=extent, cmap="gray_r",
                          vmin=0, vmax=1.6, interpolation="nearest", zorder=3)
                im = ax.contourf(X, Y, np.clip(panels[z][key], vlo, vhi),
                                 levels=levels, cmap=cmap, extend="both")
                try:
                    im.set_edgecolor("face")
                except AttributeError:
                    for coll in im.collections:
                        coll.set_edgecolor("face")
                ims[key] = im
                # The vents are the only thing that varies across this dataset,
                # so put them on every panel rather than in a separate plan view.
                draw_vents(ax, sup_boxes, ret_boxes)
            if r == 0:
                ax.set_title(f"z = {z:.2f} m", fontsize=11)
        axes[r][0].set_ylabel(label, fontsize=10)
        fig.colorbar(ims[key], ax=axes[r, :].ravel().tolist(),
                     location="right", shrink=0.9, pad=0.01)

    fig.suptitle(f"{name} — CFD data, {len(xyz):,} interior points   "
                 f"(solid outline = supply vent, dashed = return, "
                 f"grey = obstacle or no fluid)", fontsize=12)
    p1 = os.path.join(args.out_dir, f"{name}_slices.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    print(f"saved {p1}")

    # ---- vertical section through the supply row -------------------------
    y0 = float(np.median(supply[:, 1]))
    m = np.abs(xyz[:, 1] - y0) < args.slab * 3
    print(f"\nvertical section at y = {y0:.2f} m: {m.sum():,} points")
    xs = np.linspace(xyz[:, 0].min(), xyz[:, 0].max(), args.n)
    zs = np.linspace(xyz[:, 2].min(), xyz[:, 2].max(), max(8, args.n // 3))
    XS, ZS = np.meshgrid(xs, zs, indexing="xy")
    src = xyz[m][:, [0, 2]]
    d, _ = cKDTree(src).query(np.column_stack([XS.ravel(), ZS.ravel()]),
                              k=1, workers=-1)
    hole = d.reshape(XS.shape) > max(args.slab * 3, 2.0 * (xs[1] - xs[0]))
    gs = lambda vv: np.where(hole, np.nan,
                             griddata(src, vv[m], (XS, ZS), method="linear"))

    fig2, ax2 = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    sp = gs(speed)
    c0 = ax2[0].contourf(XS, ZS, sp, levels=np.linspace(0, np.nanpercentile(sp, 99), args.levels),
                         cmap="turbo", extend="max")
    ax2[0].set_ylabel("z [m]"); ax2[0].set_title(
        f"{name} — vertical section at y = {y0:.2f} m (through the supply row)")
    fig2.colorbar(c0, ax=ax2[0], label="speed |U| [m/s]")
    tt = gs(T)
    c1 = ax2[1].contourf(XS, ZS, tt, levels=np.linspace(np.nanpercentile(tt, 1),
                                                        np.nanpercentile(tt, 99), args.levels),
                         cmap="coolwarm", extend="both")
    ax2[1].set_xlabel("x [m]"); ax2[1].set_ylabel("z [m]")
    fig2.colorbar(c1, ax=ax2[1], label="temperature [K]")
    # Same reason as the plan panels: mark the vents with outlined glyphs, not
    # filled colour that competes with the field.
    import matplotlib.patheffects as pe
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    for a in ax2:
        a.set_aspect("equal")
        for boxes, mk, lab in ((sup_boxes, "v", "supply (blows down)"),
                               (ret_boxes, "^", "return (draws up)")):
            xs_b = [b[0] + b[2] / 2 for b in boxes if abs(b[1] + b[3] / 2 - y0) < 0.6]
            zs_b = [b[4] for b in boxes if abs(b[1] + b[3] / 2 - y0) < 0.6]
            if xs_b:
                s = a.scatter(xs_b, zs_b, s=70, marker=mk, facecolor="white",
                              edgecolor="black", linewidths=1.4, zorder=6,
                              label=lab)
                s.set_path_effects(halo)
    ax2[0].legend(loc="lower right", fontsize=8)
    p2 = os.path.join(args.out_dir, f"{name}_section.png")
    fig2.savefig(p2, dpi=130)
    plt.close(fig2)
    print(f"saved {p2}")


if __name__ == "__main__":
    main()
