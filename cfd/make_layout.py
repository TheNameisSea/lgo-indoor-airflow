#!/usr/bin/env python
"""Generate the geometry for one FCU layout.

A layout is six rectangles in the ceiling plane. Every other patch -- walls,
floor, furniture, the door leak -- is layout-independent and copied verbatim,
so this rebuilds only `Ceiling.stl` (a plane with six holes) and the six
`HVAC_*.stl` vent patches, then reassembles `room.stl`.

## The parameterisation, read off the reference set

The nine ver1 layouts are a 3x3 grid in two parameters, and nothing else moves:

    supply x   outer at {1.21, 1.96, 2.71}, middle pinned at 4.46,
               third mirrored:  x_outer + x_third = 8.92
    supply y   {2.30, 1.55, 0.80}
    return     mirrored about y = 3.075:  y_supply + y_return = 6.15,
               and shifted -0.04 in x

**The supply array centroid is therefore always x = 4.46.** The reference varies
the array's *spread* but never *translates* it along x. That is precisely the
degeneracy the paper identifies as the reason the model emits the
training-set mean circulation: with every layout sharing an array centre,
"distance to the nearest training vent" is nearly constant for any held-out
case, so the bulk circulation of an unseen layout is not inferable.

So `--dx` exists, and it is the parameter worth sampling: it translates all six
vents together and produces layouts the reference set cannot express.

Run:
    python cfd/make_layout.py --out cfd/geometry_L01 --spread 1.96 --row 1.55
    python cfd/make_layout.py --out cfd/geometry_L02 --spread 1.96 --row 1.55 --dx 0.75
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

# Panel edge lengths measured off the CAD (step_to_stl.py confirms these against
# case_spec.json's 0.588 / 0.597).
SUPPLY_M, RETURN_M = 0.591, 0.600
CEILING_Z = 3.20
X_MIRROR = 8.92        # x_outer + x_third
Y_MIRROR = 6.15        # y_supply + y_return
X_MIDDLE = 4.46
RETURN_DX = -0.04      # returns sit 4 cm left of the supply column
LAYOUT_PATCHES = ("Ceiling", "HVAC_01", "HVAC_02", "HVAC_03",
                  "HVAC_04", "HVAC_05", "HVAC_06")


def read_stl(path):
    """ASCII STL -> (n_tri, 3, 3) float64."""
    v = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("vertex "):
                v.append([float(x) for x in s.split()[1:4]])
    return np.asarray(v, dtype=float).reshape(-1, 3, 3)


def boundary_loops(tri, tol=1e-6):
    """Ordered vertex loops around the free edges of a triangulation."""
    key = {}
    idx = np.empty(tri.shape[:2], dtype=int)
    pts = []
    for i, t in enumerate(tri):
        for j, p in enumerate(t):
            k = tuple(np.round(p / tol).astype(np.int64))
            if k not in key:
                key[k] = len(pts)
                pts.append(p)
            idx[i, j] = key[k]
    pts = np.asarray(pts)

    count, order = {}, {}
    for a, b, c in idx:
        for u, v in ((a, b), (b, c), (c, a)):
            e = (min(u, v), max(u, v))
            count[e] = count.get(e, 0) + 1
            order.setdefault(e, (u, v))
    free = [order[e] for e, n in count.items() if n == 1]

    nxt = {}
    for u, v in free:
        nxt.setdefault(u, []).append(v)
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur = [start], start
        seen.add(start)
        while True:
            cand = [w for w in nxt.get(cur, []) if w not in seen]
            if not cand:
                break
            cur = cand[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(pts[loop])
    return loops


def area_xy(loop):
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def vent_centres(spread, row, dx=0.0, dy=0.0, offsets=None):
    """Six (name, cx, cy, edge, role) from the layout parameters.

    `offsets` is an optional {name: (ox, oy)} of INDEPENDENT per-vent shifts.
    Without it the array keeps the reference's symmetry -- third supply
    mirrored (x1 + x3 = 8.92), returns mirrored about y = 3.075 -- which is a
    property of that particular design, not of the room. Jittering the vents
    individually breaks those constraints and is the only way the dataset ever
    contains an irregular array.
    """
    xs = [spread, X_MIDDLE, X_MIRROR - spread]
    y_sup, y_ret = row, Y_MIRROR - row
    out = []
    # Odd HVAC numbers are returns, even are supplies (step_to_stl measures this
    # from panel size and confirms it against the Diffuser row).
    off = offsets or {}
    for i, x in enumerate(xs):
        for name, cx, cy, e, role in (
                (f"HVAC_0{2*i+2}", x + dx, y_sup + dy, SUPPLY_M, "supply"),
                (f"HVAC_0{2*i+1}", x + RETURN_DX + dx, y_ret + dy,
                 RETURN_M, "return")):
            ox, oy = off.get(name, (0.0, 0.0))
            out.append((name, cx + ox, cy + oy, e, role))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cfd/geometry",
                    help="geometry to take the layout-independent patches from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spread", type=float, default=1.96,
                    help="outer supply x; reference uses 1.21 / 1.96 / 2.71")
    ap.add_argument("--row", type=float, default=1.55,
                    help="supply row y; reference uses 0.80 / 1.55 / 2.30")
    ap.add_argument("--dx", type=float, default=0.0,
                    help="translate the whole array in x -- the axis the "
                         "reference set never varies")
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--offsets", type=float, nargs=12, default=None,
                    metavar="OX OY",
                    help="independent per-vent shifts, HVAC_01..HVAC_06 in "
                         "order as (ox, oy) pairs; breaks the array symmetry")
    ap.add_argument("--size", type=float, default=0.05)
    args = ap.parse_args()

    import gmsh
    os.makedirs(args.out, exist_ok=True)
    base = json.load(open(os.path.join(args.base, "patches.json")))
    offsets = None
    if args.offsets:
        offsets = {f"HVAC_0{i+1}": (args.offsets[2*i], args.offsets[2*i+1])
                   for i in range(6)}
    vents = vent_centres(args.spread, args.row, args.dx, args.dy, offsets)

    # Sanity: every vent must sit inside the ceiling, clear of its edges.
    clo, chi = (np.array(v) for v in base["patches"]["Ceiling"]["bbox_m"])
    for name, cx, cy, e, _ in vents:
        if not (clo[0] + e/2 <= cx <= chi[0] - e/2 and
                clo[1] + e/2 <= cy <= chi[1] - e/2):
            raise SystemExit(f"{name} at ({cx:.2f}, {cy:.2f}) is outside the "
                             f"ceiling {clo[:2]}..{chi[:2]}")

    outer = max(boundary_loops(read_stl(os.path.join(args.base, "Ceiling.stl"))),
                key=area_xy)
    print(f"ceiling outline: {len(outer)} vertices, area {area_xy(outer):.2f} m2")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", args.size / 5)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 2)
    gmsh.model.add("layout")

    def loop_of(poly):
        tags = [gmsh.model.geo.addPoint(x, y, CEILING_Z, args.size)
                for x, y in poly]
        lines = [gmsh.model.geo.addLine(tags[i], tags[(i + 1) % len(tags)])
                 for i in range(len(tags))]
        return gmsh.model.geo.addCurveLoop(lines)

    outer_loop = loop_of([(p[0], p[1]) for p in outer])
    holes, vent_surfs = [], []
    for name, cx, cy, e, role in vents:
        rect = [(cx - e/2, cy - e/2), (cx + e/2, cy - e/2),
                (cx + e/2, cy + e/2), (cx - e/2, cy + e/2)]
        # The SAME loop bounds the hole in the ceiling and the vent patch, so
        # the two are conformal by construction and the result stays watertight.
        lp = loop_of(rect)
        holes.append(lp)
        vent_surfs.append((name, gmsh.model.geo.addPlaneSurface([lp]), role,
                           cx, cy, e))
    ceiling_surf = gmsh.model.geo.addPlaneSurface([outer_loop] + holes)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    manifest = dict(base["patches"])
    written = []
    for name, tag, role, cx, cy, e in vent_surfs:
        gmsh.model.removePhysicalGroups()
        gmsh.model.addPhysicalGroup(2, [tag], name=name)
        gmsh.write(os.path.join(args.out, f"{name}.stl"))
        lo = [cx - e/2, cy - e/2, CEILING_Z]
        hi = [cx + e/2, cy + e/2, CEILING_Z]
        manifest[name] = {"surfaces": 1, "role": role,
                          "bbox_m": [lo, hi], "size_m": [e, e, 0.0]}
        written.append(name)
    gmsh.model.removePhysicalGroups()
    gmsh.model.addPhysicalGroup(2, [ceiling_surf], name="Ceiling")
    gmsh.write(os.path.join(args.out, "Ceiling.stl"))
    gmsh.finalize()

    for name in base["patches"]:
        if name not in LAYOUT_PATCHES:
            shutil.copy(os.path.join(args.base, f"{name}.stl"),
                        os.path.join(args.out, f"{name}.stl"))

    tri_total = 0
    with open(os.path.join(args.out, "room.stl"), "w") as out:
        for name in sorted(manifest):
            p = os.path.join(args.out, f"{name}.stl")
            body = open(p).read()
            tri_total += body.count("facet normal")
            out.write(body)
            manifest[name]["triangles"] = body.count("facet normal")

    meta = dict(base)
    meta["patches"] = manifest
    meta["layout"] = {"spread": args.spread, "row": args.row,
                      "dx": args.dx, "dy": args.dy,
                      "offsets": args.offsets,
                      "vents": {n: [cx, cy] for n, cx, cy, _, _ in vents}}
    with open(os.path.join(args.out, "patches.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"rebuilt {', '.join(sorted(written))} + Ceiling")
    for name, cx, cy, e, role in sorted(vents):
        print(f"  {name:8} {role:7} centre ({cx:5.2f}, {cy:5.2f})  {e:.3f} m")
    print(f"copied {len(manifest) - len(LAYOUT_PATCHES)} layout-independent "
          f"patches\nwrote {args.out} ({tri_total} triangles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())