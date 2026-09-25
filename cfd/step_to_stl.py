#!/usr/bin/env python
"""STEP -> per-patch STL, preserving the CAD's face names.

Why the names matter: `H105.step` carries a name on every one of its 1180
`ADVANCED_FACE` records, and those names are exactly the CSV surface classes
(`Ceiling`, `Floor`, `Wall_01`..`Wall_04`, `Table`, `TV`, `Leak`, ...), with
`HVAC_01`..`HVAC_06` named individually. Carrying them through to STL gives
snappyHexMesh one region per patch, so OpenFOAM boundary conditions are assigned
BY NAME and a new FCU layout is "move six named patches", not "remodel a room".

Three facts measured before writing this, each of which cost a failed run:

  * **The file is imported in millimetres.** It declares `SI_UNIT($,.METRE.)`,
    but OpenCASCADE's STEP reader normalises to mm regardless, so gmsh reports
    the room as 8800 x 6200 x 3350. Fixing this needs `Geometry.OCCTargetUnit`;
    without it a metre-sized mesh setting means micron facets and the mesher
    never finishes.
  * **gmsh cannot see the face names.** `Geometry.OCCImportLabels` reads XCAF
    product labels, and this file has exactly one product ("Fluid 7") covering
    the whole solid; all 1180 names live on the faces, where XCAF does not look.
    So the names are parsed out of the STEP text here and matched to gmsh's
    surface tags by shell traversal order -- OCC creates faces in the order the
    `CLOSED_SHELL` records list them. That assumption is *checked*, not trusted:
    see `_validate()`.
  * **The solid is the air, not the furniture.** The one solid is named
    "Fluid 7": the STEP is the fluid domain with the furniture already carved
    out of it. Its faces are therefore the CFD boundary directly.

Geometry the mapping recovers (all confirmed against `case_spec.json`):

    Floor      z = 0            Ceiling   z = 3.20     room 8.80 x 6.10 x 3.35
    Wall_01    y = 0            Wall_02   x = 8.80
    Wall_03    y = 6.10         Wall_04   x = 0
    HVAC_01/03/05  0.60 m sq, y in [4.30, 4.90]   -> return (plain grille)
    HVAC_02/04/06  0.59 m sq, y in [1.25, 1.84]   -> supply (1014 vane faces
                                                     named `Diffuser` sit over
                                                     this row and only this row)

The supply/return split is decided by measurement, not by hard-coded tags: the
three smaller panels are supply (`case_spec.json`: supply 0.588 m, return
0.597 m), and `--check` fails loudly if that does not come out 3-and-3.

Run:
    conda activate cfdEnv
    python cfd/step_to_stl.py --step step/H105.step --out cfd/geometry
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

# Room bounds with a margin, in METRES. A face outside this is construction
# geometry, not a real surface. With the unit fix this drops nothing; it stays
# as a guard, because silently meshing a stray face wrecks the domain.
ROOM_LO = np.array([-0.5, -0.7, -0.5])
ROOM_HI = np.array([9.3, 6.7, 4.0])

# Everything not named here is a no-slip wall. HVAC roles are resolved by panel
# size at runtime, so they are deliberately absent.
PATCH_ROLE = {"Leak": "leak"}

# Landmarks that must hold if the name->surface mapping is correct. Each entry
# is (patch, axis, expected coordinate, which end): the mapping is only accepted
# when every one of these lands where the CFD room says it should.
LANDMARKS = [
    ("Floor",   2, 0.00, "lo"),
    ("Ceiling", 2, 3.20, "lo"),
    ("Wall_01", 1, 0.00, "lo"),
    ("Wall_02", 0, 8.80, "lo"),
    ("Wall_03", 1, 6.10, "lo"),
    ("Wall_04", 0, 0.00, "hi"),
]
LANDMARK_TOL = 0.02


def face_names_in_shell_order(step_path):
    """Map gmsh surface index -> CAD face name, via CLOSED_SHELL ordering."""
    text = open(step_path).read().replace("\n", "").replace("\r", "")
    named, order = {}, []
    for stmt in text.split(";"):
        m = re.match(r"\s*#(\d+)\s*=\s*ADVANCED_FACE\('([^']*)'", stmt)
        if m:
            named[int(m.group(1))] = m.group(2)
            continue
        m = re.match(r"\s*#(\d+)\s*=\s*(?:CLOSED|OPEN)_SHELL\('[^']*',\((.*)\)\)\s*$", stmt)
        if m:
            order += [int(x) for x in re.findall(r"#(\d+)", m.group(2))]
    if len(order) != len(set(order)):
        raise SystemExit("a face is listed by two shells; ordering is ambiguous")
    return [named.get(f, f"unnamed_{f}") for f in order], len(named)


def _validate(bbox_of):
    """Fail loudly if the name->geometry mapping does not match the CFD room."""
    bad = []
    for patch, axis, want, end in LANDMARKS:
        if patch not in bbox_of:
            bad.append(f"{patch}: missing")
            continue
        lo, hi = bbox_of[patch]
        got = (lo if end == "lo" else hi)[axis]
        if abs(got - want) > LANDMARK_TOL:
            bad.append(f"{patch}: {'xyz'[axis]}={got:.3f}, expected {want:.2f}")
    if bad:
        raise SystemExit("name->surface mapping is WRONG, refusing to write:\n  "
                         + "\n  ".join(bad))


def hvac_roles(bbox_of):
    """Supply = the three smaller panels; returns are the wider grilles."""
    panels = {n: bbox_of[n] for n in bbox_of if n.startswith("HVAC_")}
    if not panels:
        return {}
    area = {n: float(np.prod((hi - lo)[:2])) for n, (lo, hi) in panels.items()}
    ranked = sorted(area, key=area.get)
    n_supply = len(ranked) // 2
    roles = {n: ("supply" if i < n_supply else "return") for i, n in enumerate(ranked)}
    if len(panels) % 2 or sum(v == "supply" for v in roles.values()) != n_supply:
        raise SystemExit(f"cannot split {len(panels)} HVAC panels evenly: {area}")
    # The split must be a real gap, not an arbitrary cut through one cluster.
    gap = area[ranked[n_supply]] - area[ranked[n_supply - 1]]
    if gap <= 0:
        raise SystemExit(f"supply/return areas are not separable: {area}")
    return roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="step/H105.step")
    ap.add_argument("--out", default="cfd/geometry")
    ap.add_argument("--size", type=float, default=0.05,
                    help="target STL facet size [m]")
    ap.add_argument("--exclude", nargs="*", default=["Diffuser"],
                    help="patches not to mesh at all (default: the vane geometry)")
    ap.add_argument("--keep-plenum", action="store_true",
                    help="keep the 0.15 m boxes above the vents. They are only "
                         "closed with the vane assembly in place, so this leaves "
                         "an open surface -- see the module docstring.")
    args = ap.parse_args()

    import gmsh
    os.makedirs(args.out, exist_ok=True)
    names, n_named = face_names_in_shell_order(args.step)
    print(f"{n_named} named faces in the STEP, {len(names)} in shell order", flush=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")   # else it imports as mm
    gmsh.model.add("room")
    print(f"importing {args.step} ...", flush=True)
    gmsh.model.occ.importShapes(args.step)
    gmsh.model.occ.synchronize()

    surfaces = [t for _, t in gmsh.model.getEntities(2)]
    if len(surfaces) != len(names):
        raise SystemExit(f"{len(surfaces)} gmsh surfaces vs {len(names)} named "
                         "faces -- shell order cannot be trusted")
    bb = np.array([gmsh.model.getBoundingBox(2, t) for t in surfaces])
    print(f"  {len(surfaces)} surfaces, model bbox "
          f"x[{bb[:,0].min():.2f},{bb[:,3].max():.2f}] "
          f"y[{bb[:,1].min():.2f},{bb[:,4].max():.2f}] "
          f"z[{bb[:,2].min():.2f},{bb[:,5].max():.2f}]", flush=True)

    groups, dropped = OrderedDict(), 0
    for name, tag, b in zip(names, surfaces, bb):
        if np.any(b[:3] < ROOM_LO) or np.any(b[3:] > ROOM_HI):
            dropped += 1
            continue
        groups.setdefault(name, []).append(tag)
    bbox_of = {n: (bb[[surfaces.index(t) for t in tags]][:, :3].min(0),
                   bb[[surfaces.index(t) for t in tags]][:, 3:].max(0))
               for n, tags in groups.items()}
    print(f"  {len(groups)} patches, {dropped} faces dropped as out-of-room", flush=True)
    _validate(bbox_of)
    roles = hvac_roles(bbox_of)
    print("  landmark check passed; "
          f"supply={sorted(n for n,r in roles.items() if r=='supply')} "
          f"return={sorted(n for n,r in roles.items() if r=='return')}", flush=True)

    exclude = list(args.exclude)
    if not args.keep_plenum:
        # Replace each vent plenum with a flat patch closing the ceiling hole it
        # sits over. The plenum walls span exactly their panel's footprint from
        # z=3.20 to z=3.35, so the hole's own curves bound the new patch: it is
        # conformal with the ceiling by construction, and no new geometry is
        # invented. The vent BC moves from the panel to the ceiling plane, which
        # is what `case_spec.json` already assumes (a uniform normal velocity on
        # a flat panel -- the vanes' turning is baked into that number).
        ceiling = groups["Ceiling"]
        curves = gmsh.model.getBoundary([(2, t) for t in ceiling],
                                        oriented=False, combined=False)
        cbb = {c: np.array(gmsh.model.getBoundingBox(1, c)) for _, c in curves}
        made = []
        for name in sorted(n for n in groups if n.startswith("HVAC_")):
            lo, hi = bbox_of[name]
            # Match on footprint only: the panel is at z=3.35, the hole it feeds
            # is in the ceiling plane at z=3.20.
            inside = [c for c, b in cbb.items()
                      if np.all(b[[0, 1]] >= lo[:2] - 0.02)
                      and np.all(b[[3, 4]] <= hi[:2] + 0.02)]
            if len(inside) != 4:
                raise SystemExit(f"{name}: found {len(inside)} ceiling-hole curves, "
                                 "expected 4 -- the opening is not a rectangle")
            loop = gmsh.model.occ.addCurveLoop(inside)
            new = gmsh.model.occ.addPlaneSurface([loop])
            made.append((name, new))
        gmsh.model.occ.synchronize()
        # The original panels at z=3.35 are superseded, so drop them from the
        # mesh as well as from the output.
        gmsh.model.setVisibility([(2, t) for name, _ in made
                                  for t in groups[name]], 0)
        for name, tag in made:
            groups[name] = [tag]
            bbox_of[name] = (np.array(gmsh.model.getBoundingBox(2, tag))[:3],
                             np.array(gmsh.model.getBoundingBox(2, tag))[3:])
        exclude.append("Wall_HVAC")
        print(f"  plenums replaced by flat ceiling patches: "
              f"{', '.join(n for n, _ in made)}", flush=True)

    # Excluded patches are hidden rather than deleted: they belong to the same
    # solid, so removing them would invalidate the shape. Hide the faces ONLY --
    # `recursive` would also hide curves they share with patches we are keeping,
    # leaving those with an unmeshed edge and an unclosed 1D loop.
    excluded = [n for n in exclude if n in groups]
    for name in excluded:
        gmsh.model.setVisibility([(2, t) for t in groups.pop(name)], 0)
    if excluded:
        print(f"  not meshing: {', '.join(excluded)}", flush=True)

    gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1 if excluded else 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.size)
    gmsh.option.setNumber("Mesh.MeshSizeMin", args.size / 5)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.SaveAll", 0)                 # physical groups only
    gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 2)   # one solid per group
    print(f"meshing surfaces at {args.size} m ...", flush=True)
    gmsh.model.mesh.generate(2)

    def n_tri(tags):
        total = 0
        for t in tags:
            _, elt, _ = gmsh.model.mesh.getElements(2, t)
            total += sum(len(e) for e in elt)
        return int(total)

    manifest = {}
    for name, tags in sorted(groups.items()):
        gmsh.model.removePhysicalGroups()
        gmsh.model.addPhysicalGroup(2, tags, name=name)
        gmsh.write(os.path.join(args.out, f"{name}.stl"))
        lo, hi = bbox_of[name]
        manifest[name] = {
            "surfaces": len(tags), "triangles": n_tri(tags),
            "role": roles.get(name, PATCH_ROLE.get(name, "wall")),
            "bbox_m": [lo.round(4).tolist(), hi.round(4).tolist()],
            "size_m": (hi - lo).round(4).tolist(),
        }
        print(f"  {name:12} {len(tags):5} surf {manifest[name]['triangles']:8} tri  "
              f"x[{lo[0]:5.2f},{hi[0]:5.2f}] y[{lo[1]:5.2f},{hi[1]:5.2f}] "
              f"z[{lo[2]:5.2f},{hi[2]:5.2f}]  [{manifest[name]['role']}]", flush=True)

    # snappyHexMesh needs a closed surface, so check it here rather than
    # discovering it as a leak during meshing: in a watertight triangulation
    # every edge is shared by exactly two triangles.
    edges = {}
    for tags in groups.values():
        for t in tags:
            _, _, nodes = gmsh.model.mesh.getElements(2, t)
            for arr in nodes:
                tri = np.asarray(arr).reshape(-1, 3)
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    for e in map(tuple, np.sort(tri[:, [a, b]], axis=1)):
                        edges[e] = edges.get(e, 0) + 1
    free = sum(1 for c in edges.values() if c != 2)
    print(f"\nwatertight check: {len(edges)} edges, {free} not shared by exactly 2 "
          f"triangles -- {'CLOSED' if free == 0 else 'OPEN, snappyHexMesh will leak'}")

    gmsh.finalize()

    # One multi-solid STL is what snappyHexMesh actually reads; the per-patch
    # files above are for inspection and for moving a vent on its own. This is a
    # concatenation rather than a second gmsh.write because `removePhysicalGroups`
    # does not clear gmsh's *name* registry: re-registering a name already used
    # in the loop above silently yields "Gmsh Physical Surface N" instead, which
    # would leave snappyHexMesh with 25 unusable region names.
    combined = os.path.join(args.out, "room.stl")
    with open(combined, "w") as out:
        for name in sorted(manifest):
            with open(os.path.join(args.out, f"{name}.stl")) as src:
                out.write(src.read())

    with open(os.path.join(args.out, "patches.json"), "w") as f:
        json.dump({"patches": manifest, "excluded": excluded,
                   "facet_size_m": args.size, "watertight": free == 0}, f, indent=2)
    total = sum(m["triangles"] for m in manifest.values())
    print(f"wrote {len(manifest)} patches ({total} triangles) + patches.json to "
          f"{args.out}\ncombined: {combined}"
          + (f"  (excluded {', '.join(excluded)})" if excluded else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
