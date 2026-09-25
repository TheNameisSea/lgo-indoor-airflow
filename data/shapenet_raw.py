#!/usr/bin/env python
"""Access to the RAW Umetani ShapeNet-Car release — the one with the volume.

WHY THIS EXISTS. The paper: the Zenodo release everyone
links is **surface pressure only**, and its `press_*.npy` sits on a point set it
does not ship coordinates for. The raw release
(`http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip`, 2.03 GB) carries
what our task actually needs, per case:

    press.npy             (3682,)      surface pressure
    quadpress_smpl.vtk    3682 points  the coordinates those pressures sit on
    velo.npy              (88494,)     volume velocity, = 29498 x 3, float64
    hexvelo_smpl.vtk      29498 points the coordinates that velocity sits on
    cd.txt, param1.txt, param2.txt

THE NAMING PROBLEM, AND HOW IT IS SOLVED. The raw release names cases by
ShapeNet hash (`1ba30d64da90ea05283ffcfc40c29975`); the Zenodo release, the
split manifests and `data/splits_shapenet.json` all use numeric ids (`001`..
`798`). Nothing ships the correspondence.

It is recovered EXACTLY, with no geometric tolerance anywhere: Zenodo copied
`press.npy` verbatim, so a case is identified by the **md5 of its pressure
array's bytes**. Verified: all 798 Zenodo arrays have distinct value-hashes
(no ambiguity possible), all 798 map to exactly one raw case, and all 611
watertight cases are covered. The reconciliation closes with nothing left over:

    893 raw directories = 889 real cases + 4 empty shells (no data files at all)
    889 real cases      = 798 kept by Zenodo + 91 dropped as corrupted

An exact byte match is used deliberately rather than nearest-shape matching --
the latter would return a plausible wrong answer on near-duplicate cars, which
is precisely the failure mode `panels()` hit twice.

The mapping is cached to `external/shapenet_car/id_to_raw.json` because building
it reads 798 + 889 arrays. Delete that file to force a rebuild.

BOTH POINT SETS CORRESPOND ACROSS CASES. Surface 16.5x, volume 20.5x
(corresponded vs randomly permuted mean distance), so index i means the same
place on every car. The volume mesh is body-fitted -- 0.0% of volume points are
identical between two cases -- so correspondence is by construction, not by a
shared grid.
"""

import glob
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "external", "shapenet_car")
RAW = os.path.join(BASE, "raw")
ZENODO = os.path.join(BASE, "processed-car-pressure-data")
MAP = os.path.join(BASE, "id_to_raw.json")

N_SURFACE = 3682
N_VOLUME = 29498
REQUIRED = ("press.npy", "velo.npy", "quadpress_smpl.vtk", "hexvelo_smpl.vtk")


def build_mapping(verbose=True):
    """{numeric id: raw case dir, relative to the repo root}. Exact, by md5."""
    by_hash = {}
    for f in sorted(glob.glob(os.path.join(ZENODO, "data", "press_*.npy"))):
        nid = os.path.basename(f)[len("press_"):-len(".npy")]
        by_hash[hashlib.md5(np.load(f).tobytes()).hexdigest()] = nid
    if len(by_hash) != 798:
        raise SystemExit(f"[shapenet_raw] expected 798 distinct pressure arrays, "
                         f"got {len(by_hash)} -- a duplicate would make the "
                         f"mapping ambiguous; do not proceed")

    mapping, unmapped, incomplete = {}, [], []
    for c in sorted(d for d in glob.glob(os.path.join(RAW, "param*", "*"))
                    if os.path.isdir(d)):
        if any(not os.path.exists(os.path.join(c, f)) for f in REQUIRED):
            incomplete.append(c)
            continue
        h = hashlib.md5(np.load(os.path.join(c, "press.npy")).tobytes()).hexdigest()
        rel = os.path.relpath(c, ROOT)
        if h in by_hash:
            mapping[by_hash[h]] = rel
        else:
            unmapped.append(rel)
    if verbose:
        print(f"[shapenet_raw] mapped {len(mapping)}/798 ids; "
              f"{len(unmapped)} raw cases dropped by Zenodo as corrupted; "
              f"{len(incomplete)} empty directories")
    if len(mapping) != 798:
        raise SystemExit(f"[shapenet_raw] only {len(mapping)}/798 ids mapped")
    return mapping


def mapping(verbose=False):
    """Cached {numeric id: raw case dir}."""
    if os.path.exists(MAP):
        return json.load(open(MAP))
    m = build_mapping(verbose=True)
    json.dump(m, open(MAP, "w"), indent=0, sort_keys=True)
    if verbose:
        print(f"[shapenet_raw] wrote {MAP}")
    return m


MAP_ALL = os.path.join(BASE, "id_to_raw_all.json")


def mapping_all(verbose=False):
    """All 889 real cases: the 798 Zenodo ids PLUS the 91 it drops as corrupted.

    WHY THIS EXISTS. The published Shape-Net Car table is computed on **789
    train / 100 test over all 889** -- not on the
    Zenodo manifest, which is the curators' own addition over 611 watertight
    cases. To reproduce that REGIME (not their exact assignment, which uses an
    unpublished seed) we need identifiers for the 91.

    They get `u001`..`u091` -- "u" for unlisted -- assigned in sorted raw-path
    order, so the numbering is deterministic and reproducible. The prefix is
    deliberately not numeric: an id that cannot be confused with a Zenodo id is
    the point, because these 91 carry a caveat the others do not.

    ⚠ THE 91 ARE FLAGGED CORRUPTED BY ZENODO AND WE CANNOT REPRODUCE THAT.
    Audited 2026-09-21: all 91 have the four data files, exact (3682,) / (88494,)
    shapes, finite values, sane speed and pressure ranges, 3,682-point surface
    topology and bounding boxes within 2x of a kept case -- **0 of 91 fail any
    check we can apply**. That is NOT a claim they are sound; Zenodo's curators
    may have used a criterion we have not tested. Disclose it wherever the 889
    population is used.
    """
    m = dict(mapping(verbose=verbose))
    known = {os.path.abspath(os.path.join(ROOT, v)) for v in m.values()}
    extra = []
    for c in sorted(glob.glob(os.path.join(RAW, "param*", "*"))):
        if not os.path.isdir(c) or os.path.abspath(c) in known:
            continue
        if any(not os.path.exists(os.path.join(c, f)) for f in REQUIRED):
            continue                      # the 4 empty shells
        extra.append(os.path.relpath(c, ROOT))
    for i, rel in enumerate(extra, 1):
        m[f"u{i:03d}"] = rel
    if verbose:
        print(f"[shapenet_raw] mapping_all: {len(m)} cases "
              f"({len(m) - len(extra)} Zenodo + {len(extra)} unlisted)")
    if len(m) != 889:
        raise SystemExit(f"[shapenet_raw] mapping_all has {len(m)}, expected 889")
    if not os.path.exists(MAP_ALL):
        json.dump(m, open(MAP_ALL, "w"), indent=0, sort_keys=True)
    return m


def case_dir(nid, m=None):
    m = m or mapping()
    if nid not in m:
        raise KeyError(f"case {nid!r} is not in the mapping")
    return os.path.join(ROOT, m[nid])


# --------------------------------------------------------------------------- #
def vtk_points(path):
    """(n, 3) float64 from a legacy ASCII VTK UNSTRUCTURED_GRID POINTS block."""
    with open(path) as f:
        toks = f.read().split()
    i = toks.index("POINTS")
    n = int(toks[i + 1])
    return np.array(toks[i + 3: i + 3 + 3 * n], dtype=np.float64).reshape(n, 3)


def surface(nid, m=None):
    """(xyz (3682,3), pressure (3682,)) -- the BOUNDARY, our model's input."""
    d = case_dir(nid, m)
    xyz = vtk_points(os.path.join(d, "quadpress_smpl.vtk"))
    p = np.load(os.path.join(d, "press.npy"))
    if xyz.shape[0] != N_SURFACE or p.shape[0] != N_SURFACE:
        raise ValueError(f"{nid}: surface shapes {xyz.shape} / {p.shape}")
    return xyz, p


def volume(nid, m=None):
    """(xyz (29498,3), velocity (29498,3)) -- the FIELD, our model's target."""
    d = case_dir(nid, m)
    xyz = vtk_points(os.path.join(d, "hexvelo_smpl.vtk"))
    v = np.load(os.path.join(d, "velo.npy")).reshape(-1, 3)
    if xyz.shape[0] != N_VOLUME or v.shape[0] != N_VOLUME:
        raise ValueError(f"{nid}: volume shapes {xyz.shape} / {v.shape}")
    return xyz, v


def velocity(nid, m=None):
    """(29498, 3) velocity only -- skips the slow ASCII VTK parse."""
    v = np.load(os.path.join(case_dir(nid, m), "velo.npy")).reshape(-1, 3)
    if v.shape[0] != N_VOLUME:
        raise ValueError(f"{nid}: velocity shape {v.shape}")
    return v


# --------------------------------------------------------------------------- #
# BANDS. The single definition, imported by BOTH evaluate_shapenet.py and
# evaluate_nulls_shapenet.py. If the two ever banded differently every
# model-vs-null comparison on this benchmark would be void, so there is one
# function and no thresholds written anywhere else.
#
# Distance to the car surface, in the dataset's own length units (car
# bounding-box diagonal 6.285). Chosen from the measured structure of the field
#, not from round numbers:
#   near   the resolved boundary layer -- mean speed 2.13 rising to 6.33
#   mid    the layer's outer part -- 14.52
#   outer  essentially freestream -- 18.62 against a 20 m/s inlet
BANDS = (("near", 0.0, 0.05),
         ("mid", 0.05, 0.2),
         ("outer", 0.2, float("inf")))


def band_masks(query_xyz, boundary_xyz, bands=BANDS):
    """{band name: boolean mask over query_xyz}, by distance to the boundary.

    PER CASE, from that case's own surface -- not a shared index assignment.
    This is stricter than reusing one band assignment across cases, and is the
    version a reported number should use.
    """
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.ascontiguousarray(boundary_xyz, np.float64)).query(
        np.ascontiguousarray(query_xyz, np.float64), k=1, workers=-1)
    return {name: (d >= lo) & (d < hi) for name, lo, hi in bands}, d


def drag(nid, m=None):
    return float(open(os.path.join(case_dir(nid, m), "cd.txt")).read().strip())


if __name__ == "__main__":
    m = mapping(verbose=True)
    print(f"[shapenet_raw] {len(m)} ids mapped")
    nid = sorted(m)[0]
    sx, sp = surface(nid, m)
    vx, vv = volume(nid, m)
    print(f"  {nid}: surface {sx.shape} p[{sp.min():.1f},{sp.max():.1f}]  "
          f"volume {vx.shape} |U| mean {np.linalg.norm(vv,axis=1).mean():.2f}  "
          f"Cd {drag(nid, m):.4f}")
