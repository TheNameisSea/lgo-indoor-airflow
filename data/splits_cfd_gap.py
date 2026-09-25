#!/usr/bin/env python
"""Layout-GAP split: hold-out sets separated from training by a minimum distance.

`splits_cfd.py` assigns by `sha1(tier:name) % 1000`, which is effectively random.
Tier B is 80 Sobol points over a CONTINUOUS box, and randomly splitting a densely
sampled continuous design gives every held-out case a training neighbour a few
centimetres away, which makes a random split an interpolation test rather than a
generalization test.

This module builds the alternative. `splits_cfd.py` is left untouched: it defines
the split the first CFD run trained on, and its numbers stay reproducible as the
"random / interpolation" arm.

DISTANCE. Vent panels are recovered by single-linkage clustering and split into
supply/return by the sign of each patch's MEAN vertical velocity -- the same net-
flux rule `pi_ginot.dataset` uses (correction A5), not the sign of a point's own
w. Supply panels are matched to supply panels and returns to returns by Hungarian
assignment, and the distance is the mean matched displacement in metres. Chamfer
over the raw HVAC clouds was tried first and is unusable HERE: partial panel
overlap makes it saturate, and every gap >= 0.20 m collapses the graph to a
single component.

GUARANTEE. Cases closer than `gap` are joined by an edge; each connected
component is assigned to ONE split. Every cross-split pair is therefore at least
`gap` apart, and no case has to be discarded into a buffer zone.

WHAT THIS SPLIT CANNOT DO. Retrieval ASHRAE R2 falls to zero only at ~0.58 m of
panel displacement (fitted on the 23 measured val cases), but the largest
nearest-neighbour distance anywhere in the 107 is 0.393 m -- so NO split of this
dataset can push retrieval to zero. At gap 0.30 the predicted retrieval R2 is
~+0.52 against +0.573 today. The layout box is sampled too densely for a gap
split alone to make this an extrapolation problem. Use it to remove the
near-duplicate leakage, and report the retrieval null alongside every model
number regardless.

STABILITY. `splits_cfd.py` guarantees that adding tier C cannot move an already
placed case, because assignment depends only on a case's own name. That property
is NOT available here: a new case lying between two components merges them, and a
merged component gets one assignment. This is inherent -- a hard distance
guarantee and assignment-stability-under-growth cannot both hold. The manifest is
therefore the source of truth: `--freeze` writes it, and a frozen manifest is
honoured for every case it lists, so tier C can only ever ADD.

    CPY=/path/to/cfd-env/bin/python
    $CPY data/splits_cfd_gap.py                      # summary at the default gap
    $CPY data/splits_cfd_gap.py --gap 0.25           # try another gap
    $CPY data/splits_cfd_gap.py --freeze             # write the manifest
    $CPY data/splits_cfd_gap.py --materialize        # build splits_cfd_gap/ links
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "cfd", "dataset")
SPLIT_ROOT = os.path.join(ROOT, "splits_cfd_gap")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "splits_cfd_gap.json")

# 0.30 m: 38 components with the largest at 10 of 107, which still leaves room to
# hit the target ratios. 0.25 gives 66 components but a smaller gap; 0.40 leaves
# 4 components (largest 90) and cannot be split at all. See the module docstring.
DEFAULT_GAP = 0.30
N_PANELS_PER_ROW = 3         # three supply and three return, in every tier
PANEL_LINK_M = 0.05          # superseded; kept so old manifests stay readable
TARGET = {"train": 0.63, "val": 0.22, "test": 0.15}
XYZ = ["X (m)", "Y (m)", "Z (m)"]


def tier_of(name):
    """Shared with splits_cfd.py -- imported rather than duplicated when possible."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import splits_cfd
        return splits_cfd.tier_of(name)
    except Exception:
        if name.startswith("C"):
            return "C"
        if name.startswith("B"):
            return "B"
        return "Am" if ("dxm" in name or "dxp075" in name) else "A"


def _kmeans_fixed_k(pts, k, iters=100):
    """Lloyd's algorithm with deterministic farthest-point seeding.

    Written out rather than imported: the CFD environment has no scikit-learn,
    and the job is small and fully specified -- k is known exactly, the clusters
    are compact and equal-sized, and the result must be reproducible.

    Partitioning by nearest centroid is what makes this immune to the touching
    panels that defeat linkage: two panels 19 mm apart at the edges still have
    centres 0.61 m apart, so every point is unambiguous about which centre it
    belongs to, even though no gap separates the point sets.
    """
    c = [pts[np.argmin(pts[:, 0])]]
    while len(c) < k:                               # farthest-point seeding
        d = np.min([np.linalg.norm(pts - q, axis=1) for q in c], axis=0)
        c.append(pts[int(np.argmax(d))])
    c = np.array(c, dtype=np.float64)
    for _ in range(iters):
        lab = np.argmin(((pts[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        new = np.array([pts[lab == j].mean(0) if np.any(lab == j) else c[j]
                        for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c


def panels(case_dir):
    """(supply centroids, return centroids) in metres, as (3, 2) xy arrays.

    TWO STAGES, EACH USING AN EXACT CRITERION.

    1. Supply versus return is decided by the SIGN OF MEAN VERTICAL VELOCITY,
       which is physical and unambiguous: supply blows down, returns draw up.
    2. Within each row, the three panels are separated by k-means on position
       with k = 3, because every layout in every tier has exactly three.

    NO DISTANCE THRESHOLD ANYWHERE. Two earlier versions used one and both broke
    on the widened design. Single-linkage at 0.05 m assumed "panels are ~1.5 m
    apart", true only while `spread` was confined to [1.00, 3.00]; adjacent
    panels now reach 0.610 m centre-to-centre at 0.591 m width, a 19 mm edge gap.
    Switching that to `maxclust=6` did not help either: intra-panel point spacing
    exceeds 19 mm, so single linkage bridges two panels before it finishes
    either, and the six clusters it returns are not the six panels (measured:
    7 of 193 cases came back 4+2 or 5+1).

    The failure is silent -- `layout_distance` compares whatever it is given, so
    a mis-recovered layout yields a plausible small distance and merges unrelated
    cases in the split graph.
    """
    d = pd.read_csv(os.path.join(case_dir, "New_HVAC.csv"))
    xyz = d[XYZ].to_numpy()
    w = d["Velocity[k] (m/s)"].to_numpy()

    # STAGE 1: separate the two ROWS by y. Supply and return rows are always at
    # least one panel width apart (`y_ret - y_sup > 2*HALF` is enforced by the
    # layout generator), so k=2 on y is unambiguous in every tier.
    rows = _kmeans_fixed_k(xyz[:, 1:2], 2)
    which = np.argmin(np.abs(xyz[:, 1:2] - rows.T), axis=1)

    # STAGE 2: label each ROW by its NET FLUX, never a single point's w. Supply
    # blows down, returns draw up, but individual points on a supply patch can
    # still read w > 0. Classifying per point instead of per patch put k-means
    # centroids between the two rows and corrupted every return position by
    # metres, which silently produced plausible-looking layout distances.
    out = {}
    for r in (0, 1):
        m = which == r
        out["sup" if w[m].mean() < 0.0 else "ret"] = xyz[m][:, :2]
    if len(out) != 2:
        raise ValueError(f"{case_dir}: both rows have the same flux sign")

    # STAGE 3: three panels per row, by nearest-centroid partition. No distance
    # threshold: adjacent panels can touch (19 mm edge gap in the widened
    # design) while their centres stay 0.61 m apart.
    res = []
    for k in ("sup", "ret"):
        c = _kmeans_fixed_k(out[k], N_PANELS_PER_ROW)
        res.append(c[np.argsort(c[:, 0])])          # ordered by x, deterministic
    return res[0], res[1]


def layout_distance(a, b):
    """Mean Hungarian-matched panel displacement [m], supply and returns apart."""
    total, n = 0.0, 0
    for pa, pb in zip(a, b):
        if len(pa) == 0 or len(pb) == 0:
            continue
        c = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
        r, cc = linear_sum_assignment(c)
        total += c[r, cc].sum()
        n += len(r)
    return total / max(n, 1)


def distance_matrix(cases, dataset_dir=DATASET, verbose=True):
    P = {c: panels(os.path.join(dataset_dir, c)) for c in cases}
    n = len(cases)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = layout_distance(P[cases[i]], P[cases[j]])
    if verbose:
        nn = np.where(np.eye(n, dtype=bool), np.inf, D).min(1)
        print(f"[gap] {n} cases; nearest-neighbour layout distance "
              f"min {nn.min():.3f} median {np.median(nn):.3f} max {nn.max():.3f} m")
    return D


def discover(dataset_dir=DATASET):
    """Exported cases, excluding the `base` reference solve."""
    out = []
    for p in sorted(glob.glob(os.path.join(dataset_dir, "*"))):
        n = os.path.basename(p)
        if os.path.isdir(p) and n != "base" and \
                os.path.exists(os.path.join(p, "New_HVAC.csv")):
            out.append(n)
    return out


def assign(cases, D, gap=DEFAULT_GAP, frozen=None):
    """{case: split}. Whole components go to one split; frozen cases are kept."""
    n = len(cases)
    adj = sp.csr_matrix((D < gap) & ~np.eye(n, dtype=bool))
    ncomp, lab = csgraph.connected_components(adj, directed=False)
    comps = [[cases[i] for i in range(n) if lab[i] == k] for k in range(ncomp)]

    # Deterministic order: a component is keyed by its smallest member name, so
    # the ordering does not depend on how `discover` happened to sort.
    comps.sort(key=lambda c: hashlib.sha1(min(c).encode()).hexdigest())

    frozen = frozen or {}
    out, count = {}, {k: 0 for k in TARGET}
    for c in comps:
        pinned = {frozen[m] for m in c if m in frozen}
        if len(pinned) == 1:
            s = pinned.pop()
        elif len(pinned) > 1:
            # A new case merged two frozen components. Honour the manifest for the
            # cases it names and put only the NEW members with the majority split.
            s = max(pinned, key=lambda p: sum(frozen.get(m) == p for m in c))
            for m in c:
                if m in frozen:
                    out[m] = frozen[m]
                    count[frozen[m]] += 1
            for m in c:
                if m not in frozen:
                    out[m] = s
                    count[s] += 1
            continue
        else:
            total = sum(count.values()) or 1
            # Whichever split is furthest below its target share takes the
            # component; ties break on the target itself.
            s = min(TARGET, key=lambda k: (count[k] / total - TARGET[k], -TARGET[k]))
        for m in c:
            out[m] = s
            count[s] += 1
    return out, comps


def assign_extend(cases, D, gap, frozen, new_to=None):
    """Place ONLY the new cases, keeping every frozen assignment and the gap.

    `assign(..., frozen=...)` cannot do this. Its `len(pinned) > 1` branch keeps
    frozen cases in their original splits and gives a bridging new case the
    component MAJORITY, which puts that new case within `gap` of a frozen case
    in a different split -- exactly the guarantee the split exists to provide.
    Measured when tier C landed: C015 sits 0.176 m from B041 (test) and 0.284 m
    from s196_r155_dxp075 (val), and `verify` refused the result.

    A case that bridges two splits cannot be placed anywhere, so it is DROPPED
    from the gap split rather than allowed to break the guarantee. It stays
    exported and remains available to `splits_cfd.py`, which has no gap to keep.

    `new_to` sends every unconstrained new case to one nominated split instead of
    dealing it by target ratio. This is what keeps a LEARNING CURVE readable: the
    rungs are only comparable if the validation set does not move, so new cases
    have to land in train. A case forced into a split by its neighbours still
    goes where the gap requires -- `new_to` only decides the free ones.

    Returns (assignment, dropped).
    """
    idx = {c: i for i, c in enumerate(cases)}
    new = [c for c in cases if c not in frozen]
    out = dict(frozen)
    count = {k: sum(1 for v in frozen.values() if v == k) for k in TARGET}

    # New cases that are themselves within `gap` of each other must share a
    # split, so they are placed as connected groups, not one at a time.
    ni = [idx[c] for c in new]
    sub = D[np.ix_(ni, ni)]
    adj = sp.csr_matrix((sub < gap) & ~np.eye(len(new), dtype=bool))
    ncomp, lab = csgraph.connected_components(adj, directed=False)
    groups = [[new[i] for i in range(len(new)) if lab[i] == k] for k in range(ncomp)]
    groups.sort(key=lambda g: hashlib.sha1(min(g).encode()).hexdigest())

    dropped = []
    for g in groups:
        near = {frozen[o] for m in g for o in frozen
                if D[idx[m], idx[o]] < gap}
        if len(near) > 1:
            dropped.extend(g)
            continue
        if len(near) == 1:
            s = near.pop()
        elif new_to is not None:
            s = new_to
        else:
            total = sum(count.values()) or 1
            s = min(TARGET, key=lambda k: (count[k] / total - TARGET[k], -TARGET[k]))
        for m in g:
            out[m] = s
            count[s] += 1
    return out, sorted(dropped)


def verify(cases, D, assignment, gap, raise_on_fail=True):
    """Smallest cross-split distance actually achieved. Must be >= gap."""
    idx = {c: i for i, c in enumerate(cases)}
    worst, pair = np.inf, None
    for a in cases:
        for b in cases:
            if a < b and assignment[a] != assignment[b]:
                d = D[idx[a], idx[b]]
                if d < worst:
                    worst, pair = d, (a, b)
    return worst, pair


def summary(assignment, comps, gap, worst, pair, raise_on_fail=True):
    tiers = sorted({tier_of(c) for c in assignment})
    print(f"\n[gap] gap = {gap:.2f} m, {len(comps)} components "
          f"(largest {max(len(c) for c in comps)})")
    print(f"{'tier':>6} " + "".join(f"{s:>8}" for s in ("train", "val", "test")) + f"{'total':>8}")
    for t in tiers:
        row = [sum(1 for c, s in assignment.items()
                   if s == k and tier_of(c) == t) for k in ("train", "val", "test")]
        print(f"{t:>6} " + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
    row = [sum(1 for s in assignment.values() if s == k)
           for k in ("train", "val", "test")]
    print(f"{'ALL':>6} " + "".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
    print(f"\n[gap] smallest cross-split distance {worst:.4f} m "
          f"({pair[0]} / {pair[1]})  -- nominal gap {gap:.2f}")
    if worst < gap - 1e-9 and raise_on_fail:
        raise SystemExit("[gap] GUARANTEE VIOLATED")


def materialize(assignment, root=SPLIT_ROOT, dataset_dir=DATASET, verbose=True):
    for s in ("train", "val", "test"):
        d = os.path.join(root, s)
        os.makedirs(d, exist_ok=True)
        for link in glob.glob(os.path.join(d, "*")):
            if os.path.islink(link):
                os.unlink(link)
    for case, s in sorted(assignment.items()):
        os.symlink(os.path.abspath(os.path.join(dataset_dir, case)),
                   os.path.join(root, s, case))
    if verbose:
        print(f"[gap] materialised {len(assignment)} links under {root}/")


def farthest_point_subsets(train, D, cases, sizes):
    """Nested train subsets that COVER the layout box, largest gap first.

    A learning curve over layout count only means something if each rung is
    spread over the design space; a random draw of 8 from 65 Sobol points
    clusters, and would understate what 8 well-chosen layouts can do. Farthest-
    point sampling picks, at each step, the layout furthest from everything
    already chosen. The subsets are nested by construction (rung N is the first N
    picks), so the curve adds information monotonically instead of resampling.

    Seeded deterministically at the layout closest to the centroid of `train`,
    so the sequence does not depend on directory order.
    """
    idx = {c: i for i, c in enumerate(cases)}
    tr = sorted(train)
    sub = np.array([idx[c] for c in tr])
    Dt = D[np.ix_(sub, sub)]
    order = [int(np.argmin(Dt.sum(1)))]                 # most central layout
    dist = Dt[order[0]].copy()
    while len(order) < len(tr):
        dist[order[-1]] = -np.inf
        nxt = int(np.argmax(dist))
        order.append(nxt)
        dist = np.minimum(dist, Dt[nxt])
    seq = [tr[i] for i in order]
    return {n: seq[:n] for n in sizes if n <= len(seq)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=DEFAULT_GAP)
    ap.add_argument("--dataset_dir", default=DATASET)
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--freeze", action="store_true",
                    help="write the manifest, pinning every case placed so far")
    ap.add_argument("--new_to", choices=("train", "val", "test"), default=None,
                    help="with --extend, send every unconstrained new case to "
                         "this split instead of dealing it by target ratio. Use "
                         "--new_to train to grow the training set while holding "
                         "the validation set fixed, which is what makes a "
                         "layout-count learning curve comparable across rungs.")
    ap.add_argument("--extend", action="store_true",
                    help="place ONLY cases absent from the manifest, keeping "
                         "every frozen assignment and the gap guarantee. A new "
                         "case within `gap` of two different splits is DROPPED "
                         "from this split rather than allowed to bridge them. "
                         "Use this to grow the dataset without re-dealing: the "
                         "default path re-deals free components and can violate "
                         "the guarantee (see assign_extend's docstring).")
    ap.add_argument("--sweep", action="store_true",
                    help="print component structure across a range of gaps")
    ap.add_argument("--subsample", type=int, nargs="+", default=None,
                    help="also materialise nested farthest-point train subsets "
                         "of these sizes, as splits_cfd_gap/train_<N>, for a "
                         "layout-count learning curve")
    args = ap.parse_args()

    cases = discover(args.dataset_dir)
    D = distance_matrix(cases, args.dataset_dir)

    if args.sweep:
        n = len(cases)
        print(f"\n{'gap':>6} {'comps':>6} {'largest':>8} {'singletons':>11}")
        for g in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            adj = sp.csr_matrix((D < g) & ~np.eye(n, dtype=bool))
            nc, lab = csgraph.connected_components(adj, directed=False)
            s = np.bincount(lab)
            print(f"{g:>6.2f} {nc:>6} {s.max():>8} {(s == 1).sum():>11}")
        return

    frozen = None
    if os.path.exists(MANIFEST):
        blob = json.load(open(MANIFEST))
        if abs(blob.get("gap", args.gap) - args.gap) < 1e-9:
            frozen = blob["assignment"]
            print(f"[gap] honouring frozen manifest ({len(frozen)} cases)")

    dropped = []
    if args.extend:
        if not frozen:
            raise SystemExit("[gap] --extend needs a frozen manifest at this gap")
        assignment, dropped = assign_extend(cases, D, args.gap, frozen,
                                            new_to=args.new_to)
        print(f"[gap] extend: {len(assignment) - len(frozen)} new cases placed, "
              f"{len(dropped)} dropped")
        for c in dropped:
            print(f"[gap]   dropped {c} -- within {args.gap} m of two splits")
        # A dropped case is not part of the split, so both the components and
        # the guarantee are measured on the KEPT cases. `verify` indexes D by
        # its own `cases` argument, so the matrix must be sliced to match.
        idx = {c: i for i, c in enumerate(cases)}
        kept = [c for c in cases if c in assignment]
        ki = [idx[c] for c in kept]
        Dk = D[np.ix_(ki, ki)]
        adj = sp.csr_matrix((Dk < args.gap) & ~np.eye(len(kept), dtype=bool))
        nc, lab = csgraph.connected_components(adj, directed=False)
        comps = [[kept[i] for i in range(len(kept)) if lab[i] == k]
                 for k in range(nc)]
        worst, pair = verify(kept, Dk, assignment, args.gap, raise_on_fail=False)
        # An extension cannot repair a violation it INHERITED. Correcting
        # `panels()` (see its docstring) shifted a few measured distances by
        # millimetres, and one already-frozen pair fell 9 mm below the nominal
        # gap: B043 [train] / s196_r155_dxp000 [val] at 0.2912 m. Nothing placed
        # now caused it and nothing placed now can fix it, so refusing to grow
        # the split over it would be wrong. New placements are still held
        # strictly: a violation involving any case absent from the manifest is
        # fatal. The achieved separation is reported as measured, never as the
        # nominal value.
        if worst < args.gap - 1e-9:
            a, b = pair
            if a in frozen and b in frozen:
                print(f"[gap] INHERITED violation, not introduced by this "
                      f"extension: {a} / {b} at {worst:.4f} m against a nominal "
                      f"{args.gap:.2f}. Quote the ACHIEVED {worst:.4f} m.")
            else:
                raise SystemExit(f"[gap] GUARANTEE VIOLATED by a NEW placement: "
                                 f"{a} / {b} at {worst:.4f} m")
    else:
        assignment, comps = assign(cases, D, args.gap, frozen)
        worst, pair = verify(cases, D, assignment, args.gap)
    summary(assignment, comps, args.gap, worst, pair,
            raise_on_fail=not args.extend)

    if args.freeze:
        json.dump({"gap": args.gap, "metric": "mean Hungarian panel displacement [m]",
                   "n_cases": len(assignment), "dropped": dropped,
                   "assignment": assignment},
                  open(MANIFEST, "w"), indent=2, sort_keys=True)
        print(f"[gap] froze {MANIFEST}")
    if args.materialize:
        materialize(assignment, dataset_dir=args.dataset_dir)

    if args.subsample:
        train = [c for c, s in assignment.items() if s == "train"]
        subs = farthest_point_subsets(train, D, cases, sorted(args.subsample))
        idx = {c: i for i, c in enumerate(cases)}
        print()
        for n, members in subs.items():
            d = os.path.join(SPLIT_ROOT, f"train_{n}")
            if args.materialize:
                os.makedirs(d, exist_ok=True)
                for link in glob.glob(os.path.join(d, "*")):
                    if os.path.islink(link):
                        os.unlink(link)
                for c in members:
                    os.symlink(os.path.abspath(os.path.join(args.dataset_dir, c)),
                               os.path.join(d, c))
            # How well the rung covers the box: the largest distance from any
            # training layout in the FULL set to its nearest member of the rung.
            sub_i = [idx[c] for c in members]
            cover = D[np.ix_([idx[c] for c in train], sub_i)].min(1).max()
            tiers = {}
            for c in members:
                tiers[tier_of(c)] = tiers.get(tier_of(c), 0) + 1
            print(f"[gap] train_{n:<3} {dict(sorted(tiers.items()))}  "
                  f"covering radius {cover:.3f} m"
                  + ("" if args.materialize else "   (not written; add --materialize)"))


if __name__ == "__main__":
    main()
