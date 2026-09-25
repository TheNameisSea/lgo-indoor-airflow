#!/usr/bin/env python
"""Shape-GAP split for the ShapeNet-Car external benchmark.

WHY THIS EXISTS. The release ships a train/test manifest (500 / 111) that
leaves every test car a near-twin in training: 611 template-fitted cars sample a
shape manifold densely, so an arbitrary split is an interpolation test.

This module builds the alternative, and the published manifest is left untouched
so its numbers stay comparable to published GINO / Transolver results. Report
BOTH: the published split for comparability, the gap split for the claim.

DISTANCE. Every car is a template fit, so vertex i corresponds across cases --
verified: mean vertex-to-vertex distance is 0.115 under the given ordering and
1.900 under a random permutation, a 16.5x ratio. Shape distance is therefore just
the MEAN VERTEX DISPLACEMENT, with no matching step at all. This is the direct
analogue of `splits_cfd_gap.layout_distance` (Hungarian-matched mean panel
displacement) minus the matching, which the correspondence makes unnecessary.

Units are the dataset's own; there is no metre here. Quote gaps against the
6.285 bounding-box diagonal, never as if they were physical lengths.

GUARANTEE. Same as `splits_cfd_gap.py`: cases closer than `gap` are joined by an
edge, each connected component goes to ONE split, so every cross-split pair is at
least `gap` apart and nothing is discarded into a buffer zone.

A VAL SPLIT IS CARVED HERE. The release has none -- only train and test. Running
without one and selecting on the test manifest would spend the test set the way
ledger table K spent ours.

    python data/splits_shapenet.py --sweep
    python data/splits_shapenet.py --gap 0.12 --freeze --materialize
"""

import argparse
import hashlib
import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "external", "shapenet_car",
                       "processed-car-pressure-data")
SPLIT_ROOT = os.path.join(ROOT, "splits_shapenet")
MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "splits_shapenet.json")
CACHE = os.path.join(ROOT, "external", "shapenet_car", "shape_distance.npy")
def manifest_fold(fold_id=0):
    """Frozen assignment for ONE published fold. The id is in the filename.

    Deliberately not a single fold-agnostic path. The published protocol has
    nine folds; a shared filename would let fold 3's assignment overwrite fold
    0's and be picked up later as "the published split" -- a well-formed wrong
    answer, which is the failure mode this repo keeps hitting.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"splits_shapenet_fold{fold_id}.json")


# Deprecated alias: fold 0's path. Use manifest_fold(fold_id).
MANIFEST_MATCHED = manifest_fold(0)
# SUPERSEDED CONSTANTS REMOVED (MATCHED_SEED / MATCHED_N_TEST / MATCHED_N_VAL).
# They configured a RANDOM 789/100 with a 50-case val carved out of train, built
# when the published split was believed to be random. It is not: it is an exact
# 9-fold leave-one-out over param0..8 (`param_fold`), which has no seed and no
# val carve -- they validate on the held-out fold itself. Keeping the constants
# would leave a second, plausible-looking way to build a split that is
# comparable to nothing.
PUBLISHED_N_TRAIN = 789
PUBLISHED_N_EVAL = 100

# Match the published proportions as closely as whole components allow:
# 500/111 is 81.8% train. We take val out of that share, not out of test.
TARGET = {"train": 0.72, "val": 0.10, "test": 0.18}


# --------------------------------------------------------------------------- #
def read_ply_vertices(path):
    """(n, 3) float64. Open3D binary-little-endian PLY, double xyz."""
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            hdr += line
        text = hdr.decode()
        nv = int([l for l in text.split("\n")
                  if l.startswith("element vertex")][0].split()[-1])
        buf = f.read(nv * 24)
    if len(buf) != nv * 24:
        raise ValueError(f"{path}: truncated vertex block")
    return np.frombuffer(buf, dtype="<f8").reshape(nv, 3)


def discover(dataset_dir=DATASET):
    """The 611 WATERTIGHT cases, in sorted order.

    The release holds 798 meshes but only 611 are watertight, and
    train.txt | test.txt is exactly that set. The other 187 carry a different
    topology (3682 verts / 3584 quad faces against 3586 / 7168 tri) so they
    have no vertex correspondence with the 611 and cannot enter this metric.
    """
    wt = sorted(open(os.path.join(dataset_dir, "watertight_meshes.txt")).read().split())
    missing = [c for c in wt
               if not os.path.exists(os.path.join(dataset_dir, "data", f"mesh_{c}.ply"))]
    if missing:
        raise SystemExit(f"[shapenet] {len(missing)} watertight meshes absent, "
                         f"e.g. {missing[:5]}")
    return wt


def published_split(dataset_dir=DATASET):
    """{case: 'train'|'test'} exactly as the release ships it."""
    out = {}
    for name, key in (("train.txt", "train"), ("test.txt", "test")):
        for c in open(os.path.join(dataset_dir, name)).read().strip().split(","):
            out[c.strip()] = key
    return out


def load_vertices(cases, dataset_dir=DATASET):
    """(n_cases, n_vertices, 3) float32, in `cases` order."""
    V = np.stack([read_ply_vertices(os.path.join(dataset_dir, "data", f"mesh_{c}.ply"))
                  for c in cases])
    return V.astype(np.float32)


def shape_distance(a, b):
    """Mean vertex displacement between two corresponded meshes."""
    return float(np.linalg.norm(a - b, axis=1).mean())


def distance_matrix(cases, dataset_dir=DATASET, cache=CACHE, verbose=True):
    """Symmetric (n, n) of mean vertex displacement. Cached -- it is 2e9 flops."""
    if cache and os.path.exists(cache):
        D = np.load(cache)
        if D.shape == (len(cases), len(cases)):
            if verbose:
                print(f"[shapenet] distance matrix from cache {cache}")
            return D
        if verbose:
            print(f"[shapenet] cache shape {D.shape} != {len(cases)}, recomputing")
    V = load_vertices(cases, dataset_dir)
    n = len(cases)
    D = np.zeros((n, n), dtype=np.float32)
    chunk = 8
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = V[s:e, None, :, :] - V[None, :, :, :]
        D[s:e] = np.sqrt((d * d).sum(-1)).mean(-1)
        if verbose and (s // chunk) % 10 == 0:
            print(f"  [shapenet] distance rows {e}/{n}", flush=True)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    if cache:
        np.save(cache, D)
    return D


# --------------------------------------------------------------------------- #
def assign(cases, D, gap, frozen=None):
    """{case: split}. Whole components to one split; frozen cases are kept."""
    n = len(cases)
    adj = sp.csr_matrix((D < gap) & ~np.eye(n, dtype=bool))
    ncomp, lab = csgraph.connected_components(adj, directed=False)
    comps = [[cases[i] for i in range(n) if lab[i] == k] for k in range(ncomp)]
    # Deterministic order keyed by the component's smallest member, so the
    # ordering never depends on how `discover` happened to sort.
    comps.sort(key=lambda c: hashlib.sha1(min(c).encode()).hexdigest())

    frozen = frozen or {}
    out, count = {}, {k: 0 for k in TARGET}
    for c in comps:
        pinned = {frozen[m] for m in c if m in frozen}
        if len(pinned) == 1:
            s = pinned.pop()
        elif len(pinned) > 1:
            s = max(pinned, key=lambda p: sum(frozen.get(m) == p for m in c))
            for m in c:
                out[m] = frozen.get(m, s)
                count[out[m]] += 1
            continue
        else:
            total = sum(count.values()) or 1
            s = min(TARGET, key=lambda k: (count[k] / total - TARGET[k], -TARGET[k]))
        for m in c:
            out[m] = s
            count[s] += 1
    return out, comps


def verify(cases, D, assignment, gap):
    """Smallest cross-split distance actually achieved. Must be >= gap."""
    idx = {c: i for i, c in enumerate(cases)}
    split = np.array([assignment[c] for c in cases])
    worst, pair = np.inf, None
    for k in ("train", "val", "test"):
        a = np.where(split == k)[0]
        b = np.where(split != k)[0]
        if not len(a) or not len(b):
            continue
        sub = D[np.ix_(a, b)]
        i, j = np.unravel_index(sub.argmin(), sub.shape)
        if sub[i, j] < worst:
            worst, pair = float(sub[i, j]), (cases[a[i]], cases[b[j]])
    return worst, pair


def leak_report(cases, D, assignment, label):
    """Nearest-training-case distance for each held-out case."""
    idx = {c: i for i, c in enumerate(cases)}
    tr = [idx[c] for c in cases if assignment.get(c) == "train"]
    rows = {}
    for k in ("val", "test"):
        ho = [idx[c] for c in cases if assignment.get(c) == k]
        if not ho or not tr:
            continue
        nn = D[np.ix_(ho, tr)].min(1)
        rows[k] = dict(n=len(ho), min=float(nn.min()),
                       p05=float(np.quantile(nn, 0.05)),
                       median=float(np.median(nn)),
                       max=float(nn.max()))
    print(f"\n[{label}] nearest-TRAIN distance for held-out cases")
    print(f"{'split':>6}{'n':>6}{'min':>10}{'p05':>10}{'median':>10}{'max':>10}")
    for k, r in rows.items():
        print(f"{k:>6}{r['n']:>6}{r['min']:>10.4f}{r['p05']:>10.4f}"
              f"{r['median']:>10.4f}{r['max']:>10.4f}")
    return rows


def summary(assignment, comps, gap, worst, pair):
    row = [sum(1 for s in assignment.values() if s == k)
           for k in ("train", "val", "test")]
    print(f"\n[shapenet] gap = {gap:.4f}, {len(comps)} components "
          f"(largest {max(len(c) for c in comps)})")
    print("".join(f"{s:>8}" for s in ("train", "val", "test")) + f"{'total':>8}")
    print("".join(f"{v:>8}" for v in row) + f"{sum(row):>8}")
    print(f"\n[shapenet] smallest cross-split distance {worst:.4f} "
          f"({pair[0]} / {pair[1]}) -- nominal gap {gap:.4f}")
    if worst < gap - 1e-9:
        raise SystemExit("[shapenet] GUARANTEE VIOLATED")


def materialize(assignment, root=SPLIT_ROOT, dataset_dir=DATASET, verbose=True):
    """Symlink trees, one directory per case, mirroring splits_cfd_gap layout."""
    data = os.path.join(dataset_dir, "data")
    for k in ("train", "val", "test"):
        os.makedirs(os.path.join(root, k), exist_ok=True)
    made = 0
    for case, split in sorted(assignment.items()):
        d = os.path.join(root, split, case)
        os.makedirs(d, exist_ok=True)
        for src, dst in ((f"mesh_{case}.ply", "mesh.ply"),
                         (f"press_{case}.npy", "press.npy")):
            link = os.path.join(d, dst)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.join(data, src), link)
        made += 1
    if verbose:
        print(f"[shapenet] materialized {made} cases under {root}/")


# --------------------------------------------------------------------------- #
def param_fold(fold_id=0):
    """THE PUBLISHED SPLIT, EXACTLY — not a random approximation of it.

    The Shape-Net Car table everyone cites is
    NOT a random split. `Car-Design-ShapeNetCar/dataset/load_dataset.py` in the
    Transolver release builds **9-fold leave-one-out over the raw release's own
    `param0`..`param8` directories**:

        folds = [f'param{i}' for i in range(9)]
        if i == args.fold_id: continue
        trainlst += samples[i]
        vallst = samples[args.fold_id]

    with `--fold_id` defaulting to 0. So the published 789/100 is **param1..8 for
    training and param0 for evaluation**, and it is exactly reproducible -- there
    is no unpublished seed to guess.

    ⚠ THEY REPORT ON THE FOLD THEY VALIDATE ON. `main_evaluation.py` loads
    `vallst` and scores it; there is no separate test set. We mirror that here
    BECAUSE the point is comparability -- and say so: the gate number is a
    validation number under their protocol, not a held-out one. Our own claim is
    made on the gap split, which does hold out a test set.

    Their fold sizes are stated in-code as
    `100 + 99 + 97 + 100 + 100 + 96 + 100 + 98 + 99 = 889`. Ours reconcile with
    that EXACTLY once the 4 empty directories are excluded (param2 99-2=97,
    param5 97-1=96, param8 100-1=99), which `mapping_all()` already does.
    """
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from data.shapenet_raw import mapping_all
    m = mapping_all()
    out, sizes = {}, {}
    for nid, rel in m.items():
        # rel looks like external/shapenet_car/raw/paramK/<hash>
        parts = rel.replace(os.sep, "/").split("/")
        param = next(q for q in parts if q.startswith("param"))
        k = int(param[len("param"):])
        sizes[k] = sizes.get(k, 0) + 1
        out[nid] = "val" if k == fold_id else "train"
    return out, sizes, m


PUBLISHED_FOLD_SIZES = [100, 99, 97, 100, 100, 96, 100, 98, 99]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=0.12)
    ap.add_argument("--dataset_dir", default=DATASET)
    ap.add_argument("--sweep", action="store_true",
                    help="component structure vs gap; pick a gap, then re-run")
    ap.add_argument("--freeze", action="store_true",
                    help=f"write the assignment to {MANIFEST}")
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--fold_id", type=int, default=0,
                    help="which param folder is the evaluation fold (published "
                         "default: 0)")
    ap.add_argument("--param_fold", action="store_true",
                    help="build THE PUBLISHED SPLIT exactly: 9-fold "
                         "leave-one-out over param0..8, --fold_id picking the "
                         "evaluation fold (published default 0). Refuses if the "
                         "fold sizes disagree with the published ones.")
    ap.add_argument("--matched_regime", action="store_true",
                    help="DEPRECATED alias for --param_fold. It once named a "
                         "RANDOM 789/100 approximation, built before the "
                         "published split was established as an exact 9-fold "
                         "LOO. Same behaviour as --param_fold now.")
    ap.add_argument("--published", action="store_true",
                    help="report the leak of the RELEASE's own split and stop")
    a = ap.parse_args()

    if a.matched_regime and not a.param_fold:
        print("[shapenet] ⚠ --matched_regime is a DEPRECATED alias for "
              "--param_fold. It once named a RANDOM 789/100 approximation, "
              "built before we established the published split is an exact "
              "9-fold LOO. Same behaviour now; use --param_fold.", flush=True)

    if a.param_fold or a.matched_regime:
        assignment, sizes, _m = param_fold(a.fold_id)
        n_tr = sum(1 for v in assignment.values() if v == "train")
        n_va = len(assignment) - n_tr
        ours = [sizes[k] for k in range(9)]
        print(f"[shapenet] PUBLISHED SPLIT, fold_id={a.fold_id} "
              f"(param{a.fold_id} is the evaluation fold)")
        print(f"[shapenet] fold sizes ours      {ours} = {sum(ours)}")
        print(f"[shapenet] fold sizes published {PUBLISHED_FOLD_SIZES} = "
              f"{sum(PUBLISHED_FOLD_SIZES)}")
        if ours != PUBLISHED_FOLD_SIZES:
            raise SystemExit("[shapenet] FOLD SIZES DISAGREE with the published "
                             "release -- the split would not be theirs. Refusing.")
        print(f"[shapenet] EXACT MATCH -> {n_tr} train / {n_va} eval")
        unl = sum(1 for c in assignment if c.startswith("u"))
        print(f"[shapenet] {unl} Zenodo-unlisted cases included (their setup "
              f"reads the raw directories, so they are in it too -- see 5.8)")
        if a.freeze:
            mf = manifest_fold(a.fold_id)
            json.dump(assignment, open(mf, "w"), indent=0, sort_keys=True)
            print(f"[shapenet] froze -> {mf}")
        return

    cases = discover(a.dataset_dir)
    print(f"[shapenet] {len(cases)} watertight cases")
    D = distance_matrix(cases, a.dataset_dir)

    if a.published:
        pub = published_split(a.dataset_dir)
        leak_report(cases, D, pub, "published")
        off = D[np.triu_indices(len(cases), 1)]
        print(f"\n[published] typical pair {off.mean():.4f}, "
              f"median {np.median(off):.4f}")
        return

    if a.sweep:
        off = D[np.triu_indices(len(cases), 1)]
        print(f"[shapenet] pair distances: min {off.min():.4f} "
              f"p01 {np.quantile(off, .01):.4f} median {np.median(off):.4f} "
              f"max {off.max():.4f}")
        print(f"\n{'gap':>8}{'comps':>8}{'largest':>10}{'singletons':>12}")
        for g in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20):
            n = len(cases)
            adj = sp.csr_matrix((D < g) & ~np.eye(n, dtype=bool))
            nc, lab = csgraph.connected_components(adj, directed=False)
            sizes = np.bincount(lab)
            print(f"{g:>8.2f}{nc:>8}{sizes.max():>10}{int((sizes == 1).sum()):>12}")
        return

    frozen = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else None
    assignment, comps = assign(cases, D, a.gap, frozen)
    worst, pair = verify(cases, D, assignment, a.gap)
    summary(assignment, comps, a.gap, worst, pair)
    leak_report(cases, D, assignment, f"gap {a.gap:.3f}")

    if a.freeze:
        json.dump(assignment, open(MANIFEST, "w"), indent=0, sort_keys=True)
        print(f"[shapenet] froze {len(assignment)} assignments -> {MANIFEST}")
    if a.materialize:
        materialize(assignment, dataset_dir=a.dataset_dir)


if __name__ == "__main__":
    main()
