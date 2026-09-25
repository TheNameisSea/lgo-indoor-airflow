#!/usr/bin/env python
"""Layout-gap split at 0.58 m: the separation at which retrieval actually fails.

WHY A SECOND GAP SPLIT. `splits_cfd_gap.py` holds out layouts at 0.30 m, which
removes near-duplicate leakage but does not defeat the nearest-layout baseline:
with the donor restricted to training layouts at least M metres away, retrieval's
ASHRAE R2 falls +0.385 (nearest) -> +0.230 (0.45 m) -> +0.023 (0.58 m) ->
-0.077 (0.70 m). It collapses at roughly the vent panel width. A split at 0.58 m
therefore turns the comfort-band comparison from "beat a strong lookup" into
"beat nothing", which is the only configuration in which a generalisation claim
rests on something a lookup table cannot do.

WHY IT IS CONSTRUCTIBLE ONLY NOW. At 0.58 m the 127 base layouts collapse into a
SINGLE connected component -- no partition of them can achieve this separation,
because the most isolated base layout has a neighbour at 0.405 m. The 66 tier D
layouts are each their own component: mutually >= 0.603 m apart and >= 0.609 m
from every base layout. Every tier D case is therefore independently assignable,
and ANY partition of them satisfies the guarantee automatically.

THE DESIGN CHOICE THIS FILE MAKES. Because the base set must go to train, the
only free parameter is how many tier D layouts join it (`--n_train_d`).

    k = 0    train 127            every held-out layout lies OUTSIDE any region
                                  seen in training. That is EXTRAPOLATION, and it
                                  confounds two different failures: "cannot
                                  generalise 0.58 m" and "was never shown this
                                  part of the design space at all". The
                                  donor-floor measurement licenses only the first.
    k = 16   train 143 (default)  training covers the widened region, so a
                                  held-out layout is a generalisation test WITHIN
                                  the design rather than beyond it.
    k = 44   train 171            held-out sets of 11 are too small: per-case
                                  ASHRAE R2 has a spread near 0.2, so the
                                  standard error would swamp the effect.

The k training layouts are chosen by FARTHEST-POINT sampling over tier D, so they
spread over the widened region rather than clustering in one corner of it -- the
point of putting them in train is coverage. The remaining layouts are dealt to
val and test by a hash of the case name, which is stable under regeneration.

NOTE ON INDEPENDENCE. The training layouts here overlap those of the 0.30 m
split. Results on the two splits are therefore not statistically independent, and
that has to be stated rather than left implicit.

    CPY=/path/to/cfd-env/bin/python
    $CPY data/splits_cfd_gap58.py                        # summary only
    $CPY data/splits_cfd_gap58.py --freeze --materialize
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from splits_cfd_gap import (DATASET, discover, distance_matrix,  # noqa: E402
                            materialize, verify)

GAP = 0.58
SPLIT_ROOT = os.path.join(ROOT, "splits_cfd_gap58")
MANIFEST = os.path.join(HERE, "splits_cfd_gap58.json")
N_TRAIN_D = 16


def farthest_point(idx, D, k):
    """k indices spread over `idx`, seeded at the member nearest the centroid."""
    sub = D[np.ix_(idx, idx)]
    order = [int(np.argmin(sub.sum(1)))]          # most central member first
    dist = sub[order[0]].copy()
    while len(order) < k:
        dist[order[-1]] = -np.inf
        nxt = int(np.argmax(dist))
        order.append(nxt)
        dist = np.minimum(dist, sub[nxt])
    return [idx[i] for i in order[:k]]


def build(cases, D, n_train_d=N_TRAIN_D, gap=GAP):
    """Assign whole components: base -> train, the rest split train/val/test.

    Components, not individual cases. An earlier version assumed every layout
    outside the base set was its own component, which was true of the ANALYTIC
    panel positions but not of the ones recovered from the solved cases: under
    the corrected `panels()` a few widened-region layouts fall within `gap` of
    each other and must travel together, exactly as the guarantee requires.
    """
    n = len(cases)
    adj = sp.csr_matrix((D < gap) & ~np.eye(n, dtype=bool))
    ncomp, lab = csgraph.connected_components(adj, directed=False)
    sizes = np.bincount(lab)

    # The base set is the one giant component; deriving it from the graph rather
    # than from name prefixes keeps this correct as the dataset grows.
    big = int(np.argmax(sizes))
    base = [i for i in range(n) if lab[i] == big]
    free_comps = [[i for i in range(n) if lab[i] == c]
                  for c in range(ncomp) if c != big]
    n_free = sum(len(c) for c in free_comps)
    print(f"[gap58] {n} cases: {len(base)} in the base component, "
          f"{n_free} in {len(free_comps)} further components "
          f"(sizes {sorted(len(c) for c in free_comps)[-5:]} largest)")

    # Farthest-point over component CENTROIDS, so the training picks spread over
    # the widened region instead of clustering in one corner of it.
    cen = np.array([D[c].mean(0) for c in free_comps])
    Dc = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
    order = [int(np.argmin(Dc.sum(1)))]
    dist = Dc[order[0]].copy()
    while len(order) < len(free_comps):
        dist[order[-1]] = -np.inf
        nxt = int(np.argmax(dist))
        order.append(nxt)
        dist = np.minimum(dist, Dc[nxt])

    out, n_train_extra, train_comps = {}, 0, []
    for i in base:
        out[cases[i]] = "train"
    ci = 0
    while ci < len(order) and n_train_extra < n_train_d:
        comp = free_comps[order[ci]]
        for i in comp:
            out[cases[i]] = "train"
        train_comps.append(order[ci]); n_train_extra += len(comp); ci += 1

    held = [free_comps[j] for j in order[ci:]]
    held.sort(key=lambda c: hashlib.sha1(min(cases[i] for i in c).encode()).hexdigest())
    half = sum(len(c) for c in held) // 2
    got = 0
    for comp in held:
        s = "val" if got < half else "test"
        for i in comp:
            out[cases[i]] = s
        got += len(comp)
    if not any(v == "val" for v in out.values()) or \
            not any(v == "test" for v in out.values()):
        raise SystemExit("[gap58] val or test came out empty; lower --n_train_d")
    return out, sorted(cases[i] for j in train_comps for i in free_comps[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=GAP)
    ap.add_argument("--n_train_d", type=int, default=N_TRAIN_D,
                    help="how many independently assignable (tier D) layouts "
                         "join the base set in train; see the module docstring")
    ap.add_argument("--dataset_dir", default=DATASET)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--materialize", action="store_true")
    args = ap.parse_args()

    cases = discover(args.dataset_dir)
    D = distance_matrix(cases, args.dataset_dir)
    assignment, train_d = build(cases, D, args.n_train_d, args.gap)

    worst, pair = verify(cases, D, assignment, args.gap)
    counts = {k: sum(1 for v in assignment.values() if v == k)
              for k in ("train", "val", "test")}
    print(f"\n[gap58] gap = {args.gap:.2f} m")
    print(f"{'':>6}" + "".join(f"{s:>8}" for s in counts) + f"{'total':>8}")
    print(f"{'ALL':>6}" + "".join(f"{v:>8}" for v in counts.values())
          + f"{sum(counts.values()):>8}")
    print(f"\n[gap58] {len(train_d)} widened-region layouts in train, chosen by "
          f"farthest-point:\n        {', '.join(train_d)}")
    print(f"\n[gap58] smallest cross-split distance {worst:.3f} m "
          f"({pair[0]} / {pair[1]})  -- must be >= {args.gap:.2f}")
    if worst < args.gap - 1e-9:
        raise SystemExit("[gap58] GUARANTEE VIOLATED")

    if args.freeze:
        json.dump({"gap": args.gap,
                   "metric": "mean Hungarian panel displacement [m]",
                   "n_cases": len(assignment), "n_train_widened": len(train_d),
                   "train_widened": train_d, "assignment": assignment},
                  open(MANIFEST, "w"), indent=2, sort_keys=True)
        print(f"[gap58] froze {MANIFEST}")
    if args.materialize:
        materialize(assignment, root=SPLIT_ROOT, dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
