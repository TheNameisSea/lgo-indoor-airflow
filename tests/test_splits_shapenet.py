"""Guards on data/splits_shapenet.py — the ShapeNet-Car gap split.

Each of these is a property that can fail while the module still returns a
well-formed answer, which is the failure mode `panels()` hit twice and the
reason `test_sweep_tiers.py` exists.

  1. DISCOVERY. Exactly the 611 watertight cases, and the release's own
     train.txt | test.txt is exactly that set. If the release ever ships a
     different manifest, the gap split and the published split stop describing
     the same population and every comparison between them is void.

  2. CORRESPONDENCE. The distance metric is mean vertex displacement with NO
     matching step, which is only legitimate because the meshes are template
     fits. If vertex ordering were arbitrary the metric would still return
     plausible positive numbers -- and be meaningless. Measured separation
     between corresponded and shuffled must stay large.

  3. THE GAP GUARANTEE. Every cross-split pair in the FROZEN manifest is at
     least the nominal gap apart. This is the whole point of the split; it is
     also exactly what `splits_cfd_gap.verify()` refuses to let slide.

  4. THE MANIFEST MATCHES THE CODE. Re-running `assign` at the frozen gap must
     reproduce the committed manifest. Component ordering is seeded by a sha1
     of each component's smallest member precisely so this is deterministic;
     if it drifts, results computed under the old assignment are mislabelled
     with nothing warning -- the `sweep.layouts()` trap, one dataset removed.

  5. THE RAW RELEASE JOINS CORRECTLY. Every numeric id resolves to a raw case
     carrying both fields at their documented shapes, and the volume points
     correspond across cases -- the property that makes the volume nulls and
     the distance banding definable at all.
     The id mapping is exact (md5 of the pressure bytes), so it cannot be
     silently approximate -- but its COVERAGE can drift if the extraction is
     partial, and a partial extraction looks like a smaller dataset, not an
     error.

  6. THE MATCHED-REGIME SPLIT IS WHAT IT CLAIMS. 789/100 over all 889, the
     published setup's population and sizes.
     It exists ONLY as the verification gate against published GINO's 0.0386;
     it is a random split of a dense shape manifold and carries no claim. The
     danger it guards is quiet: if it silently covered 611 instead of 889, or
     drifted from 789/100, the gate would still run and still print a number --
     one that could not be compared with anything.

  7. THE SPLIT BUYS WHAT IT CLAIMS. The frozen split's minimum held-out-to-
     train distance must beat the published split's by a wide margin. A gap
     split that does not remove the near-duplicates has no reason to exist.

    python -m tests.test_splits_shapenet
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.splits_shapenet import (DATASET, MANIFEST, assign, discover,
                                  distance_matrix, load_vertices,
                                  published_split, verify)

FROZEN_GAP = 0.045          # the value data/splits_shapenet.json was frozen at
N_WATERTIGHT = 611

_fail = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        _fail.append(name)


def main():
    if not os.path.isdir(DATASET):
        print(f"SKIP: dataset absent at {DATASET}\n"
              f"      see external/shapenet_car/DOWNLOAD.txt.")
        return 0

    print("\n[1] discovery")
    cases = discover(DATASET)
    check("611 watertight cases", len(cases) == N_WATERTIGHT, f"got {len(cases)}")
    pub = published_split(DATASET)
    check("train.txt | test.txt == watertight set",
          set(pub) == set(cases), f"{len(set(pub) ^ set(cases))} symmetric diff")
    check("train and test disjoint",
          len([c for c in pub if pub[c] == "train"]) +
          len([c for c in pub if pub[c] == "test"]) == len(pub))

    print("\n[2] vertex correspondence")
    V = load_vertices(cases[:2], DATASET)
    rng = np.random.default_rng(0)
    perm = rng.permutation(V.shape[1])
    d_corr = float(np.linalg.norm(V[0] - V[1], axis=1).mean())
    d_shuf = float(np.linalg.norm(V[0] - V[1][perm], axis=1).mean())
    check("corresponded << shuffled", d_shuf / d_corr > 5.0,
          f"ratio {d_shuf / d_corr:.1f}x ({d_corr:.4f} vs {d_shuf:.4f})")

    if not os.path.exists(MANIFEST):
        print(f"\nSKIP [3-5]: no frozen manifest at {MANIFEST}")
        return 1 if _fail else 0

    frozen = json.load(open(MANIFEST))
    D = distance_matrix(cases, DATASET, verbose=False)

    print("\n[3] the gap guarantee, on the FROZEN manifest")
    check("every case assigned", set(frozen) == set(cases))
    worst, pair = verify(cases, D, frozen, FROZEN_GAP)
    check(f"min cross-split distance >= {FROZEN_GAP}", worst >= FROZEN_GAP - 1e-9,
          f"{worst:.4f} ({pair[0]} / {pair[1]})")

    print("\n[4] the manifest reproduces from the code")
    rebuilt, _ = assign(cases, D, FROZEN_GAP, frozen=None)
    differ = [c for c in cases if rebuilt[c] != frozen[c]]
    check("assign() reproduces the frozen assignment", not differ,
          f"{len(differ)} differ, e.g. {differ[:5]}")

    print("\n[5] the raw release joins correctly")
    try:
        from data.shapenet_raw import RAW, mapping, surface, velocity, volume
    except ImportError as e:
        check("data.shapenet_raw imports", False, str(e))
        RAW = None
    if RAW is not None and not os.path.isdir(RAW):
        print(f"  SKIP: raw release absent at {RAW}")
    elif RAW is not None:
        m = mapping()
        check("all 798 numeric ids map to a raw case", len(m) == 798, f"got {len(m)}")
        check("every watertight case is covered",
              set(cases) <= set(m), f"{len(set(cases) - set(m))} uncovered")
        nid = cases[0]
        sx, sp = surface(nid, m)
        vx, vv = volume(nid, m)
        check("surface shapes (3682,3)/(3682,)",
              sx.shape == (3682, 3) and sp.shape == (3682,), f"{sx.shape} {sp.shape}")
        check("volume shapes (29498,3)/(29498,3)",
              vx.shape == (29498, 3) and vv.shape == (29498, 3), f"{vx.shape} {vv.shape}")
        # volume correspondence: the banding in 5.4 reuses one band assignment
        # for every case, which is only legitimate if index i means the same
        # place on every car.
        vx2, _ = volume(cases[1], m)
        rng2 = np.random.default_rng(0)
        pv = rng2.permutation(vx.shape[0])
        dc = float(np.linalg.norm(vx - vx2, axis=1).mean())
        ds = float(np.linalg.norm(vx - vx2[pv], axis=1).mean())
        check("volume points correspond across cases", ds / dc > 5.0,
              f"ratio {ds / dc:.1f}x ({dc:.4f} vs {ds:.4f})")
        check("velocity() agrees with volume()",
              np.array_equal(velocity(nid, m), vv))

    print("\n[6] the PUBLISHED split, reproduced exactly")
    try:
        from data.splits_shapenet import (PUBLISHED_FOLD_SIZES,
                                          PUBLISHED_N_EVAL, PUBLISHED_N_TRAIN,
                                          manifest_fold, param_fold)
        from data.shapenet_raw import RAW as _RAW
    except ImportError as e:
        check("param_fold imports", False, str(e))
        _RAW = None
    if _RAW is not None and not os.path.isdir(_RAW):
        print("  SKIP: raw release absent")
    elif _RAW is not None:
        am, sizes, _mm = param_fold(0)
        n_tr = sum(1 for v in am.values() if v == "train")
        n_ev = len(am) - n_tr
        # THE point of this fold: it is not an approximation of the published
        # split, it IS the published split. If the fold sizes drift from the
        # ones stated in their loader, it silently stops being theirs and the
        # gate compares our number to someone else's setup.
        check("fold sizes match the published ones exactly",
              [sizes[k] for k in range(9)] == PUBLISHED_FOLD_SIZES,
              f"{[sizes[k] for k in range(9)]} vs {PUBLISHED_FOLD_SIZES}")
        check(f"fold 0 is {PUBLISHED_N_TRAIN} train / {PUBLISHED_N_EVAL} eval",
              n_tr == PUBLISHED_N_TRAIN and n_ev == PUBLISHED_N_EVAL,
              f"got {n_tr}/{n_ev}")
        check("covers all 889 real cases", len(am) == 889, f"got {len(am)}")
        check("includes the 91 Zenodo-unlisted cases",
              sum(1 for c in am if c.startswith("u")) == 91)
        check("there is NO test split -- they report on the eval fold",
              set(am.values()) == {"train", "val"}, f"{sorted(set(am.values()))}")
        mf = manifest_fold(0)
        if os.path.exists(mf):
            check("frozen fold-0 manifest reproduces from the code",
                  json.load(open(mf)) == am)
        check("fold manifest is a DIFFERENT file from the gap split",
              os.path.abspath(mf) != os.path.abspath(MANIFEST))
        # A fold-agnostic filename would let fold 3 overwrite fold 0 and later
        # be read as "the published split".
        check("fold id is in the manifest filename", "fold0" in os.path.basename(mf),
              os.path.basename(mf))

    print("\n[7] the split removes the near-duplicates")
    idx = {c: i for i, c in enumerate(cases)}

    def min_nn(assignment, holdout):
        tr = [idx[c] for c in cases if assignment.get(c) == "train"]
        ho = [idx[c] for c in cases if assignment.get(c) == holdout]
        return float(D[np.ix_(ho, tr)].min(1).min())

    pub_min = min_nn(pub, "test")
    gap_min = min_nn(frozen, "test")
    check("gap split's closest held-out case beats the published one 10x+",
          gap_min > 10 * pub_min,
          f"published {pub_min:.4f} -> gap {gap_min:.4f} ({gap_min / pub_min:.0f}x)")

    print(f"\n{'ALL PASS' if not _fail else 'FAILURES: ' + ', '.join(_fail)}")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
