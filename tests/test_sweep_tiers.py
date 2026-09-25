"""Guards on cfd/sweep.py's layout generation.

The claim that must hold before tier D is worth solving:

  1. Adding tier D does not move tiers A/A'/B/C. `layouts()` regenerates the
     Sobol sequence on every call and there is NO committed manifest, so a
     change to SPREAD_RANGE / ROW_RANGE / DX_LIMIT would silently re-deal
     B000-B079 and C000-C019 to different parameter values while
     cfd/dataset/B000 still holds the old solve. Every existing result would
     become mislabelled with nothing failing and nothing warning.
  2. The analytic panel positions agree with the exported CSVs, since tier D
     is selected on `panel_xy` but the split is built from `panels()`, which
     reads a solved case. If they disagree, the separation tier D was chosen
     to guarantee is not the separation the split will measure.
  3. Tier D actually delivers the separation it exists for: every new layout
     clears the ~0.58 m panel width from every pre-existing one.

    python -m tests.test_sweep_tiers
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cfd"))

import sweep  # noqa: E402

# Tiers A/A'/B/C as generated BEFORE tier D was added. Spot values, not the
# whole list: enough to catch a re-deal, small enough to read.
FROZEN = {
    "s121_r080_dxm075": (1.21, 0.80, -0.75),
    "s271_r230_dxp075": (2.71, 2.30, 0.75),
    "B000": (2.9305, 2.1775, 0.9066),
    "B079": (1.4324, 0.7658, 0.6458),
    "C000": (1.9887, 1.8989, -0.1898),
    "C019": (2.668, 0.8282, 0.891),
}
# Tier D is SELECTED at N_TIER_D and then filtered: layouts intersecting the
# corner column are dropped after selection (dropping them before would re-deal
# every D-name and orphan the cases already solved under them). The delivered
# count is therefore smaller than the selected count, and the test pins the
# DELIVERED one -- that is what the dataset and the splits actually contain.
N_TIER_D_DELIVERED = 66
COUNTS = {"A": 9, "A'": 18, "B": 80, "C": 20, "D": N_TIER_D_DELIVERED}
PANEL_WIDTH_M = 0.58     # retrieval fails at roughly the vent panel width
# 70 cases over the feasible wedge, from a 2**17 pool, achieve 0.604 m.
# Packing more points in costs separation monotonically: 75 reach 0.575 m and
# 85 only 0.545 m. The count was cut from 85 to 70 once the vent-overlap
# constraint shrank the wedge: separation from the existing layouts is the
# point of this tier.
TIER_D_MIN_SEP_M = 0.58


def test_tier_ac_unchanged():
    L = {d["name"]: d for d in sweep.layouts()}
    for name, (sp, rw, dx) in FROZEN.items():
        d = L[name]
        got = (d["spread"], d["row"], d["dx"])
        assert np.allclose(got, (sp, rw, dx), atol=1e-4), \
            f"{name} moved: {got} != {(sp, rw, dx)} -- tier D re-dealt an " \
            f"existing tier; every solved case under that name is now mislabelled"
    # and generating without tier D must be a strict prefix of generating with
    base, full = sweep.layouts(with_tier_d=False), sweep.layouts()
    assert full[:len(base)] == base, "tier D perturbed the earlier tiers"
    print(f"  tiers A/A'/B/C unchanged ({len(base)} layouts), "
          f"tier D appended ({len(full) - len(base)})")


def test_tier_counts():
    from collections import Counter
    c = Counter(d["tier"].split(" ")[0] for d in sweep.layouts())
    assert dict(c) == COUNTS, (
        f"{dict(c)} != {COUNTS}. Tier D is selected at N_TIER_D="
        f"{sweep.N_TIER_D} and filtered down to what layouts() returns; if the "
        f"delivered count changed, the dataset and both split manifests are "
        f"stale.")
    print(f"  tier counts {dict(c)}")


def test_panel_xy_matches_exported():
    """Two independent recoveries of the same six panels must agree.

    `sweep.panel_xy` computes panel centres ANALYTICALLY from a layout's three
    parameters. `data.splits_cfd_gap.panels` recovers them from the SOLVED case
    by clustering the exported vent cloud. The splits are built from the second
    and the tier D design was verified with the first, so a disagreement means
    the separation a tier was selected for is not the separation its split will
    measure.

    RUN THIS AFTER ANY CHANGE TO `panels()`. It has caught two real regressions,
    and in both the broken version returned a WELL-FORMED answer -- three supply
    and three return centroids, plausible layout distances, no exception:

      * a 0.05 m single-linkage threshold merged panels whose edges are 19 mm
        apart in the widened design (it assumed "panels are ~1.5 m apart");
      * classifying supply vs return by a single point's vertical velocity
        instead of a patch's NET FLUX put return centroids metres out of place,
        which shifted split distances enough to invent a guarantee violation
        and to hide the real component structure at 0.58 m.

    Supply and return are checked separately: a whole-layout metric averages the
    two, so a corrupted return row can hide behind an intact supply row.
    """
    from data.splits_cfd_gap import DATASET, layout_distance, panels
    L = sweep.layouts()
    if not any(os.path.isdir(os.path.join(DATASET, d["name"])) for d in L):
        print(f"  SKIP: no exported cases under {DATASET}")
        return
    err, sup_err, ret_err, shape = [], [], [], []
    for d in L:
        case = os.path.join(DATASET, d["name"])
        if not os.path.isdir(case):
            continue
        P = sweep.panel_xy(d["spread"], d["row"], d["dx"], d["offsets"])
        sup, ret = panels(case)
        if len(sup) != 3 or len(ret) != 3:
            shape.append((d["name"], len(sup), len(ret)))
            continue
        err.append(layout_distance((sup, ret), (P[:3], P[3:])))
        sup_err.append(float(np.abs(np.sort(sup, 0) - np.sort(P[:3], 0)).max()))
        ret_err.append(float(np.abs(np.sort(ret, 0) - np.sort(P[3:], 0)).max()))
    assert not shape, f"panels() did not return 3 supply + 3 return for {shape[:5]}"
    assert err, "no exported cases found -- cannot verify"
    m, ms, mr = float(np.max(err)), float(np.max(sup_err)), float(np.max(ret_err))
    assert ms < 0.05, f"SUPPLY centroids disagree with the analytic twin by {ms:.4f} m"
    assert mr < 0.05, f"RETURN centroids disagree with the analytic twin by {mr:.4f} m"
    assert m < 0.05, f"analytic panels disagree with exported by {m:.4f} m"
    print(f"  analytic vs exported panels over {len(err)} cases: "
          f"matched {m:.4f} m, supply {ms:.4f} m, return {mr:.4f} m")


def test_tier_d_separation():
    L = sweep.layouts()
    isD = np.array([d["name"].startswith("D") for d in L])
    P = np.array([sweep.panel_xy(d["spread"], d["row"], d["dx"], d["offsets"])
                  for d in L])
    M = sweep.layout_distance_xy(P, P)
    cross = M[np.ix_(isD, ~isD)].min()
    assert cross >= TIER_D_MIN_SEP_M, \
        f"tier D sits {cross:.3f} m from an existing layout, below the " \
        f"{TIER_D_MIN_SEP_M} m this design achieves (target {PANEL_WIDTH_M} m). " \
        f"A drop here usually means panel_xy and the exported panels disagree " \
        f"-- check test_panel_xy_matches_exported first."
    # and no tier-D case may merge two existing components at the 0.30 m gap
    assert cross >= 0.30, "a tier D case could merge two frozen gap components"
    print(f"  min tier-D to pre-existing distance {cross:.3f} m "
          f"(>= {PANEL_WIDTH_M} m panel width, and >= 0.30 m gap)")


if __name__ == "__main__":
    for fn in (test_tier_ac_unchanged, test_tier_counts,
               test_panel_xy_matches_exported, test_tier_d_separation):
        print(f"{fn.__name__}:")
        fn()
    print("\nall sweep tier guards passed")
