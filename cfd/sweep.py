#!/usr/bin/env python
"""Generate the layout sweep: geometry -> mesh -> fields -> ramped solve.

## What is in the sweep, and why

Two tiers, and both are wanted:

**Tier A, `dx = 0` (9 layouts).** Exactly the STAR-CCM+ grid --
`spread in {1.21, 1.96, 2.71}` x `row in {0.80, 1.55, 2.30}`. Keeping these is
deliberate even though we no longer compare against that dataset: they are a
real HVAC designer's layout choices for this room, and having them in our set
means the generated data covers the configurations someone actually built.

**Tier B, `dx != 0` (18 layouts).** The same grid translated bodily along x.
The reference set pins the supply array's centroid at x = 4.46 in every one of
its nine layouts -- it varies the array's *spread* but never *moves* it. That
degeneracy is what the paper blames for the model emitting the
training-set mean circulation: with every layout sharing an array centre,
"distance to the nearest training vent" barely varies, so the bulk circulation
of an unseen layout is not inferable. `dx` is the axis with no prior coverage
at all, which makes it the most valuable thing in the sweep.

## The leak stays

The room is a real one and the STEP carries its door undercut, so the sweep
keeps it. Two reasons beyond fidelity:

* `data/knn_cache.py` treats `leak` as a first-class boundary type with its own
  neighbour group (K=8), on the argument that air flows THROUGH an opening and
  folding it into `solid` would force velocity to zero where air actually
  moves. A dataset with no leak would leave that group empty in every case and
  never exercise an input the architecture explicitly models.
* Resolving the 10 mm slot costs ~177k of 568k cells, which was ~45 min of
  serial solve and is ~8 min now that MPI works. That is a fair price.

It stays PRESCRIBED at the measured 0.0009 m3/s rather than pressure-driven:
that keeps it layout-independent, which is what "layout is the only variable"
requires, and avoids the pressure-outlet failure that gave 34x too much flow.

## Concurrency: ONE job x 8 ranks -- there is an 8-core quota

The machine this was tuned on allowed 8 cores per account (a cgroup quota). Every attempt to
exceed it made things worse, not merely no better, because MPI busy-waits:

    1 job x  8 ranks   0.309 s/iter   <- optimal, fills the quota exactly
    1 job x 16 ranks   0.500 s/iter
    4 jobs x 16 ranks  stalled outright, 0 iterations in 60 s
    6 jobs x  8 ranks  10 s/iter each   (298 h projected)
    12 serial jobs     31 s/iter each   (414 h projected)

So: one case at a time, 8 ranks. ~15 min/case at 3000 iterations.

## Superseded: concurrency notes from before the quota was found

8 ranks per case is not a compromise, it is FASTER than 16: benchmarked alone on
one case, 16 ranks gives 0.500 s/iteration and 8 ranks gives 0.309. At 568k
cells a 16-way split leaves only 35k cells per rank, small enough that MPI
communication dominates and scaling runs backwards.

Total ranks must also stay well under the core count. This box has 64 cores and
other users on it; running 4 jobs x 16 ranks asked for all 64 and the sweep
STALLED COMPLETELY -- zero iterations in 60 s with the machine 80% idle, because
MPI busy-waits on communication and so does not degrade gracefully when
oversubscribed. 6 x 8 = 48 ranks leaves 16 cores of headroom.

Run:
    conda activate cfdEnv
    python cfd/sweep.py --list
    python cfd/sweep.py --jobs 6 --procs 8
"""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess

import numpy as np
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sanity

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

SPREADS = (1.21, 1.96, 2.71)          # tier A: the reference grid
ROWS = (0.80, 1.55, 2.30)
DXS = (-0.75, 0.0, 0.75)

# Tier B/C sample the layout space CONTINUOUSLY and far wider than the
# reference grid. A 3-level lattice in 3 dimensions is the bare minimum to see
# curvature, and every held-out point on it is either a lattice point (which can
# be memorised) or an extrapolation. Sobol covers the same box far better per
# case, and at ~7.7 min/case the cost is not the binding constraint -- the
# number of LAYOUTS is, since that is the effective sample size for learning
# layout dependence (27 cases, not 27 x 568k points).
SPREAD_RANGE = (1.00, 3.00)
ROW_RANGE = (0.70, 2.50)
DX_LIMIT = 1.00
JITTER_M = 0.15                        # tier C: per-vent, breaks the symmetry
N_SOBOL, N_JITTER = 80, 20
SOBOL_SEED = 20260903

# --- tier D: the newly opened region -------------------------------------
# THE RANGES ABOVE ARE FROZEN. They are not a design opinion any more -- they
# are the identity of the 107 solved cases. `layouts()` regenerates the Sobol
# sequence on every call and there is no committed manifest, so widening
# SPREAD_RANGE / ROW_RANGE / DX_LIMIT would re-deal B000-B079 and C000-C019 to
# DIFFERENT parameter values while cfd/dataset/B000 still holds the old solve.
# Every existing result would silently become mislabelled. Tier D is therefore
# ADDITIVE: a new tier over the wider space, leaving A/A'/B/C byte-identical.
# `test_tier_ac_unchanged()` is the guard.
#
# WHY IT EXISTS. The box above samples 18.2% of the geometrically feasible
# layout space, measured against valid(). The waste is concentrated in
# translation: dx's admissible range is [0.34 - spread, spread - 0.42], so at
# spread = 3.00 m the geometry permits dx = +-2.6 m and the box uses +-1.0.
# Translation of the array centroid is exactly the degree of freedom the paper
# identifies as under-determined, and the paper shows the
# consequence: no split of the current 107 separates a held-out layout by more
# than 0.392 m, against the ~0.58 m panel width at which retrieval fails. Adding
# cases INSIDE the box makes that worse (0.392 -> 0.352 m at n=192); only the
# wider region reaches it (67 of 192 cases clear 0.58 m).
D_SPREAD_RANGE = (0.30, 4.30)
D_ROW_RANGE = (0.35, 2.77)
D_DX_LIMIT = 3.20
N_TIER_D = 70
D_SOBOL_SEED = 20260907
# 2**17 raw Sobol draws, rejected down to ~62k feasible candidates. The pool
# size is not cosmetic: greedy farthest-point can only place a case where a
# candidate exists, so a thin pool costs separation directly. At N_TIER_D = 70
# the achieved minimum separation is 0.576 m from a 2**15 pool and 0.604 m from
# 2**17 -- the difference between missing and clearing the ~0.58 m panel width.
#
# N_TIER_D was cut from 85 to 70 when the vent-overlap rule above shrank the
# wedge: separation falls monotonically with count (70 -> 0.604 m, 75 -> 0.575,
# 85 -> 0.545). 70 is the largest count that still clears 0.58 m, and clearing
# it is the point of the tier.
D_POOL_M = 17

# Geometric margins: a vent must sit inside the ceiling with its half-width
# clear of the edge. Returns are the larger panel (0.600) and sit 0.04 m left.
CEIL_X = (0.0, 8.80)
HALF = 0.30
# Vents must also not overlap EACH OTHER. Supplies sit at
# [spread, X_MIDDLE, X_MIRROR - spread], so both adjacent gaps are exactly
# |X_MIDDLE - spread|, and the panels are 0.591 m (supply) / 0.600 m (return)
# wide. The original box (spread 1.00-3.00) kept that gap at 1.46 m or more, so
# the check was never needed and was never written. Tier D reaches spread 4.30,
# where the gap falls to 0.16 m and all three panels merge into one patch:
# D001 (spread 4.22, gap 0.24 m) meshed and ran for 68 minutes, then failed
# sanity with supply_flux -0.0013 against an expected 0.6019 and return_flux
# -274.3 -- the flux BC applied to a self-intersecting patch. 32 of the first
# 85 tier D layouts were degenerate this way.
PANEL_M = 0.600

# A vent must also not be cut into FIXED interior geometry. Exactly one object
# in this room reaches the ceiling plane: the corner Column, x [0.02, 0.19] x
# y [5.91, 6.08], rising to z = 3.18 against a ceiling at 3.20. (Inlet_AC,
# Outlet_AC and Wall_AC stop at 2.95 m and cannot be reached by a ceiling
# panel; Wall_03 already starts at x = 0.20 to clear the column.)
#
# A panel overlapping it is cut into a solid pillar. snappyHexMesh cannot
# resolve that intersection: it loses the Ceiling and the affected HVAC patch
# into a default `background` patch, so the return BC is never applied and the
# case still runs to t=3000 -- D002 finished with supply_flux -0.0016 against
# an expected 0.6019 and return_flux -237.1, after 70 minutes.
#
# Never reachable in the old box: returns sit at y = 6.15 - row with row >= 0.70,
# so y <= 5.45, always clear. Tier D's row reaches 0.35 and dx reaches -3.20,
# which puts a return panel in that corner. 0 of the solved 127 collide; 4 of
# the 70 tier D layouts do. Tested against every attempted case, this predicts
# pass/fail exactly (2 failures caught, 19 passes cleared, no false positives).
COLUMN_X, COLUMN_Y = (0.02, 0.19), (5.91, 6.08)


def hits_column(spread, row, dx, offsets=None, margin=0.0):
    """Does any vent panel overlap the floor-to-ceiling corner column?"""
    P = panel_xy(spread, row, dx, offsets)
    half = np.array([0.591 / 2] * 3 + [0.600 / 2] * 3) + margin
    for (x, y), hw in zip(P, half):
        if (x - hw < COLUMN_X[1] and x + hw > COLUMN_X[0]
                and y - hw < COLUMN_Y[1] and y + hw > COLUMN_Y[0]):
            return True
    return False


def tag(spread, row, dx):
    return (f"s{round(spread*100):03d}_r{round(row*100):03d}"
            f"_dx{'m' if dx < 0 else 'p'}{round(abs(dx)*100):03d}")


def vent_xs(spread, dx):
    """Supply and return x positions for a layout, for the validity check."""
    xs = [spread + dx, 4.46 + dx, 8.92 - spread + dx]
    return xs + [x - 0.04 for x in xs]


def valid(spread, row, dx, jitter=0.0):
    lo = min(vent_xs(spread, dx)) - jitter
    hi = max(vent_xs(spread, dx)) + jitter
    if lo < CEIL_X[0] + HALF or hi > CEIL_X[1] - HALF:
        return False
    # Vents must not overlap each other along x. Both adjacent gaps equal
    # |X_MIDDLE - spread|; below one panel width the three patches merge and
    # the flux BC is applied to a self-intersecting surface (see PANEL_M).
    if abs(4.46 - spread) < PANEL_M + jitter:
        return False
    # supply and return rows must not overlap or leave the ceiling
    y_sup, y_ret = row, 6.15 - row
    return (y_sup - HALF - jitter > 0.0 and y_ret + HALF + jitter < 6.10
            and y_ret - y_sup > 2 * HALF + 2 * jitter)


_PERM3 = None


def panel_xy(spread, row, dx, offsets=None):
    """The six vent panel centres as (6, 2) xy: rows 0-2 supply, 3-5 return.

    The analytic twin of `data/splits_cfd_gap.panels()`, which can only read a
    SOLVED case's exported CSV. Verified against all 107 exported cases: the
    two agree to 5.4 mm, three orders below the 0.30 m gap.
    """
    xs = np.array(vent_xs(spread, dx)[:3])
    P = np.concatenate([np.stack([xs, np.full(3, row)], 1),
                        np.stack([xs - 0.04, np.full(3, 6.15 - row)], 1)])
    if offsets:
        # make_layout.py maps --offsets pairwise onto HVAC_01..HVAC_06, and ODD
        # HVAC numbers are RETURNS while even are supplies. The pairs therefore
        # interleave return/supply per x position -- they are NOT three supply
        # pairs followed by three return pairs. Getting this wrong displaces a
        # tier C panel by up to 0.25 m, which is most of the 0.30 m split gap.
        o = np.asarray(offsets, float).reshape(6, 2)
        for i in range(3):
            P[i] += o[2 * i + 1]        # supply i is HVAC_0{2i+2}
            P[3 + i] += o[2 * i]        # return i is HVAC_0{2i+1}
    return P


def layout_distance_xy(A, B):
    """(na, nb) mean Hungarian-matched panel displacement [m] for (n, 6, 2)."""
    global _PERM3
    if _PERM3 is None:
        _PERM3 = np.array(list(itertools.permutations(range(3))))
    out = np.zeros((len(A), len(B)))
    for sl in (slice(0, 3), slice(3, 6)):
        a, b = A[:, sl][:, None], B[:, sl][None, :]
        # 3 panels per group, so all 3! assignments IS the exact Hungarian.
        best = None
        for p in _PERM3:
            c = np.linalg.norm(a - b[:, :, p, :], axis=3).sum(2)
            best = c if best is None else np.minimum(best, c)
        out += best
    return out / 6.0


def tier_d_layouts(existing, n=N_TIER_D):
    """Maximin-select `n` layouts over the full feasible wedge.

    Greedy farthest-point: repeatedly take the candidate whose nearest chosen
    or pre-existing neighbour is furthest. That puts every new case in the
    newly opened region -- packing them next to the existing 107 would shrink
    the separation the tier exists to create.
    """
    from scipy.stats import qmc
    raw = qmc.Sobol(d=3, scramble=True, seed=D_SOBOL_SEED).random_base2(m=D_POOL_M)
    lo = np.array([D_SPREAD_RANGE[0], D_ROW_RANGE[0], -D_DX_LIMIT])
    hi = np.array([D_SPREAD_RANGE[1], D_ROW_RANGE[1], +D_DX_LIMIT])
    cand = np.array([r for r in qmc.scale(raw, lo, hi) if valid(*r)])
    CP = np.array([panel_xy(*r) for r in cand])
    P0 = np.array([panel_xy(d["spread"], d["row"], d["dx"], d["offsets"])
                   for d in existing])
    d = layout_distance_xy(CP, P0).min(1)
    out = []
    for i in range(n):
        j = int(d.argmax())
        sp, rw, dxv = (round(float(v), 4) for v in cand[j])
        out.append(dict(name=f"D{i:03d}", spread=sp, row=rw, dx=dxv,
                        offsets=None, tier="D (widened box)"))
        d = np.minimum(d, layout_distance_xy(CP, CP[j:j + 1])[:, 0])
        d[j] = -1.0

    # Column collisions are dropped AFTER selection, deliberately. Filtering the
    # candidate pool before the greedy loop would change every subsequent pick
    # and therefore re-deal every D-name, orphaning cases already solved under
    # those names -- the same trap as widening the ranges (see the tier D
    # comment block). Dropping selected entries leaves the survivors' names and
    # their >= 0.58 m separation untouched; the tier is simply smaller.
    keep = [d_ for d_ in out if not hits_column(d_["spread"], d_["row"], d_["dx"])]
    if len(keep) != len(out):
        dropped = [d_["name"] for d_ in out if d_ not in keep]
        print(f"[sweep] tier D: dropped {len(dropped)} layout(s) intersecting "
              f"the corner column: {', '.join(dropped)}")
    return keep


def layouts(with_tier_d=True):
    """Tier A: the reference grid. Tiers B/C: Sobol over a widened box."""
    out = [dict(name=tag(s, r, d), spread=s, row=r, dx=d, offsets=None,
                tier="A (STAR-CCM+ grid)" if d == 0 else "A' (translated grid)")
           for s, r, d in itertools.product(SPREADS, ROWS, DXS)]

    from scipy.stats import qmc
    rng = np.random.default_rng(SOBOL_SEED)
    # Draw generously and filter: dx's admissible range depends on spread (a
    # wide array cannot also be translated far), so a fixed box contains
    # infeasible corners. Rejecting them beats distorting the sampler.
    eng = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED)
    # Sobol balance wants a power of two; draw 512 and filter down.
    raw = eng.random_base2(m=9)
    lo = np.array([SPREAD_RANGE[0], ROW_RANGE[0], -DX_LIMIT])
    hi = np.array([SPREAD_RANGE[1], ROW_RANGE[1], +DX_LIMIT])
    cand = qmc.scale(raw, lo, hi)

    n_b = n_c = 0
    for sp, rw, dxv in cand:
        want_jitter = n_b >= N_SOBOL
        j = JITTER_M if want_jitter else 0.0
        if not valid(sp, rw, dxv, j):
            continue
        if want_jitter and n_c >= N_JITTER:
            break
        offs = (np.round(rng.uniform(-JITTER_M, JITTER_M, 12), 4).tolist()
                if want_jitter else None)
        idx = n_c if want_jitter else n_b
        out.append(dict(name=f"{'C' if want_jitter else 'B'}{idx:03d}",
                        spread=round(float(sp), 4), row=round(float(rw), 4),
                        dx=round(float(dxv), 4), offsets=offs,
                        tier="C (irregular array)" if want_jitter
                             else "B (Sobol, symmetric)"))
        if want_jitter:
            n_c += 1
        else:
            n_b += 1
    if with_tier_d:
        out += tier_d_layouts(out)
    return out


def sh(cmd, log, cwd=None):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        return subprocess.call(cmd, cwd=cwd, stdout=f, stderr=f)


def final_time(case):
    if not os.path.isdir(case):
        return None
    t = [d for d in os.listdir(case)
         if re.fullmatch(r"\d+", d) and os.path.isdir(os.path.join(case, d))]
    return max(t, key=float) if t else None


def build(job, args):
    """One layout, end to end. Returns a status dict."""
    name = job["name"]
    geo = os.path.join(args.root, f"geometry_{name}")
    case = os.path.join(args.root, "run", name)
    log = os.path.join(args.logs, f"{name}.log")
    os.makedirs(args.logs, exist_ok=True)
    t0 = time.time()

    done = final_time(case)
    if done and float(done) >= args.expect and not args.force:
        return dict(job, status="skipped", t=done, mins=0.0)

    # A PARTIAL case must be wiped, not rebuilt over. The pipeline always
    # restarts from make_layout, so nothing in a half-finished directory is
    # reused on purpose -- but `snappyHexMesh -overwrite` run against a stale
    # constant/polyMesh can collapse the mesh instead of rebuilding it. D019,
    # interrupted mid-solve and then re-run, came back with ONE patch
    # (`background`): every HVAC patch gone, and solve.py died on
    # `KeyError: 'HVAC_02'`. Cleanup after a successful case (below) removes
    # only intermediate time directories, which does not help here.
    if os.path.isdir(case) or os.path.isdir(geo):
        shutil.rmtree(case, ignore_errors=True)
        shutil.rmtree(geo, ignore_errors=True)
        if done:
            print(f"  {name:<22} discarded partial case at t={done}", flush=True)

    steps = [
        (["python", os.path.join(HERE, "make_layout.py"), "--out", geo,
          "--spread", str(job["spread"]), "--row", str(job["row"]),
          "--dx", str(job["dx"])]
         + (["--offsets"] + [str(v) for v in job["offsets"]]
            if job.get("offsets") else []), None),
        ([PY, os.path.join(HERE, "make_case.py"), "--out", case,
          "--geometry", geo, "--stage", "mesh", "--procs", str(args.procs)], None),
        (["blockMesh"], case),
        (["snappyHexMesh", "-overwrite"], case),
        ([PY, os.path.join(HERE, "make_case.py"), "--out", case,
          "--geometry", geo, "--stage", "fields", "--procs", str(args.procs)], None),
        ([PY, os.path.join(HERE, "solve.py"), "--case", case,
          "--procs", str(args.procs), "--warmup", str(args.warmup),
          "--per-step", str(args.per_step), "--final", str(args.final)], None),
    ]
    env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))
    for cmd, cwd in steps:
        cmd = [PY if c == "python" else c for c in cmd]
        with open(log, "a") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            rc = subprocess.call(cmd, cwd=cwd, stdout=f, stderr=f, env=env)
        if rc != 0:
            return dict(job, status=f"FAILED ({cmd[0].split('/')[-1]} rc={rc})",
                        t=final_time(case), mins=(time.time() - t0) / 60)
    # Sanity BEFORE cleanup, so a failing case keeps its ramp history for
    # diagnosis. This is checked per case rather than at the end of the sweep:
    # the --geometry bug it caught on L01 would otherwise have corrupted all 27
    # runs before anyone looked.
    try:
        s = sanity.check(case, verbose=False)
    except Exception as e:                      # a broken case must not stop the sweep
        s = {"ok": False, "fail": [f"sanity raised {type(e).__name__}: {e}"]}
    if not s["ok"]:
        return dict(job, status="SANITY: " + "; ".join(s["fail"])[:60],
                    t=final_time(case), mins=(time.time() - t0) / 60,
                    sanity=s)

    # The ramp's intermediate states are scaffolding: only 0/ (mesh, BCs,
    # cell centres) and the converged field are wanted, and the intermediates
    # are half the case's ~900 MB.
    if not args.keep_intermediate:
        keep = {"0", final_time(case)}
        for d in os.listdir(case):
            if re.fullmatch(r"\d+", d) and d not in keep:
                shutil.rmtree(os.path.join(case, d), ignore_errors=True)
    return dict(job, status="ok", t=final_time(case),
                mins=(time.time() - t0) / 60, sanity=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="cfd")
    ap.add_argument("--logs", default="cfd/logs/sweep")
    ap.add_argument("--jobs", type=int, default=1, help="cases in parallel")
    ap.add_argument("--procs", type=int, default=8, help="MPI ranks per case")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--per-step", type=int, default=300)
    ap.add_argument("--final", type=int, default=1500)
    ap.add_argument("--expect", type=float, default=3000,
                    help="a case with a time >= this is considered done")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="keep the ramp's intermediate time directories")
    ap.add_argument("--only", nargs="*", help="run only these layout names")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    jobs = layouts()
    if args.only:
        jobs = [j for j in jobs if j["name"] in args.only]

    if args.list:
        print(f"{'name':22} {'spread':>7} {'row':>6} {'dx':>6}  tier")
        for j in jobs:
            print(f"{j['name']:22} {j['spread']:7.2f} {j['row']:6.2f} "
                  f"{j['dx']:+6.2f}  {j['tier']}")
        from collections import Counter
        print()
        for t, n in sorted(Counter(j["tier"] for j in jobs).items()):
            print(f"  {n:4} {t}")
        print(f"  {len(jobs):4} total")
        print(f"at {args.jobs} concurrent x {args.procs} ranks "
              f"= {args.jobs*args.procs} cores")
        return 0

    print(f"sweep: {len(jobs)} layouts, {args.jobs} at a time, "
          f"{args.procs} ranks each ({args.jobs*args.procs} cores)", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for r in ex.map(lambda j: build(j, args), jobs):
            results.append(r)
            print(f"  {r['name']:22} {r['status']:28} t={r['t']}  "
                  f"{r['mins']:.0f} min", flush=True)

    ok = [r for r in results if r["status"] in ("ok", "skipped")]
    print(f"\n{len(ok)}/{len(results)} complete")
    for r in results:
        if r["status"] not in ("ok", "skipped"):
            print(f"  FAILED {r['name']}: {r['status']}  (see {args.logs}/{r['name']}.log)")
    with open(os.path.join(args.logs, "sweep_status.json"), "w") as f:
        json.dump(results, f, indent=2)
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
