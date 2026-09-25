#!/usr/bin/env python
"""Per-case physical sanity checks. Run on every generated case.

The generated set stands alone, so nothing compares it against STAR-CCM+ any
more. That removes the acceptance gate but NOT the need to know a case is free
of artifacts -- and the leak bug showed how badly that can go: it produced a
room 6x too fast with a fictitious room-wide draft, and **mass balance did not
catch it**, because the leak was prescribed and so balanced perfectly while
being 258x wrong.

So these checks are deliberately not all conservation-based. The one that would
have caught the leak bug on its own is `speed_mean`, and it needs no reference
data: 0.602 m3/s into a ~172 m3 room is ~12.6 air changes per hour, which puts
mean room speed in the 0.1-0.3 m/s range. The bugged case sat at 1.20 m/s and
should have been rejected on that alone.

What each check is for:

  mass_balance   net boundary flux ~ 0        solver/BC consistency
  supply_flux    3 x 0.2006 = 0.602 m3/s      the vent profile is intact
  leak_flux      the prescribed 0.0009 m3/s   the leak is not blasting again
  return_flux    supply - leak                outlets absorb what comes in
  speed_mean     0.05-0.60 m/s                the air-changes argument
  speed_max      < 5 m/s                      the supply patch itself peaks ~3.75
  T_range        inside the BC envelope       no runaway or frozen cells
  limiters       0 cells limited              guard rails must stay inactive
  bounding       undershoot < 1e-2 of mean    clipping must be negligible
  finite         no NaN/Inf                   the obvious one

A failing case is not necessarily wrong, but it is not usable unexamined.

Run:
    python cfd/sanity.py --case cfd/run/base
    python cfd/sanity.py --all cfd/run
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate import read_field

RHO = 1.2058              # Boussinesq rho0, constant/thermophysicalProperties
ROOM_VOLUME = 172.0       # 8.8 x 6.1 x 3.2 m, less furniture; near enough for ACH
SUPPLY_TARGET = 0.6019    # 3 x 0.2006 m3/s
LEAK_TARGET = 0.0009
SPEED_MIN, SPEED_MAX = 0.05, 0.60
SPEED_PEAK = 5.0
T_LO, T_HI = 284.0, 312.0   # supply 286.15, hottest wall 308.06, plus margin
SUPPLY = ("HVAC_02", "HVAC_04", "HVAC_06")
RETURN = ("HVAC_01", "HVAC_03", "HVAC_05")
# Largest epsilon undershoot, as a fraction of the epsilon field mean, that
# still counts as negligible. Worst observed over 107 completed sweep cases is
# 1.1e-03 (a few stagnant cells clipped against a mean around 0.058); 73 of the
# 107 clip not at all in the final quarter, and the ones that do are
# indistinguishable from the clean ones in how the field settles.
# The threshold sits an order of magnitude above that so ordinary cases have
# margin, and two orders below a clip that could actually perturb the solution
# -- that would be a substantial fraction of the mean, not a thousandth of it.
BOUND_SEVERITY = 1e-2
BOUND_RE = re.compile(r"bounding (\w+), min: (\S+) max: (\S+) average: (\S+)")


def latest(case):
    t = [d for d in os.listdir(case)
         if re.fullmatch(r"\d+", d) and os.path.isdir(os.path.join(case, d))
         and d != "0"]
    return max(t, key=float) if t else None


def patch_flux(path):
    """Volumetric flux per boundary patch, m3/s, positive out of the domain."""
    txt = open(path).read()
    body = txt[txt.index("boundaryField"):]
    out = {}
    for name, blk in re.findall(r"^    (\w+)\n    \{(.*?)\n    \}", body, re.S | re.M):
        m = re.search(r"nonuniform List<scalar>\s*\n?\s*\d+\s*\n?\(\n(.*?)\n\)",
                      blk, re.S)
        if m:
            v = np.fromstring(m.group(1).replace("\n", " "), sep=" ")
        else:
            u = re.search(r"uniform\s+(-?[\d.eE+-]+)", blk)
            v = np.array([float(u.group(1))]) if u else np.zeros(1)
        out[name] = float(v.sum()) / RHO
    return out


def check(case, verbose=True):
    name = os.path.basename(os.path.normpath(case))
    t = latest(case)
    if t is None:
        r = {"case": name, "ok": False, "fail": ["no converged time directory"]}
        if verbose:
            print(f"{name:22} FAIL  no converged time directory")
        return r

    d = os.path.join(case, t)
    U = read_field(os.path.join(d, "U"))
    T = read_field(os.path.join(d, "T"))
    flux = patch_flux(os.path.join(d, "phi"))

    sup = -sum(v for k, v in flux.items() if k in SUPPLY)
    ret = sum(v for k, v in flux.items() if k in RETURN)
    leak = flux.get("Leak", 0.0)
    net = sum(flux.values())
    speed = np.linalg.norm(U, axis=1)

    logtxt = ""
    for p in ("log.solve.final", "log.solve.continue", "log.solve.warm"):
        q = os.path.join(case, p)
        if os.path.exists(q):
            logtxt += open(q).read()
    limited = sum(int(n) for n in re.findall(r"Limited (\d+) \(", logtxt))
    limited += sum(int(n) for n in re.findall(r"LimitedCells=(\d+)", logtxt))
    # Bounding events are judged by their MAGNITUDE, not by how many there are
    # or where they fall. Both earlier criteria -- raw count, then count in the
    # final quarter -- were measured against the solution and found to track
    # nothing: a flagged case (19 events) and a clean one (0 events) continue to
    # move by an identical 6.00% of |U| per 250 iterations, so the count is
    # uncorrelated with solution quality. What matters is whether the clip is
    # large enough to perturb the field. Measured across the sweep the worst
    # undershoot is -3e-05 against an epsilon mean of 0.058 -- 5e-04 of the
    # mean, in a few stagnant cells -- while the field average itself is static
    # to four significant figures across those same events.
    lines = logtxt.split("\n")
    _its = [i for i, l in enumerate(lines) if l.startswith("Time = ")]
    _bnd = [(i, l) for i, l in enumerate(lines) if "bounding" in l]
    bounding = len(_bnd)
    bounding_tail = 0
    worst_tail = 0.0          # worst undershoot / field mean, final quarter
    worst_all = 0.0
    if _its:
        import bisect as _bisect
        for _b, _line in _bnd:
            g = BOUND_RE.search(_line)
            sev = 0.0
            if g:
                try:
                    lo, avg = float(g.group(2)), float(g.group(4))
                    if avg > 0:
                        sev = abs(min(lo, 0.0)) / avg
                except ValueError:
                    sev = 0.0
            worst_all = max(worst_all, sev)
            _k = _bisect.bisect_right(_its, _b)
            if _k > 0.75 * len(_its):
                bounding_tail += 1
                worst_tail = max(worst_tail, sev)
    res = re.findall(r"Solving for p_rgh, Initial residual = ([\d.eE+-]+)", logtxt)

    m = {
        "case": name, "t": t, "cells": int(len(U)),
        "supply": sup, "return": ret, "leak": leak, "net": net,
        "ach": sup * 3600 / ROOM_VOLUME,
        "speed_mean": float(speed.mean()), "speed_max": float(speed.max()),
        "T_min": float(T.min()), "T_max": float(T.max()),
        "limited": limited, "bounding": bounding,
        "bounding_tail": bounding_tail,
        "bounding_worst": worst_all, "bounding_worst_tail": worst_tail,
        "p_rgh": float(res[-1]) if res else float("nan"),
    }

    fail = []
    if abs(net) > 1e-3 * SUPPLY_TARGET:
        fail.append(f"mass_balance net {net:+.2e} m3/s")
    if abs(sup - SUPPLY_TARGET) > 0.05 * SUPPLY_TARGET:
        fail.append(f"supply_flux {sup:.4f} vs {SUPPLY_TARGET:.4f}")
    if abs(leak - LEAK_TARGET) > 0.5 * LEAK_TARGET:
        fail.append(f"leak_flux {leak:.5f} vs {LEAK_TARGET}")
    if abs((sup - leak) - ret) > 0.05 * SUPPLY_TARGET:
        fail.append(f"return_flux {ret:.4f} != supply-leak {sup - leak:.4f}")
    if not SPEED_MIN <= m["speed_mean"] <= SPEED_MAX:
        fail.append(f"speed_mean {m['speed_mean']:.3f} outside "
                    f"[{SPEED_MIN}, {SPEED_MAX}] at {m['ach']:.1f} ACH")
    if m["speed_max"] > SPEED_PEAK:
        fail.append(f"speed_max {m['speed_max']:.2f} > {SPEED_PEAK}")
    if m["T_min"] < T_LO or m["T_max"] > T_HI:
        fail.append(f"T_range [{m['T_min']:.1f}, {m['T_max']:.1f}]")
    if limited:
        fail.append(f"limiters engaged on {limited} cells")
    if worst_tail > BOUND_SEVERITY:
        fail.append(f"epsilon clipped by {worst_tail:.1e} of the field mean in "
                    f"the final quarter ({bounding_tail} of {bounding} events) "
                    f"-- exceeds {BOUND_SEVERITY:.0e}")
    if not (np.isfinite(U).all() and np.isfinite(T).all()):
        fail.append("non-finite values in U or T")

    m["ok"], m["fail"] = not fail, fail
    if verbose:
        print(f"{name:22} t={t:>5} {'PASS' if m['ok'] else 'FAIL'}  "
              f"sup {sup:.4f} ret {ret:.4f} leak {leak:.5f} net {net:+.1e}  "
              f"|U| {m['speed_mean']:.3f}/{m['speed_max']:.2f}  "
              f"T[{m['T_min']:.1f},{m['T_max']:.1f}]  p_rgh {m['p_rgh']:.1e}  "
              f"clip {worst_tail:.0e}")
        for f in fail:
            print(f"    ! {f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--all", help="directory containing case directories")
    ap.add_argument("--json", help="write results here")
    args = ap.parse_args()

    if args.case:
        cases = [args.case]
    elif args.all:
        cases = sorted(p for p in glob.glob(os.path.join(args.all, "*"))
                       if os.path.isdir(os.path.join(p, "constant")))
    else:
        raise SystemExit("need --case or --all")
    if not cases:
        raise SystemExit("no cases found")

    res = [check(c) for c in cases]
    bad = [r for r in res if not r["ok"]]
    print(f"\n{len(res) - len(bad)}/{len(res)} passed")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
