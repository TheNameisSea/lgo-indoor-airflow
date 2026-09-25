#!/usr/bin/env python
"""Run a case, ramping the diffuser's radial throw in.

Why this exists. The supply BC is the momentum method: the patch carries the
jet the deleted vanes made, which is ~3.8 m/s of *tangential* velocity against
a normal component of only 0.58. Applied cold, that is a velocity jump against
a stagnant first cell ~6 mm away -- a shear of order 600 1/s -- and k, epsilon
follow it to 1e9 within three iterations, after which the enthalpy equation
returns T = -46000 and the thermo model aborts. Measured, not guessed: with the
radial component zeroed and nothing else changed, the same case runs clean with
zero bounding events and a continuity error of 1e-6.

Refining the vent does not help, it makes the jump worse (the wall distance
shrinks). Coded BCs are not available either -- this OpenFOAM build cannot
compile them. So the jet is introduced by continuation: solve the vertical-only
problem first, then raise the radial component in steps, restarting each time
from the previous field. Once air is already moving, the jump the boundary has
to impose is small.

Run:
    conda activate cfdEnv
    python cfd/solve.py --case cfd/run/base
"""

import argparse
import atexit
import glob
import json
import os
import shutil
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_case import patch_face_centres



def acquire(case):
    """Refuse to run two drivers on one case.

    Two of these sharing a case directory silently corrupt each other: they
    write the same system/controlDict, the same time directories, and each
    rewrites the supply BC at every stage boundary. That happened once and cost
    ~50 minutes of solver time before the interleaved stage logs gave it away.
    An atomic mkdir is enough.
    """
    d = os.path.join(case, ".solve.lock")
    try:
        os.mkdir(d)
    except FileExistsError:
        raise SystemExit(
            f"{d} exists: another solve.py is running on this case. "
            f"If you are sure it is not, remove the directory.")
    with open(os.path.join(d, "pid"), "w") as f:
        f.write(f"{os.getpid()}\n")
    atexit.register(lambda: shutil.rmtree(d, ignore_errors=True))


def latest_time(case):
    times = [d for d in os.listdir(case)
             if re.fullmatch(r"\d+(\.\d+)?", d) and
             os.path.isdir(os.path.join(case, d))]
    return max(times, key=float) if times else "0"



def replace_patch_value(txt, name, body):
    """Replace one patch's `value` entry, whatever formatting wrote the file.

    This has to survive BOTH layouts: the one make_case.py writes into 0/, and
    the one OpenFOAM writes when it saves a time directory. They differ by a
    single trailing space (`nonuniform List<vector> ` vs `nonuniform
    List<vector>`), which is enough to make a literal regex match 0/U and miss
    every later time -- so the ramp died on its first step after the warmup.
    Find the patch block by brace, then substitute inside it.
    """
    m = re.search(r"^    " + re.escape(name) + r"\n    \{\n", txt, re.M)
    if not m:
        return txt, 0
    start, end = m.end(), txt.index("\n    }", m.end())
    block, n = re.subn(
        r"(value\s+)(?:uniform\s+\([^)]*\)|nonuniform\s+List<vector>\s*\d+\s*\(.*?\))\s*;",
        lambda mm: mm.group(1) + body + ";", txt[start:end], count=1, flags=re.S)
    return (txt[:start] + block + txt[end:], n)



def warm_start(case, donor, log=None):
    """Initialise this case from a converged donor field, keeping OUR BCs.

    Every case is the same room with the vents moved, so a converged neighbour
    is a good initial guess -- good enough that the throw ramp becomes
    unnecessary. The ramp exists only because a cold start at full throw
    diverges within three iterations; starting from a developed field does not.

    mapFields maps boundary values too, and the donor's supply patches are in
    the WRONG PLACE for this layout, so its boundary data must not survive.
    Each field is therefore spliced: mapped `internalField`, our own
    `boundaryField`. That keeps the per-face supply profile this layout needs
    (the one thing a mis-set BC has already silently corrupted once).
    """
    fields = ("U", "T", "p_rgh", "p", "k", "epsilon", "nut", "alphat")
    zero = os.path.join(case, "0")
    ours = {f: open(os.path.join(zero, f)).read()
            for f in fields if os.path.exists(os.path.join(zero, f))}

    t = latest_time(donor)
    cmd = ["mapFields", os.path.relpath(os.path.abspath(donor), os.path.abspath(case)),
           "-sourceTime", t, "-consistent"]
    with open(log or os.devnull, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}   (donor {donor} @ {t})\n")
        fh.flush()
        rc = subprocess.call(cmd, cwd=case, stdout=fh, stderr=fh)
    if rc != 0:
        raise SystemExit(f"mapFields from {donor} failed (rc={rc})")

    for f, mine in ours.items():
        path = os.path.join(zero, f)
        mapped = open(path).read()
        i, j = mapped.find("boundaryField"), mine.find("boundaryField")
        if i < 0 or j < 0:
            continue                      # nothing to splice; leave as mapped
        open(path, "w").write(mapped[:i] + mine[j:])
    return t


def set_supply(case, time, patches, prof, wsup, tsup, fraction):
    """Rewrite the supply velocity in <time>/U with u_r scaled by `fraction`."""
    path = os.path.join(case, time, "U")
    txt = open(path).read()
    centres = patch_face_centres(case)
    for name, p in sorted(patches.items()):
        if p.get("role") != "supply":
            continue
        lo, hi = p["bbox_m"]
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        c = centres[name]
        dx, dy = c[:, 0] - cx, c[:, 1] - cy
        r = np.hypot(dx, dy)
        inv = np.where(r > 1e-9, 1.0 / np.maximum(r, 1e-30), 0.0)
        uz = np.interp(r, prof["r"], prof["u_z"]) * wsup
        ur = np.interp(r, prof["r"], prof["u_r"]) * wsup * fraction
        vecs = np.c_[ur * dx * inv, ur * dy * inv, uz]
        rows = "\n".join(f"({v[0]:.6g} {v[1]:.6g} {v[2]:.6g})" for v in vecs)
        body = f"nonuniform List<vector>\n{len(vecs)}\n(\n{rows}\n)"
        txt, n = replace_patch_value(txt, name, body)
        if n != 1:
            raise SystemExit(f"could not rewrite supply {name} in {path}")
    open(path, "w").write(txt)


def run(case, iters, label, procs=1):
    """One solver stage. `procs` > 1 decomposes, runs under MPI, reconstructs.

    Decompose/reconstruct happens per stage rather than once, because the ramp
    rewrites <time>/U between stages and that edit has to land on reconstructed
    fields -- rewriting it inside each processor*/ directory instead would mean
    reproducing OpenFOAM's decomposition addressing, for no gain: decomposePar
    costs seconds against stages that cost tens of minutes.
    """
    ctrl = os.path.join(case, "system/controlDict")
    txt = open(ctrl).read()
    start = float(latest_time(case))
    end = start + iters
    txt = re.sub(r"^endTime         .*;$", f"endTime         {end:g};", txt, flags=re.M)
    # writeInterval must divide endTime, not equal the stage length: writeControl
    # timeStep fires on the ABSOLUTE time index, so a stage running 1501->4500
    # with writeInterval 3000 writes at 3000 and never at 4500 -- the last 1500
    # iterations are computed and then thrown away. Writing once, at endTime,
    # is always aligned.
    txt = re.sub(r"^writeInterval   .*;$", f"writeInterval   {end:g};", txt, flags=re.M)
    open(ctrl, "w").write(txt)
    log = os.path.join(case, f"log.solve.{label}")
    with open(log, "w") as f:
        if procs > 1:
            for p in glob.glob(os.path.join(case, "processor*")):
                shutil.rmtree(p)
            rc = subprocess.call(["decomposePar", "-force", "-latestTime"],
                                 cwd=case, stdout=f, stderr=f)
            if rc == 0:
                rc = subprocess.call(
                    ["mpiexec", "-n", str(procs), "buoyantSimpleFoam", "-parallel"],
                    cwd=case, stdout=f, stderr=f)
            if rc == 0:
                rc = subprocess.call(["reconstructPar", "-latestTime"],
                                     cwd=case, stdout=f, stderr=f)
            for p in glob.glob(os.path.join(case, "processor*")):
                shutil.rmtree(p)
        else:
            rc = subprocess.call(["buoyantSimpleFoam"], cwd=case,
                                 stdout=f, stderr=f)
    text = open(log).read()
    bounded = len(re.findall(r"bounding (k|epsilon)", text))
    conv = "SIMPLE solution converged" in text
    res = re.findall(r"Solving for p_rgh, Initial residual = ([\d.eE+-]+)", text)
    tail = float(res[-1]) if res else float("nan")
    print(f"  {label:>10}: exit {rc}, {latest_time(case):>6} iters, "
          f"p_rgh {tail:.2e}, {bounded} bounding events"
          + (", CONVERGED" if conv else ""), flush=True)
    return rc == 0, conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="cfd/run/base")
    ap.add_argument("--geometry", default="cfd/geometry")
    ap.add_argument("--profile", default="cfd/vent_profile.json")
    ap.add_argument("--spec", default="cfd/case_spec.json")
    ap.add_argument("--warmup", type=int, default=300,
                    help="vertical-only iterations before the ramp")
    ap.add_argument("--ramp", type=float, nargs="*",
                    default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--per-step", type=int, default=300)
    ap.add_argument("--final", type=int, default=3000)
    ap.add_argument("--procs", type=int, default=1,
                    help="MPI ranks; 1 runs serially")
    ap.add_argument("--warm-from", default=None,
                    help="converged case to initialise from; skips the ramp")
    ap.add_argument("--continue-only", action="store_true",
                    help="skip warmup and ramp: the case is already at full "
                         "throw, just run on from the latest time. Without this, "
                         "main() resets the supply to vertical first.")
    args = ap.parse_args()

    acquire(args.case)
    # The case's own stamped manifest wins. Falling back to a --geometry
    # default silently solves one layout with another's vent centres.
    stamped = os.path.join(args.case, "constant", "patches.json")
    manifest = stamped if os.path.exists(stamped) else os.path.join(
        args.geometry, "patches.json")
    patches = json.load(open(manifest))["patches"]
    prof = json.load(open(args.profile))
    spec = json.load(open(args.spec))
    wsup = abs(spec["supply"]["w_m_s"])
    tsup = spec["supply"]["T_K"]

    if args.warm_from:
        log = os.path.join(args.case, "log.warmstart")
        t = warm_start(args.case, args.warm_from, log)
        print(f"warm-started {args.case} from {args.warm_from} @ {t}", flush=True)
        set_supply(args.case, latest_time(args.case), patches, prof, wsup, tsup, 1.0)
        ok, conv = run(args.case, args.final, "warm", args.procs)
        print("done" + (" (converged)" if conv else " (hit iteration cap)"))
        return 0 if ok else 1

    if args.continue_only:
        set_supply(args.case, latest_time(args.case), patches, prof, wsup, tsup, 1.0)
        ok, conv = run(args.case, args.final, "continue", args.procs)
        print("done" + (" (converged)" if conv else " (hit iteration cap)"))
        return 0 if ok else 1

    print(f"ramping the diffuser throw in {args.case}")
    set_supply(args.case, latest_time(args.case), patches, prof, wsup, tsup, 0.0)
    ok, _ = run(args.case, args.warmup, "vertical", args.procs)
    if not ok:
        return 1
    for f in args.ramp:
        set_supply(args.case, latest_time(args.case), patches, prof, wsup, tsup, f)
        ok, conv = run(args.case, args.per_step, f"radial{f:g}", args.procs)
        if not ok:
            print(f"  diverged at radial fraction {f}", flush=True)
            return 1
    ok, conv = run(args.case, args.final, "final", args.procs)
    print("done" + (" (converged)" if conv else " (hit iteration cap)"), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
