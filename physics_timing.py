#!/usr/bin/env python
"""Test 3 -- inference cost, for one run.

    $PY physics_timing.py --run runs/<run> --split splits_cfd_gap/val \
        --out results/physics_timing_<run>.json

REQUIRED BEFORE THE WORD "SURROGATE" IS USED. This project does not otherwise
have a timing measurement: the inherited 3.6 s figure is different hardware and
a different model and does not transfer.

THE PROTOCOL S3.5.3 ASKS FOR, AND WHY EACH PART IS THERE:

  * documented hardware, driver, torch build -- recorded into the JSON, because
    a speed-up claim is a claim about a measurement;
  * an explicit WARM-UP -- the first call pays cuDNN autotuning and lazy CUDA
    context creation, and including it inflates the number several-fold;
  * REPEATED timed inference at a FIXED query count -- median and IQR over
    `--repeats`, not a single sample;
  * cost broken out PER STAGE -- case setup (the KD-tree index, once per case),
    geometry encoding (once per case), and query decoding (per query point), so
    the addon's share is visible rather than inferred.

HOW THE LOCAL BRANCH'S SHARE IS ISOLATED. Not by toggling `model.use_local`:
mutating that after construction bypasses the `__init__` guard and crashes on
`lgo_ginot`. Instead the ablation pairs do it -- a hybrid and its
baseline are bit-identical apart from the branch, so the DIFFERENCE of their
decode times is the branch's cost, measured rather than instrumented.

WALL CLOCK, NOT CUDA EVENTS. The local branch gathers its neighbours through
scipy on the CPU, so a GPU-only timer would be blind to a real part of the cost.
`torch.cuda.synchronize()` brackets each region and `perf_counter` measures it.
That also means the CPU side is exposed to whatever else is on the box: the
system load average is recorded alongside every number.
"""

import argparse
import glob
import json
import os
import statistics as st
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

import evaluate as ev  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    sync()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t0)
    return out


def summarise(samples):
    s = sorted(samples)
    q1, q3 = s[len(s) // 4], s[(3 * len(s)) // 4]
    return {"median_s": st.median(s), "min_s": s[0], "max_s": s[-1],
            "iqr_s": q3 - q1, "n": len(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="splits_cfd_gap/val")
    ap.add_argument("--out", default="results/physics_timing.json")
    ap.add_argument("--queries", type=int, nargs="+", default=[10000, 100000])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=16384)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip().split("\n")[0],
        "cpu_count": os.cpu_count(),
        "knn_workers": os.environ.get("LGO_KNN_WORKERS", "unset"),
        "loadavg_at_start": open("/proc/loadavg").read().split()[:3],
        "note": ("Shared machine. The GPU carried no other process for this run; "
                 "the CPU is exposed to other users, which the load average records."),
    }

    a = json.load(open(os.path.join(args.run, "args.json")))
    local = os.path.join(args.run, "thermo_stats.pt")
    if os.path.exists(local):
        a["stats_path"] = local

    from thermo.dataset import ThermoDataset
    folders = sorted(glob.glob(os.path.join(args.split, "*/")))[:1]   # one case is enough
    ds = ThermoDataset(case_dirs=[f.rstrip("/") for f in folders],
                       saved_stats=a["stats_path"], stats_path=a["stats_path"],
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    model, a = ev.build_from_args(args.run, ds, device)
    uses_branch = bool(getattr(model, "use_local", False))

    # --- stage 1: per-case setup (the KD-tree index). Branch arms only. -------
    setup = None
    if uses_branch or getattr(model, "use_hbc", False):
        t0 = time.perf_counter()
        model.register_cases(ds, verbose=False,
                             normals_k=a.get("normals_k", 16),
                             with_normals=not a.get("no_normals", False))
        setup = {"median_s": time.perf_counter() - t0, "n": 1,
                 "note": "one-off per case, not per query; single sample"}

    room = ds.cases[0]
    pc = room["pc_full"].unsqueeze(0).to(device)

    # --- stage 2: geometry encoding, once per case ---------------------------
    with torch.no_grad():
        enc = summarise(timed(lambda: model.encode_geometry(pc),
                              args.repeats, args.warmup))
        lat = model.encode_geometry(pc)

        # --- stage 3: query decoding, at each fixed query count --------------
        xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
        dec = {}
        for nq in args.queries:
            if nq > len(xyz_n):
                continue
            q = xyz_n[:nq].to(device)

            def run_decode(q=q):
                return torch.cat([
                    model.decode_query(lat, q[s:s + args.chunk].unsqueeze(0)).squeeze(0)
                    for s in range(0, len(q), args.chunk)], 0)

            s = summarise(timed(run_decode, args.repeats, args.warmup))
            s["queries"] = nq
            s["us_per_query"] = s["median_s"] / nq * 1e6
            s["queries_per_s"] = nq / s["median_s"]
            dec[str(nq)] = s

    out = {"run": args.run, "split": args.split, "model": a.get("model"),
           "uses_local_branch": uses_branch, "hardware": hw,
           "chunk": args.chunk, "repeats": args.repeats, "warmup": args.warmup,
           "case_setup": setup, "geometry_encoding": enc, "query_decoding": dec,
           "cfd_reference_s": {"median": 24 * 60, "mean": 28 * 60, "max": 70 * 60,
                               "note": "CFD solve time over the 127-case set"}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    print(f"\n=== {a.get('model')}  ({hw['device']}) ===")
    if setup:
        print(f"  case setup (KD-tree index) : {setup['median_s']:8.3f} s   once per case")
    print(f"  geometry encoding          : {enc['median_s']*1e3:8.2f} ms  once per case"
          f"   (IQR {enc['iqr_s']*1e3:.2f} ms)")
    for nq, s in dec.items():
        print(f"  decode {int(nq):>7,} queries    : {s['median_s']*1e3:8.2f} ms"
              f"   {s['us_per_query']:6.2f} us/query   {s['queries_per_s']/1e3:8.1f} k q/s")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
