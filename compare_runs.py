#!/usr/bin/env python
"""Summarise runs from their `history.json` — cheap, no GPU, no re-scoring.

This reads the validation metrics the trainer already recorded (50k points per
case) so a ladder or a sweep can be read while it is still running. It is NOT a
substitute for `evaluate.py`, which scores every interior CFD point through the
prior project's protocol and is what any reported number must come from: the
trainer's validation is a subsample, and the prior project measured that a
train-vs-held-out gap read off subsamples is partly artifact (their weakest
model moved by 0.12 between the two protocols).

    python compare_runs.py                    # every run in runs/
    python compare_runs.py --runs runs/ladder_*  --metric vel_r2
"""

import argparse
import glob
import json
import os


def load(run):
    hp = os.path.join(run, "history.json")
    ap = os.path.join(run, "args.json")
    if not os.path.exists(hp):
        return None
    hist = json.load(open(hp))
    vals = [h for h in hist if isinstance(h.get("val"), dict)]
    if not vals:
        return None
    args = json.load(open(ap)) if os.path.exists(ap) else {}
    best = max(vals, key=lambda h: h["val"].get("score", float("-inf")))
    return {"name": os.path.basename(run.rstrip("/")), "args": args,
            "last": vals[-1], "best": best, "n_val": len(vals),
            "epoch": hist[-1]["epoch"] if hist else 0,
            "target": args.get("epochs", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--sort", default="score")
    args = ap.parse_args()

    dirs = args.runs or sorted(glob.glob("runs/*/"))
    rows = [r for r in (load(d) for d in dirs) if r]
    if not rows:
        print("no runs with validation history yet")
        return
    rows.sort(key=lambda r: -r["best"]["val"].get(args.sort, 0))

    print(f"{'run':30} {'lr':>7} {'ep':>10} {'score':>7} {'v_R2':>7} {'v_rho':>6} "
          f"{'v_s':>5} {'T_mae':>6} {'T_rho':>6}")
    print("-" * 92)
    for r in rows:
        b, v = r["best"], r["best"]["val"]
        prog = f"{r['epoch']}/{r['target']}" if r["target"] else str(r["epoch"])
        print(f"{r['name']:30} {r['args'].get('lr', 0):7.0e} {prog:>10} "
              f"{v.get('score', 0):7.4f} {v.get('vel_r2', 0):+7.3f} "
              f"{v.get('vel_rho', 0):6.3f} {v.get('vel_s', 0):5.2f} "
              f"{v.get('temp_mae', 0):6.3f} {v.get('temp_rho', 0):6.3f}"
              f"{'' if r['epoch'] >= r['target'] else '   (running)'}")
    print(f"\nbest epoch per run is selected on joint Taylor `score`, the same rule "
          f"the trainer checkpoints on.")


if __name__ == "__main__":
    main()
