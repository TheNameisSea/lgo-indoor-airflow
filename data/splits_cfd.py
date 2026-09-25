#!/usr/bin/env python
"""Train/val/test split for the generated CFD dataset in `cfd/dataset/`.

Kept separate from the split over the prior
project's STAR-CCM+ cases and is left untouched so its numbers stay
reproducible. Nothing here reads that dataset.

## Why the assignment is a hash and not a shuffle

The sweep produces cases over hours, so the set of exported cases grows while
work is already under way. A shuffle-and-slice split reassigns cases every time
the set changes -- which silently moves a case from test into train between two
runs, the exact failure that makes a held-out number meaningless.

Each case is therefore assigned by hashing **its own name**:

    bucket = sha1(f"{TIER}:{name}") % 1000    ->  <150 test, <300 val, else train

The assignment of any case depends on nothing but its name, so adding tier C
later cannot move a single case that is already placed.

## Layouts

In the prior project's split several cases share one FCU layout, and the split has to
enforce layout-disjointness explicitly. Here **every case is a distinct layout**
by construction -- the layout is what the sweep varies -- so disjointness is
automatic, and the check that remains is simply that no name repeats.

## Tiers

    A   reference grid, dx = 0                 9 layouts
    Am  reference grid translated, dx = +-0.75 m    18
    B   Sobol over the widened continuous box       80
    C   Sobol + independent per-vent jitter         20

Stratifying by tier keeps each of them represented in all three splits, so a
held-out score is not accidentally a score on one tier.

Run:
    python data/splits_cfd.py                 # summary only
    python data/splits_cfd.py --materialize   # write splits_cfd/{train,val,test}
"""

import argparse
import hashlib
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "cfd", "dataset")

TEST_PCT, VAL_PCT = 150, 300      # per mille: test < 150, val < 300, else train
TIERS = ("A", "Am", "B", "C")

# Exported directories that are not sweep layouts, with the reason kept next to
# the exclusion.
EXCLUDED = {
    "base": "development case used while building the pipeline; not one of the "
            "127 designed layouts",
}


def tier_of(name):
    """Which design tier a case name belongs to."""
    if name.startswith("B"):
        return "B"
    if name.startswith("C"):
        return "C"
    m = re.match(r"s\d+_r\d+_dx([mp])(\d+)$", name)
    if m:
        return "A" if int(m.group(2)) == 0 else "Am"
    raise ValueError(f"cannot classify case name {name!r}")


def split_of(name):
    """'train' | 'val' | 'test', from the case name alone."""
    key = f"{tier_of(name)}:{name}".encode()
    bucket = int(hashlib.sha1(key).hexdigest(), 16) % 1000
    if bucket < TEST_PCT:
        return "test"
    if bucket < VAL_PCT:
        return "val"
    return "train"


def discover(dataset_dir=None):
    """Exported cases on disk, as {split: [names]}. A case counts as exported
    only once it has the interior file the loader needs."""
    d = dataset_dir or DATASET_DIR
    names = sorted(n for n in os.listdir(d)
                   if os.path.isdir(os.path.join(d, n))
                   and n not in EXCLUDED
                   and os.path.exists(os.path.join(d, n, "Fluid_data.csv")))
    if len(names) != len(set(names)):
        raise ValueError("duplicate case names -- layouts would not be disjoint")
    out = {"train": [], "val": [], "test": []}
    for n in names:
        out[split_of(n)].append(n)
    return out


def summary(dataset_dir=None, verbose=True):
    sp = discover(dataset_dir)
    total = sum(len(v) for v in sp.values())
    if verbose:
        print(f"{total} exported cases in {dataset_dir or DATASET_DIR}\n")
        print(f"  {'':6} {'train':>6} {'val':>6} {'test':>6} {'all':>6}")
        for t in TIERS:
            row = [sum(1 for n in sp[s] if tier_of(n) == t)
                   for s in ("train", "val", "test")]
            if sum(row):
                print(f"  {t:6} {row[0]:6} {row[1]:6} {row[2]:6} {sum(row):6}")
        print(f"  {'total':6} {len(sp['train']):6} {len(sp['val']):6} "
              f"{len(sp['test']):6} {total:6}")
        print(f"\n  test cases: {' '.join(sp['test'])}")
    return sp


def materialize(root=None, dataset_dir=None, verbose=True):
    """Create splits_cfd/<split>/<case> symlinks. Idempotent."""
    root = root or os.path.join(REPO_ROOT, "splits_cfd")
    d = dataset_dir or DATASET_DIR
    sp = discover(d)
    for split, names in sp.items():
        sub = os.path.join(root, split)
        os.makedirs(sub, exist_ok=True)
        # Drop links whose case is no longer exported, so the tree cannot go stale.
        for stale in os.listdir(sub):
            if stale not in names:
                os.unlink(os.path.join(sub, stale))
        for n in names:
            link = os.path.join(sub, n)
            if os.path.islink(link):
                os.unlink(link)
            os.symlink(os.path.join(d, n), link)
        if verbose:
            print(f"  splits_cfd/{split}: {len(names)} links")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=None)
    ap.add_argument("--materialize", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    summary(a.dataset_dir)
    if a.materialize:
        print()
        materialize(a.root, a.dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
