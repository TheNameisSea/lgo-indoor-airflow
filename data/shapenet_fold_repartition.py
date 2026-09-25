#!/usr/bin/env python
"""Build a published ShapeNet-Car fold from the already-exported cases.

The nine published folds partition the same 889 cases nine ways
(`data/splits_shapenet.param_fold`). Fold 0 is exported to
`splits_shapenet_matched_csv/`; any other fold is built here as a tree of
symlinks into that export, so no case is exported twice.

    python data/shapenet_fold_repartition.py --fold_id 1
    python data/shapenet_fold_repartition.py --fold_id 1 --check
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SOURCE = os.path.join(ROOT, "splits_shapenet_matched_csv")


def source_index(source=SOURCE):
    """case id -> the directory holding its CSVs, wherever it currently sits."""
    idx = {}
    for share in ("train", "val", "test"):
        d = os.path.join(source, share)
        if not os.path.isdir(d):
            continue
        for case in os.listdir(d):
            p = os.path.join(d, case)
            if os.path.isdir(p):
                idx[case] = p
    return idx


def build(fold_id, source=SOURCE, verbose=True):
    from data.splits_shapenet import PUBLISHED_FOLD_SIZES, param_fold

    assignment, sizes, _ = param_fold(fold_id=fold_id)
    idx = source_index(source)

    # Every case in the fold must already be exported. A missing one means the
    # export and the manifest disagree, which is exactly the failure that
    # `--manifest` auto-selection in shapenet_export.py exists to prevent --
    # refuse rather than silently build a short split.
    missing = sorted(set(assignment) - set(idx))
    if missing:
        raise SystemExit(
            f"[repartition] REFUSING: {len(missing)} of {len(assignment)} cases "
            f"are not exported under {source} (first few: {missing[:5]}). "
            f"The export and the published manifest disagree.")

    out_root = os.path.join(ROOT, f"splits_shapenet_fold{fold_id}_csv")
    n = {"train": 0, "val": 0}
    for case, share in assignment.items():
        d = os.path.join(out_root, share)
        os.makedirs(d, exist_ok=True)
        link = os.path.join(d, case)
        if os.path.islink(link) or os.path.exists(link):
            os.unlink(link)
        os.symlink(os.path.relpath(idx[case], d), link)
        n[share] += 1

    expect_val = PUBLISHED_FOLD_SIZES[fold_id]
    if n["val"] != expect_val:
        raise SystemExit(
            f"[repartition] REFUSING: fold {fold_id} val is {n['val']}, the "
            f"published size is {expect_val}. The fold reproduction is wrong.")

    manifest = os.path.join(ROOT, "data", f"splits_shapenet_fold{fold_id}.json")
    with open(manifest, "w") as f:
        json.dump(assignment, f, indent=1, sort_keys=True)

    if verbose:
        print(f"[repartition] fold {fold_id}: {n['train']} train / {n['val']} val "
              f"(published val size {expect_val}) -> {out_root}")
        print(f"[repartition] manifest {manifest}")
        print(f"[repartition] symlinks into {source} -- no bytes copied")
    return out_root, n


def check(fold_id, source=SOURCE):
    """The links resolve, the shares are disjoint, and nothing was copied."""
    out_root = os.path.join(ROOT, f"splits_shapenet_fold{fold_id}_csv")
    seen, broken, real = {}, [], 0
    for share in ("train", "val"):
        d = os.path.join(out_root, share)
        for case in sorted(os.listdir(d)):
            p = os.path.join(d, case)
            if not os.path.islink(p):
                real += 1
            if not os.path.exists(p):
                broken.append(p)
            if case in seen:
                raise SystemExit(f"[repartition] {case} is in BOTH shares")
            seen[case] = share
    print(f"[repartition] fold {fold_id}: {len(seen)} cases, {len(broken)} broken "
          f"links, {real} non-symlink entries")
    if broken:
        raise SystemExit(f"[repartition] broken: {broken[:5]}")
    # A case must carry the same CSVs the loader expects.
    any_case = os.path.join(out_root, "val", sorted(os.listdir(
        os.path.join(out_root, "val")))[0])
    files = sorted(os.listdir(any_case))
    print(f"[repartition] sample case carries {len(files)} files: {files[:3]} ...")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_id", type=int, default=1)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--check", action="store_true",
                    help="verify an existing re-partition instead of building")
    a = ap.parse_args()
    if a.check:
        check(a.fold_id, a.source)
    else:
        build(a.fold_id, a.source)
        check(a.fold_id, a.source)
