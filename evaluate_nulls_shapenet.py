#!/usr/bin/env python
"""Zero-parameter baselines on ShapeNet-Car -- surface pressure and volume velocity.

The ShapeNet-Car counterpart of `evaluate_nulls.py`. Two nulls, neither with a
parameter:

  retrieval   copy the field of the nearest TRAINING car, where "nearest" is
              mean vertex displacement (data/splits_shapenet.py).
  train_mean  ignore the geometry: predict the per-point mean over training.

Pressure lives in the release's canonical 3,682-vertex quad-mesh ordering, which
is shared across every car (template fit), so both nulls are well defined
without the vertex coordinates.

    python evaluate_nulls_shapenet.py --split published
    python evaluate_nulls_shapenet.py --split gap --holdout test
"""

import argparse
import json
import os

import numpy as np

from data.shapenet_raw import BANDS, band_masks, mapping, surface, velocity, volume
from data.splits_shapenet import (DATASET, MANIFEST, discover,
                                  distance_matrix, published_split)


def load_pressure(cases, dataset_dir=DATASET):
    """(n, 3682) surface pressure, from the Zenodo release."""
    return np.stack([np.load(os.path.join(dataset_dir, "data", f"press_{c}.npy"))
                     for c in cases])


def load_velocity(cases):
    """(n, 29498*3) volume velocity, from the RAW release, flattened per case.

    Flattened because a null predicts the whole field and every metric below is
    computed over a case's full set of numbers -- u, v and w together, which is
    the right unit here: the freestream lives almost entirely in one component,
    so scoring components separately would flatter w and punish u for no
    physical reason. Index i is the same place on every car (correspondence is
    20.5x; see data/shapenet_raw.py), which is what makes both nulls definable.
    """
    m = mapping()
    return np.stack([velocity(c, m).reshape(-1) for c in cases])


def per_case_stats(pred, truth):
    """Per-case R2, rho, amplitude ratio s, and relative L2.

    Relative L2 is carried because it is what the ShapeNet-Car literature
    reports, and because R2 badly flatters this benchmark: a field with a large
    shape-independent common mode (surface pressure, and the freestream in the
    volume) hands any null an R2 near 0.9 while leaving the entire published
    field's worth of headroom underneath.
    """
    r2, rho, s, rl2 = [], [], [], []
    for k in range(len(truth)):
        t, p = truth[k], pred[k]
        ss = ((t - t.mean()) ** 2).sum()
        r2.append(1.0 - ((t - p) ** 2).sum() / ss)
        rho.append(np.corrcoef(t, p)[0, 1])
        s.append(p.std() / t.std())
        rl2.append(np.linalg.norm(p - t) / np.linalg.norm(t))
    return np.array(r2), np.array(rho), np.array(s), np.array(rl2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("published", "gap"), default="published")
    ap.add_argument("--field", choices=("pressure", "velocity"), default="pressure",
                    help="pressure = surface, from Zenodo; velocity = volume, "
                         "from the raw release (data/shapenet_raw.py)")
    ap.add_argument("--holdout", choices=("val", "test"), default="test")
    ap.add_argument("--dataset_dir", default=DATASET)
    ap.add_argument("--banded", action="store_true",
                    help="also report per-band metrics (velocity only). The "
                         "bands come from data.shapenet_raw.BANDS, the SAME "
                         "definition evaluate_shapenet.py uses -- if the two "
                         "banded differently every model-vs-null comparison on "
                         "this benchmark would be void. Costs one VTK parse per "
                         "held-out case.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cases = discover(a.dataset_dir)
    if a.split == "published":
        assignment = published_split(a.dataset_dir)
        holdout = "test"                     # the release has no val
        if a.holdout != "test":
            print("[nulls] the published split has no val; using test")
    else:
        if not os.path.exists(MANIFEST):
            raise SystemExit(f"[nulls] no manifest at {MANIFEST}. Run "
                             f"data/splits_shapenet.py --gap 0.045 --freeze")
        assignment = json.load(open(MANIFEST))
        holdout = a.holdout

    tr = [c for c in cases if assignment.get(c) == "train"]
    ho = [c for c in cases if assignment.get(c) == holdout]
    print(f"[nulls] split={a.split} train={len(tr)} {holdout}={len(ho)}")

    D = distance_matrix(cases, a.dataset_dir)
    idx = {c: i for i, c in enumerate(cases)}
    sub = D[np.ix_([idx[c] for c in ho], [idx[c] for c in tr])]
    nn = sub.argmin(1)
    nnd = sub.min(1)

    load = (load_pressure if a.field == "pressure"
            else lambda cs, _d=None: load_velocity(cs))
    print(f"[nulls] field={a.field}: loading ...", flush=True)
    Ptr = load(tr, a.dataset_dir)
    Pho = load(ho, a.dataset_dir)
    print(f"[nulls] train {Ptr.shape} holdout {Pho.shape}")

    preds = {"retrieval": Ptr[nn],
             "train_mean": np.tile(Ptr.mean(0), (len(ho), 1))}

    res = {"split": a.split, "field": a.field, "holdout": holdout,
           "n_train": len(tr), "n_holdout": len(ho),
           "nn_dist_median": float(np.median(nnd)),
           "nn_dist_min": float(nnd.min())}
    print(f"\n{'null':>12}{'R2 mean':>10}{'R2 med':>10}{'rho':>10}{'s':>8}"
          f"{'R2 min':>10}{'relL2':>10}")
    for name, pred in preds.items():
        r2, rho, s, rl2 = per_case_stats(pred, Pho)
        res[name] = {"r2_mean": float(r2.mean()), "r2_median": float(np.median(r2)),
                     "rho_mean": float(rho.mean()), "s_mean": float(s.mean()),
                     "r2_min": float(r2.min()),
                     "rel_l2_mean": float(rl2.mean()),
                     "per_case_r2": r2.tolist()}
        print(f"{name:>12}{r2.mean():>10.4f}{np.median(r2):>10.4f}"
              f"{rho.mean():>10.4f}{s.mean():>8.3f}{r2.min():>10.4f}"
              f"{rl2.mean()*100:>9.2f}%")

    rr = np.array(res["retrieval"]["per_case_r2"])
    res["corr_nn_dist_retrieval_r2"] = float(np.corrcoef(nnd, rr)[0, 1])
    print(f"\n[nulls] nearest-train distance: median {np.median(nnd):.4f} "
          f"min {nnd.min():.4f}")
    print(f"[nulls] corr(nn distance, retrieval R2) = "
          f"{res['corr_nn_dist_retrieval_r2']:.3f}")
    print(f"[nulls] retrieval - train_mean = "
          f"{res['retrieval']['r2_mean'] - res['train_mean']['r2_mean']:+.4f} R2")

    if a.banded and a.field == "velocity":
        print(f"\n[nulls] banding {len(ho)} held-out cases by distance to the "
              f"car surface ...", flush=True)
        mp = mapping()
        # (n_cases, n_points, 3) so a band mask can index points directly.
        T = Pho.reshape(len(ho), -1, 3)
        preds3 = {k: v.reshape(len(ho), -1, 3) for k, v in preds.items()}
        acc = {b: {k: [] for k in preds} for b, _, _ in BANDS}
        frac = {b: [] for b, _, _ in BANDS}
        for i, nid in enumerate(ho):
            vx, _ = volume(nid, mp)
            sx, _ = surface(nid, mp)
            masks, _ = band_masks(vx, sx)
            for b, mask in masks.items():
                if mask.sum() < 100:
                    continue
                frac[b].append(float(mask.mean()))
                for k in preds:
                    t = T[i][mask].reshape(-1)
                    q = preds3[k][i][mask].reshape(-1)
                    ss = ((t - t.mean()) ** 2).sum()
                    acc[b][k].append((1.0 - ((t - q) ** 2).sum() / ss,
                                      float(np.linalg.norm(q - t)
                                            / np.linalg.norm(t))))
        print(f"\n{'band':>8}{'pts':>8}{'null':>12}{'R2':>10}{'relL2':>9}")
        res["bands"] = {}
        for b, _, _ in BANDS:
            if not frac[b]:
                continue
            res["bands"][b] = {"point_frac": float(np.mean(frac[b]))}
            for k in preds:
                r2 = float(np.mean([x[0] for x in acc[b][k]]))
                rl = float(np.mean([x[1] for x in acc[b][k]]))
                res["bands"][b][k] = {"r2_mean": r2, "rel_l2_mean": rl}
                print(f"{b:>8}{np.mean(frac[b]) * 100:>7.1f}%{k:>12}"
                      f"{r2:>10.4f}{rl * 100:>8.2f}%")
        near = res["bands"].get("near", {})
        if "retrieval" in near and "train_mean" in near:
            ratio = ((1 - near["retrieval"]["r2_mean"])
                     / max(1 - near["train_mean"]["r2_mean"], 1e-12))
            print(f"\n[nulls] NEAR band retrieval/train-mean error ratio "
                  f"{ratio:.3f}  (1.000 = retrieval is worth nothing over "
                  f"ignoring geometry)")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"[nulls] -> {a.out}")


if __name__ == "__main__":
    main()
