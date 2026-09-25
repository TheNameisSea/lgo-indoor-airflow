"""Vent extraction: inlet cloud -> per-vent anchors and conditioning vectors.

Implements PLAN_EQUIJET_UPGRADE.md §3.1. Everything is computed in PHYSICAL units
(metres, m/s, kelvin): the dataset's coordinates are per-axis min-max normalized
and its targets are z-scored, so a local offset or a speed is only meaningful once
de-normalized.

Per vent k, the conditioning vector is

    c_k = [s_k, d_k(3), a_k, dT_k, Ri_k]        (cond_dim = 7)

with s_k the mean inlet speed, d_k the mean unit direction, a_k an area proxy,
dT_k = T_in,k - T_ref the supply-temperature deficit and Ri_k = g*beta*dT_k*L/s_k^2
the Richardson number (beta = 1/T_ref, ideal gas; L = vent width).

MEASURED DEVIATIONS FROM THE PLAN — both from running the numbers on all 18 cases:

1. **K = 3 supply jets, after filtering return-patch artifacts.** Raw clustering of
   the dataset's inlet pool returns SIX clusters, but only three are supply vents.
   Measured on Case_10:

       3 clusters at y=2.30, n=267 each, mean w = -1.160 m/s   <- real supply jets
       3 clusters at y~3.78, n=28-56,    mean w = -0.08..-0.13 m/s
       3 OUTLET clusters at y~3.86, n=211-239, mean w = +0.87 m/s

   The weak y~3.78 clusters are **fringe points of the RETURN patches**: the loader
   splits `New_HVAC.csv` purely by the sign of `Velocity[k]`
   (`pi_ginot/dataset.py`), so the minority of return-patch points with slightly
   negative w land in the inlet pool. They are co-located with the outlets, not
   separate vents. Anchoring jet templates there would place jets at return
   locations with near-zero speed conditioning — wrong physics and wasted capacity
   (plan: "outlets are sinks, not jets ... handled by R").

   `extract_vents` therefore drops inlet clusters lying within `outlet_exclude_m`
   of any outlet point. Verified: K = 3 on all 16 train+val cases, with per-vent
   speeds falling into exactly the two fan-speed groups (0.58 / 1.16 m/s, the
   documented ~2x ratio). `n_vents_expected` defaults to 3.

2. **cluster threshold 0.7 m, not the plan's 0.5 m.** Measured cluster count vs
   single-linkage threshold over all 18 cases:

       thr   0.4    0.5    0.6    0.8    1.0
       K     6-12   6-10   6      6      3-6

   At the plan's 0.5 m, **8 of 18 cases give K = 7, 9 or 10** and the K assertion
   would fire. K = 6 is stable across every case only on the 0.6-0.8 m plateau; at
   1.0 m the y-pairs merge and K collapses to 3. The default is the plateau
   midpoint, 0.7 m.

3. **dT is very nearly constant** (measured T_in = 286.42-287.45 K over 18 cases,
   std 0.36 K), exactly the risk flagged in plan §5. It is retained in `c_k` for
   forward-compatibility but is not expected to carry signal. `Ri` is therefore
   driven almost entirely by SPEED, which does vary 2.1x between the fan-speed
   groups and is read from the dataset rather than the CSV.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage

G = 9.81
COND_DIM = 7

# Same rule the dataset loader uses to identify supply (inlet) rows.
_COL_VK = "Velocity[k] (m/s)"
_COL_T = "Temperature (K)"
_COLS_XYZ = ["X (m)", "Y (m)", "Z (m)"]


def _read_hvac(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def build_temperature_table(case_dirs, verbose=True):
    """Map mean-inlet-speed -> mean supply temperature, over every case folder.

    Keyed by SPEED rather than by folder order on purpose: the dataset does not
    retain the source folder of a room, `os.scandir` ordering is not guaranteed,
    and reflection-augmented rooms have no folder at all. Mean inlet speed is
    invariant under reflection and is distinct per case here (measured 0.509-1.143
    m/s across the 18 cases), so it identifies a case robustly.
    """
    if isinstance(case_dirs, str):
        folders = sorted(glob.glob(os.path.join(case_dirs, "*/")))
    else:
        folders = list(case_dirs)

    table = []
    for folder in folders:
        path = os.path.join(folder, "New_HVAC.csv")
        if not os.path.exists(path):
            continue
        df = _read_hvac(path)
        if _COL_VK not in df.columns:
            continue
        inlet = df[df[_COL_VK] < 0]
        if len(inlet) == 0:
            continue
        # Keep only STRONG downward flow. The weak tail is return-patch fringe that
        # the sign rule misfiles as inlet; including it biases the case speed
        # (measured 0.54 vs the true 0.58) far enough to break the speed-keyed
        # lookup used below. This mirrors the outlet filter in `extract_vents`.
        w = inlet[_COL_VK].abs()
        inlet = inlet[w >= 0.5 * w.max()]
        vel = inlet[["Velocity[i] (m/s)", "Velocity[j] (m/s)", _COL_VK]].values
        speed = float(np.linalg.norm(vel, axis=1).mean())
        if _COL_T in df.columns and inlet[_COL_T].notna().any():
            t_in = float(inlet[_COL_T].mean())
        else:
            warnings.warn(f"{folder}: no usable '{_COL_T}' column; dT will be 0.")
            t_in = None
        table.append({"folder": folder, "speed": speed, "t_in": t_in})

    if verbose and table:
        temps = [r["t_in"] for r in table if r["t_in"] is not None]
        if temps:
            print(f"[equijet] supply temperature over {len(temps)} case folders: "
                  f"mean={np.mean(temps):.2f} K, std={np.std(temps):.2f} K, "
                  f"range=[{min(temps):.2f}, {max(temps):.2f}]")
            if np.std(temps) < 1.0:
                print("[equijet] NOTE: supply temperature is ~constant across cases, "
                      "so dT/Ri carry little between-case signal (plan §5 risk).")
    return table


def lookup_t_in(table, speed, tol=0.02):
    """Nearest-speed lookup into the temperature table. Returns None if no match."""
    best, best_d = None, float("inf")
    for row in table:
        d = abs(row["speed"] - speed)
        if d < best_d:
            best, best_d = row, d
    if best is None or best_d > tol * max(1e-6, speed) + tol:
        return None
    return best["t_in"]


def extract_vents(room, coord_min, coord_scale, target_mean, target_std,
                  cluster_m=0.7, n_expected=3, t_table=None, t_ref=294.0,
                  vent_width_m=0.6, outlet_exclude_m=0.5):
    """Cluster one room's inlet cloud into vents. Returns (anchors, cond).

    anchors : (K, 3) float32, metres
    cond    : (K, COND_DIM) float32 = [s, d(3), a, dT, Ri]
    """
    xyz_n = room["pool_in_xyz"]
    tgt_n = room["pool_in_tgt"]
    if len(xyz_n) == 0:
        raise ValueError("room has no inlet points; cannot build vent templates")

    # -> physical units
    xyz = (xyz_n * coord_scale + coord_min).cpu().numpy().astype(np.float64)
    vel = (tgt_n[:, 0:3] * target_std[0:3] + target_mean[0:3]).cpu().numpy().astype(np.float64)

    labels = fcluster(linkage(xyz, method="single"), t=cluster_m, criterion="distance")

    # Drop clusters co-located with a RETURN patch: those are sign-rule artifacts,
    # not supply vents (see module docstring). Done before the K assertion so the
    # assertion counts real jets.
    keep = []
    out_xyz = room.get("pool_out_xyz")
    if out_xyz is not None and len(out_xyz) > 0:
        out_m = (out_xyz * coord_scale + coord_min).cpu().numpy().astype(np.float64)
        for k in range(1, int(labels.max()) + 1):
            c = xyz[labels == k].mean(0)
            if np.linalg.norm(out_m - c, axis=1).min() > outlet_exclude_m:
                keep.append(k)
    else:
        keep = list(range(1, int(labels.max()) + 1))

    remap = {old: new + 1 for new, old in enumerate(keep)}
    mask_keep = np.isin(labels, keep)
    xyz, vel = xyz[mask_keep], vel[mask_keep]
    labels = np.array([remap[l] for l in labels[mask_keep]])

    K = int(labels.max()) if len(labels) else 0
    if n_expected is not None and K != n_expected:
        raise ValueError(
            f"vent clustering found K={K} supply jets, expected {n_expected} "
            f"(cluster_m={cluster_m}, outlet_exclude_m={outlet_exclude_m}). A wrong "
            f"K is a data bug, not a tuning knob: inspect the inlet cloud before "
            f"changing --n_vents_expected. Measured for this dataset: raw clustering "
            f"gives 6, of which 3 are return-patch artifacts; after the outlet "
            f"filter K=3 on all 16 train+val cases, stable for cluster_m in "
            f"[0.6, 0.8].")

    # Order clusters deterministically so that "vent k" means the same thing in
    # every case — the shared template depends on it.
    #
    # A plain lexsort on (x, y) is NOT tolerance-aware and gets this wrong: the two
    # vents of an FCU pair sit at the same x only to within a centroid jitter of
    # ~0.01-0.1 m, so x already separates them and the y key is never consulted,
    # leaving the pair order effectively random. Instead, group the x coordinates
    # with the same linkage threshold used for the vents themselves, order the
    # groups by mean x, then order within a group by y.
    cents = np.stack([xyz[labels == k].mean(0) for k in range(1, K + 1)])
    if K > 1:
        xg = fcluster(linkage(cents[:, :1], method="single"),
                      t=cluster_m, criterion="distance")
        gmean = {g: cents[xg == g, 0].mean() for g in np.unique(xg)}
        rank = {g: i for i, g in enumerate(sorted(gmean, key=lambda g: gmean[g]))}
        primary = np.array([rank[g] for g in xg])
    else:
        primary = np.zeros(K, dtype=int)
    order = np.lexsort((cents[:, 1], primary))

    # Supply temperature is looked up ONCE per case, from the case-mean inlet
    # speed, and shared by every vent. Keying the lookup on a per-vent speed is
    # wrong: the table is built from case-mean speeds, so vents whose own speed
    # deviates (the small patches differ from the large ones) fall outside the
    # tolerance and silently degrade to dT = 0 — measured as a spurious
    # dT = -2.19 +/- 2.28 K instead of the correct ~-7.1 K. There is no per-vent
    # temperature resolution to recover here anyway, and dT is near-constant
    # across cases, so a per-case value is both correct and sufficient.
    case_speed = float(np.linalg.norm(vel, axis=1).mean())
    t_in_case = lookup_t_in(t_table, case_speed) if t_table else None
    if t_table and t_in_case is None:
        warnings.warn(f"no temperature match for case-mean inlet speed "
                      f"{case_speed:.4f} m/s; dT and Ri set to 0 for this case.")

    anchors, cond = [], []
    for k in order:
        sel = labels == (k + 1)
        pts, v = xyz[sel], vel[sel]

        centroid = pts.mean(0)
        speed = float(np.linalg.norm(v, axis=1).mean())
        mean_v = v.mean(0)
        nv = np.linalg.norm(mean_v)
        direction = mean_v / nv if nv > 1e-9 else np.array([0.0, 0.0, -1.0])

        # Area proxy: bounding-box extent of the patch in its two widest axes.
        ext = pts.max(0) - pts.min(0)
        area = float(np.sort(ext)[-2:].prod())

        # dT and Ri. Ri is ALWAYS computed from physical speed, so it stays
        # meaningful when --nondim scales the stored targets. dT is per case;
        # Ri still varies per vent through that vent's own speed.
        if t_in_case is None:
            d_t, ri = 0.0, 0.0
        else:
            d_t = float(t_in_case - t_ref)
            beta = 1.0 / t_ref
            ri = float(G * beta * d_t * vent_width_m / max(speed, 1e-6) ** 2)

        anchors.append(centroid)
        cond.append([speed, *direction, area, d_t, ri])

    return (torch.tensor(np.stack(anchors), dtype=torch.float32),
            torch.tensor(np.array(cond), dtype=torch.float32))


def normalize_cond(cond):
    """Scale conditioning to O(1) for the FiLM inputs (plan §3.2)."""
    out = cond.clone()
    out[:, 5] = out[:, 5] / 20.0                                   # dT / ~20 K
    out[:, 6] = torch.sign(out[:, 6]) * torch.log1p(out[:, 6].abs())  # log1p(Ri)
    return out
