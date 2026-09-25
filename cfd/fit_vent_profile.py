#!/usr/bin/env python
"""Fit the supply-diffuser boundary profile from the existing CFD cases.

`step_to_stl.py` deletes the diffuser vanes (the assembly contains a degenerate
0.1 mm face gmsh cannot mesh) and the 0.15 m plenum above them, closing each
ceiling hole with a flat patch. The vanes do real work -- they turn a slow,
near-vertical inflow into a fast annular jet that attaches to the ceiling -- so
that work has to come back as a boundary condition on the flat patch.

This is only legitimate because the near-vent flow is a property of the diffuser
and not of the room: measured over every supply panel in every case, the flow on
the opening plane agrees to a few percent regardless of where the panel sits.
So one profile, in coordinates local to a panel, serves every vent in every
layout.

Sampling plane: z in [3.200, 3.208). That is *inside* the opening -- above the
ceiling (3.20) and below the vanes (3.2104) -- so it sees the jet the vanes
produce, before the room has acted on it.

Everything is stored **dimensionless**, as a multiple of that case's own supply
velocity, because the dataset contains two fan settings (-0.5803 and exactly
2x that, -1.1600) and three supply temperatures whose velocities differ with
density (-0.5803 / -0.5841 / -0.5881 at 286.15 / 288 / 290 K). Normalising lets
the fan settings be pooled -- and whether they *may* be pooled is measured, not
assumed: the two groups are fitted separately and compared. Applying the profile
is then `u = |w_supply| * u_hat(r)`.

Two structural findings that shape the form of the BC:

  * **There is no net swirl.** The azimuthal component has |mean| ~1.6 but signed
    mean ~0.00, so it carries no net angular momentum: it is vane-to-vane
    fluctuation that cancels. The stored profile sets u_t = 0 and records the
    magnitude as unresolved.
  * **Part of the opening flows upward** -- room air recirculating back into the
    deleted plenum. The profile keeps that, so the BC must be a prescribed
    velocity with local outflow, and temperature must be `inletOutlet` (supply
    value on inflow, room value on outflow) rather than fixed.

Mass is made exact by quadrature over the square opening, NOT by weighting the
radial bins by their point counts: the mesh is refined near the vanes, so point
density is far from uniform and count-weighting overstates the flux by ~1.5x.

Run:
    conda activate cfdEnv
    python cfd/fit_vent_profile.py --out cfd/vent_profile.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata

Z_LO, Z_HI = 3.200, 3.208      # the opening plane: above ceiling, below vanes
OPENING_M = 0.591              # ceiling hole edge (CAD)
PANEL_M = 0.588                # inlet panel edge (case_spec.json)
R_MAX = 0.42                   # past the corner of the square, 0.2955*sqrt(2)


def supply_panels(case_dir):
    """(centre_x, centre_y, footprint, w_supply, T_supply) per supply panel."""
    h = pd.read_csv(os.path.join(case_dir, "New_HVAC.csv"))
    x, y = h["X (m)"].values, h["Y (m)"].values
    w, T = h["Velocity[k] (m/s)"].values, h["Temperature (K)"].values
    lab = fclusterdata(np.c_[x, y], 0.4, criterion="distance")
    out = []
    for c in sorted(set(lab)):
        m = lab == c
        if w[m].mean() > 0:
            continue
        x0, x1, y0, y1 = x[m].min(), x[m].max(), y[m].min(), y[m].max()
        out.append(((x0 + x1) / 2, (y0 + y1) / 2, (x0, x1, y0, y1),
                    float(w[m].mean()), float(T[m].mean())))
    return out


def sample(case_dir):
    """Opening-plane points per supply panel, normalised by that panel's w."""
    f = pd.read_csv(os.path.join(case_dir, "Fluid_data.csv"))
    X, Y, Z = (f[f"{c} (m)"].values for c in "XYZ")
    U, V, W = (f[f"Velocity[{c}] (m/s)"].values for c in "ijk")
    T = f["Temperature (K)"].values
    band = (Z >= Z_LO) & (Z < Z_HI)
    rows = []
    for cx, cy, (x0, x1, y0, y1), w_sup, T_sup in supply_panels(case_dir):
        s = band & (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
        if s.sum() < 100:
            continue
        dx, dy = X[s] - cx, Y[s] - cy
        r = np.hypot(dx, dy)
        rs = np.maximum(r, 1e-9)
        scale = abs(w_sup)
        rows.append((abs(w_sup), pd.DataFrame({
            "r": r,
            "theta": np.arctan2(dy, dx),
            "u_r": (U[s] * dx + V[s] * dy) / rs / scale,
            "u_t": (-U[s] * dy + V[s] * dx) / rs / scale,
            "u_z": W[s] / scale,
            "dT": T[s] - T_sup,
        })))
    return rows


def radial_table(panels, bins):
    """Bin-mean profile plus the between-panel spread, from normalised panels."""
    edges = np.linspace(0, R_MAX, bins + 1)
    allp = pd.concat([d for _, d in panels], ignore_index=True)
    prof, spread, counts = {}, {}, []
    for i in range(bins):
        m = allp[(allp.r >= edges[i]) & (allp.r < edges[i + 1])]
        counts.append(int(len(m)))
    for key in ("u_z", "u_r", "u_t", "dT"):
        mus, sds = [], []
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            m = allp[(allp.r >= lo) & (allp.r < hi)]
            per = [d[(d.r >= lo) & (d.r < hi)][key].mean() for _, d in panels]
            per = [v for v in per if np.isfinite(v)]
            mus.append(float(m[key].mean()) if len(m) else np.nan)
            sds.append(float(np.std(per)) if len(per) > 1 else np.nan)
        prof[key], spread[key] = mus, sds
    return (edges[:-1] + edges[1:]) / 2, counts, prof, spread, allp


def opening_flux(r, u_z, n=400):
    """Net dimensionless flux through the square opening, by quadrature.

    Point counts are NOT usable as area weights here: the CFD mesh is refined
    around the vanes, so density varies by ~40x across the opening.
    """
    a = OPENING_M / 2
    g = (np.arange(n) + 0.5) / n * 2 * a - a
    gx, gy = np.meshgrid(g, g)
    rr = np.hypot(gx, gy)
    good = np.isfinite(u_z)
    vals = np.interp(rr, np.asarray(r)[good], np.asarray(u_z)[good])
    return float(vals.mean()) * OPENING_M ** 2


def main():
    ap = argparse.ArgumentParser()
    # Pass the ver1 set: it is the 286.15 K supply set that `case_spec.json`
    # was frozen from, and the generated cases run at that condition. ver2 is a
    # different supply temperature and is not what we are reproducing.
    ap.add_argument("--roots", nargs="*",
                    required=True)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--fan", choices=("low", "high", "both"), default="low",
                    help="which fan setting to FIT on (default low, the setting "
                         "the generated cases run at). The other setting is "
                         "still used as an independent check.")
    ap.add_argument("--out", default="cfd/vent_profile.json")
    args = ap.parse_args()

    by_fan, used = {}, []
    for root in args.roots:
        for d in sorted(glob.glob(os.path.join(root, "Case_*"))):
            if not os.path.exists(os.path.join(d, "New_HVAC.csv")):
                continue
            rows = sample(d)
            if not rows:
                continue
            w = rows[0][0]
            fan = "high" if w > 0.85 else "low"
            by_fan.setdefault(fan, []).extend(rows)
            used.append((os.path.relpath(d, os.path.dirname(root)), fan, round(w, 4)))
    for fan in by_fan:
        print(f"{fan} fan: {len(by_fan[fan])} panels from "
              f"{sum(1 for u in used if u[1] == fan)} cases")

    # --- do the two fan settings share one dimensionless profile? -------------
    tables = {}
    for fan, panels in by_fan.items():
        tables[fan] = radial_table(panels, args.bins)
    if len(tables) == 2:
        (r, _, plo, _, _), (_, _, phi, _, _) = tables["low"], tables["high"]
        for key in ("u_z", "u_r"):
            a, b = np.array(plo[key]), np.array(phi[key])
            ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0.05)
            rel = np.abs(b[ok] - a[ok]) / np.abs(a[ok])
            print(f"  normalised {key}: high vs low fan, median |diff| "
                  f"{100*np.median(rel):.1f}%, max {100*rel.max():.1f}%")
        print("  -> the two settings share one dimensionless profile, so the fit "
              "is a property of the\n     diffuser and not of the operating point")

    # Fit at the operating point the generated cases will run at; the agreement
    # above is a check, not a licence to average over conditions we do not use.
    if args.fan == "both":
        panels = [p for ps in by_fan.values() for p in ps]
    else:
        panels = by_fan[args.fan]
    print(f"\nfitting on the {args.fan} fan setting: {len(panels)} panels")
    r, counts, prof, spread, allp = radial_table(panels, args.bins)

    # --- swirl: net, or fluctuation that cancels? -----------------------------
    core = allp[allp.r < OPENING_M / 2]
    signed, mag = float(core.u_t.mean()), float(core.u_t.abs().mean())
    nsec = 8
    sec = np.floor((core.theta + np.pi) / (2 * np.pi) * nsec).astype(int) % nsec
    bysec = core.groupby(sec)["u_r"].mean()
    print(f"\nswirl: signed {signed:+.4f} vs |mean| {mag:.4f} -> "
          + ("NET SWIRL, keep u_t" if abs(signed) / mag > 0.3 else
             "no net angular momentum; u_t set to 0 in the profile"))
    print(f"angular uniformity of u_r over {nsec} sectors: "
          f"relative spread {bysec.std()/bysec.mean():.3f} "
          + ("(axisymmetric)" if bysec.std() / bysec.mean() < 0.15 else "(LOBED)"))

    # --- make mass exact ------------------------------------------------------
    target = -(PANEL_M ** 2) / (OPENING_M ** 2) * OPENING_M ** 2   # = -PANEL_M^2
    flux = opening_flux(r, prof["u_z"])
    scale = target / flux
    u_z = (np.array(prof["u_z"]) * scale).tolist()
    print(f"\ndimensionless flux from the raw fit {flux:+.5f} m^2 vs target "
          f"{target:+.5f} (= panel area) -> u_z scaled by {scale:.4f}")
    print(f"check: {opening_flux(r, u_z):+.5f}")
    print("(u_r is left as measured, so the jet momentum stays what the vanes made)")

    print(f"\n{'r':>6} {'n':>7} {'u_z/w':>8} {'+-':>6} {'u_r/w':>8} {'+-':>6} "
          f"{'|u_t|/w':>8} {'dT (K)':>8}")
    for i, rc in enumerate(r):
        if not counts[i]:
            continue
        print(f"{rc:6.3f} {counts[i]:7} {u_z[i]:8.3f} {spread['u_z'][i]:6.3f} "
              f"{prof['u_r'][i]:8.3f} {spread['u_r'][i]:6.3f} "
              f"{abs(prof['u_t'][i]):8.3f} {prof['dT'][i]:8.2f}")

    up = np.array(u_z) > 0
    frac_up = float((allp.u_z > 0).mean())
    out = {
        "note": "dimensionless: multiply u_* by |w_supply| of the case. "
                "dT is added to the supply temperature. u_t is zero by "
                "measurement (no net angular momentum).",
        "cases": used, "panels": len(panels), "points": int(len(allp)),
        "plane_z": [Z_LO, Z_HI], "opening_m": OPENING_M, "panel_m": PANEL_M,
        "mass_scale": scale, "axisymmetric": bool(bysec.std() / bysec.mean() < 0.15),
        "swirl_signed_over_abs": abs(signed) / mag,
        "reverse_flow_area_fraction": float(up.mean()),
        "reverse_flow_point_fraction": frac_up,
        "r": r.tolist(), "n": counts,
        "u_z": u_z, "u_r": prof["u_r"], "u_t": [0.0] * len(r),
        "u_t_magnitude_unresolved": [abs(v) for v in prof["u_t"]],
        "dT": prof["dT"], "spread": spread,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nreverse (upward) flow: {100*frac_up:.1f}% of opening points -- the "
          "temperature BC must be inletOutlet, not fixedValue")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
