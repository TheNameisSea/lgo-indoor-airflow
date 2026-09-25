#!/usr/bin/env python
"""Provenance exports, no compute.

Writes `exports/`:

    layout_manifest.csv   one row per DESIGNED layout (tiers A/A'/B/C/D)
    case_manifest.csv     one row per SOLVED+EXPORTED case, with its QC record
    model_config.json     architecture, per run
    training_config.json  optimiser/schedule/loss/seed/hardware, per run
    normalization.json    the exact coordinate and target transforms
    README.md             what each column means, and what is NOT here

Reads only files already on disk. No inference, no CFD, no GPU: safe to run
while the sweep holds the core quota.

    $CPY export_manifests.py

TWO THINGS THE REQUEST ASSUMES THAT ARE NOT TRUE OF THIS DATASET, both recorded
in the written manifest rather than silently encoded:

  * There are no B-parent / C-child families. `layouts()` draws tier C from the
    SAME Sobol sequence, continuing past the 80th accepted tier B point, and
    then adds per-vent jitter. Every tier C case is an independent design point.
    Measured: 0 of 20 tier C cases lie within the 0.15 m jitter magnitude of any
    tier B case (min 0.177 m, median 0.266). `parent_id` is therefore empty for
    every row, and "reserve their B parents as matched tests" has no
    referent.
  * Cell volumes, face areas and outward normals are NOT exported anywhere in
    this dataset, so every integral diagnostic is blocked. See README.
"""

import csv
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cfd"))

import sweep  # noqa: E402

OUT = os.path.join(ROOT, "exports")
DATASET = os.path.join(ROOT, "cfd", "dataset")
RUNDIR = os.path.join(ROOT, "cfd", "run")
SWEEP_LOG = os.path.join(ROOT, "cfd", "logs", "sweep_main.log")
X_MIDDLE, PANEL_SUPPLY_M, PANEL_RETURN_M = 4.46, 0.591, 0.600
ROOM = {"x": [0.0, 8.80], "y": [0.0, 6.10], "z": [0.0, 3.20]}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def solve_minutes():
    """{case: minutes} from the sweep log. Last entry wins -- the log appends."""
    if not os.path.exists(SWEEP_LOG):
        return {}
    out = {}
    for name, mins in re.findall(r"^\s+(\S+)\s+.*?t=\s*3000\s+(\d+) min",
                                 open(SWEEP_LOG).read(), re.M):
        out[name] = int(mins)
    return out


def sanity_records():
    """{case: dict} parsed from `cfd/sanity.py --all`, if a capture exists."""
    for p in (os.path.join(OUT, "sanity_all.txt"),
              os.path.join(ROOT, "results", "sanity_all.txt")):
        if os.path.exists(p):
            txt = open(p).read()
            break
    else:
        return {}
    out = {}
    pat = (r"^(\S+)\s+t=\s*\d+\s+(PASS|FAIL).*?sup\s+(\S+)\s+ret\s+(\S+)\s+"
           r"leak\s+(\S+)\s+net\s+(\S+)\s+\|U\|\s+([\d.]+)/([\d.]+)\s+"
           r"T\[([\d.]+),([\d.]+)\]\s+p_rgh\s+(\S+)\s+clip\s+(\S+)")
    for m in re.finditer(pat, txt, re.M):
        out[m.group(1)] = dict(
            sanity=m.group(2), supply_flux=float(m.group(3)),
            return_flux=float(m.group(4)), leak_flux=float(m.group(5)),
            net_flux=float(m.group(6)), speed_mean=float(m.group(7)),
            speed_max=float(m.group(8)), T_min=float(m.group(9)),
            T_max=float(m.group(10)), p_rgh=float(m.group(11)),
            clip_severity=float(m.group(12)))
    return out


def splits():
    """{case: split} for every split definition."""
    gap = {}
    p = os.path.join(ROOT, "data", "splits_cfd_gap.json")
    if os.path.exists(p):
        blob = json.load(open(p))
        gap = blob["assignment"]
        gap_dropped = set(blob.get("dropped") or [])
    else:
        gap_dropped = set()
    rnd = {}
    for s in ("train", "val", "test"):
        for d in glob.glob(os.path.join(ROOT, "splits_cfd", s, "*")):
            rnd[os.path.basename(d)] = s
    g58, g58_train_widened = {}, []
    p58 = os.path.join(ROOT, "data", "splits_cfd_gap58.json")
    if os.path.exists(p58):
        blob = json.load(open(p58))
        g58 = blob["assignment"]
        g58_train_widened = blob.get("train_widened") or []
    return rnd, gap, gap_dropped, g58, set(g58_train_widened)


def layout_rows():
    L = sweep.layouts()
    P = np.array([sweep.panel_xy(d["spread"], d["row"], d["dx"], d["offsets"])
                  for d in L])
    M = sweep.layout_distance_xy(P, P)
    np.fill_diagonal(M, np.inf)
    rnd, gap, dropped, g58, g58_tw = splits()
    mins, san = solve_minutes(), sanity_records()
    rows = []
    for i, d in enumerate(L):
        n = d["name"]
        j = int(M[i].argmin())
        off = d["offsets"]
        r = dict(
            layout_id=n, tier=d["tier"].split(" ")[0], sobol_index=i,
            spread_m=d["spread"], row_m=d["row"], dx_m=d["dx"],
            has_jitter=int(off is not None), parent_id="",
            vent_gap_m=round(abs(X_MIDDLE - d["spread"]), 4),
            feasible=int(sweep.valid(d["spread"], d["row"], d["dx"],
                                     sweep.JITTER_M if off else 0.0)),
            panel_supply_m=PANEL_SUPPLY_M, panel_return_m=PANEL_RETURN_M,
            nearest_layout=L[j]["name"],
            nearest_distance_m=round(float(M[i, j]), 4),
            split_random=rnd.get(n, ""), split_gap=gap.get(n, ""),
            gap_dropped=int(n in dropped),
            split_gap58=g58.get(n, ""),
            gap58_widened_in_train=int(n in g58_tw),
            solved=int(os.path.isdir(os.path.join(RUNDIR, n))),
            exported=int(os.path.isdir(os.path.join(DATASET, n))),
            solve_minutes=mins.get(n, ""),
            sanity=san.get(n, {}).get("sanity", ""))
        for k in range(3):
            r[f"supply{k}_x"] = round(float(P[i][k][0]), 4)
            r[f"supply{k}_y"] = round(float(P[i][k][1]), 4)
            r[f"return{k}_x"] = round(float(P[i][3 + k][0]), 4)
            r[f"return{k}_y"] = round(float(P[i][3 + k][1]), 4)
        for k in range(12):
            r[f"offset_{k}"] = (round(float(off[k]), 4) if off else "")
        rows.append(r)
    return rows


def case_rows():
    san = solve_minutes(), sanity_records()
    mins, sanr = san
    rnd, gap, dropped, g58, g58_tw = splits()
    rows = []
    for case in sorted(os.path.basename(p.rstrip("/"))
                       for p in glob.glob(os.path.join(DATASET, "*/"))):
        if case == "base":
            continue
        d = os.path.join(DATASET, case)
        fluid = os.path.join(d, "Fluid_data.csv")
        n_int = sum(1 for _ in open(fluid)) - 1 if os.path.exists(fluid) else ""
        n_bnd, n_patch = 0, 0
        for f in glob.glob(os.path.join(d, "*.csv")):
            if os.path.basename(f) == "Fluid_data.csv":
                continue
            n_patch += 1
            n_bnd += sum(1 for _ in open(f)) - 1
        r = dict(case_id=case, tier=("C" if case.startswith("C") else
                                     "D" if case.startswith("D") else
                                     "B" if case.startswith("B") else "A/Am"),
                 parent_id="", n_interior_points=n_int,
                 n_boundary_points=n_bnd, n_patches=n_patch,
                 split_random=rnd.get(case, ""), split_gap=gap.get(case, ""),
                 gap_dropped=int(case in dropped),
                 split_gap58=g58.get(case, ""),
                 gap58_widened_in_train=int(case in g58_tw),
                 solve_minutes=mins.get(case, ""),
                 fluid_md5=md5(fluid) if os.path.exists(fluid) else "")
        r.update({k: sanr.get(case, {}).get(k, "") for k in
                  ("sanity", "supply_flux", "return_flux", "leak_flux",
                   "net_flux", "speed_mean", "speed_max", "T_min", "T_max",
                   "p_rgh", "clip_severity")})
        rows.append(r)
    return rows


def write_csv(path, rows):
    if not rows:
        print(f"  (no rows for {path})")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows, {len(rows[0])} columns)")


ARCH_KEYS = ("model", "global_query", "local_query", "embed_dim",
             "branch_latent_d", "branch_width", "branch_n_point",
             "branch_radius", "cross_attn_layers", "local_heads",
             "local_hidden", "k_supply", "k_return", "k_solid", "k_leak",
             "k_solid_far", "far_voxels", "jet_bands", "jet_depth",
             "jet_hidden", "jet_radial", "jet_envelope_init",
             "jet_envelope_max", "jet_min_length", "freq_spacing", "ell_wall",
             "pos_encoding", "sdf", "hbc")
TRAIN_KEYS = ("train_dir", "val_dir", "out_dir", "lr", "epochs", "seed",
              "accum_cases", "batch", "n_colloc", "n_super", "n_pc", "pc_mode",
              "val_every", "log_every", "ckpt_every", "augment", "amp",
              "lambda_data", "lambda_p", "lambda_t", "lambda_wall",
              "lambda_inlet", "lambda_outlet", "lambda_leak", "g", "chunk_knn")


def run_configs():
    import torch
    models, trains, norms = {}, {}, {}
    for run in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "args.json"))):
        d = os.path.dirname(run)
        name = os.path.basename(d)
        a = json.load(open(run))
        models[name] = {k: a[k] for k in ARCH_KEYS if k in a}
        t = {k: a[k] for k in TRAIN_KEYS if k in a}
        for ck in ("best.pth", "last.pth"):
            p = os.path.join(d, ck)
            if os.path.exists(p):
                t[f"{ck}_md5"] = md5(p)
        h = os.path.join(d, "history.json")
        if os.path.exists(h):
            hh = json.load(open(h))
            rows = [e for e in hh if e.get("val")]
            t["epochs_completed"] = hh[-1]["epoch"]
            if rows:
                b = max(rows, key=lambda e: e["val"]["score"])
                t["best_epoch"] = b["epoch"]
                t["best_val_score"] = b["val"]["score"]
                t["selection_metric"] = ("mean of velocity and temperature "
                                         "Taylor scores, GLOBAL (~94% near-jet "
                                         "by variance)")
        t["unstated_in_args"] = {
            "hardware": "1x NVIDIA RTX A5000, 8-core cgroup quota",
            "precision": "fp32 unless --amp is set in this run's args",
            "steps_per_epoch": "= number of training cases (accum_cases=1)",
        }
        trains[name] = t
        sp = os.path.join(d, "thermo_stats.pt")
        if os.path.exists(sp):
            s = torch.load(sp, map_location="cpu", weights_only=False)
            norms[name] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                           for k, v in s.items()}
            norms[name]["_convention"] = (
                "xyz_normalised = (xyz_m - coord_min) / coord_scale ; "
                "target_physical = target_normalised * target_std + target_mean ; "
                "target channel order = (u, v, w, p_rgh, T), T_IDX = 4 ; "
                "SI units throughout (m, m/s, Pa, K)")
    return models, trains, norms


README = """# exports/ — P0 provenance package

Generated by `export_manifests.py`. Reads only files already on disk: no
inference, no CFD, no GPU.

| file | contents |
|---|---|
| `layout_manifest.csv` | every DESIGNED layout (A/A'/B/C/D), its parameters, its six vent centres, feasibility, nearest other layout, split assignment, solved/exported/QC status |
| `case_manifest.csv` | every SOLVED and EXPORTED case, point counts, both
split assignments, full sanity record, `Fluid_data.csv` checksum |
| `model_config.json` | architecture per run |
| `training_config.json` | optimiser/schedule/loss/seed, checkpoint md5s, best
epoch and the metric it was selected on |
| `normalization.json` | exact coordinate and target transforms, channel order, units |

## Three things worth knowing about this dataset

**1. There are no B-parent / C-child families.** It is tempting to "keep
B-parent/C-child families out of training together" and to "reserve their B
parents as matched tests"; the note above P2 says "if all 20 C layouts are
reserved, reserve their B parents as matched tests too". No such relationship
exists. `cfd/sweep.py:layouts()` draws tier C from the **same Sobol sequence**,
continuing past the 80th accepted tier B point, and then adds per-vent jitter.
Each tier C case is an independent design point that happens to be irregular.

Measured: **0 of 20** tier C cases lie within the 0.15 m jitter magnitude of any
tier B case (min 0.177 m, median 0.266, max 0.517). `parent_id` is present but
empty in both manifests rather than being invented.

The related-layout leakage the request is trying to prevent is real, but it is
not a family relationship — it is proximity in a continuous design space, and it
is handled by the gap split (`data/splits_cfd_gap.py`), whose guarantee is that
every cross-split pair is >= 0.30 m apart in matched panel displacement.
`nearest_layout` / `nearest_distance_m` in `layout_manifest.csv` let you audit
that directly for any proposed split.

**2. Every integral diagnostic is blocked, and not by effort.** Wall
compliance, per-opening flux and local continuity all need
face areas `A_f`, outward normals `n_f` and cell volumes `V_c`. The exported
CSVs contain only `Velocity[i,j,k]`, `Pressure`, `Temperature` and `X,Y,Z` — no
area, no normal, no volume, for any patch or cell. The request already forbids
the obvious workaround ("CSV row counts are not integration weights"), and it is
right to: this is a boundary-refined mesh, so point density varies by orders of
magnitude across a patch.

Fixing it requires an OpenFOAM post-processing pass over every case, not a
script against the CSVs. Until it is run, none of those three numbers can be
produced honestly. The gradient comparison is partly available today
(`evaluate.py` already
reports relative L2, R2, per-zone and per-case statistics with pre-declared
worst cases) but its volume-weighted variants are blocked for the same reason.

**3. `solve_minutes` is per case but the sweep log APPENDS across runs.** Where
a case was solved more than once the last value wins. Two tier D layout names
(`D001`, `D002`, `D003`) appear in `cfd/logs/sweep/*.log` against parameters that
no longer match those names, because tier D was regenerated after a
vent-overlap bug; only entries after the final regeneration are meaningful. The
manifests are generated from the CURRENT `layouts()`, so they are self-consistent.

## Units and conventions

SI throughout: metres, m/s, Pa (`p_rgh`), K. Room bounds
x [0, 8.80], y [0, 6.10], z [0, 3.20] m. Vent panels are 0.591 m (supply) and
0.600 m (return) square. Supply blows down; returns draw up. `vent_gap_m` is
`|4.46 - spread|`, the centre-to-centre distance between adjacent supplies,
which must stay >= 0.600 m or the panels overlap.
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    print("layout and case manifests")
    write_csv(os.path.join(OUT, "layout_manifest.csv"), layout_rows())
    write_csv(os.path.join(OUT, "case_manifest.csv"), case_rows())
    print("model, training and normalization configs")
    models, trains, norms = run_configs()
    for fn, blob in (("model_config.json", models),
                     ("training_config.json", trains),
                     ("normalization.json", norms)):
        json.dump(blob, open(os.path.join(OUT, fn), "w"), indent=2,
                  sort_keys=True)
        print(f"  wrote {os.path.join(OUT, fn)}  ({len(blob)} runs)")
    open(os.path.join(OUT, "README.md"), "w").write(README)
    print(f"  wrote {os.path.join(OUT, 'README.md')}")


if __name__ == "__main__":
    main()
