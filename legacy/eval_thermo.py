"""Full-field evaluation for runs_thermo_final/ (and runs_ablation/ if needed).

Loads the best.pth checkpoint, predicts every interior point of every val/train
case, and reports joint velocity + temperature metrics (rho, s, Taylor, mae, R2)
per case, mean over cases, and with uniform-cell subsampling to remove mesh-bias.

    python eval_thermo.py --run runs_thermo_final/ginot_se
    python eval_thermo.py --run runs_thermo_final/ginot_se --split split_v2/train
    python eval_thermo.py --runs runs_thermo_final  # all finished runs in a dir
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from thermo.dataset import ThermoDataset
from thermo.metrics_joint import joint_metrics, summarise, fmt


BASELINES = ("transolver", "gno", "deeponet", "gino")

# ---- reuse the model-loader from the vis script so they stay in sync -------
def load_model(run, ds, device):
    from pi_ginot import build_model
    a = json.load(open(os.path.join(run, "args.json")))
    mln = a["min_length_m"] / ds.coord_scale.max().item()

    if a["model"] in BASELINES:
        from baselines.models import build_baseline
        cfg = json.loads(a.get("cfg", "{}"))
        model = build_baseline(a["model"], cfg, pc_channels=ds.pc_channels,
                               out_channels=5,
                               min_length_norm=mln).to(device)
        model.load_state_dict(torch.load(os.path.join(run, "best.pth"),
                                         map_location=device))
        model.eval()
        return model, a

    trunk = build_model(
        {"input_channels": ds.pc_channels, "out_c": a["embed_dim"],
         "width": a["branch_width"], "latent_d": a["branch_latent_d"],
         "n_point": a["branch_n_point"], "radius": a["branch_radius"]},
        {"in_channels": 3, "out_channels": 5, "embed_dim": a["embed_dim"],
         "cross_attn_layers": a["cross_attn_layers"], "min_length_norm": mln,
         "num_bands": a["num_bands"], "freq_spacing": a["freq_spacing"]},
        verbose=False).to(device)
    model = trunk
    hbc = None
    if "hbc" in a["model"]:
        from hbc import HardBCModel
        hbc = HardBCModel(trunk, ds.target_mean, ds.target_std, ds.coord_min,
                          ds.coord_scale, ell_wall=a["ell_wall"], stage=1).to(device)
        model = hbc
    if "se" in a["model"]:
        from equijet import EquiJetModel, JetTemplate, build_temperature_table
        from equijet.vents import COND_DIM
        tt = (build_temperature_table(a["train_dir"], verbose=False)
              + build_temperature_table(a["val_dir"], verbose=False))
        tpl = JetTemplate(num_bands=a["jet_bands"], min_length=a["jet_min_length"],
                          hidden=a["jet_hidden"], depth=a["jet_depth"],
                          cond_dim=COND_DIM, s_init=a["jet_envelope_init"],
                          s_max=a["jet_envelope_max"], radial=a["jet_radial"],
                          n_radial=a["n_radial"]).to(device)
        model = EquiJetModel(trunk, tpl, ds.target_mean, ds.target_std, ds.coord_min,
                             ds.coord_scale, hbc=hbc, cluster_m=a["vent_cluster_m"],
                             n_vents_expected=a["n_vents_expected"], t_table=tt,
                             t_ref=a["t_ref"], vent_width_m=a["vent_width_m"]).to(device)
    model.load_state_dict(torch.load(os.path.join(run, "best.pth"), map_location=device))
    model.eval()
    if hasattr(model, "register_cases"):
        try:
            model.register_cases(ds, verbose=False)
        except TypeError:
            model.register_cases(ds)
    return model, a


def uniform_cells(xyz, cell=0.05):
    """One point per 5 cm cube — removes mesh-density bias from global metrics."""
    key = np.floor(xyz / cell).astype(np.int64)
    key -= key.min(axis=0)
    span = key.max(axis=0) + 1
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    _, idx = np.unique(flat, return_index=True)
    return np.sort(idx)


@torch.no_grad()
def predict_room(model, room, ds, device, chunk=16384):
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    tgt   = torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], 0)
    pc    = room["pc_full"].unsqueeze(0).to(device)
    tm, ts = ds.target_mean.to(device), ds.target_std.to(device)
    lat = model.encode_geometry(pc)
    out = torch.cat([
        model.decode_query(lat, xyz_n[s:s+chunk].unsqueeze(0).to(device)).squeeze(0)
        for s in range(0, len(xyz_n), chunk)], 0)
    pred_phys = (out * ts[:5] + tm[:5]).cpu()
    true_phys = (tgt.to(device) * ts + tm).cpu()
    xyz_m     = (xyz_n * ds.coord_scale + ds.coord_min).numpy()
    return xyz_m, true_phys.numpy(), pred_phys.numpy()


def eval_run(run, split_dir, device):
    a = json.load(open(os.path.join(run, "args.json")))
    folders = sorted(glob.glob(os.path.join(split_dir, "*/")))
    names   = [os.path.basename(f.rstrip("/")) for f in folders]
    ds = ThermoDataset(case_dirs=[f.rstrip("/") for f in folders],
                       saved_stats=a["stats_path"], stats_path=a["stats_path"],
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    model, _ = load_model(run, ds, device)

    per_case, per_case_u = [], []
    t0 = time.time()
    for i, (room, name) in enumerate(zip(ds.cases, names)):
        t_room = time.time()
        xyz, true, pred = predict_room(model, room, ds, device)
        elapsed = time.time() - t_room
        m  = joint_metrics(true, pred)
        u  = uniform_cells(xyz)
        mu = joint_metrics(true[u], pred[u])
        per_case.append(m)
        per_case_u.append(mu)
        pts_s = len(xyz) / elapsed
        print(f"  {name}  {len(xyz):>8,} pts  {elapsed:.1f}s  ({pts_s/1e6:.2f} M pts/s)")
        print(f"    raw:     {fmt(m)}")
        print(f"    uniform: {fmt(mu)}")

    mean  = summarise(per_case)
    meanu = summarise(per_case_u)
    wall  = time.time() - t0
    print(f"\n--- MEAN over {len(names)} cases ---")
    print(f"  raw:     {fmt(mean)}")
    print(f"  uniform: {fmt(meanu)}")
    print(f"  total wall {wall:.0f}s")
    return {"per_case": per_case, "per_case_uniform": per_case_u,
            "mean": mean, "mean_uniform": meanu, "cases": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run",   default=None, help="single run dir")
    ap.add_argument("--runs",  default=None, help="dir of run dirs (all finished)")
    ap.add_argument("--split", default="split_v2_valclean")
    ap.add_argument("--out",   default=None, help="save JSON results here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run:
        run_dirs = [args.run]
    elif args.runs:
        run_dirs = sorted(d for d in glob.glob(os.path.join(args.runs, "*/"))
                          if os.path.exists(os.path.join(d, "best.pth")))
    else:
        ap.error("provide --run or --runs")

    all_results = {}
    for run in run_dirs:
        name = os.path.basename(run.rstrip("/"))
        print(f"\n{'='*60}")
        print(f"  {name}  ({args.split})")
        print(f"{'='*60}")
        res = eval_run(run, args.split, device)
        all_results[name] = res

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(all_results, open(args.out, "w"), indent=2)
        print(f"\nsaved {args.out}")

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY (val mean, raw points)")
        print(f"{'='*60}")
        rows = sorted(all_results.items(), key=lambda kv: -kv[1]["mean"]["score"])
        print(f"{'model':20} {'score':>7} {'v_rho':>6} {'v_s':>5} {'v_S':>6} "
              f"{'T_rho':>6} {'T_s':>5} {'T_mae':>7} {'v_R2':>7}")
        print("-" * 80)
        for n, r in rows:
            m = r["mean"]
            print(f"{n:20} {m['score']:7.4f} {m['vel_rho']:+6.3f} {m['vel_s']:5.2f} "
                  f"{m['vel_taylor']:6.3f} {m['temp_rho']:+6.3f} {m['temp_s']:5.2f} "
                  f"{m['temp_mae']:7.3f} {m['vel_r2']:+7.3f}")


if __name__ == "__main__":
    main()