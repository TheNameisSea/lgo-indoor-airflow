"""Train ANY model in the comparison to predict velocity AND temperature.

One entry point for all seven architectures — the four GINOT variants and the
three Tier-1 baselines — so nothing in the comparison can differ because of how a
model was trained. `pi_ginot/`, `ginot/`, `train.py` and `baselines/models/` are
NOT modified; this composes them.

    --model ginot | ginot_se | ginot_hbc | ginot_se_hbc      (pi_ginot family)
            transolver | gno | deeponet                       (Tier-1 baselines)

Outputs are `[u, v, w, p, T]`. Pressure keeps index 3 so every existing index is
unchanged; it is carried but unsupervised (`--lambda_p 0`), as in every
velocity-only run, because it has ~0 skill on this data.

Both wrappers compose with a 5-channel trunk unchanged: `HardBCModel` and
`EquiJetModel` only ever touch channels 0:3, and take `target_mean[0:3]` /
`target_std[0:3]`, so temperature passes through them untouched. That is the
correct behaviour — the jet template is a *velocity* prior and the wall mask is a
*velocity* boundary condition; neither has anything to say about temperature.

    python train_thermo.py --model ginot_se --lambda_t 1.0 --lr 1e-3 \
        --out_dir runs_thermo/ginot_se --epochs 2000
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from pi_ginot import build_model
from thermo import ThermoDataset, ThermoTrainer

GINOT_FAMILY = ("ginot", "ginot_se", "ginot_hbc", "ginot_se_hbc")
BASELINES = ("transolver", "gno", "deeponet", "gino")
OUT_CHANNELS = 5                      # u, v, w, p, T


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=GINOT_FAMILY + BASELINES)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_dir", default="split_v2/train")
    p.add_argument("--val_dir", default="split_v2/val")
    p.add_argument("--stats_path", default=None)
    p.add_argument("--cfg", default="{}", help="JSON knobs for the baselines")
    # pools
    p.add_argument("--n_case_pool", type=int, default=1_200_000)
    p.add_argument("--n_boundary_pool", type=int, default=250_000)
    p.add_argument("--n_pc", type=int, default=15000)
    p.add_argument("--n_collocation_batch", type=int, default=1024)
    p.add_argument("--n_supervised_batch", type=int, default=4096)
    # Boundary batches default to 0. Every boundary lambda is 0 in this program
    # (hard BC replaces the soft penalties), so those ~4,900 points per step were
    # decoded and then discarded -- pure waste for six of the seven models, and
    # fatal for GNO, whose memory is (query x neighbour) PAIRS: it OOMs at 11.7 GiB
    # trying to decode them. Set them non-zero only if a boundary loss is enabled.
    p.add_argument("--n_boundary_batch", type=int, default=0)
    p.add_argument("--n_inlet_batch", type=int, default=0)
    p.add_argument("--n_outlet_batch", type=int, default=0)
    p.add_argument("--n_leak_batch", type=int, default=0)
    p.add_argument("--g", type=float, default=9.81)
    p.add_argument("--pressure_gauge", action="store_true", default=True)
    # optimisation
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_t", type=float, default=1.0)
    p.add_argument("--lambda_data", type=float, default=10.0)
    p.add_argument("--lambda_wall", type=float, default=0.0)
    p.add_argument("--lambda_inlet", type=float, default=0.0)
    p.add_argument("--lambda_outlet", type=float, default=0.0)
    p.add_argument("--lambda_leak", type=float, default=0.0)
    p.add_argument("--lambda_p", type=float, default=0.0)
    p.add_argument("--vel_floor", type=float, default=0.2)
    p.add_argument("--accum_cases", type=int, default=1)
    # trunk
    p.add_argument("--pc_mode", default="full", choices=["full", "geom"])
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--cross_attn_layers", type=int, default=5)
    p.add_argument("--branch_width", type=int, default=128)
    p.add_argument("--branch_latent_d", type=int, default=1024)
    p.add_argument("--branch_n_point", type=int, default=1024)
    p.add_argument("--branch_radius", type=float, default=0.08)
    p.add_argument("--min_length_m", type=float, default=0.15)
    p.add_argument("--num_bands", type=int, default=8)
    p.add_argument("--freq_spacing", default="log", choices=["linear", "log"])
    # SE / HBC
    p.add_argument("--ell_wall", type=float, default=0.002)
    p.add_argument("--jet_radial", action="store_true", default=False)
    p.add_argument("--jet_bands", type=int, default=6)
    p.add_argument("--jet_min_length", type=float, default=0.08)
    p.add_argument("--jet_hidden", type=int, default=128)
    p.add_argument("--jet_depth", type=int, default=4)
    p.add_argument("--jet_envelope_init", type=float, default=1.5)
    p.add_argument("--jet_envelope_max", type=float, default=3.0)
    p.add_argument("--n_radial", type=int, default=10)
    p.add_argument("--vent_cluster_m", type=float, default=0.7)
    p.add_argument("--n_vents_expected", type=int, default=3)
    p.add_argument("--t_ref", type=float, default=294.0)
    p.add_argument("--vent_width_m", type=float, default=0.6)
    # bookkeeping
    p.add_argument("--val_every", type=int, default=25)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tf32", action="store_true", default=True)
    p.add_argument("--debug", action="store_true", default=False)
    return p.parse_args()


def build(args, ds, device):
    """Return the model, already wrapped for SE / HBC as `--model` requires."""
    min_length_norm = args.min_length_m / ds.coord_scale.max().item()

    if args.model in BASELINES:
        from baselines.models import build_baseline
        cfg = json.loads(args.cfg)
        return build_baseline(args.model, cfg, pc_channels=ds.pc_channels,
                              out_channels=OUT_CHANNELS,
                              min_length_norm=min_length_norm).to(device)

    trunk = build_model(
        {"input_channels": ds.pc_channels, "out_c": args.embed_dim,
         "width": args.branch_width, "latent_d": args.branch_latent_d,
         "n_point": args.branch_n_point, "radius": args.branch_radius},
        {"in_channels": 3, "out_channels": OUT_CHANNELS,
         "embed_dim": args.embed_dim, "cross_attn_layers": args.cross_attn_layers,
         "min_length_norm": min_length_norm, "num_bands": args.num_bands,
         "freq_spacing": args.freq_spacing}, verbose=False).to(device)
    if args.model == "ginot":
        return trunk

    hbc = None
    if "hbc" in args.model:
        from hbc import HardBCModel
        hbc = HardBCModel(trunk, ds.target_mean, ds.target_std, ds.coord_min,
                          ds.coord_scale, ell_wall=args.ell_wall, stage=1).to(device)
    if "se" not in args.model:
        return hbc

    from equijet import EquiJetModel, JetTemplate, build_temperature_table
    from equijet.vents import COND_DIM
    t_table = (build_temperature_table(args.train_dir, verbose=False)
               + build_temperature_table(args.val_dir, verbose=False))
    template = JetTemplate(
        num_bands=args.jet_bands, min_length=args.jet_min_length,
        hidden=args.jet_hidden, depth=args.jet_depth, cond_dim=COND_DIM,
        s_init=args.jet_envelope_init, s_max=args.jet_envelope_max,
        radial=args.jet_radial, n_radial=args.n_radial).to(device)
    return EquiJetModel(trunk, template, ds.target_mean, ds.target_std,
                        ds.coord_min, ds.coord_scale, hbc=hbc,
                        cluster_m=args.vent_cluster_m,
                        n_vents_expected=args.n_vents_expected, t_table=t_table,
                        t_ref=args.t_ref, vent_width_m=args.vent_width_m).to(device)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = args.stats_path or os.path.join(args.out_dir, "thermo_stats.pt")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for lam, nb, nm in ((args.lambda_wall, args.n_boundary_batch, "wall"),
                        (args.lambda_inlet, args.n_inlet_batch, "inlet"),
                        (args.lambda_outlet, args.n_outlet_batch, "outlet"),
                        (args.lambda_leak, args.n_leak_batch, "leak")):
        if lam and nb == 0:
            raise SystemExit(f"--lambda_{nm} is {lam} but --n_{nm}_batch is 0, so "
                             f"that loss would silently never be applied.")
    if "hbc" in args.model and args.lambda_wall:
        raise SystemExit("hard BC enforces no-slip by construction; use "
                         f"--lambda_wall 0 (got {args.lambda_wall})")

    with open(os.path.join(args.out_dir, "args.json"), "w") as fh:
        json.dump({**vars(args), "predict_temperature": True,
                   "out_channels": OUT_CHANNELS}, fh, indent=2)

    ds_kwargs = dict(
        n_case_pool=args.n_case_pool, n_boundary_pool=args.n_boundary_pool,
        n_pc=args.n_pc, n_collocation_batch=args.n_collocation_batch,
        n_supervised_batch=args.n_supervised_batch,
        n_boundary_batch=args.n_boundary_batch, n_inlet_batch=args.n_inlet_batch,
        n_outlet_batch=args.n_outlet_batch, n_leak_batch=args.n_leak_batch,
        g=args.g, pressure_gauge=args.pressure_gauge, pc_mode=args.pc_mode)

    # A thermo stats file has 5 target channels; it must never be shared with the
    # 4-channel velocity-only files.
    #
    # Reuse the file when it already exists. Recomputing per run would (a) cost
    # ~8 min each, and (b) race: parallel sweep runs sharing one --stats_path all
    # write it. The values are deterministic given the training set, so reusing is
    # also what keeps every run in a sweep on an identical normalisation.
    have_stats = os.path.exists(stats_path)
    if have_stats:
        print(f"[thermo] reusing stats {stats_path}")
    train_ds = ThermoDataset(case_dirs=args.train_dir,
                             saved_stats=stats_path if have_stats else None,
                             stats_path=stats_path, augment=[], **ds_kwargs)
    val_ds = ThermoDataset(case_dirs=args.val_dir, saved_stats=stats_path,
                           stats_path=stats_path, **ds_kwargs) if args.val_dir else None

    model = build(args, train_ds, device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.model}] outputs u,v,w,p,T | pc {train_ds.pc_channels} ch | "
          f"{n_par:,} params | lr {args.lr} lambda_t {args.lambda_t}", flush=True)

    trainer = ThermoTrainer(model=model, device=device, out_dir=args.out_dir,
                            debug=args.debug, lambda_t=args.lambda_t)
    # HardBCModel.register_cases takes no `verbose`; EquiJetModel's does. Probe
    # rather than branch on the model name, which is what broke here first.
    for d in (train_ds, val_ds):
        if d is None or not hasattr(model, "register_cases"):
            continue
        try:
            model.register_cases(d, verbose=False)
        except TypeError:
            model.register_cases(d)

    trainer.train(
        dataset=train_ds, val_dataset=val_ds, epochs=args.epochs, lr=args.lr,
        lambda_data=args.lambda_data, lambda_wall=args.lambda_wall,
        lambda_inlet=args.lambda_inlet, lambda_outlet=args.lambda_outlet,
        lambda_leak=args.lambda_leak, lambda_p=args.lambda_p,
        vel_floor=args.vel_floor, val_every=args.val_every,
        ckpt_every=args.ckpt_every, log_every=args.log_every,
        accum_cases=args.accum_cases)


if __name__ == "__main__":
    main()
