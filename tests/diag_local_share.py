"""How much is the local branch actually contributing?

The local stream is zero-initialised, so it starts silent and has to earn its
way in. That is what makes the A/B honest, but it also means a real failure mode
exists: if the stream never grows, LGO is just GINOT with extra parameters and
a 3x step cost, and we would not find out until a 3-hour run ended.

Reports, for a checkpoint:

  |h_local| / |x_global|   in the fused representation, before output_proj
  d(output)                the change in predicted (u,v,w,T) when the local
                           stream is zeroed -- the quantity that actually matters
  per-group attention      where the local branch is looking. If it spends
                           everything on `solid`, the stratified gather is not
                           buying what it was built to buy.

CPU by default so it does not take a GPU slot from a training run.

    python -m tests.diag_local_share --run runs/ladder_lgo_lr1e-3
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

from data.knn_cache import GROUPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--case", default=os.path.join(ROOT, "splits", "val", "Case_07"))
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--ckpt", default="best.pth")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from thermo.dataset import ThermoDataset
    from model.lgo import build_lgo

    a = json.load(open(os.path.join(args.run, "args.json")))
    dev = torch.device(args.device)
    ds = ThermoDataset(case_dirs=[args.case], saved_stats=a["stats_path"],
                       stats_path=a["stats_path"], n_case_pool=200_000,
                       n_boundary_pool=250_000, n_pc=a["n_pc"], pc_mode="full")

    model = build_lgo(
        ds.pc_channels, ds.coord_min, ds.coord_scale, ds.target_mean, ds.target_std,
        out_channels=5, embed_dim=a["embed_dim"],
        cross_attn_layers=a["cross_attn_layers"], branch_width=a["branch_width"],
        branch_latent_d=a["branch_latent_d"], branch_n_point=a["branch_n_point"],
        branch_radius=a["branch_radius"],
        min_length_norm=a["min_length_m"] / ds.coord_scale.max().item(),
        num_bands=a["num_bands"], freq_spacing=a["freq_spacing"],
        n_heads=a.get("local_heads", 8),
        k={"supply": a.get("k_supply", 32), "return": a.get("k_return", 16),
           "solid": a.get("k_solid", 32), "leak": a.get("k_leak", 8)},
        n_local_layers=a.get("n_local_layers", 1), n_rbf=a.get("n_rbf", 12),
        local_hidden=a.get("local_hidden", 128), rbf_max=a.get("rbf_max", 4.0),
        local_query=a.get("local_query", "learned"),
        global_query=a.get("global_query", "fourier"),
        use_local=True, use_global="local" not in a["model"],
        use_hbc=a["model"].endswith("hbc"), ell_wall=a["ell_wall"],
        verbose=False).to(dev)
    sd = torch.load(os.path.join(args.run, args.ckpt), map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    model.register_cases(ds, verbose=False)

    room = ds.cases[0]
    xyz = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    sel = torch.randperm(len(xyz))[:args.n]
    xyt = xyz[sel].unsqueeze(0).to(dev)
    pc = room["pc_full"].unsqueeze(0).to(dev)

    with torch.no_grad():
        lat = model.encode_geometry(pc)
        idx = model._active
        xm = model._metres(xyt[0], idx)
        # global stream
        xg = model.trunk.Q_encoder(model.trunk.fourier_enc(xyt))
        for blk in model.trunk.resblocks:
            xg = blk(xg, lat)
        # local stream
        nb = model._gather(xm, idx, dev)
        tok, val = model.local.encode(nb)
        h = torch.zeros_like(xg[0])
        for l in range(model.n_local_layers):
            h = h + model.local(h, tok, val, layer=l)
        out_both = model.trunk.output_proj(xg + h.unsqueeze(0))[0]
        out_glob = model.trunk.output_proj(xg)[0]

    ng, nl = xg[0].norm(dim=-1), h.norm(dim=-1)
    print(f"\n  checkpoint: {args.run}/{args.ckpt}   epoch of best: "
          f"{max((e['epoch'] for e in json.load(open(os.path.join(args.run,'history.json'))) if isinstance(e.get('val'),dict)), default='?')}")
    print(f"\n  REPRESENTATION (before output_proj)")
    print(f"    |global| mean {ng.mean():.4f}   |local| mean {nl.mean():.4f}"
          f"   ratio {nl.mean()/ng.mean().clamp_min(1e-9):.4f}")
    print(f"    local is >1% of global at {float((nl > 0.01*ng).float().mean())*100:.1f}% of queries")

    d = (out_both - out_glob)
    ts = ds.target_std[:5].to(dev)
    print(f"\n  EFFECT ON OUTPUT (physical units)")
    print(f"    velocity change  mean {(d[:, 0:3]*ts[0:3]).norm(dim=-1).mean():.4f} m/s"
          f"   max {(d[:, 0:3]*ts[0:3]).norm(dim=-1).max():.4f}")
    print(f"    temperature      mean {(d[:, 4]*ts[4]).abs().mean():.4f} K"
          f"   max {(d[:, 4]*ts[4]).abs().max():.4f}")
    spd = (out_both[:, 0:3]*ts[0:3] + ds.target_mean[:3].to(dev)).norm(dim=-1)
    print(f"    for scale, predicted speed mean {spd.mean():.4f} m/s")

    # Where does the attention go? Recompute weights for layer 0.
    with torch.no_grad():
        att = model.local.attn[0]
        N, K, _ = tok.shape
        # The layer-0 query IS zero in the two-stream design. Size it from
        # q_proj, not from the tokens: tokens carry d_token, which the local
        # branch allows to be narrower than the trunk's d_model.
        q0 = torch.zeros(N, att.q_proj.in_features, device=dev)
        qh = att.q_proj(q0).view(N, att.h, 1, att.dk)
        kh = att.k_proj(tok).view(N, K, att.h, att.dk).transpose(1, 2)
        sc = (qh * kh).sum(-1) / att.dk ** 0.5
        sc = sc.masked_fill(~val[:, None, :], float("-inf"))
        w = sc.softmax(-1).mean(1)                      # (N, K) averaged over heads
    ks = [a.get(f"k_{g}", d_) for g, d_ in zip(GROUPS, (32, 16, 32, 8))]
    print(f"\n  ATTENTION MASS BY GROUP (layer 0, uniform would be K/88)")
    s = 0
    for g, k in zip(GROUPS, ks):
        share = float(w[:, s:s + k].sum(-1).mean())
        print(f"    {g:7} K={k:<3} mass {share:6.3f}   (uniform {k/sum(ks):.3f})")
        s += k


if __name__ == "__main__":
    main()
