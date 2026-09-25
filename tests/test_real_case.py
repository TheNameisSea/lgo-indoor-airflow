"""End-to-end check of LGO on a real CFD case, through the legacy eval path.

Confirms on real data what the synthetic tests confirm in miniature, and
measures the two costs that decide whether the design is usable: full-field
inference time (the prior project: GINOT 3.0 s, GINOT+HBC 32.6 s per 1.16 M
points) and the kNN gather that the local branch adds.

Run: python -m tests.test_real_case [--case splits/val/Case_07]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

STATS = os.path.join(ROOT, "stats", "thermo_stats.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=os.path.join(ROOT, "splits", "val", "Case_07"))
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--full", action="store_true",
                    help="predict every interior point, as eval_thermo does")
    args = ap.parse_args()

    from thermo.dataset import ThermoDataset
    from model.lgo import build_lgo

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}   case: {os.path.basename(args.case)}")

    t0 = time.time()
    ds = ThermoDataset(case_dirs=[args.case], saved_stats=STATS, stats_path=STATS,
                       n_case_pool=1_500_000, n_boundary_pool=250_000,
                       n_pc=15000, pc_mode="full")
    print(f"\ndataset loaded in {time.time() - t0:.1f}s   pc_channels={ds.pc_channels}")

    model = build_lgo(ds.pc_channels, ds.coord_min, ds.coord_scale,
                      ds.target_mean, ds.target_std, out_channels=5,
                      use_local=True, use_hbc=True, verbose=True).to(dev)
    t0 = time.time()
    model.register_cases(ds)
    print(f"boundary index built in {time.time() - t0:.1f}s")
    model.eval()

    room = ds.cases[0]
    xyz_n = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)
    n_pts = len(xyz_n) if args.full else min(len(xyz_n), 200_000)
    xyz_n = xyz_n[:n_pts]
    pc = room["pc_full"].unsqueeze(0).to(dev)
    print(f"\npredicting {n_pts:,} interior points, chunk={args.chunk}")

    with torch.no_grad():
        lat = model.encode_geometry(pc)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        t0 = time.time()
        out = torch.cat([
            model.decode_query(lat, xyz_n[s:s + args.chunk].unsqueeze(0).to(dev)).squeeze(0)
            for s in range(0, n_pts, args.chunk)], 0)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        dt = time.time() - t0

    print(f"  {dt:.1f}s  ({n_pts / dt / 1e6:.3f} M pts/s)"
          f"   [prior project: GINOT 3.0s, GINOT+HBC 32.6s per 1.16M]")
    print(f"  projected full-field (1.16M): {dt * 1.16e6 / n_pts:.1f}s")

    pred = (out * ds.target_std[:5].to(dev) + ds.target_mean[:5].to(dev)).cpu()
    true = (torch.cat([room["pool_stream_tgt"], room["pool_bg_tgt"]], 0)[:n_pts]
            * ds.target_std + ds.target_mean)
    print(f"\n  prediction is finite: {bool(torch.isfinite(pred).all())}")
    print(f"  pred speed  mean {pred[:, 0:3].norm(dim=1).mean():.3f} m/s "
          f"(truth {true[:, 0:3].norm(dim=1).mean():.3f})")
    print(f"  pred T      mean {pred[:, 4].mean():.2f} K "
          f"(truth {true[:, 4].mean():.2f})")

    # No-slip on REAL wall points, in physical units.
    wall = room["pool_wall_xyz"][:20000]
    with torch.no_grad():
        w = model.decode_query(lat, wall.unsqueeze(0).to(dev)).squeeze(0)
    vel = (w[:, 0:3] * model.sigma + model.mu).abs().max().item()
    # Not bit-zero, and it should not be: the mask makes the PHYSICAL velocity
    # exactly 0, but the model returns normalized units, so the value round-trips
    # as ((0 - mu)/sigma)*sigma + mu and leaves ~eps*|mu| ~ 1e-8. That is the
    # same floor the prior project reported (1.5e-08) for the same reason.
    print(f"\n  worst |velocity| on {len(wall):,} real wall points: {vel:.3e} m/s"
          f"   [prior project receipt: 1.5e-08]")
    ok = vel < 1e-7
    print(f"  {'ok' if ok else 'FAIL'}  exact no-slip on real geometry")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
