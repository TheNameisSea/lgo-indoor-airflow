"""Gate 1 unit checks for the hard-BC upgrade (PLAN_HARD_BC_UPGRADE.md §4.1).

Run BEFORE any training:

    python -m hbc.tests            # synthetic only (seconds, no data needed)
    python -m hbc.tests --real     # also checks against a real case (loads split/train)

Checks, in the plan's order:
  1. phi = 0 on the cloud's own points, > 0 off it, and matches a scipy KD-tree.
  2. stage=0 is bitwise identical to the bare trunk.
  3. stage=1 at wall points returns exactly -mu/sigma (i.e. physical zero).
  4. stage=2 at inlet points reproduces the inlet velocity.
  5. backward() through the mask product yields finite gradients.
  6. subsampling error is << ell (the plan's stated tolerance for cloud_max).
"""

import argparse
import sys

import torch

from .distance import mask, min_dist, phys_coords
from .wrapper import HardBCModel

TOL = 1e-5
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)


class _FakeTrunk(torch.nn.Module):
    """Stand-in for the real Trunk: same interface, trivial deterministic output."""

    def __init__(self, out_channels=3, width=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, width), torch.nn.SiLU(),
            torch.nn.Linear(width, out_channels))

    def encode_geometry(self, pc, sample_ids=None):
        return pc.mean(dim=1, keepdim=True)

    def decode_query(self, latent, xyt):
        return self.net(xyt) + latent[..., :1] * 0.0


def _make_case(device, n_wall=800, n_inlet=60, n_pc=2000):
    """Synthetic room: unit cube, walls on z=0, inlets in a patch on z=1."""
    g = torch.Generator().manual_seed(0)
    wall = torch.rand(n_wall, 3, generator=g)
    wall[:, 2] = 0.0
    inlet = torch.rand(n_inlet, 3, generator=g) * 0.2 + 0.4
    inlet[:, 2] = 1.0
    inlet_tgt = torch.zeros(n_inlet, 4)
    inlet_tgt[:, 0:3] = torch.tensor([0.0, 0.0, -2.0])   # 2 m/s downward jet

    # pc_full layout: xyz | one-hot(5) | u,v,w,p
    def block(xyz, cls, uvwp):
        oh = torch.zeros(len(xyz), 5)
        oh[:, cls] = 1.0
        return torch.cat([xyz, oh, uvwp], dim=1)

    pc = torch.cat([
        block(wall, 1, torch.zeros(n_wall, 4)),
        block(inlet, 2, inlet_tgt),
    ], dim=0)
    if len(pc) < n_pc:
        pad = torch.rand(n_pc - len(pc), 3, generator=g)
        pc = torch.cat([pc, block(pad, 1, torch.zeros(len(pad), 4))], dim=0)

    return pc.to(device), wall.to(device), inlet.to(device), inlet_tgt.to(device)


def _wrapper(device, stage, ell=0.05, out_channels=3):
    torch.manual_seed(0)
    trunk = _FakeTrunk(out_channels=out_channels).to(device)
    # Non-trivial stats: this is what makes "physical zero != normalized zero".
    tmean = torch.tensor([0.3, -0.2, 0.15, 0.0], device=device)
    tstd = torch.tensor([0.8, 0.7, 0.9, 1.0], device=device)
    cmin = torch.tensor([0.0, 0.0, 0.0], device=device)
    cscale = torch.tensor([4.0, 3.0, 2.5], device=device)   # anisotropic on purpose
    m = HardBCModel(trunk, tmean, tstd, cmin, cscale, ell_wall=ell, ell_inlet=ell,
                    stage=stage).to(device)          # cloud_max=None -> exact
    return m, trunk, tmean, tstd, cmin, cscale


def test_distance(device):
    print("\n1. distance correctness")
    pc, wall, inlet, _ = _make_case(device)
    cloud = wall

    d_on = min_dist(cloud, cloud)
    check("phi = 0 on the cloud's own points", d_on.abs().max().item() < 1e-6,
          f"max={d_on.abs().max().item():.2e}")

    off = cloud.clone()
    off[:, 2] += 0.37
    d_off = min_dist(off, cloud)
    check("phi > 0 off the cloud", d_off.min().item() > 1e-3,
          f"min={d_off.min().item():.4f}")
    check("phi equals the known offset for a planar cloud",
          (d_off - 0.37).abs().max().item() < 1e-4,
          f"max|err|={(d_off - 0.37).abs().max().item():.2e}")

    try:
        from scipy.spatial import cKDTree
        q = torch.rand(500, 3, generator=torch.Generator().manual_seed(1)).to(device)
        ref = torch.tensor(cKDTree(cloud.cpu().numpy()).query(q.cpu().numpy())[0],
                           device=device, dtype=q.dtype)
        ours = min_dist(q, cloud)
        err = (ours - ref).abs().max().item()
        check("matches scipy cKDTree", err < 1e-4, f"max|err|={err:.2e}")
    except ImportError:
        check("matches scipy cKDTree", True, "scipy unavailable — skipped")

    # Chunking must not change the answer.
    q = torch.rand(3000, 3, generator=torch.Generator().manual_seed(2)).to(device)
    a = min_dist(q, cloud, chunk=4096)
    b = min_dist(q, cloud, chunk=137)
    check("chunk size does not affect result", (a - b).abs().max().item() < 1e-6)

    print("\n   mask shape")
    phi = torch.tensor([0.0, 0.05, 1e6], device=device)
    m = mask(phi, 0.05)
    check("m(0) = 0", abs(m[0].item()) < 1e-12)
    check("m(ell) = 0.5", abs(m[1].item() - 0.5) < 1e-6)
    check("m -> 1 in the far field", m[2].item() > 0.999)


def test_stage0_identity(device):
    print("\n2. stage 0 is bitwise identical to the bare trunk")
    m, trunk, *_ = _wrapper(device, stage=0)
    pc, *_ = _make_case(device)
    pc = pc.unsqueeze(0)
    xyt = torch.rand(1, 256, 3, generator=torch.Generator().manual_seed(3)).to(device)

    with torch.no_grad():
        ref = trunk.decode_query(trunk.encode_geometry(pc), xyt)
        got = m.forward(xyt, pc)
    check("stage 0 == trunk (bitwise)", torch.equal(ref, got),
          f"max|diff|={(ref - got).abs().max().item():.2e}")


def test_stage1_walls(device):
    print("\n3. stage 1 gives physical zero on wall points")
    m, _, tmean, tstd, cmin, cscale = _wrapper(device, stage=1)
    pc, wall, _, _ = _make_case(device)
    pc = pc.unsqueeze(0)

    m.register_cases(type("DS", (), {"cases": [{
        "pc_full": pc[0].cpu(),
        "pool_wall_xyz": wall.cpu(),
        "pool_in_xyz": torch.zeros(0, 3),
        "pool_in_tgt": torch.zeros(0, 4),
    }]})())

    xyt = wall.unsqueeze(0)
    with torch.no_grad():
        out = m.forward(xyt, pc)

    expected = (-tmean[0:3] / tstd[0:3]).expand_as(out[0])
    err = (out[0] - expected).abs().max().item()
    check("normalized output == -mu/sigma on walls", err < TOL, f"max|err|={err:.2e}")

    u_phys = out[0] * tstd[0:3] + tmean[0:3]
    check("physical |u| == 0 on walls", u_phys.abs().max().item() < 1e-5,
          f"max|u|={u_phys.abs().max().item():.2e} m/s")

    # Interior must NOT be zeroed out.
    q = torch.rand(1, 200, 3, generator=torch.Generator().manual_seed(4)).to(device)
    q[0, :, 2] = 0.5
    with torch.no_grad():
        interior = m.forward(q, pc)
    check("interior prediction is not collapsed",
          (interior[0] * tstd[0:3] + tmean[0:3]).abs().max().item() > 1e-3)


def test_stage2_inlets(device):
    print("\n4. stage 2 reproduces the inlet velocity on inlet points")
    m, _, tmean, tstd, *_ = _wrapper(device, stage=2)
    pc, wall, inlet, inlet_tgt = _make_case(device)
    pc = pc.unsqueeze(0)

    m.register_cases(type("DS", (), {"cases": [{
        "pc_full": pc[0].cpu(),
        "pool_wall_xyz": wall.cpu(),
        "pool_in_xyz": inlet.cpu(),
        "pool_in_tgt": inlet_tgt.cpu(),
    }]})())

    with torch.no_grad():
        out = m.forward(inlet.unsqueeze(0), pc)
    u_phys = out[0] * tstd[0:3] + tmean[0:3]
    want = inlet_tgt[:, 0:3] * tstd[0:3] + tmean[0:3]
    err = (u_phys - want).abs().max().item()
    check("physical u == u_in on inlet points", err < 1e-4,
          f"max|err|={err:.2e} m/s (u_in={want[0].tolist()})")


def test_gradients(device):
    print("\n5. gradients flow through the mask product")
    for stage in (1, 2):
        m, _, tmean, tstd, *_ = _wrapper(device, stage=stage)
        pc, wall, inlet, inlet_tgt = _make_case(device)
        pc = pc.unsqueeze(0)
        m.register_cases(type("DS", (), {"cases": [{
            "pc_full": pc[0].cpu(), "pool_wall_xyz": wall.cpu(),
            "pool_in_xyz": inlet.cpu(), "pool_in_tgt": inlet_tgt.cpu(),
        }]})())

        q = torch.rand(1, 512, 3, generator=torch.Generator().manual_seed(5)).to(device)
        out = m.forward(q, pc)
        out.square().mean().backward()

        grads = [p.grad for p in m.trunk.parameters() if p.grad is not None]
        n_finite = sum(int(torch.isfinite(g).all()) for g in grads)
        check(f"stage {stage}: finite grads on all trunk params",
              len(grads) > 0 and n_finite == len(grads),
              f"{n_finite}/{len(grads)} tensors finite")
        check(f"stage {stage}: gradient is non-zero",
              any(g.abs().sum().item() > 0 for g in grads))


def test_subsample_error(device):
    print("\n6. cloud subsampling error << ell")
    g = torch.Generator().manual_seed(6)
    cloud = torch.rand(60000, 3, generator=g).to(device)
    cloud[:, 2] = 0.0
    q = torch.rand(2000, 3, generator=g).to(device)

    # The plan's claim is specifically "cap >= 20k pts; error << ell". Only that is
    # asserted; smaller caps are measured and reported so the budget is visible
    # (they are expected to be worse — that is why the cap exists).
    full = min_dist(q, cloud)
    for cap in (20000, 5000):
        idx = torch.randperm(len(cloud), generator=g)[:cap].to(device)
        err = (min_dist(q, cloud[idx]) - full).abs().max().item()
        if cap >= 20000:
            check(f"cap={cap}: distance error << ell=0.05 m", err < 0.05 / 5,
                  f"err={err * 1000:.1f} mm")
        else:
            print(f"      (info) cap={cap}: error {err * 1000:.1f} mm "
                  f"— below the recommended 20k cap, error grows as expected")


def test_real_case(device):
    print("\n7. real case from split/train")
    sys.path.insert(0, ".")
    from baselines.common.data import build_datasets
    ds, _ = build_datasets("split/train", "", "baselines/runs/shared_stats.pt")

    m, *_ = _wrapper(device, stage=1)
    m.mu.copy_(ds.target_mean[0:3].to(device))
    m.sigma.copy_(ds.target_std[0:3].to(device))
    m.coord_min.copy_(ds.coord_min.to(device))
    m.coord_scale.copy_(ds.coord_scale.to(device))
    n = m.register_cases(ds)
    check("register_cases indexed every room", n == len(ds.cases),
          f"{n}/{len(ds.cases)}")

    room = ds.cases[0]
    pc = room["pc_full"].unsqueeze(0).to(device)
    wall = room["pool_wall_xyz"].to(device)
    print(f"      wall pool: {len(wall)} points, cloud_max={m.cloud_max}")

    # Sample wall points from ACROSS the pool, not just the first block, so this
    # catches any cloud that fails to cover the pool the trainer draws from.
    sel = torch.linspace(0, len(wall) - 1, 4096, device=device).long()
    with torch.no_grad():
        out = m.forward(wall[sel].unsqueeze(0), pc)
    u_phys = out[0] * m.sigma + m.mu
    worst = u_phys.norm(dim=-1).max().item()
    check("physical |u| == 0 on real wall points (exact no-slip)", worst < 1e-4,
          f"max|u|={worst:.2e} m/s")

    # Timing: the plan budgets "microseconds to low-ms, negligible next to
    # attention". Verify with the full (uncapped) cloud actually in use.
    import time
    q = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)[:9570].to(device)
    case_t = m._resolve(pc)
    for _ in range(2):
        min_dist(phys_coords(q, m.coord_min, m.coord_scale), case_t["wall_phys"])
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        min_dist(phys_coords(q, m.coord_min, m.coord_scale), case_t["wall_phys"])
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 5 * 1000
    print(f"      distance cost: {ms:.1f} ms for {len(q)} queries vs "
          f"{len(case_t['wall_phys'])}-point cloud")
    check("distance cost is acceptable per forward (<100 ms)", ms < 100,
          f"{ms:.1f} ms")

    # ---- Calibrate ell from the DATA, not from the plan's guess. ----
    # The CFD mesh is boundary-refined, so uniformly sampling mesh cells puts most
    # interior points close to a surface. That sets an upper bound on ell: the mask
    # must be ~1 over the interior or it simply damps the whole field.
    interior = torch.cat([room["pool_stream_xyz"], room["pool_bg_xyz"]], 0)[:8192].to(device)
    x_phys = phys_coords(interior, m.coord_min, m.coord_scale)
    case = m._resolve(pc)
    phi = min_dist(x_phys, case["wall_phys"])

    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    vals = [phi.quantile(torch.tensor(q, device=device)).item() for q in qs]
    print("      phi_wall over interior points [m]: "
          + "  ".join(f"p{int(q * 100)}={v:.3f}" for q, v in zip(qs, vals)))

    # The plan's real criterion is not a global median: it is "does ell damp the
    # WALL JET?". Ceiling/wall jets sit at ~0.1-0.3 m, and near-wall damping is
    # physically correct there (the true field also goes to zero at the surface).
    # So report mean m_w per distance band, using eval_hbc.py's bands.
    bands = [(0.00, 0.05), (0.05, 0.20), (0.20, 0.60), (0.60, 9.99)]
    print("      mean m_w by distance-to-wall band (jet lives in 0.05-0.30 m):")
    header = "  ".join(f"{lo:.2f}-{hi:.2f}m" for lo, hi in bands)
    print(f"        {'ell':>8}   {header}")
    ok_ell = []
    for ell in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        mw = mask(phi, ell)
        cells = []
        jet_ok = True
        for lo, hi in bands:
            sel = (phi >= lo) & (phi < hi)
            v = mw[sel].mean().item() if bool(sel.any()) else float("nan")
            cells.append(f"{v:9.3f}")
            if lo >= 0.05 and v < 0.9:      # jet band and beyond must be undamped
                jet_ok = False
        print(f"        {ell:8.3f}   {'  '.join(cells)}"
              + ("   <- jet preserved" if jet_ok else ""))
        if jet_ok:
            ok_ell.append(ell)

    check("some ell leaves the wall-jet band (>=0.05 m) undamped (mean m_w>0.9)",
          bool(ok_ell),
          f"ell <= {max(ok_ell)} m works" if ok_ell else "none of the tested values")
    print(f"\n      => sweep ell_wall in {ok_ell} m. The plan's 0.02-0.2 m range is "
          f"mostly unusable here: at ell=0.05 the 0.05-0.20 m jet band is damped to "
          f"mean m_w={mask(phi, 0.05)[(phi >= 0.05) & (phi < 0.20)].mean():.2f}.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true", help="also test a real case")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"hbc gate-1 unit checks on {device}")

    test_distance(device)
    test_stage0_identity(device)
    test_stage1_walls(device)
    test_stage2_inlets(device)
    test_gradients(device)
    test_subsample_error(device)
    if args.real:
        test_real_case(device)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    if n_fail:
        print("FAILED:")
        for name, ok, detail in results:
            if not ok:
                print(f"  - {name} ({detail})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
