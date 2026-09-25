"""Gate-1 unit checks for EquiJet (PLAN_EQUIJET_UPGRADE.md §4.1).

    python -m equijet.tests            # synthetic only
    python -m equijet.tests --real     # also clusters real cases and checks dT/Ri

Checks, in the plan's order:
  1. clustering returns a constant K on all cases, anchors inside inlet bboxes
  2. zero-init  =>  wrapper output bitwise identical to the bare trunk
  3. gradients finite through envelope + FiLM
  4. TRANSLATION: shifting a vent by delta shifts its contribution by delta exactly
  5. dT / Ri extraction matches a hand-computed value; a missing temperature
     column degrades to dT = Ri = 0 with a warning rather than crashing
  6. COMPOSED zero-init  =>  output equals the MASKED baseline, not the bare trunk
     (plan §6.3 — the EquiJet-only bitwise test must not be silently reused)
"""

import argparse
import sys

import numpy as np
import torch

from .template import JetTemplate
from .vents import COND_DIM, extract_vents, lookup_t_in, normalize_cond
from .wrapper import EquiJetModel

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""),
          flush=True)


class _FakeTrunk(torch.nn.Module):
    def __init__(self, out_channels=3, width=16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, width), torch.nn.SiLU(),
            torch.nn.Linear(width, out_channels))

    def encode_geometry(self, pc, sample_ids=None):
        return pc.mean(dim=1, keepdim=True)

    def decode_query(self, latent, xyt):
        return self.net(xyt) + latent[..., :1] * 0.0


def _stats(device):
    return (torch.tensor([0.3, -0.2, 0.15, 0.0], device=device),   # target_mean
            torch.tensor([0.8, 0.7, 0.9, 1.0], device=device),     # target_std
            torch.tensor([0.0, 0.0, 0.0], device=device),           # coord_min
            torch.tensor([9.0, 6.0, 3.5], device=device))           # coord_scale


def _synth_room(device, k_vents=6, per_vent=40):
    """Room whose inlet cloud has k_vents well-separated ceiling patches."""
    g = torch.Generator().manual_seed(0)
    centres = torch.tensor([[1.2, 2.3, 3.35], [1.2, 3.8, 3.35],
                            [4.5, 2.3, 3.35], [4.5, 3.8, 3.35],
                            [7.7, 2.3, 3.35], [7.7, 3.8, 3.35]])[:k_vents]
    pts, vel = [], []
    for c in centres:
        pts.append(c + 0.1 * torch.rand(per_vent, 3, generator=g))
        v = torch.tensor([0.0, 0.0, -0.54]).repeat(per_vent, 1)
        vel.append(v)
    xyz_m = torch.cat(pts)
    vel_m = torch.cat(vel)

    tmean, tstd, cmin, cscale = _stats("cpu")
    xyz_n = (xyz_m - cmin) / cscale
    tgt_n = torch.zeros(len(vel_m), 4)
    tgt_n[:, 0:3] = (vel_m - tmean[0:3]) / tstd[0:3]

    pc = torch.cat([xyz_n, torch.zeros(len(xyz_n), 9)], dim=1)
    pc[:, 3 + 2] = 1.0                                    # inlet one-hot
    return {"pool_in_xyz": xyz_n, "pool_in_tgt": tgt_n, "pc_full": pc}, centres


def _model(device, hbc=None, k_vents=6):
    torch.manual_seed(0)
    tmean, tstd, cmin, cscale = _stats(device)
    trunk = _FakeTrunk().to(device)
    tpl = JetTemplate(cond_dim=COND_DIM).to(device)
    m = EquiJetModel(trunk, tpl, tmean, tstd, cmin, cscale, hbc=hbc,
                     n_vents_expected=k_vents).to(device)
    return m, trunk


def test_clustering(device):
    print("\n1. clustering")
    room, centres = _synth_room("cpu")
    tmean, tstd, cmin, cscale = _stats("cpu")
    a, c = extract_vents(room, cmin, cscale, tmean, tstd, cluster_m=0.7, n_expected=6)
    check("K matches n_expected", len(a) == 6, f"K={len(a)}")
    check("cond has the documented width", c.shape[1] == COND_DIM,
          f"cond_dim={c.shape[1]}")

    # Anchors must land inside their patches (order is lexsorted, as are centres).
    d = (a - centres).norm(dim=-1).max().item()
    check("anchors sit at the vent patches", d < 0.15, f"max|diff|={d:.3f} m")

    speeds = c[:, 0]
    check("speed recovered from inlet velocities",
          abs(speeds.mean().item() - 0.54) < 1e-3, f"mean s={speeds.mean():.4f} m/s")
    dirs = c[:, 1:4]
    check("direction is downward (0,0,-1)",
          torch.allclose(dirs, torch.tensor([0.0, 0.0, -1.0]), atol=1e-4))

    # Wrong K must raise, not silently proceed.
    try:
        extract_vents(room, cmin, cscale, tmean, tstd, cluster_m=0.7, n_expected=3)
        check("wrong K raises", False, "no exception")
    except ValueError:
        check("wrong K raises loudly", True)


def test_outlet_filter(device):
    """Return-patch fringe must be filtered out of the supply-jet anchors."""
    print("\n1b. return-patch filtering")
    room, centres = _synth_room("cpu", k_vents=6)
    tmean, tstd, cmin, cscale = _stats("cpu")

    # Mark the three y~3.8 patches as a RETURN: place outlet points on them. The
    # loader's sign rule leaks such points into the inlet pool, so extract_vents
    # must drop the co-located inlet clusters.
    room = dict(room)
    room["pool_out_xyz"] = ((centres[1::2] - cmin) / cscale)

    a, c = extract_vents(room, cmin, cscale, tmean, tstd,
                         cluster_m=0.7, n_expected=3)
    check("outlet-co-located clusters are dropped", len(a) == 3, f"K={len(a)}")
    kept_y = sorted(round(float(v), 1) for v in a[:, 1])
    check("survivors are the supply row, not the return row",
          all(abs(y - 2.3) < 0.2 for y in kept_y), f"y={kept_y}")


def test_zero_init(device):
    print("\n2. zero-init => bitwise identical to the bare trunk")
    m, trunk = _model(device)
    room, _ = _synth_room("cpu")
    m.register_cases(type("DS", (), {"cases": [room]})(), verbose=False)

    pc = room["pc_full"].unsqueeze(0).to(device)
    xyt = torch.rand(1, 256, 3, generator=torch.Generator().manual_seed(3)).to(device)
    with torch.no_grad():
        ref = trunk.decode_query(trunk.encode_geometry(pc), xyt)
        got = m.forward(xyt, pc)
    check("EquiJet(zero-init) == trunk (bitwise)", torch.equal(ref, got),
          f"max|diff|={(ref - got).abs().max().item():.2e}")

    jets = m.jets(xyt)
    check("template output is exactly zero at init", jets.abs().max().item() == 0.0)


def test_gradients(device):
    print("\n3. gradients through envelope + FiLM")
    m, _ = _model(device)
    room, _ = _synth_room("cpu")
    m.register_cases(type("DS", (), {"cases": [room]})(), verbose=False)
    pc = room["pc_full"].unsqueeze(0).to(device)
    xyt = torch.rand(1, 256, 3, generator=torch.Generator().manual_seed(4)).to(device)

    # Break the zero-init so the head carries gradient.
    with torch.no_grad():
        m.template.head.weight.normal_(0, 0.01)
    m.forward(xyt, pc).square().mean().backward()

    tg = [p.grad for p in m.template.parameters() if p.grad is not None]
    check("template gets finite gradients",
          len(tg) > 0 and all(torch.isfinite(g).all() for g in tg),
          f"{len(tg)} tensors")
    check("envelope scale s gets a gradient",
          m.template.s_raw.grad is not None and torch.isfinite(m.template.s_raw.grad))
    check("some template gradient is non-zero",
          any(g.abs().sum().item() > 0 for g in tg))


def test_translation(device):
    print("\n4. translation equivariance")
    torch.manual_seed(0)
    tpl = JetTemplate(cond_dim=COND_DIM).to(device)
    with torch.no_grad():                       # un-zero the head so J != 0
        tpl.head.weight.normal_(0, 0.05)
        tpl.head.bias.normal_(0, 0.05)

    q = torch.rand(1, 512, 3, device=device) * torch.tensor([9.0, 6.0, 3.5], device=device)
    anchor = torch.tensor([[4.5, 2.3, 3.35]], device=device)
    c = torch.zeros(1, COND_DIM, device=device)
    c[0, 0], c[0, 3] = 0.54, -1.0

    delta = torch.tensor([1.7, -0.6, 0.0], device=device)
    with torch.no_grad():
        a = tpl((q - anchor.unsqueeze(1)), c)                        # vent at anchor
        b = tpl(((q + delta) - (anchor + delta).unsqueeze(1)), c)    # both shifted
    d = (a - b).abs().max().item()
    check("shifting vent and query by the same delta leaves J unchanged",
          d < 1e-5, f"max|diff|={d:.2e}")

    with torch.no_grad():
        c2 = tpl((q - (anchor + delta).unsqueeze(1)), c)
    moved = (a - c2).abs().max().item()
    check("moving only the vent DOES change J (template is not constant)",
          moved > 1e-4, f"max|diff|={moved:.2e}")


def test_envelope(device):
    print("\n5. envelope locality")
    tpl = JetTemplate(cond_dim=COND_DIM, s_init=1.5, s_max=3.0).to(device)
    xi = torch.zeros(1, 4, 3, device=device)
    xi[0, 1, 0], xi[0, 2, 0], xi[0, 3, 0] = 1.5, 3.0, 6.0
    e = tpl.envelope(xi).squeeze()
    check("envelope = 1 at the vent", abs(e[0].item() - 1.0) < 1e-6)
    check("envelope ~0.61 at s", abs(e[1].item() - np.exp(-0.5)) < 1e-5)
    check("envelope < 1% beyond 3s", e[3].item() < 0.01, f"e(6 m)={e[3].item():.2e}")
    with torch.no_grad():
        tpl.s_raw.fill_(99.0)
    check("s is clamped to s_max", abs(float(tpl.s) - 3.0) < 1e-6, f"s={float(tpl.s)}")


def test_composed(device):
    print("\n6. composed EquiJet+HBC zero-init => MASKED baseline (plan §6.3)")
    from hbc.wrapper import HardBCModel
    tmean, tstd, cmin, cscale = _stats(device)

    room, _ = _synth_room("cpu")
    # Give the HBC wrapper a wall cloud to mask against.
    wall = torch.rand(400, 3, generator=torch.Generator().manual_seed(7))
    wall[:, 2] = 0.0
    room = dict(room)
    room["pool_wall_xyz"] = wall
    room["pool_in_xyz"], room["pool_in_tgt"] = room["pool_in_xyz"], room["pool_in_tgt"]

    torch.manual_seed(0)
    trunk = _FakeTrunk().to(device)
    hbc = HardBCModel(trunk, tmean, tstd, cmin, cscale, ell_wall=0.002, stage=1).to(device)
    tpl = JetTemplate(cond_dim=COND_DIM).to(device)
    m = EquiJetModel(trunk, tpl, tmean, tstd, cmin, cscale, hbc=hbc,
                     n_vents_expected=6).to(device)

    ds = type("DS", (), {"cases": [room]})()
    m.register_cases(ds, verbose=False)

    pc = room["pc_full"].unsqueeze(0).to(device)
    xyt = torch.rand(1, 256, 3, generator=torch.Generator().manual_seed(5)).to(device)

    with torch.no_grad():
        composed = m.forward(xyt, pc)
        masked_baseline = hbc.forward(xyt, pc)          # trunk + wall mask, no jets
        bare = trunk.decode_query(trunk.encode_geometry(pc), xyt)

    d_masked = (composed - masked_baseline).abs().max().item()
    check("composed(zero-init) == masked baseline", d_masked < 1e-6,
          f"max|diff|={d_masked:.2e}")
    check("composed(zero-init) != bare trunk (the mask IS applied)",
          (composed - bare).abs().max().item() > 1e-6)


def test_real(device):
    print("\n7. real cases")
    sys.path.insert(0, ".")
    from baselines.common.data import build_datasets
    from .vents import build_temperature_table

    ds, _ = build_datasets("split/train", "", "baselines/runs/shared_stats.pt")
    table = build_temperature_table("split/train")

    ks, dts, ris = [], [], []
    for room in ds.cases:
        a, c = extract_vents(room, ds.coord_min, ds.coord_scale,
                             ds.target_mean, ds.target_std,
                             cluster_m=0.7, n_expected=3, t_table=table, t_ref=294.0)
        ks.append(len(a)); dts.append(c[:, 5].mean().item()); ris.append(c[:, 6].mean().item())
    check("K constant = 3 supply jets across all training cases",
          set(ks) == {3}, f"Ks={sorted(set(ks))}")
    print(f"      dT per case: mean={np.mean(dts):.2f} K  std={np.std(dts):.3f} K")
    print(f"      Ri per case: mean={np.mean(ris):.3f}    std={np.std(ris):.3f}")
    check("dT is non-zero (temperature table was found)", abs(np.mean(dts)) > 1e-6,
          f"mean dT={np.mean(dts):.2f} K")
    if np.std(dts) < 1.0:
        print("      NOTE: dT is near-constant across cases, as plan §5 predicted — "
              "kept for forward-compatibility, expected to carry no signal.")

    # Missing temperature column must degrade, not crash.
    a, c = extract_vents(ds.cases[0], ds.coord_min, ds.coord_scale,
                         ds.target_mean, ds.target_std,
                         cluster_m=0.7, n_expected=3, t_table=None)
    check("no temperature table => dT = Ri = 0, no crash",
          float(c[:, 5].abs().max()) == 0.0 and float(c[:, 6].abs().max()) == 0.0)

    # Hand-check Ri against the formula.
    a, c = extract_vents(ds.cases[0], ds.coord_min, ds.coord_scale,
                         ds.target_mean, ds.target_std, cluster_m=0.7, n_expected=3,
                         t_table=table, t_ref=294.0)
    s, d_t, ri = c[0, 0].item(), c[0, 5].item(), c[0, 6].item()
    hand = 9.81 * (1.0 / 294.0) * d_t * 0.6 / s ** 2
    check("Ri reproduces g*beta*dT*L/s^2 by hand", abs(ri - hand) < 1e-4,
          f"got {ri:.4f}, hand {hand:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"equijet gate-1 unit checks on {device}")

    test_clustering(device)
    test_outlet_filter(device)
    test_zero_init(device)
    test_gradients(device)
    test_translation(device)
    test_envelope(device)
    test_composed(device)
    if args.real:
        test_real(device)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name} ({detail})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
