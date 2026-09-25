"""Integration tests for model/lgo_ginot.py -- the SUPERSEDED arm.

The claims that must hold before any training run is worth doing:

  1. LGO with the local branch OFF is bit-identical to the baseline GINOT
     trunk, so "global-only" is literally the prior project's model.
  2. LGO at initialisation is bit-identical to global-only, because the local
     branch is zero-initialised. The A/B therefore starts from one model.
  3. With HBC on, velocity is EXACTLY zero at solid points -- in physical
     units, which is where the condition actually lives.
  4. The model satisfies the legacy Trunk interface, so legacy/eval_thermo.py
     scores it through the same path as every baseline.

Run: python -m tests.test_lgo_ginot
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

from model.lgo_ginot import LGOGinot, build_lgo_ginot, fingerprint
from tests.test_knn_cache import make_room

PASS, FAIL = [], []
PC_C = 13          # xyz(3) + class one-hot(5) + u,v,w,p,T(5)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}"
          f"{('  -- ' + detail) if detail and not cond else ''}")


class FakeDataset:
    """Minimum surface a dataset needs to register cases."""

    def __init__(self, room, pc):
        room = dict(room)
        room["pc_full"] = pc
        self.cases = [room]
        self.coord_min = torch.zeros(3)
        self.coord_scale = torch.ones(3)
        self.target_mean = torch.tensor([0.1, -0.2, 0.3, 0.0, 293.0])
        self.target_std = torch.tensor([0.9, 0.9, 0.6, 12.0, 4.5])


def make_pc(room, n=3000, seed=0):
    """A pc_full-shaped cloud drawn from the room's own surfaces."""
    g = torch.Generator().manual_seed(seed)
    xyz = torch.cat([room["pool_wall_xyz"], room["pool_in_xyz"],
                     room["pool_out_xyz"], room["pool_leak_xyz"]], 0)
    idx = torch.randperm(len(xyz), generator=g)[:n]
    return torch.cat([xyz[idx], torch.zeros(len(idx), PC_C - 3)], 1)


def main():
    torch.manual_seed(0)
    room = make_room(n_wall=3000, n_supply=300, n_return=300, n_leak=100)
    pc = make_pc(room)
    ds = FakeDataset(room, pc)

    common = dict(embed_dim=64, cross_attn_layers=2, branch_width=32,
                  branch_latent_d=128, branch_n_point=128, num_bands=4,
                  verbose=False)
    kw = dict(coord_min=ds.coord_min, coord_scale=ds.coord_scale,
              target_mean=ds.target_mean, target_std=ds.target_std)

    model = build_lgo_ginot(PC_C, out_channels=5, use_local=True, n_heads=4,
                      local_hidden=32, **kw, **common)
    model.register_cases(ds, verbose=False)
    model.eval()

    xyt = torch.rand(1, 256, 3) * 0.8 + 0.1
    pcb = pc.unsqueeze(0)

    print("\n[1] global-only LGO == the baseline GINOT trunk")
    model.use_local = False
    with torch.no_grad():
        lat = model.encode_geometry(pcb)
        a = model.decode_query(lat, xyt)
        b = model.trunk.decode_query(model.trunk.encode_geometry(pcb), xyt)
    check("outputs are bit-identical", torch.equal(a, b),
          f"max diff {float((a - b).abs().max()):.3e}")

    print("\n[2] zero-init: LGO == global-only at step 0")
    model.use_local = True
    with torch.no_grad():
        c = model.decode_query(model.encode_geometry(pcb), xyt)
    check("local branch contributes exactly nothing at init", torch.equal(a, c),
          f"max diff {float((a - c).abs().max()):.3e}")

    print("\n[3] once trained, the local branch DOES change the output")
    with torch.no_grad():
        for att in model.local.attn:
            att.out.weight.normal_(std=0.05)
            att.out.bias.normal_(std=0.05)
        d = model.decode_query(model.encode_geometry(pcb), xyt)
    delta = float((a - d).abs().max())
    check("a non-zero local branch moves the prediction", delta > 1e-3, f"{delta:.3e}")

    print("\n[4] hard no-slip is exact, in physical units")
    hbc = build_lgo_ginot(PC_C, out_channels=5, use_local=True, use_hbc=True, n_heads=4,
                    local_hidden=32, ell_wall=0.002, **kw, **common)
    hbc.register_cases(ds, verbose=False)
    hbc.eval()
    with torch.no_grad():
        for att in hbc.local.attn:      # defeat the trivial all-zero case
            att.out.weight.normal_(std=0.05)
        wall_pts = room["pool_wall_xyz"][:400].unsqueeze(0)
        out_w = hbc.decode_query(hbc.encode_geometry(pcb), wall_pts)
        vel_phys = out_w[0, :, 0:3] * hbc.sigma + hbc.mu
        interior = torch.rand(1, 200, 3) * 0.6 + 0.2
        out_i = hbc.decode_query(hbc.encode_geometry(pcb), interior)
        vel_i = out_i[0, :, 0:3] * hbc.sigma + hbc.mu
    worst = float(vel_phys.abs().max())
    print(f"        worst wall speed {worst:.3e} m/s"
          f"   [prior project HBC receipt: 1.5e-08 m/s]")
    # Machine precision, not bit-zero: the mask zeroes the PHYSICAL velocity,
    # which round-trips through normalized units as ((0-mu)/sigma)*sigma + mu.
    check("velocity is 0 m/s on solid points to machine precision",
          worst < 1e-7, f"{worst:.3e} m/s")
    check("the interior is NOT clamped", float(vel_i.abs().max()) > 1e-3,
          "the mask suppressed the whole field")
    check("temperature is untouched by the velocity mask",
          not torch.allclose(out_w[0, :, 4], torch.zeros(1)))

    print("\n[5] the legacy Trunk interface is satisfied")
    for m in ("encode_geometry", "decode_query", "forward", "register_cases"):
        check(f"has {m}()", callable(getattr(model, m, None)))
    with torch.no_grad():
        f = model(xyt, pcb)
    check("forward(xyt, pc) returns (B, N, 5)", tuple(f.shape) == (1, 256, 5),
          str(tuple(f.shape)))
    with torch.no_grad():
        chunked = torch.cat([model.decode_query(lat, xyt[:, s:s + 64])
                             for s in range(0, 256, 64)], dim=1)
        whole = model.decode_query(lat, xyt)
    check("chunked decoding matches whole-batch (eval path)",
          torch.allclose(chunked, whole, atol=1e-5),
          f"max diff {float((chunked - whole).abs().max()):.3e}")

    print("\n[6] an unregistered case fails loudly, not silently")
    stray = build_lgo_ginot(PC_C, out_channels=5, use_local=True, n_heads=4,
                      local_hidden=32, **kw, **common)
    try:
        stray.encode_geometry(pcb)
        check("raises when the case was never registered", False, "no error raised")
    except RuntimeError as e:
        check("raises when the case was never registered", "register_cases" in str(e))

    print("\n[8] EQUIVARIANCE OF THE LOCAL TERM (the architecture's claim)")
    # The global path reads absolute coordinates and is not equivariant by
    # design, so the full output MUST move under translation. What must not move
    # is the local term. It is read straight off the model rather than inferred
    # as out(on) - out(off): `output_proj` is a nonlinear MLP, so that difference
    # is f(x+h) - f(x) and drifts with the global path even when h is identical.
    d = np.array([0.137, -0.081, 0.023])
    room64 = {k: (v.double() if k.endswith("_xyz") else v) for k, v in room.items()}
    room_s = {k: (v + torch.tensor(d) if k.endswith("_xyz") else v)
              for k, v in room64.items()}
    pc_s = pc.clone()
    pc_s[:, :3] += torch.tensor(d).float()

    def local_stream(mdl, rm, pcx, q):
        mdl._index.clear()
        mdl.register_cases(FakeDataset(rm, pcx), verbose=False)
        mdl.debug_keep_local = True
        with torch.no_grad():
            mdl(q, pcx.unsqueeze(0))
        return mdl._last_local

    for mode in ("learned", "global"):
        m = build_lgo_ginot(PC_C, out_channels=5, use_local=True, n_heads=4,
                      local_hidden=32, local_query=mode, **kw, **common)
        m.eval()
        with torch.no_grad():                  # undo zero-init: a silent branch
            for a in m.local.attn:             # is trivially equivariant
                a.out.weight.normal_(std=0.05)
        base = local_stream(m, room64, pc, xyt)
        moved = local_stream(m, room_s, pc_s, xyt + torch.tensor(d).float())
        delta, scale = float((moved - base).abs().max()), float(base.abs().max())
        print(f"        local_query={mode:8}  max|delta| = {delta:.3e}   "
              f"(stream magnitude {scale:.3e})")
        if mode == "learned":
            check("two-stream: local term survives translation unchanged",
                  delta < 1e-5 * max(scale, 1e-12), f"delta {delta:.3e} vs {scale:.3e}")
            check("...and the term is not trivially zero", scale > 1e-3,
                  f"magnitude {scale:.3e}")
        else:
            check("interleaved mode is measurably NOT equivariant (documents D6)",
                  delta > 1e-3 * max(scale, 1e-12), f"delta {delta:.3e} vs {scale:.3e}")

    print("\n[7] gradients reach both paths")
    model.train()
    out = model(xyt, pcb)
    out.pow(2).mean().backward()
    g_local = sum(float(p.grad.abs().sum()) for p in model.local.parameters()
                  if p.grad is not None)
    g_trunk = sum(float(p.grad.abs().sum()) for p in model.trunk.parameters()
                  if p.grad is not None)
    check("local branch receives gradient", g_local > 0, f"{g_local:.3e}")
    check("GINOT trunk receives gradient", g_trunk > 0, f"{g_trunk:.3e}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
