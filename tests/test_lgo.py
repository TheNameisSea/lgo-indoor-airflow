"""Integration tests for model/lgo.py -- THE MAIN ARCHITECTURE.

The claims that must hold before any training run is worth doing:

  1. At initialisation the hybrid is BIT-IDENTICAL to the plain `deeponet`
     baseline, for both injection points, because the local branch is
     zero-initialised. The A/B therefore starts from one model and every gain
     has to be earned -- the same discipline that makes "LGO minus local ==
     GINOT" hold bit-for-bit.
  2. Once the local branch is non-zero it DOES move the prediction, for both
     injection points. Without this, (1) is satisfied by a dead code path.
  3. `inject='basis'` is exactly basis augmentation through the SHARED head:
     trunk_out(h + h_local) == trunk_out(h) + W_out @ h_local. This is the claim
     the docstring makes to justify adding zero parameters, so it is checked
     rather than asserted.
  4. The local term is translation-equivariant: translating the room and the
     query together leaves the local stream unchanged, while moving the vents
     alone does not. This is the property the whole local branch exists for and
     it must survive the change of decoder.
  5. `inject` changes behaviour but NOT parameter names, so a checkpoint trained
     under one setting loads without complaint under the other. The test pins it so the rebuild path is never allowed to default.
  6. The legacy Trunk interface is satisfied, so legacy/eval_thermo.py and
     evaluate.py score this model through the same path as every baseline.

Run: python -m tests.test_lgo
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

from baselines.models import build_baseline
from model.lgo import LGO
from tests.test_knn_cache import make_room
from tests.test_lgo_ginot import PC_C, FakeDataset, make_pc

PASS, FAIL = [], []

# Small enough to run on CPU in seconds; the properties under test are
# structural and do not depend on width.
CFG = dict(n_basis=32, trunk_width=32, trunk_layers=2, branch_out_c=32,
           branch_width=32, branch_latent_d=64, branch_n_point=64, num_bands=4)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}"
          f"{('  -- ' + detail) if detail else ''}")


def build(ds, inject="basis", film=True, seed=0, **kw):
    torch.manual_seed(seed)
    deeponet = build_baseline("deeponet", dict(CFG, film=film),
                              pc_channels=PC_C, out_channels=5,
                              min_length_norm=0.05)
    model = LGO(deeponet, ds.coord_min, ds.coord_scale, ds.target_mean,
                        ds.target_std,
                        k={"supply": 8, "return": 8, "solid": 8, "leak": 4},
                        n_heads=4, local_hidden=32, inject=inject, **kw)
    model.register_cases(ds, verbose=False)
    model.eval()
    return model, deeponet


def excite(model, std=0.05, seed=1):
    """Defeat the zero-init so the local branch actually contributes."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for att in model.local.attn:
            att.out.weight.normal_(std=std)
            att.out.bias.normal_(std=std)


def main():
    torch.manual_seed(0)
    room = make_room(n_wall=3000, n_supply=300, n_return=300, n_leak=100)
    pc = make_pc(room)
    ds = FakeDataset(room, pc)
    xyt = torch.rand(1, 256, 3) * 0.8 + 0.1
    pcb = pc.unsqueeze(0)

    print("\n[1] zero-init: the hybrid == the plain deeponet baseline at step 0")
    for inject in ("basis", "seed"):
        model, deeponet = build(ds, inject=inject)
        with torch.no_grad():
            base = deeponet.decode_query(deeponet.encode_geometry(pcb), xyt)
            hyb = model.decode_query(model.encode_geometry(pcb), xyt)
        check(f"inject={inject!r}: bit-identical to the baseline",
              torch.equal(base, hyb),
              f"max diff {float((base - hyb).abs().max()):.3e}")

    print("\n[2] once trained, the local branch DOES change the output")
    for inject in ("basis", "seed"):
        model, deeponet = build(ds, inject=inject)
        with torch.no_grad():
            base = model.decode_query(model.encode_geometry(pcb), xyt)
        excite(model)
        with torch.no_grad():
            moved = model.decode_query(model.encode_geometry(pcb), xyt)
        d = float((base - moved).abs().max())
        check(f"inject={inject!r}: a non-zero local branch moves the prediction",
              d > 1e-3, f"{d:.3e}")

    print("\n[3] inject='basis' IS basis augmentation through the shared head")
    model, deeponet = build(ds, inject="basis")
    excite(model)
    with torch.no_grad():
        lat = model.encode_geometry(pcb)
        coeff, shape_vec = lat
        # the hybrid's own output
        got = model.decode_query(lat, xyt)
        # rebuilt by hand: trunk_out(h) + W_out @ h_local, contracted with coeff
        h = deeponet.trunk_in(deeponet.fourier(xyt))
        for layer in deeponet.trunk_layers:
            h = layer(h, shape_vec) if deeponet.film else layer(h) + h
        xyz_m = model._metres(xyt[0], model._active)
        h_local = model._local_stream(xyz_m, xyt.device, 1, dtype=coeff.dtype)
        basis = deeponet.trunk_out(h) + (h_local @ deeponet.trunk_out.weight.T)
        want = torch.bmm(basis, coeff) / deeponet.n_basis ** 0.5 + deeponet.bias
    d = float((got - want).abs().max())
    check("trunk_out(h + h_local) == trunk_out(h) + W_out @ h_local",
          d < 1e-5, f"max diff {d:.3e}")

    print("\n[4] the local term is translation-equivariant")
    # Build the three indices EXPLICITLY and set `_active` by hand, rather than
    # going through register_cases/_resolve. Two reasons, both of which bit an
    # earlier version of this test and produced a plausible-looking failure:
    #   * the vents-alone control does not change `pc_full`, so it has the SAME
    #     cloud fingerprint as the reference and register_cases silently SKIPS
    #     it -- the control then re-measures whichever index was registered
    #     last, which is not the one it names.
    #   * surface normals are oriented against the INTERIOR pools
    #     (pool_stream_xyz / pool_bg_xyz), so a "translated room" that shifts
    #     only the boundary changes the normals and fails equivariance for a
    #     reason that has nothing to do with the local branch.
    # coord_scale is 1 in the fake dataset, so a shift in normalized units is a
    # shift in metres.
    from data.knn_cache import BoundaryIndex

    shift = torch.tensor([0.05, -0.03, 0.02])

    def index_for(r):
        return BoundaryIndex(r, ds.coord_min, ds.coord_scale, ds.target_mean,
                             ds.target_std, groups=model.groups,
                             far_voxels=model.far_voxels)

    # EVERY position pool moves -- boundary and interior alike.
    moved_room = {k: (v + shift if k.endswith("_xyz") else v)
                  for k, v in room.items()}
    # The control: the supply vents move, the room does not.
    vent_room = dict(room, pool_in_xyz=room["pool_in_xyz"] + shift)

    model, _ = build(ds, inject="basis")
    excite(model)
    with torch.no_grad():
        model._active = index_for(room)
        h_a = model._local_stream(model._metres(xyt[0], model._active),
                                  xyt.device, 1)
        model._active = index_for(moved_room)
        h_b = model._local_stream(
            model._metres((xyt + shift)[0], model._active), xyt.device, 1)
        model._active = index_for(vent_room)
        h_c = model._local_stream(model._metres(xyt[0], model._active),
                                  xyt.device, 1)
    joint = float((h_a - h_b).abs().max())
    # If `vents` were also ~0 the test would be vacuous -- it would mean the
    # branch ignores geometry, not that it is equivariant.
    vents = float((h_a - h_c).abs().max())
    print(f"        joint translation {joint:.3e}   vents alone {vents:.3e}")
    check("the local stream is unchanged under joint translation",
          joint < 1e-4, f"{joint:.3e}")
    check("moving the vents ALONE does change it (the test is not vacuous)",
          vents > 1e-3, f"{vents:.3e}")

    print("\n[5] inject changes behaviour but NOT parameter names")
    m_basis, _ = build(ds, inject="basis", seed=3)
    m_seed, _ = build(ds, inject="seed", seed=3)
    check("the two state_dicts have identical keys",
          list(m_basis.state_dict()) == list(m_seed.state_dict()))
    excite(m_basis)
    m_seed.load_state_dict(m_basis.state_dict())    # loads WITHOUT complaint
    with torch.no_grad():
        a = m_basis.decode_query(m_basis.encode_geometry(pcb), xyt)
        b = m_seed.decode_query(m_seed.encode_geometry(pcb), xyt)
    d = float((a - b).abs().max())
    check("...yet the SAME weights predict differently, so inject must come "
          "from args.json", d > 1e-3, f"{d:.3e}")

    print("\n[6] the legacy Trunk interface is satisfied")
    model, _ = build(ds)
    for m in ("encode_geometry", "decode_query", "forward", "register_cases"):
        check(f"has {m}()", callable(getattr(model, m, None)))
    with torch.no_grad():
        f = model(xyt, pcb)
    check("forward(xyt, pc) returns (B, N, 5)", tuple(f.shape) == (1, 256, 5),
          str(tuple(f.shape)))
    with torch.no_grad():
        f2 = model.decode_query(model.encode_geometry(pcb), xyt[0])
    check("decode_query accepts an unbatched query", tuple(f2.shape) == (256, 5),
          str(tuple(f2.shape)))
    check("evaluate.py's registration gate is set", model.use_local is True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
