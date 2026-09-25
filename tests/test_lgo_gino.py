"""Integration tests for model/lgo_gino.py -- the local branch on a GINO host.

The claims that must hold before a training run is worth doing. Each maps to a
sentence in `model/lgo_gino.py`'s docstring that would otherwise be an
assertion nobody checked:

  1. At initialisation the hybrid is BIT-IDENTICAL to the plain `gino`
     baseline, because the local branch is zero-initialised. "Hybrid minus
     local branch" is therefore literally the baseline arm and the A/B carries
     no confound -- the same discipline `tests/test_lgo.py` pins for the
     DeepONet host.
  2. Once the local branch is non-zero it DOES move the prediction. Without
     this, (1) is satisfied by a dead code path.
  3. The local term is translation-equivariant: translating the room and the
     query together leaves the local stream unchanged, while moving the vents
     alone does not. This is the property the branch exists for and it must
     survive the change of host.
  4. A checkpoint round-trips. For this host that means neuralop's `_metadata`
     key, which wrapping buries at `gino.net._metadata`, does not break
     `load_state_dict` -- see LGOGino.load_state_dict.
  5. The legacy Trunk interface is satisfied, so evaluate.py scores this model
     through the same path as every baseline.

Run: python -m tests.test_lgo_gino
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

from baselines.models import build_baseline
from model.lgo_gino import LGOGino
from tests.test_knn_cache import make_room
from tests.test_lgo_ginot import PC_C, FakeDataset, make_pc

PASS, FAIL = [], []

# Small enough to run on CPU in seconds. The properties under test are
# structural and do not depend on width -- but note grid_res stays a real 3-D
# grid, because the injection point is downstream of the output GNO and a
# degenerate grid would not exercise it.
CFG = dict(grid_res=8, fno_hidden=32, fno_layers=2, fno_modes=4,
           gno_mlp_hidden=32, n_pc_tokens=512, in_radius=0.25, out_radius=0.25)

# MUST stay <= CFG["n_pc_tokens"]. GINOBaseline.encode_geometry draws a fresh
# torch.randperm subsample whenever the cloud is larger, which makes its own
# forward NONDETERMINISTIC -- two calls on identical weights then differ by
# ~2e-3 and the bit-identity test below fails for a reason that has nothing to
# do with the local branch. Test [0] pins this so the next person gets a
# diagnosis instead of a mystery.
N_CLOUD = 400


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}"
          f"{('  -- ' + detail) if detail and not cond else ''}")


def build(ds, seed=0):
    torch.manual_seed(seed)
    gino = build_baseline("gino", dict(CFG), pc_channels=PC_C, out_channels=5)
    torch.manual_seed(seed)
    bare = build_baseline("gino", dict(CFG), pc_channels=PC_C, out_channels=5)
    bare.load_state_dict(gino.state_dict())

    model = LGOGino(gino, ds.coord_min, ds.coord_scale, ds.target_mean,
                    ds.target_std,
                    k={"supply": 8, "return": 8, "solid": 8, "leak": 4},
                    n_heads=4, local_hidden=32)
    model.register_cases(ds, verbose=False)
    model.eval()
    return model, bare.eval()


def excite(model, scale=0.05, seed=7):
    """Make the local branch non-zero, or every test below passes vacuously."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.local.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))


def main():
    torch.manual_seed(0)
    room = make_room(n_wall=3000, n_supply=300, n_return=300, n_leak=100)
    pc = make_pc(room, n=N_CLOUD)
    ds = FakeDataset(room, pc)

    xyt = torch.rand(1, 64, 3)
    pc_in = pc.unsqueeze(0)

    print("\n[0] the bare host is deterministic (precondition for [1] and [4])")
    model, bare = build(ds)
    with torch.no_grad():
        d0 = float((bare(xyt, pc_in) - bare(xyt, pc_in)).abs().max())
    check("gino baseline is deterministic across calls", d0 == 0.0,
          f"max diff {d0:.3e} -- N_CLOUD ({N_CLOUD}) exceeds n_pc_tokens "
          f"({CFG['n_pc_tokens']}), so encode_geometry is resampling")

    print("\n[1] zero-init: hybrid == bare gino, bit for bit")
    with torch.no_grad():
        ref = bare(xyt, pc_in)
        got = model(xyt, pc_in)
    d = float((ref - got).abs().max())
    check("LGOGino(init) == gino baseline", d == 0.0, f"max diff {d:.3e}")
    check("output shape", tuple(got.shape) == (1, 64, 5), str(tuple(got.shape)))

    print("\n[2] an excited local branch moves the prediction")
    excite(model)
    with torch.no_grad():
        moved = model(xyt, pc_in)
    d2 = float((moved - ref).abs().max())
    # Guards against (1) passing because the injection is wired to nothing.
    check("local branch changes the output", d2 > 1e-6, f"max diff {d2:.3e}")

    print("\n[3] the local term is translation-equivariant")
    # Indices built EXPLICITLY and `_active` set by hand, for the two reasons
    # documented in tests/test_lgo.py: the vents-alone control shares a cloud
    # fingerprint with the reference so register_cases would silently skip it,
    # and normals are oriented against the interior pools so a boundary-only
    # shift would break equivariance for an unrelated reason.
    from data.knn_cache import BoundaryIndex

    shift = torch.tensor([0.05, -0.03, 0.02])

    def index_for(r):
        return BoundaryIndex(r, ds.coord_min, ds.coord_scale, ds.target_mean,
                             ds.target_std, groups=model.groups,
                             far_voxels=model.far_voxels)

    moved_room = {k: (v + shift if k.endswith("_xyz") else v)
                  for k, v in room.items()}
    vent_room = dict(room, pool_in_xyz=room["pool_in_xyz"] + shift)

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
    vents = float((h_a - h_c).abs().max())
    print(f"        joint translation {joint:.3e}   vents alone {vents:.3e}")
    check("room+query translate together -> unchanged", joint < 1e-5,
          f"{joint:.3e}")
    # Without this the test would be satisfied by a branch that ignores
    # geometry entirely, which is equivariant and useless.
    check("vents alone -> changed (test is not vacuous)", vents > 1e-4,
          f"{vents:.3e}")

    print("\n[4] checkpoint round-trips past neuralop's nested _metadata")
    sd = model.state_dict()
    has_meta = any(k.endswith("_metadata") for k in sd)
    # If neuralop stops injecting this, the strip in load_state_dict is dead
    # code and should be deleted rather than left as false reassurance.
    check("state_dict carries a _metadata key (the trap exists)", has_meta)
    fresh, _ = build(ds, seed=3)
    try:
        fresh.load_state_dict(sd)          # strict=True: must not raise
        loaded = True
        err = ""
    except Exception as e:                 # noqa: BLE001 - reporting the reason
        loaded, err = False, f"{type(e).__name__}: {e}"
    check("load_state_dict(strict=True) accepts it", loaded, err)
    if loaded:
        fresh.eval()
        with torch.no_grad():
            d4 = float((fresh(xyt, pc_in) - model(xyt, pc_in)).abs().max())
        check("round-tripped weights reproduce the output", d4 == 0.0,
              f"max diff {d4:.3e}")

    print("\n[5] the Trunk interface evaluate.py needs")
    for m in ("encode_geometry", "decode_query", "forward", "register_cases"):
        check(f"has {m}()", callable(getattr(model, m, None)))
    for attr in ("use_local", "use_hbc", "grad_wrt_query"):
        check(f"declares {attr}", hasattr(model, attr))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
