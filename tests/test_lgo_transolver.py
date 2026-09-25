"""Guards for LGO-Transolver.

⚠ THE FIRST TEST IS THE IMPORTANT ONE AND IT EXISTS FOR A SPECIFIC RISK.
`model/lgo_transolver.py` may not edit `baselines/models/transolver.py` (user
instruction 2026-09-22), and Transolver exposes no `h_delta` hook, so its
`decode_query` RESTATES the baseline's composition. That duplication can drift
silently: change `TransolverBaseline.decode_query` and the wrapper keeps running,
keeps training, and quietly stops being the same trunk -- which would turn the
ablation pair into two different architectures while every banner and log looked
normal. Test [1] pins it: local branch OFF must be BIT-IDENTICAL to the baseline.

Run after ANY change to either file:
    /path/to/ml-env/bin/python -m tests.test_lgo_transolver
"""

import os
import sys

import torch

# `legacy/` must be on the path before baselines import `ginot.*` --
# same two lines as tests/test_lgo_gino.py.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

from baselines.models import build_baseline
from model.lgo_transolver import LGOTransolver
from tests.test_knn_cache import make_room
from tests.test_lgo_ginot import PC_C, FakeDataset, make_pc

torch.manual_seed(0)
OK = []
N_CLOUD = 400

# Built once: `register_cases` needs a real room, and the local branch needs the
# FULL boundary pools (it refuses to derive them from `pc` -- see the error text
# in BoundaryLookup). Reusing the repo's own fixtures rather than inventing a
# stub keeps this test honest about the interface the trainer really uses.
_ROOM = make_room(n_wall=3000, n_supply=300, n_return=300, n_leak=100)
_PC = make_pc(_ROOM, n=N_CLOUD)
_DS = FakeDataset(_ROOM, _PC)


def _build(seed=0, cfg=None):
    """The arm and a BARE baseline carrying identical trunk weights."""
    cfg = cfg or {"hidden_dim": 64, "n_layers": 2, "n_heads": 4,
                  "slice_num": 8, "mlp_ratio": 2, "n_pc_tokens": 4096}
    torch.manual_seed(seed)
    trans = build_baseline("transolver", cfg, pc_channels=PC_C, out_channels=5)
    torch.manual_seed(seed)
    bare = build_baseline("transolver", cfg, pc_channels=PC_C, out_channels=5)
    bare.load_state_dict(trans.state_dict())

    model = LGOTransolver(trans, _DS.coord_min, _DS.coord_scale,
                          _DS.target_mean, _DS.target_std,
                          k={"supply": 8, "return": 8, "solid": 8, "leak": 4},
                          n_heads=4, local_hidden=32)
    model.register_cases(_DS, verbose=False)
    model.eval()
    return bare.eval(), model


def _case(n_q=64):
    return _PC.unsqueeze(0), torch.rand(1, n_q, 3)


def test_1_branch_off_is_bit_identical():
    """[1] use_local=False MUST reproduce TransolverBaseline exactly."""
    trans, model = _build()
    pc, xyz = _case()
    model.eval(); trans.eval()
    with torch.no_grad():
        ref = trans.decode_query(trans.encode_geometry(pc), xyz)
        model.use_local = False
        # _resolve needs a registered case; encode_geometry does it.
        got = model.decode_query(model.encode_geometry(pc), xyz)
    d = (ref - got).abs().max().item()
    assert d == 0.0, (
        f"[1] FAIL: branch-off output differs from the baseline by {d:.3e}. "
        "decode_query has DRIFTED from TransolverBaseline.decode_query -- "
        "re-sync it before trusting any lgo_transolver number.")
    OK.append("[1] branch off == baseline, bit-identical")


def test_2_branch_on_changes_output():
    """[2] The injection must be WIRED -- checked on PERTURBED local weights.

    ⚠ The branch is ZERO-INITIALISED, which is the whole point: at init the pair
    is bit-identical, so the ablation starts from the same model. That means a
    naive "on vs off" check passes trivially for a branch that is wired to
    NOTHING -- it would read zero difference either way. So perturb the local
    parameters first and only then demand movement. Same guard, same reason, as
    tests/test_lgo_gino.py.
    """
    _bare, model = _build()
    pc, xyz = _case()
    model.eval()
    with torch.no_grad():
        model.use_local = True
        before = model.decode_query(model.encode_geometry(pc), xyz)
        zero = (before - _bare_ref(model, pc, xyz)).abs().max().item()
        g = torch.Generator().manual_seed(7)
        for prm in model.local.parameters():
            prm.add_(0.5 * torch.randn(prm.shape, generator=g))
        after = model.decode_query(model.encode_geometry(pc), xyz)
    d = (after - before).abs().max().item()
    assert zero == 0.0, (
        f"[2] FAIL: the branch is not zero-init at construction ({zero:.3e}); "
        "the pair is then NOT bit-identical at init.")
    assert d > 1e-6, (
        f"[2] FAIL: perturbing the local branch moved the output by {d:.3e}. "
        "The injection is wired to nothing -- exactly the silently-inert arm "
        "`--global_query local` shows on `lgo_ginot_local`.")
    OK.append(f"[2] zero-init at construction, and wired "
              f"(perturbed -> max |d| {d:.3e})")


def _bare_ref(model, pc, xyz):
    """Baseline output through the same wrapper, branch off."""
    model.use_local = False
    try:
        return model.decode_query(model.encode_geometry(pc), xyz)
    finally:
        model.use_local = True


def test_3_head_is_shared():
    """[3] The wrapper adds ONLY local-branch parameters -- the trunk is shared."""
    trans, model = _build()
    n_trans = sum(p.numel() for p in trans.parameters())
    n_local = sum(p.numel() for p in model.local.parameters())
    n_tot = sum(p.numel() for p in model.parameters())
    assert n_tot == n_trans + n_local, (
        f"[3] FAIL: {n_tot} != {n_trans} + {n_local}. The wrapper introduced "
        "parameters outside the local branch, so the pair is not an ablation.")
    OK.append(f"[3] params = trunk {n_trans:,} + local {n_local:,} exactly "
              f"(+{100*n_local/n_trans:.2f}%)")


def test_4_local_width_matches_trunk():
    """[4] The branch emits the trunk width -- no projection, no reshape."""
    trans, model = _build()
    assert model.local.d_token == trans.placeholder.shape[0], (
        f"[4] FAIL: local d_token {model.local.d_token} != trunk width "
        f"{trans.placeholder.shape[0]}.")
    OK.append(f"[4] local width == trunk width ({model.local.d_token})")


def test_5_boundary_rows_untouched():
    """[5] Seed injection must condition QUERY rows only.

    Adding a query-conditioned vector to the boundary tokens would leak the
    query into the geometry encoding -- the tokens the branch gathers FROM.
    """
    _trans, model = _build()
    pc, xyz = _case(n_q=16)
    model.eval()
    with torch.no_grad():
        latent = model.encode_geometry(pc)
        t = model.transolver
        b_tok = t._tokens(latent[:, :, :3], latent[:, :, 3:], is_query=False)
        n_b = b_tok.shape[1]
        # Two different query sets over the SAME geometry: the boundary half of
        # the token stream must be identical for both.
        outs = []
        for _ in range(2):
            q = torch.rand(1, 16, 3)
            q_extra = torch.zeros(1, 16, t.pc_channels - 3)
            q_tok = t._tokens(q, q_extra, is_query=True)
            fx = t.preprocess(torch.cat([b_tok, q_tok], dim=1)) + t.placeholder
            outs.append(fx[:, :n_b].clone())
    d = (outs[0] - outs[1]).abs().max().item()
    assert d == 0.0, f"[5] FAIL: boundary rows depend on the query ({d:.3e})."
    OK.append("[5] boundary rows independent of the query set")


if __name__ == "__main__":
    tests = [test_1_branch_off_is_bit_identical, test_2_branch_on_changes_output,
             test_3_head_is_shared, test_4_local_width_matches_trunk,
             test_5_boundary_rows_untouched]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  {t.__name__} ERROR: {type(e).__name__}: {e}")
            failed += 1
    for line in OK:
        print(" ", line)
    print(f"\n[test_lgo_transolver] {len(OK)}/{len(tests)} pass, {failed} fail")
    raise SystemExit(1 if failed else 0)
