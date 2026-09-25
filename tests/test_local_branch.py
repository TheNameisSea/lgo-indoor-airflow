"""Equivariance receipt and invariants for model/local_branch.py.

The headline test mirrors the prior project's SE receipt: translating the room
and the query TOGETHER must leave the branch unchanged to floating-point
round-off, while translating a vent ALONE must change it materially. Both halves
matter -- the first passes trivially for a branch that ignores its input.

The property lives in `encode()`: neighbour tokens are built from relative
quantities only. Whether the FULL model is equivariant depends on what the trunk
feeds in as `q`, which is a trunk decision and is tested there.

Run: python -m tests.test_local_branch
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.knn_cache import BoundaryIndex
from model.local_branch import LocalBranch, neighbours_to_torch
from tests.test_knn_cache import make_room

PASS, FAIL = [], []
K = {"supply": 16, "return": 8, "solid": 16, "leak": 4}


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def index_of(room):
    return BoundaryIndex(room, torch.zeros(3), torch.ones(3), torch.zeros(5), torch.ones(5))


def shift_room(room, d):
    """Translate a room, in float64.

    float64 on purpose: this is the receipt, and storing a room translated 100 m
    in float32 loses ~4e-6 m of the offsets we are trying to prove exact -- and
    can flip which neighbour lands in which slot. The limiting error would then
    be the test's own storage, not the branch.
    """
    t = torch.tensor(d, dtype=torch.float64)
    return {k: (v.double() + t if k.endswith("_xyz") else v) for k, v in room.items()}


def tokens_for(branch, room, xq):
    nb = neighbours_to_torch(index_of(room).query(xq, K, dtype=np.float64), "cpu",
                             dtype=torch.float64)
    with torch.no_grad():
        # encode() grew a third return (the raised-cosine window) with
        # --soft_knn; it is None on the default hard gather and none of the
        # assertions below depend on it.
        tokens, valid, _soft = branch.encode(nb)
    return tokens, valid


def main():
    torch.manual_seed(0)
    branch = LocalBranch(d_model=64, n_heads=4, n_layers=2).double()

    room = {k: (v.double() if k.endswith("_xyz") else v)
            for k, v in make_room(seed=1).items()}
    xq = np.random.default_rng(7).uniform(0.05, 0.95, (128, 3))
    base, valid = tokens_for(branch, room, xq)

    print("\n[1] translation equivariance (the receipt)")
    d = np.array([0.137, -0.081, 0.023])
    moved, _ = tokens_for(branch, shift_room(room, d), xq + d)
    d_equi = float((moved - base).abs().max())
    check("room + query shifted together: tokens unchanged", d_equi < 1e-12,
          f"max |delta| = {d_equi:.3e}")

    far = np.array([100.0, -50.0, 25.0])
    far_tok, _ = tokens_for(branch, shift_room(room, far), xq + far)
    d_far = float((far_tok - base).abs().max())
    check("identical 100 m away (no absolute coordinate reaches the branch)",
          d_far < 1e-12, f"max |delta| = {d_far:.3e}")

    print("\n[2] the receipt is not passing trivially")
    room_v = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in room.items()}
    room_v["pool_in_xyz"] = room_v["pool_in_xyz"] + torch.tensor([0.4, 0.0, 0.0])
    vent_tok, _ = tokens_for(branch, room_v, xq)
    d_sens = float((vent_tok - base).abs().max())
    check("moving the vent alone changes the tokens", d_sens > 1e-3,
          f"max |delta| = {d_sens:.3e}")
    check("sensitivity exceeds round-off by >1e6x",
          d_sens / max(d_equi, 1e-300) > 1e6,
          f"ratio = {d_sens / max(d_equi, 1e-300):.1e}")

    print("\n[3] attention invariants")
    q = torch.randn(len(xq), 64, dtype=torch.float64)
    for a in branch.attn:                       # undo zero-init to test the math
        torch.nn.init.normal_(a.out.weight, std=0.05)
    with torch.no_grad():
        out = branch(q, base, valid)
        perm = torch.randperm(base.shape[1])
        out_p = branch(q, base[:, perm], valid[:, perm])
    check("permuting neighbours leaves the output unchanged",
          float((out - out_p).abs().max()) < 1e-9,
          f"max |delta| = {float((out - out_p).abs().max()):.3e}")

    garbage = base.clone()
    garbage[~valid] = 1e3
    with torch.no_grad():
        out_g = branch(q, garbage, valid)
    check("garbage in padded slots does not leak", float((out - out_g).abs().max()) < 1e-9,
          f"max |delta| = {float((out - out_g).abs().max()):.3e}")

    print("\n[4] degenerate inputs stay finite")
    with torch.no_grad():
        out_dead = branch(q, base, torch.zeros_like(valid))
    check("query with no valid neighbours is finite, not NaN", torch.isfinite(out_dead).all())
    check("...and contributes exactly zero", float(out_dead.abs().max()) == 0.0,
          f"max |out| = {float(out_dead.abs().max()):.3e}")

    room_nl = {k: (v.double() if k.endswith("_xyz") else v)
               for k, v in make_room(n_leak=0, seed=1).items()}
    nb_nl = neighbours_to_torch(index_of(room_nl).query(xq[:8], K, dtype=np.float64),
                                "cpu", dtype=torch.float64)
    with torch.no_grad():
        t_nl, v_nl, _ = branch.encode(nb_nl)
        o_nl = branch(q[:8], t_nl, v_nl)
    check("a case with an empty group still works", torch.isfinite(o_nl).all())

    print("\n[5] zero-init: the branch is a no-op at step 0")
    fresh = LocalBranch(d_model=64, n_heads=4, n_layers=2).double()
    with torch.no_grad():
        z = fresh(q, base, valid)
    check("output is exactly zero at init", float(z.abs().max()) == 0.0,
          f"max |out| = {float(z.abs().max()):.3e}")

    print("\n[6] gradients reach the neighbour encoder once it starts learning")
    tr = LocalBranch(d_model=64, n_heads=4, n_layers=1).double()
    with torch.no_grad():
        tr.attn[0].out.weight.normal_(std=0.02)
    nb = neighbours_to_torch(index_of(room).query(xq, K, dtype=np.float64), "cpu",
                             dtype=torch.float64)
    tok, val, _ = tr.encode(nb)
    tr(torch.randn(len(xq), 64, dtype=torch.float64), tok, val).square().mean().backward()
    g = tr.encoder.mlp[0].weight.grad
    check("encoder receives gradient", g is not None and float(g.abs().sum()) > 0,
          "" if g is not None else "grad is None")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
