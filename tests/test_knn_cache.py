"""Tests for data/knn_cache.py, on synthetic rooms where the answer is known.

Run: python -m tests.test_knn_cache
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.knn_cache import GROUPS, BoundaryIndex, surface_normals

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail and not cond else ''}")


def make_room(n_wall=2000, n_supply=200, n_return=200, n_leak=50, seed=0):
    """A 1x1x1 m box in NORMALIZED coords (coord_scale=1) so metres == units.

    Floor at z=0, supply patch at z=1 blowing down (w<0, cold), return patch at
    z=1 (w>0, warm), a leak strip on a wall.
    """
    g = torch.Generator().manual_seed(seed)
    r = lambda n, d=2: torch.rand(n, d, generator=g)

    wall = torch.cat([torch.cat([r(n_wall), torch.zeros(n_wall, 1)], 1)], 0)  # floor z=0
    sup = torch.cat([r(n_supply) * 0.2 + 0.1, torch.ones(n_supply, 1)], 1)    # x,y in [.1,.3]
    ret = torch.cat([r(n_return) * 0.2 + 0.7, torch.ones(n_return, 1)], 1)    # x,y in [.7,.9]
    leak = torch.cat([torch.zeros(n_leak, 1), r(n_leak)], 1)                  # x=0 wall

    def tgt(n, uvw, temp):
        t = torch.zeros(n, 5)
        t[:, 0:3] = torch.tensor(uvw)
        t[:, 4] = temp
        return t

    return {
        "pool_wall_xyz": wall, "pool_wall_tgt": tgt(len(wall), (0, 0, 0), 0.0),
        "pool_in_xyz": sup,    "pool_in_tgt": tgt(len(sup), (0, 0, -1.16), -1.7),
        "pool_out_xyz": ret,   "pool_out_tgt": tgt(len(ret), (0, 0, 0.87), +0.9),
        "pool_leak_xyz": leak, "pool_leak_tgt": tgt(len(leak), (0.3, 0, 0), 0.4),
        "pool_leak_mag": torch.full((len(leak),), 0.3),
        "pool_stream_xyz": torch.rand(500, 3, generator=g),
        "pool_bg_xyz": torch.rand(500, 3, generator=g),
    }


def index_for(room):
    return BoundaryIndex(room, torch.zeros(3), torch.ones(3),
                         torch.zeros(5), torch.ones(5))


def main():
    print("\n[1] surface normals")
    # A plane at z=0 must have normals along +/-z; oriented toward fluid => +z.
    plane = np.random.default_rng(0).uniform(0, 1, (2000, 3)); plane[:, 2] = 0.0
    fluid = np.random.default_rng(1).uniform(0.1, 1, (500, 3))
    n = surface_normals(plane, fluid, k=16)
    check("plane normals are unit", np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5))
    check("plane normals align with z", np.abs(n[:, 2]).mean() > 0.99,
          f"mean |n_z| = {np.abs(n[:, 2]).mean():.4f}")
    check("plane normals point at the fluid (+z)", (n[:, 2] > 0).all(),
          f"{(n[:, 2] <= 0).sum()} of {len(n)} point away")

    room = make_room()
    bi = index_for(room)
    print(f"\n[2] index construction  {bi}")
    check("all four groups present", all(bi.count[g] > 0 for g in GROUPS))

    print("\n[3] kNN correctness vs brute force")
    xq = np.random.default_rng(2).uniform(0, 1, (64, 3))
    out = bi.query(xq, {"supply": 8, "return": 8, "solid": 16, "leak": 4})
    ok_all = True
    for g, k in (("supply", 8), ("return", 8), ("solid", 16), ("leak", 4)):
        dxyz, feat, valid = out[g]
        d_got = np.sort(np.linalg.norm(dxyz, axis=2), axis=1)
        d_bru = np.sort(np.linalg.norm(bi.xyz[g][None, :, :] - xq[:, None, :], axis=2), axis=1)[:, :k]
        ok_all &= np.allclose(d_got, d_bru, atol=1e-6)
        check(f"{g}: k={k} nearest match brute force", np.allclose(d_got, d_bru, atol=1e-6),
              f"max diff {np.abs(d_got - d_bru).max():.2e}")
    check("offsets are boundary MINUS query", np.allclose(
        out["supply"][0][0, 0], bi.xyz["supply"][bi.tree["supply"].query(xq[:1], k=1)[1][0]] - xq[0],
        atol=1e-6))

    print("\n[4] wall distance is exact where exactness is claimed")
    on_wall = bi.xyz["solid"][:200]
    phi = bi.wall_distance(on_wall)
    check("phi == 0 exactly on solid points", (phi == 0.0).all(),
          f"max phi on wall = {phi.max():.3e}")
    ell = 0.002
    check("no-slip mask == 0 exactly on solid points", ((phi / (phi + ell)) == 0.0).all())
    interior = np.array([[0.5, 0.5, 0.5]])
    check("phi is the true distance in the interior",
          np.isclose(bi.wall_distance(interior)[0],
                     np.linalg.norm(bi.xyz["solid"] - interior, axis=1).min()))
    check("leaks are NOT walls", bi.wall_distance(bi.xyz["leak"][:20]).min() > 1e-6,
          "a leak point returned phi=0, so it was treated as no-slip")

    print("\n[5] blinding: only supply temperature survives")
    T = {g: bi.feat[g][:, 6] for g in GROUPS}
    check("supply temperature kept", np.abs(T["supply"]).max() > 0.1,
          f"max |T_supply| = {np.abs(T['supply']).max():.3f}")
    check("return temperature blinded", np.abs(T["return"]).max() == 0.0,
          f"max |T_return| = {np.abs(T['return']).max():.3f} -- LEAKS THE ANSWER")
    check("solid temperature blinded", np.abs(T["solid"]).max() == 0.0)
    check("leak temperature blinded", np.abs(T["leak"]).max() == 0.0)
    check("leak velocity blinded, magnitude kept",
          np.abs(bi.feat["leak"][:, 3:6]).max() == 0.0 and np.allclose(bi.feat["leak"][:, 7], 0.3))
    check("supply velocity kept", np.isclose(bi.feat["supply"][:, 5].mean(), -1.16))

    print("\n[6] short groups pad instead of crashing")
    small = index_for(make_room(n_leak=3))
    d, f, v = small.query(np.random.rand(5, 3), {"leak": 8})["leak"]
    check("padded to k", d.shape == (5, 8, 3) and f.shape == (5, 8, 8))
    check("validity flags mark the padding", v[:, :3].all() and not v[:, 3:].any())

    print("\n[7] translation equivariance of the gathered offsets")
    shift = np.array([0.13, -0.07, 0.02])
    room_s = {k: (v + torch.tensor(shift, dtype=torch.float32) if k.endswith("_xyz") else v)
              for k, v in make_room().items()}
    bi_s = BoundaryIndex(room_s, torch.zeros(3), torch.ones(3), torch.zeros(5), torch.ones(5))
    a = bi.query(xq, {"supply": 8})["supply"][0]
    b = bi_s.query(xq + shift, {"supply": 8})["supply"][0]
    check("shifting room and query together leaves offsets unchanged",
          np.abs(a - b).max() < 1e-5, f"max diff {np.abs(a - b).max():.2e}")

    print("\n[8] orientation on a surface with air on BOTH sides (regression)")
    # The room ceiling has the HVAC plenum above it: 21% of interior points sit
    # above the ceiling plane, so "nearest fluid point" is a coin flip and left
    # 28% of real ceiling normals pointing the wrong way. Reproduce that shape:
    # a plate at z=0 with dense air below and a thin sparse layer just above.
    # Volumetric density is matched on the two sides -- what differs is DEPTH:
    # the room below is deep, the plenum above is a thin slab. That is the real
    # asymmetry, and the reason averaging over neighbours recovers the right sign
    # while the single nearest neighbour is near a coin flip.
    rng = np.random.default_rng(3)
    plate = np.column_stack([rng.uniform(0, 1, 3000), rng.uniform(0, 1, 3000), np.zeros(3000)])
    dens, deep, thin = 20000, 0.5, 0.02
    n_below, n_above = int(dens * deep), int(dens * thin)
    below = np.column_stack([rng.uniform(0, 1, n_below), rng.uniform(0, 1, n_below),
                             -rng.uniform(0, deep, n_below)])
    above = np.column_stack([rng.uniform(0, 1, n_above), rng.uniform(0, 1, n_above),
                             rng.uniform(0, thin, n_above)])
    both = np.vstack([below, above])
    n_majority = surface_normals(plate, both, k=16, orient_m=8)
    n_single = surface_normals(plate, both, k=16, orient_m=1)
    frac_maj = float(np.mean(n_majority[:, 2] < 0))
    frac_one = float(np.mean(n_single[:, 2] < 0))
    print(f"        majority(m=8) {frac_maj:.1%} correct vs single-nearest {frac_one:.1%}"
          f"   [real Case_07 ceiling: 98.2% vs 71.8%]")
    check("majority rule orients a two-sided surface toward the bulk air",
          frac_maj > 0.90, f"only {frac_maj:.1%} correct")
    check("single-nearest rule is the weaker one here (documents why m>1)",
          frac_one < frac_maj - 0.1, f"single={frac_one:.1%} vs majority={frac_maj:.1%}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
