"""Per-case boundary index: surface normals, stratified kNN, exact wall distance.

This is the geometric substrate the LGO local branch stands on. For each case we
build one index over the case's FULL boundary cloud, split by type, and answer
three questions for arbitrary query points:

  1. which boundary points are near this query, *per type*  (local attention)
  2. what is the offset to each of them, in metres         (equivariance)
  3. how far is the nearest solid surface                   (hard BC)

WHY STRATIFY BY TYPE. A plain kNN over the whole boundary returns, for a query
at head height in the middle of the room, ~32 floor points and nothing else --
the vents that *cause* the flow would be invisible to the local branch at
exactly the distances where the prior project's models lose all skill. Querying
each type separately guarantees every query sees THE NEAREST vent of each type,
however far away, and lets attention decide what matters using the distance we
hand it.

  CORRECTED 2026-09-15: this used to say "every vent". It does not. A supply
  panel carries 2,209 boundary points at 1.26 cm spacing, so the 32 nearest
  supply points of ANY query lie on ONE panel (measured on B005: span 0.15 m at
  K=32, still one panel at K=512; reaching a second needs K > 2,209). The
  guarantee is that the supply group cannot be crowded out by floor points --
  the nearest jet source is always visible with its true offset. Knowledge of
  how the OTHER panels are arranged reaches a query only through the global
  latent. K is a resolution knob on the nearest face, not a coverage knob.

A COARSE SOLID GROUP (`solid_far`, off by default). The same wall cloud
voxel-downsampled to a coarse shell, so its K nearest span metres instead of
centimetres. Enabled by --k_solid_far / --far_voxels; leave them unset.

FOUR GROUPS, NOT THREE. The implementation plan folded `leak` into `solid`;
that is wrong and this module deviates deliberately. A leak is an opening that
air flows THROUGH, so including leak points in the solid set would (a) put them
in the no-slip distance and force velocity to zero at an opening where air
actually moves, and (b) contradict the prior project, whose HBC wall cloud is
`pool_wall_xyz` alone (hbc/wrapper.py:107). Leak gets its own small group.

BLINDING. The prior loader blinds inputs only on `pc_full`, NOT on the pools
this module reads (thermo/dataset.py:112 touches `pc_full` only). Reading BC
features straight from the pools would hand the model the return-vent
temperature -- which is the room's mixed mean air temperature, i.e. most of the
answer. We re-apply the same rule here: supply keeps its temperature, everything
else is blinded to the normalized mean (0.0).
"""

import os

import numpy as np
import torch
from scipy.spatial import cKDTree

# scipy spawns one thread per core by default. Several training runs share this
# machine, so 3 runs x 64 threads on 64 cores is pure contention; set
# LGO_KNN_WORKERS to divide the box between them.
WORKERS = int(os.environ.get("LGO_KNN_WORKERS", "-1"))

# Group order is fixed and public: models index into it.
# The four groups every model has had since the start. Kept as its own name
# because the group one-hot width is baked into a trained NeighbourEncoder:
# appending a group changes that width, so a model trained on BASE_GROUPS can
# only be rebuilt with BASE_GROUPS. `solid_far` is therefore opt-in.
BASE_GROUPS = ("supply", "return", "solid", "leak")
GROUPS = BASE_GROUPS + ("solid_far",)          # single-scale far field
FAR_PREFIX = "solid_far"


def make_groups(far_voxels):
    """Group tuple for a given far-field configuration.

    `far_voxels` is a list of voxel sizes in metres, coarsest last. One entry
    keeps the historical name `solid_far` so existing checkpoints stay loadable;
    several produce `solid_far0..N`, a HIERARCHY of shells. One shell has a
    single reach (K=32 over a 0.8 m voxelisation spans 2.14 m); a hierarchy lets
    a query see fine structure nearby and the room's outline further out at the
    same time, which is what a recirculation cell spanning several metres needs.
    """
    if not far_voxels:
        return BASE_GROUPS
    if len(far_voxels) == 1:
        return GROUPS
    return BASE_GROUPS + tuple(f"{FAR_PREFIX}{i}" for i in range(len(far_voxels)))


def far_voxel_map(groups, far_voxels):
    """{group name: voxel size} for the far groups in `groups`."""
    far = [g for g in groups if g.startswith(FAR_PREFIX)]
    if not far:
        return {}
    vs = list(far_voxels) if far_voxels else [0.5]
    if len(vs) != len(far):
        raise ValueError(f"{len(far)} far groups but {len(vs)} voxel sizes")
    return dict(zip(far, vs))

# Pool names in the room dict produced by MultiCase_GINOT_Dataset._process_single_room.
_POOL = {"supply": "pool_in", "return": "pool_out", "solid": "pool_wall",
         "leak": "pool_leak"}


def _pool_of(g):
    """Every far shell is a downsampling of the same wall cloud."""
    return "pool_wall" if g.startswith(FAR_PREFIX) else _POOL[g]

T_IDX = 4          # temperature channel in a target row [u, v, w, p, T]
FEAT_DIM = 8       # normal(3) + uvw(3) + T(1) + leak_mag(1)


def surface_normals(xyz, interior_xyz=None, k=16, orient_m=8, chunk=200_000):
    """Unit normals by local PCA, oriented toward the fluid.

    The smallest-eigenvalue direction of a point's k-neighbourhood covariance is
    the surface normal, up to sign. Sign is fixed by pointing each normal into
    the air, which orients walls, ceiling, floor and furniture consistently
    without assuming the room is convex or the surfaces closed.

    ORIENTATION USES A MAJORITY OF NEIGHBOURS, NOT THE SINGLE NEAREST. Pointing
    at the one nearest fluid point fails on any surface with air on both sides,
    and this room has a big one: 21% of interior points sit ABOVE the ceiling
    plane, in the HVAC plenum feeding the diffusers. Measured on Case_07, the
    nearest fluid point lies above the ceiling for 47% of ceiling points -- a
    coin flip -- leaving only 72% of ceiling normals pointing down into the
    room. Averaging the direction to the `orient_m` nearest fluid points instead
    puts 98% of them right, because the boundary layer under the ceiling is
    finely meshed while the plenum is further away. Larger `orient_m` reaches
    into the plenum and degrades again (95.8% at m=64), so the default is small.

    KNOWN LIMIT. This is a density majority, so it resolves the sign only when
    one side is better represented near the surface. Sweeping the near-surface
    density ratio (air above : air below) on a synthetic plate, m=8 scores 0.94
    at ratio 0.25, 0.78 at 0.75 and 0.41 at 1.25: with matched densities on both
    sides of a thin plate the outward direction is genuinely undetermined from
    points alone, and no choice of `orient_m` recovers it. The real geometry is
    well inside the good regime (a 0.15 m plenum against a 3.2 m room), but a
    future dataset with thin free-standing partitions would need either a signed
    distance field or the mesh connectivity to do better. Normals are an
    ablation switch (`with_normals`) partly for this reason.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    n = len(xyz)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    k = min(k, n)
    tree = cKDTree(xyz)
    out = np.zeros((n, 3), dtype=np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _, idx = tree.query(xyz[s:e], k=k, workers=WORKERS)
        nb = xyz[idx]                                   # (m, k, 3)
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("mki,mkj->mij", nb, nb) / max(k - 1, 1)
        # Symmetric 3x3: eigh returns ascending eigenvalues, so column 0 is the
        # least-variance direction = the normal.
        out[s:e] = np.linalg.eigh(cov)[1][:, :, 0]

    if interior_xyz is not None and len(interior_xyz):
        fluid = np.ascontiguousarray(interior_xyz, dtype=np.float64)
        itree = cKDTree(fluid)
        m = min(orient_m, len(fluid))
        for s_ in range(0, n, chunk):
            e_ = min(s_ + chunk, n)
            _, near = itree.query(xyz[s_:e_], k=m, workers=WORKERS)
            near = near.reshape(e_ - s_, m)
            to_fluid = (fluid[near] - xyz[s_:e_, None, :]).mean(axis=1)
            flip = (out[s_:e_] * to_fluid).sum(1) < 0.0
            out[s_:e_][flip] *= -1.0

    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(nrm, 1e-12)).astype(np.float32)


class BoundaryIndex:
    """Type-stratified KD-trees over one case's boundary, in METRES.

    Metres, not normalized units, on purpose: the local branch's whole claim is
    that "0.4 m below a supply vent" means the same thing everywhere, and that
    only holds if the offsets it sees are physical.
    """

    def __init__(self, room, coord_min, coord_scale, target_mean, target_std,
                 normals_k=16, normals_orient_m=8, with_normals=True,
                 interior_for_normals=200_000, far_voxel=0.5,
                 groups=GROUPS, far_voxels=None):
        cmin = np.asarray(coord_min, dtype=np.float64)
        cscale = np.asarray(coord_scale, dtype=np.float64)
        self.coord_min, self.coord_scale = cmin, cscale

        interior = None
        if with_normals:
            pools = [room[k] for k in ("pool_stream_xyz", "pool_bg_xyz") if len(room.get(k, []))]
            if pools:
                interior = torch.cat(pools, 0).numpy().astype(np.float64) * cscale + cmin
                if len(interior) > interior_for_normals:      # normals only need coverage
                    sel = np.random.default_rng(0).choice(
                        len(interior), interior_for_normals, replace=False)
                    interior = interior[sel]

        self.xyz, self.feat, self.tree, self.count = {}, {}, {}, {}
        self.groups = tuple(groups)
        self.far_voxel = float(far_voxel)
        self.far_voxels = far_voxel_map(self.groups,
                                        far_voxels or [far_voxel])
        for g in self.groups:
            xyz_n = room.get(f"{_pool_of(g)}_xyz")
            tgt_n = room.get(f"{_pool_of(g)}_tgt")
            if xyz_n is None or len(xyz_n) == 0:
                self.xyz[g] = np.zeros((0, 3))
                self.feat[g] = np.zeros((0, FEAT_DIM), dtype=np.float32)
                self.tree[g] = None
                self.count[g] = 0
                continue

            xyz_m = xyz_n.numpy().astype(np.float64) * cscale + cmin
            tgt = tgt_n.numpy().astype(np.float32)
            if g.startswith(FAR_PREFIX):
                # One point per voxel: cheap, deterministic, and it preserves
                # coverage of every surface rather than concentrating on the
                # boundary-refined regions the way random subsampling would.
                key = np.floor(xyz_m / self.far_voxels[g]).astype(np.int64)
                _, keep = np.unique(key, axis=0, return_index=True)
                keep = np.sort(keep)
                xyz_m, tgt = xyz_m[keep], tgt[keep]

            uvw = tgt[:, 0:3].copy()
            temp = tgt[:, T_IDX].copy() if tgt.shape[1] > T_IDX else np.zeros(len(tgt), np.float32)
            # Same blinding the encoder gets: only a SUPPLY vent's temperature is
            # a boundary condition. 0.0 is the normalized mean, matching the
            # convention the prior loader uses when it blinds pressure.
            if g != "supply":
                temp[:] = 0.0
            # A leak carries only a speed magnitude (Leak.csv has no components),
            # so its velocity slots are blinded and the magnitude rides its own
            # channel, in physical m/s.
            leak_mag = np.zeros(len(tgt), np.float32)
            if g == "leak":
                uvw[:] = 0.0
                lm = room.get("pool_leak_mag")
                if lm is not None and len(lm) == len(tgt):
                    leak_mag = lm.numpy().astype(np.float32)

            nrm = (surface_normals(xyz_m, interior, k=normals_k,
                                   orient_m=normals_orient_m) if with_normals
                   else np.zeros((len(xyz_m), 3), dtype=np.float32))

            self.xyz[g] = xyz_m
            self.feat[g] = np.concatenate(
                [nrm, uvw, temp[:, None], leak_mag[:, None]], axis=1).astype(np.float32)
            self.tree[g] = cKDTree(xyz_m)
            self.count[g] = len(xyz_m)

    # ------------------------------------------------------------------ #
    def query(self, xq_m, k, dtype=np.float32, with_abs=False):
        """Gather the k nearest boundary points per group.

        `k` is an int or {group: int}. Returns {group: (dxyz, feat, valid)} with
        dxyz = x_boundary - x_query in metres, shape (N, k, 3); feat (N, k, 8);
        valid (N, k) False where the group holds fewer than k points.

        `dtype` is float32 by default because these arrays dominate memory at
        inference (1.16 M queries x ~44 neighbours). The search and the
        subtraction are always done in float64; only the returned offsets are
        narrowed. Pass float64 to measure the equivariance receipt, where float32
        storage of a translated room is itself the limiting error.
        """
        xq = np.ascontiguousarray(xq_m, dtype=np.float64)
        ks = {g: (k if isinstance(k, int) else k.get(g, 0)) for g in self.groups}
        out = {}
        for g in self.groups:
            kg = ks[g]
            if kg <= 0:
                continue
            n_avail, tree = self.count[g], self.tree[g]
            if n_avail == 0:
                z = (np.zeros((len(xq), kg, 3), dtype),
                     np.zeros((len(xq), kg, FEAT_DIM), dtype),
                     np.zeros((len(xq), kg), bool))
                out[g] = z + (np.zeros((len(xq), kg, 3), np.float64),) if with_abs else z
                continue
            kq = min(kg, n_avail)
            _, idx = tree.query(xq, k=kq, workers=WORKERS)
            idx = idx.reshape(len(xq), kq)
            # Offsets in metres, computed by explicit subtraction. Never via a
            # matmul distance formula: it cancels catastrophically at ~0 distance
            # (prior project trap T2) and these offsets feed the no-slip mask.
            dxyz = (self.xyz[g][idx] - xq[:, None, :]).astype(dtype)
            feat = self.feat[g][idx].astype(dtype)
            valid = np.ones((len(xq), kq), bool)
            if kq < kg:                                   # pad short groups
                pad = kg - kq
                dxyz = np.concatenate([dxyz, np.zeros((len(xq), pad, 3), dtype)], 1)
                feat = np.concatenate([feat, np.zeros((len(xq), pad, FEAT_DIM), dtype)], 1)
                valid = np.concatenate([valid, np.zeros((len(xq), pad), bool)], 1)
            if with_abs:
                # The ABSOLUTE neighbour positions, so a caller can rebuild
                # dxyz = x_boundary - x_query in torch and keep the query's
                # gradient. The subtraction here is numpy, which severs that
                # path: `dxyz` arrives as a constant and autograd through the
                # local branch reports nothing. Neighbour SELECTION is correctly
                # non-differentiable (it is piecewise constant in the query), but
                # the OFFSETS are not, and they carry all the spatial variation.
                xb = self.xyz[g][idx].astype(np.float64)
                if kq < kg:
                    xb = np.concatenate(
                        [xb, np.zeros((len(xq), kg - kq, 3), np.float64)], 1)
                out[g] = (dxyz, feat, valid, xb)
            else:
                out[g] = (dxyz, feat, valid)
        return out

    def wall_distance(self, xq_m):
        """Exact distance [m] to the nearest SOLID point. Leaks are not walls.

        Exact in the sense that matters for the hard BC: a query point that is
        itself in the solid cloud returns 0.0, so the mask phi/(phi+ell) is
        exactly 0 there. cKDTree computes true Euclidean distances in float64;
        the failure mode to avoid is the matmul-based formula, not this one.
        """
        if self.tree["solid"] is None:
            return np.full(len(xq_m), np.inf, dtype=np.float64)
        d, _ = self.tree["solid"].query(
            np.ascontiguousarray(xq_m, dtype=np.float64), k=1, workers=WORKERS)
        return d

    def to_metres(self, xyz_norm):
        x = xyz_norm.numpy() if torch.is_tensor(xyz_norm) else np.asarray(xyz_norm)
        return x.astype(np.float64) * self.coord_scale + self.coord_min

    def __repr__(self):
        return "BoundaryIndex(" + ", ".join(f"{g}={self.count[g]:,}" for g in self.groups) + ")"


def build_indices(dataset, verbose=True, **kw):
    """One BoundaryIndex per case, in dataset order."""
    idxs = []
    for i, room in enumerate(dataset.cases):
        bi = BoundaryIndex(room, dataset.coord_min, dataset.coord_scale,
                           dataset.target_mean, dataset.target_std, **kw)
        if verbose:
            print(f"  case {i}: {bi}")
        idxs.append(bi)
    return idxs
