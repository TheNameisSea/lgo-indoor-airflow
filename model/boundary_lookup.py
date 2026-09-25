"""The geometry-lookup half of LGO, factored out so it has ONE implementation.

Everything the local branch needs in order to run on top of *some* decoder --
registering a KD-tree index per case, recognising which case a forward pass is
for, converting normalized query coordinates to metres in the index's own
float64 arithmetic, and gathering stratified neighbours -- is bookkeeping, not
architecture. `model/local_branch.py` already knows nothing about GINOT: it
takes `{group: (dxyz, feat, valid)}` and returns a `d_model` vector. This mixin
is the other half of that decoupling, and it is what lets the same local branch
sit on a GINOT trunk (`model/lgo_ginot.py`) or on a DeepONet
(`model/lgo.py`).

THIS IS A MOVE, NOT A REWRITE. Every method below came out of `LGO` verbatim;
`LGO` now inherits them. The alternative -- copying them into the second model
-- is the failure mode this project has already paid for twice: `panels()` broke
in two different ways across three versions, each time returning a well-formed
answer, because an assumption was restated instead of shared. Two copies of the
float64 `to_metres` path would be the same bug waiting to happen, and the thing
it protects (T2: a distance that should be zero coming back as 1e-6) is silent.

CONTRACT. A model mixing this in must call `_init_lookup()` from its `__init__`
and, if it uses a smooth k-NN window, expose the extra-neighbour count through
`_soft_extra()` (the default reads `self.local.soft_knn`, which is where both
current users keep it).
"""

import numpy as np
import torch

from data.knn_cache import BASE_GROUPS, BoundaryIndex
from model.local_branch import neighbours_to_torch


class BoundaryLookup:
    """Per-case boundary indices, query->metres, and the stratified gather."""

    def _init_lookup(self, k, groups=BASE_GROUPS, far_voxels=None,
                     chunk_knn=200_000):
        self.k = dict(k) if isinstance(k, dict) else k
        self.groups = tuple(groups)
        self.far_voxels = list(far_voxels) if far_voxels else None
        self.chunk_knn = int(chunk_knn)
        self._index = {}          # fingerprint -> BoundaryIndex
        self._active = None
        # The trainer re-creates the batch each step but the boundary cloud of a
        # given case is bit-identical every time, so remember the last lookup and
        # skip the hash when the cloud has not changed.
        self._fp_cache = (None, None)

    # ------------------------------------------------------------------ #
    #  Case registry
    # ------------------------------------------------------------------ #
    def register_cases(self, dataset, verbose=True, **kw):
        """Build one BoundaryIndex per case, keyed by boundary-cloud fingerprint.

        Indexes the PROCESSED room dicts, so the full boundary pools are used --
        not the 15k subsample the encoder sees. The local branch and the no-slip
        mask both need the complete cloud: the mask is exactly zero only at
        points that are IN the set it searches.
        """
        n_new = 0
        for room in dataset.cases:
            key = fingerprint(room["pc_full"][:, :3])
            if key in self._index:
                continue
            self._index[key] = BoundaryIndex(
                room, dataset.coord_min, dataset.coord_scale,
                dataset.target_mean, dataset.target_std,
                groups=self.groups, far_voxels=self.far_voxels, **kw)
            n_new += 1
        if verbose and n_new:
            print(f"[{type(self).__name__}] registered {n_new} case(s); "
                  f"{len(self._index)} total")
        return n_new

    def _resolve(self, pc):
        pc0 = pc[0] if pc.dim() == 3 else pc
        cached_sig, cached_key = self._fp_cache
        sig = (pc0.data_ptr(), pc0.shape[0], float(pc0[0, 0]), float(pc0[-1, 2]))
        if cached_sig == sig:
            key = cached_key
        else:
            key = fingerprint(pc0[:, :3])
            self._fp_cache = (sig, key)
        entry = self._index.get(key)
        if entry is None:
            raise RuntimeError(
                f"{type(self).__name__}.decode_query needs this case's boundary "
                "index, and the cloud fingerprint is unknown. Call "
                "register_cases(dataset) after building the model. Unlike the "
                "encoder's 15k subsample, the local branch and no-slip mask need "
                "the FULL pools, so there is no meaningful fallback that derives "
                "them from `pc`.")
        return entry

    # ------------------------------------------------------------------ #
    #  Geometry lookups
    # ------------------------------------------------------------------ #
    def _metres(self, xyt, index):
        """Normalized query coords -> metres (float64), in the INDEX's arithmetic.

        This must go through `index.to_metres` rather than scaling on the GPU in
        float32. The KD-trees were built by casting normalized coordinates to
        float64 and scaling there; the same multiply-add in float32 rounds by
        ~1e-7 relative, which at an 8.8 m coordinate is ~1e-6 m of disagreement.
        A wall point then lands 1e-6 m away from its own tree entry, phi comes
        back as 1e-6 instead of 0, and the no-slip mask becomes
        1e-6/(1e-6 + 0.002) ~ 5e-4 rather than 0 -- measured as a 6e-5 m/s leak
        through solid walls on real geometry before both users of these
        coordinates were routed through this one function. It is the prior
        project's trap T2 (exactness lost to a distance that should be zero)
        reached by a different road, so the fix is the same in spirit: one code
        path, in the wider dtype.
        """
        return index.to_metres(xyt.detach().cpu().numpy())

    def _soft_extra(self):
        """Extra neighbours gathered per group purely to be faded to zero."""
        return getattr(getattr(self, "local", None), "soft_knn", 0) or 0

    def _gather(self, xyz_m, index, device, xq_grad=None):
        """Stratified kNN on CPU (scipy KD-tree), chunked, moved to device.

        `xq_grad` is the query in METRES as a live torch tensor. When given, the
        offsets are rebuilt as `x_boundary - xq_grad` in torch, so d(prediction)
        /d(query) survives into the local branch. Without it the offsets come
        back from numpy as constants and that derivative is identically zero --
        autograd then reports only the global path, and on a local-only arm it
        raises. Every earlier gradient claim in this project was retracted for
        exactly that reason, so anything differentiating the output with respect
        to position (a curl head, an equivariance receipt) must pass this in.

        Neighbour SELECTION stays non-differentiable, which is correct: it is
        piecewise constant in the query, so its derivative is zero almost
        everywhere and undefined on the measure-zero set where the k-th
        neighbour changes.
        """
        want_abs = xq_grad is not None
        # With the smooth window on, gather EXTRA neighbours per group: they
        # exist only to be faded to zero, so that a neighbour entering or
        # leaving the set carries no weight when it crosses.
        extra = self._soft_extra()
        kq = ({g: v + extra for g, v in self.k.items()} if isinstance(self.k, dict)
              else self.k + extra) if extra else self.k
        outs = [index.query(xyz_m[s:s + self.chunk_knn], kq, with_abs=want_abs)
                for s in range(0, len(xyz_m), self.chunk_knn)]
        n_f = 4 if want_abs else 3
        merged = {g: tuple(np.concatenate([o[g][i] for o in outs], 0) for i in range(n_f))
                  for g in outs[0]}
        if not want_abs:
            return neighbours_to_torch(merged, device)
        base = neighbours_to_torch({g: v[:3] for g, v in merged.items()}, device)
        out = {}
        for g, (dxyz, feat, valid) in base.items():
            # SUBTRACT IN float64, THEN NARROW. Both operands are ~8.8 m and
            # their difference is centimetres for a near neighbour, so a float32
            # subtraction throws away most of the mantissa -- the same
            # cancellation `knn_cache.query` avoids by doing this in float64.
            xb = torch.as_tensor(merged[g][3], dtype=torch.float64, device=device)
            d64 = xb - xq_grad[:, None, :].to(torch.float64)
            out[g] = (d64.to(xq_grad.dtype), feat, valid)
        return out

    def wall_mask(self, xyz_m, index, device):
        """phi/(phi+ell): exactly 0 on the solid cloud, ~1 in the far field."""
        phi = np.concatenate([index.wall_distance(xyz_m[s:s + self.chunk_knn])
                              for s in range(0, len(xyz_m), self.chunk_knn)])
        phi = torch.as_tensor(phi, dtype=torch.float32, device=device)
        return (phi / (phi + self.ell_wall)).unsqueeze(-1)

    # ------------------------------------------------------------------ #
    #  The local stream itself
    # ------------------------------------------------------------------ #
    def _local_stream(self, xyz_m, device, n_layers, dtype=None, xq_grad=None):
        """Run the zero-seeded local branch -> (N, d_model).

        This is the `local_query='learned'` arrangement and the only one shared
        between users: the layer-0 query is `q_proj`'s bias (a learned constant)
        and every later query is the stream's own output, so no absolute
        coordinate ever reaches the attention and the term is exactly
        translation-equivariant. `LocalCrossAttention.out` is zero-initialised,
        so the returned tensor is EXACTLY zero at step 0.
        """
        tokens, valid, soft = self.local.encode(
            self._gather(xyz_m, self._active, device, xq_grad))
        h = torch.zeros(tokens.shape[0], self.local.attn[0].out.out_features,
                        device=device, dtype=dtype or tokens.dtype)
        for l in range(n_layers):
            h = h + self.local(h, tokens, valid, layer=l, soft=soft)
        return h


def fingerprint(pc_xyz):
    """Order-invariant id of a boundary cloud. Same scheme as legacy/hbc.

    Reduced on the tensor's OWN device, so only two scalars cross the bus. The
    legacy version copied the whole cloud to CPU first, which on the hot path
    (once per training step) cost ~20 ms of the step -- the reduction itself is
    trivial, the transfer and its implied sync were not. Registration runs on
    CPU tensors and lookup on GPU ones, and the two must agree: they do, because
    float32 -> float64 -> *1e6 -> round -> int64 is exact and IEEE-deterministic
    on both, and int64 sums carry no rounding.
    """
    q = (pc_xyz.detach().double() * 1e6).round().to(torch.int64)
    return (int(q.shape[0]), int(q.sum().item()), int((q % 9973).sum().item()))
