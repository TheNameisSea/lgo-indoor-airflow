"""TransolverBaseline plus a per-point distance-to-boundary input channel.

A subclass, so `--model transolver` without the flag is unchanged. Query tokens
carry the unsigned distance to the nearest boundary point, computed on the GPU in
chunks; boundary tokens carry 0. This is the quantity the ShapeNet-Car benchmark
computes as "sdf" with a nearest-neighbour search. It is unsigned: the export
carries no surface normals to build an inside/outside test from.

Only `preprocess` gains an input column and is rebuilt, so every other weight
initialises exactly as in the parent.

USE:  --model transolver     --cfg '{"mlp_ratio": 2, "point_sdf": true}'
      --model lgo_transolver --cfg '{"mlp_ratio": 2, "point_sdf": true}'
Both route through `build_baseline("transolver", cfg, ...)`, so the hybrid picks
this up automatically.
"""

import torch

from .transolver import TransolverBaseline


class TransolverSdfBaseline(TransolverBaseline):
    """`TransolverBaseline` with one extra token channel: distance to boundary.

    Boundary tokens get 0 (they ARE the boundary, which is what their
    `sdf_surf` amounts to); query tokens get the distance to the nearest
    boundary point.
    """

    # Chunk so the (queries x boundary) distance matrix stays bounded. 3,682
    # boundary points x 16k queries is ~59M floats = 236 MB; larger query sets
    # arrive whole from the evaluator, so this is not optional.
    QCHUNK = 16384

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # ONE extra input channel. `preprocess` is rebuilt at the widened width;
        # every other module is inherited untouched, so the only parameter
        # difference from the baseline is this layer's first weight matrix.
        old = self.preprocess
        in_dim = old.linear_pre[0].in_features + 1
        hidden = old.linear_pre[0].out_features
        out = old.linear_post.out_features
        from .transolver import MLP
        self.preprocess = MLP(in_dim, hidden, out, n_layers=0, res=False, act="gelu")
        # Re-apply the official init to the replacement only -- `initialize_weights`
        # would re-init the whole model and silently break seed-matching with the
        # baseline everywhere else.
        self.preprocess.apply(self._init)
        self._boundary_xyz = None

    def encode_geometry(self, pc):
        """Stash the (subsampled) boundary positions the distance is measured to.

        Deliberately AFTER the parent's subsample, so the distance is measured to
        exactly the cloud the model is also given as tokens -- otherwise the two
        halves of the input would describe different geometry.
        """
        pc = super().encode_geometry(pc)
        self._boundary_xyz = pc[0, :, :3].detach()
        return pc

    def _distance_to_boundary(self, xyz):
        """(B, N, 1) unsigned distance to the nearest boundary point."""
        b = self._boundary_xyz
        if b is None:
            raise RuntimeError(
                "TransolverSdfBaseline.decode_query needs the boundary cloud; "
                "call encode_geometry first. (The parent subsamples there, and "
                "the distance must be measured to the SAME cloud the tokens "
                "carry.)")
        out = []
        q = xyz[0]
        for i in range(0, q.shape[0], self.QCHUNK):
            d = torch.cdist(q[i:i + self.QCHUNK].to(b.dtype), b)
            out.append(d.min(dim=1).values)
        return torch.cat(out).to(xyz.dtype)[None, :, None]

    def _tokens(self, xyz, extra, is_query):
        """Parent's token vector, plus the distance column."""
        tok = super()._tokens(xyz, extra, is_query)
        if is_query:
            d = self._distance_to_boundary(xyz)
        else:
            # Boundary points sit ON the surface. Their own `sdf_surf` is the
            # surface's distance to itself, i.e. zero.
            d = torch.zeros(xyz.shape[0], xyz.shape[1], 1,
                            device=xyz.device, dtype=xyz.dtype)
        return torch.cat([tok, d], dim=-1)
