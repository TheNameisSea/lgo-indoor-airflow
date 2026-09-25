"""GINO baseline — Geometry-Informed Neural Operator (Li et al., NeurIPS 2023).

Uses the official `neuraloperator` implementation (`neuralop.models.GINO`,
neuraloperator 2.0.0) unchanged; this module only splits its `forward` into the
`encode_geometry` / `decode_query` pair the comparison harness expects, so the
GNO-encoder + FNO stages run once per case and the output GNO can be queried in
chunks. The split is a verbatim transcription of GINO.forward.

Adaptation note: GINO was designed around an SDF
sampled on the latent grid. By DEFAULT we have none (geometry arrives as surface
point clouds only), so the latent grid carries no `latent_features` and the
geometry reaches the FNO purely through the input GNO's kernel integral from the
boundary point cloud — the "no-SDF" GINO variant. The grid spans the normalized
[0,1]^3 room box, which is the same box the dataset min-max maps every case into.

`latent_sdf=True` adds that input: the distance from each latent-grid point to
the nearest boundary point, fed to the FNO as an extra latent channel. The
ShapeNet-Car benchmark supplies this quantity to every model it compares, as
`[position, sdf, normal]`.

  * so this is a DISTANCE TO THE NEAREST BOUNDARY POINT, UNSIGNED — which is
    what the benchmark's nearest-neighbour "sdf" computes too. A true signed field would need an inside/outside test, and our export
    carries no surface normals to build one from (`Car_surface.csv` is
    X,Y,Z + velocity + pressure + temperature). Say "unsigned distance", not
    "SDF", in any number that comes out of this.

It is OFF by default: with `latent_sdf=False` this
class is byte-identical to what it was. Turning it on changes parameter SHAPES
(`latent_feature_channels=1` widens the FNO's first layer), so a checkpoint
mismatch fails loudly at load rather than silently predicting differently — which is
why this knob is safe to carry in `--cfg`.
"""

import torch
import torch.nn as nn
from neuralop.models import GINO as _GINO

from baselines.common.fast_segment import patch as _patch_segment

# neuralop's segment_csr fallback loops in Python over every output point; see
# baselines/common/fast_segment.py. Without this GINO costs ~6.6 s/case.
_patch_segment()


class GINOBaseline(nn.Module):
    def __init__(self, pc_channels=12, out_channels=3, grid_res=32,
                 fno_modes=8, fno_hidden=64, fno_layers=4,
                 in_radius=0.12, out_radius=0.12, gno_mlp_hidden=64,
                 gno_mlp_layers=2, n_pc_tokens=4096, eval_chunk=8192,
                 latent_sdf=False):
        super().__init__()
        self.latent_sdf = bool(latent_sdf)
        # The input GNO runs a channel MLP over every (boundary, grid) PAIR, so
        # cost is n_pairs * mlp_hidden -- with the full 15k cloud and a 32^3 grid
        # that is ~3.1e6 pairs and 28 s/case. The boundary subsample and grid
        # resolution are therefore the two dominant cost knobs, both swept.
        self.n_pc_tokens = n_pc_tokens
        self.eval_chunk = eval_chunk
        mlp = [gno_mlp_hidden] * gno_mlp_layers
        self.net = _GINO(
            in_channels=pc_channels,
            out_channels=out_channels,
            gno_coord_dim=3,
            in_gno_radius=in_radius,
            out_gno_radius=out_radius,
            in_gno_transform_type="nonlinear_kernelonly",
            out_gno_transform_type="linear",
            in_gno_channel_mlp_hidden_layers=mlp,
            out_gno_channel_mlp_hidden_layers=mlp,
            fno_n_modes=(fno_modes,) * 3,
            fno_hidden_channels=fno_hidden,
            fno_n_layers=fno_layers,
            # neuralop concatenates `latent_features` onto the grid embedding
            # before the FNO (GINO.forward: `in_p = torch.cat((in_p,
            # latent_features), dim=-1)`). That is GINO's own SDF channel, so
            # we use its mechanism rather than bolting one on beside it.
            **({"latent_feature_channels": 1} if latent_sdf else {}),
        )

        # Latent regular grid over the normalized room box [0, 1]^3.
        g = torch.linspace(0.0, 1.0, grid_res)
        grid = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), dim=-1)
        self.register_buffer("latent_queries", grid)   # (R, R, R, 3)

    @torch.no_grad()
    def _grid_distance(self, xyz_full):
        """Unsigned distance from every latent-grid point to the nearest
        boundary point. (R,R,R,1), in the SAME normalized [0,1]^3 units as the
        grid, so it needs no separate scaling.

        Computed on the GPU in chunks, deliberately: the CPU is usually the
        binding resource during training, and a scipy
        cKDTree here would compete with the local branch's own KNN gather,
        which already runs through scipy. `torch.cdist` on 32^3 grid points
        against a few thousand boundary points is milliseconds on an A5000.

        Takes the FULL cloud, NOT the `n_pc_tokens` subsample: the subsample is
        re-drawn every call, so building the distance from it would make a
        per-case constant fluctuate between epochs — a silent input noise
        source. The distance field is a property of the geometry alone.
        """
        lq = self.latent_queries
        g = lq.view(-1, lq.shape[-1])                  # (R^3, 3)
        out = torch.empty(g.shape[0], device=g.device, dtype=g.dtype)
        step = 4096
        for i in range(0, g.shape[0], step):
            out[i:i + step] = torch.cdist(
                g[i:i + step], xyz_full).min(dim=1).values
        return out.view(*lq.shape[:-1], 1)

    def encode_geometry(self, pc):
        """pc: (1, N_pc, C) -> latent embedding on the grid, cached per case."""
        net = self.net
        # BEFORE the subsample -- see _grid_distance's docstring.
        sdf = (self._grid_distance(pc[0, :, :3].contiguous())
               if self.latent_sdf else None)
        n = pc.shape[1]
        if n > self.n_pc_tokens:
            sel = torch.randperm(n, device=pc.device)[:self.n_pc_tokens]
            pc = pc[:, sel]
        geom = pc[0, :, :3].contiguous()               # (N_pc, 3)
        feats = pc[:, :, :].contiguous()               # (1, N_pc, C)
        lq = self.latent_queries

        in_p = net.gno_in(y=geom, x=lq.view(-1, lq.shape[-1]), f_y=feats)
        in_p = in_p.view((1, *lq.shape[:-1], -1))
        if sdf is not None:
            in_p = torch.cat((in_p, sdf.unsqueeze(0).to(in_p.dtype)), dim=-1)
        latent_embed = net.latent_embedding(in_p=in_p, ada_in=None)
        latent_embed = latent_embed.permute(
            0, *net.in_coord_dim_reverse_order, 1
        ).reshape(1, -1, net.fno_hidden_channels)
        return latent_embed

    def decode_query(self, latent, xyz, h_delta=None):
        """latent: grid embedding; xyz: (1, M, 3) -> (1, M, out_channels).

        `h_delta` is the hook `model/lgo_gino.py` uses, and is the exact
        analogue of `deeponet.py`'s: a (1, M, fno_hidden) tensor added to the
        per-query state produced by the output GNO, BEFORE the shared
        `projection` head. Left at None -- which is every call the bare `gino`
        baseline makes -- this method is byte-identical to what it was, so the
        baseline's numbers are untouched and "hybrid minus local branch" is
        literally this class.

        Note the asymmetry with `deeponet.py`: there, `trunk_out` is a single
        Linear, so injection is exactly basis augmentation
        (`trunk_out(h + d) = trunk_out(h) + W d`). Here `projection` is a
        two-layer ChannelMLP, so the local stream passes through a
        nonlinearity -- the same situation as `lgo_ginot`'s `output_proj`. The
        head is still SHARED, which is what the ablation needs.
        """
        net = self.net
        lq = self.latent_queries
        out = net.gno_out(y=lq.reshape(-1, lq.shape[-1]),
                          x=xyz.squeeze(0), f_y=latent)
        if h_delta is not None:
            out = out + h_delta
        out = out.permute(0, 2, 1)
        out = net.projection(out).permute(0, 2, 1)
        return out

    def forward(self, xyz, pc):
        return self.decode_query(self.encode_geometry(pc), xyz)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Drop neuralop's `_metadata` entry before the standard load.

        `neuralop.models.base_model.BaseModel.state_dict` injects a `_metadata`
        key holding the model's own init kwargs. It is not a parameter, so a
        round-trip through `torch.save`/`load_state_dict` fails with
        "Unexpected key(s): _metadata" -- every GINO checkpoint this repo has
        written carries it. Two of those kwargs are function objects and one is
        a class, which is separately why `baselines/models/__init__.py` has to
        allowlist them for `torch.load`. Strip it here so `strict=True` still
        means what it says for the actual weights.
        """
        if "_metadata" in state_dict:
            state_dict = {k: v for k, v in state_dict.items() if k != "_metadata"}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
