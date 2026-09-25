"""GNO baseline — Graph Neural Operator (Li et al. 2020), the reference point.

Uses the official `neuralop.layers.gno_block.GNOBlock` kernel-integral layer.
Architecture: a radius-graph kernel integral straight from the boundary point
cloud onto the query points, then optional query-to-query kernel integral layers
(the "interior" propagation GNO normally gets from its graph), then a pointwise
head to (u, v, w).

This is expected to be the weakest entry — that is its purpose. It is in GINOT's
own baseline table (Table 4 of the GINOT paper), and it establishes what a plain
kernel-integral operator achieves under the identical protocol.

`encode_geometry` caches nothing expensive (the encoder GNO is query-dependent),
so it returns the raw point cloud and all work happens in `decode_query`. That is
correct but means per-chunk cost is higher than GINOT's — noted here.
"""

import torch
import torch.nn as nn
from neuralop.layers.gno_block import GNOBlock

from baselines.common.fast_segment import patch as _patch_segment

# neuralop's segment_csr fallback loops in Python over every output point; see
# baselines/common/fast_segment.py. Without this GNO costs ~9.4 s/case.
_patch_segment()


class GNOBaseline(nn.Module):
    def __init__(self, pc_channels=12, out_channels=3, width=64, radius=0.15,
                 n_query_layers=1, query_radius=0.10, mlp_hidden=128,
                 mlp_layers=2, n_pc_tokens=4096, eval_chunk=4096,
                 head_hidden=1024, head_layers=3):
        super().__init__()
        self.radius = radius
        # The kernel integral materializes one row per (query, neighbour) pair, so
        # cost is n_query * n_boundary * density(radius). With the full 15k cloud
        # and radius 0.15 that is ~12k neighbours per query and a 20k-point eval
        # chunk needs >11 GiB. Subsampling the cloud and capping the eval chunk
        # keeps GNO inside 24 GB; both are swept as tuning knobs.
        self.n_pc_tokens = n_pc_tokens
        self.eval_chunk = eval_chunk
        mlp = [mlp_hidden] * mlp_layers

        # Boundary -> query kernel integral.
        self.gno_in = GNOBlock(
            in_channels=pc_channels, out_channels=width, coord_dim=3,
            radius=radius, transform_type="nonlinear_kernelonly",
            channel_mlp_layers=mlp, pos_embedding_type="transformer",
            use_torch_scatter_reduce=False,
        )

        # Query -> query propagation layers.
        self.query_layers = nn.ModuleList([
            GNOBlock(
                in_channels=width, out_channels=width, coord_dim=3,
                radius=query_radius, transform_type="linear",
                channel_mlp_layers=mlp, pos_embedding_type="transformer",
                use_torch_scatter_reduce=False,
            ) for _ in range(n_query_layers)
        ])
        self.query_norms = nn.ModuleList(
            [nn.LayerNorm(width) for _ in range(n_query_layers)])

        # Pointwise head. Kernel-MLP width has to stay small because those layers
        # act on (query, neighbour) PAIRS (~1e6 rows); the head acts on queries
        # only (~5e3 rows), so capacity here is essentially free. This is how GNO
        # is brought into GINOT's parameter band without exploding memory.
        head = [nn.Linear(width, head_hidden), nn.GELU()]
        for _ in range(head_layers - 1):
            head += [nn.Linear(head_hidden, head_hidden), nn.GELU()]
        head += [nn.Linear(head_hidden, out_channels)]
        self.head = nn.Sequential(*head)

    def encode_geometry(self, pc):
        # The encoder GNO is query-conditioned, so there is nothing to precompute;
        # this only fixes the boundary subsample for the whole case.
        n = pc.shape[1]
        if n > self.n_pc_tokens:
            sel = torch.randperm(n, device=pc.device)[:self.n_pc_tokens]
            pc = pc[:, sel]
        return pc

    def decode_query(self, latent, xyz):
        pc = latent
        geom = pc[0, :, :3].contiguous()
        feats = pc[:, :, :].contiguous()
        q = xyz.squeeze(0).contiguous()

        h = self.gno_in(y=geom, x=q, f_y=feats)          # (M, width)
        if h.dim() == 3:
            h = h.squeeze(0)

        for layer, norm in zip(self.query_layers, self.query_norms):
            delta = layer(y=q, x=q, f_y=norm(h).unsqueeze(0))
            if delta.dim() == 3:
                delta = delta.squeeze(0)
            h = h + delta

        return self.head(h).unsqueeze(0)

    def forward(self, xyz, pc):
        return self.decode_query(self.encode_geometry(pc), xyz)
