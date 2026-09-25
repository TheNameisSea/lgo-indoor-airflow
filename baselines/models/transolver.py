"""Transolver baseline (Wu et al., ICML 2024) — the direct architectural rival.

`Physics_Attention_Irregular_Mesh`, `Transolver_block` and the MLP are
transcribed from the official implementation
(https://github.com/thuml/Transolver, `Physics-Attention` / irregular-mesh
variant) with no changes to the mechanism: learned slice weights over tokens,
attention among slice tokens, then de-slicing back to tokens.

Adaptation to this task: Transolver predicts at its
own input tokens, but we need predictions at arbitrary interior query points
conditioned on a boundary point cloud. Since Physics-Attention is permutation
invariant and mixes all tokens globally through the slice bottleneck, we feed a
SINGLE token set = [boundary tokens ; query tokens] and read the output off the
query tokens only. Boundary tokens carry their 12 dataset channels (xyz + class
one-hots + blinded BC flow); query tokens carry xyz with the remaining channels
zeroed, plus a flag channel distinguishing the two. This is the minimal faithful
way to make Transolver geometry-conditioned and mesh-free here.

Because the slice statistics are computed over whichever tokens are present, the
prediction at a query point weakly depends on the rest of its chunk. `eval_chunk`
is therefore pinned to the training query count so train and eval see the same
token budget.
"""

import numpy as np
import torch
import torch.nn as nn

ACT = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class MLP(nn.Module):
    """Official Transolver MLP block."""

    def __init__(self, n_input, n_hidden, n_output, n_layers=0, act="gelu", res=True):
        super().__init__()
        act_fn = ACT[act]
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn())
            for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class PhysicsAttentionIrregularMesh(nn.Module):
    """Official Physics-Attention for irregular meshes / point clouds."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        # Orthogonal init on the slice projection is part of the official recipe.
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape
        H, D = self.heads, self.dim_head

        fx_mid = self.in_project_fx(x).reshape(B, N, H, D).permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).reshape(B, N, H, D).permute(0, 2, 1, 3)

        # Slice: soft assignment of every token to a physics-aware slice.
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))
        slice_norm = slice_weights.sum(2)                      # (B, H, G)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm + 1e-5)[:, :, :, None]

        # Attention among slice tokens (G << N -> linear in N).
        q, k, v = self.to_q(slice_token), self.to_k(slice_token), self.to_v(slice_token)
        attn = self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v)

        # De-slice back to tokens.
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = out_x.permute(0, 2, 1, 3).reshape(B, N, H * D)
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu", mlp_ratio=4,
                 last_layer=False, out_dim=3, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttentionIrregularMesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                       n_layers=0, res=False, act=act)
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class TransolverBaseline(nn.Module):
    def __init__(self, pc_channels=12, out_channels=3, hidden_dim=128, n_layers=8,
                 n_heads=8, slice_num=32, mlp_ratio=2, dropout=0.0,
                 n_pc_tokens=4096, num_bands=8, min_length_norm=0.05,
                 use_fourier=True, eval_chunk=5120):
        super().__init__()
        self.n_pc_tokens = n_pc_tokens
        self.pc_channels = pc_channels
        self.eval_chunk = eval_chunk
        self.use_fourier = use_fourier

        if use_fourier:
            from pi_ginot.model import MildFourierEncoder
            self.fourier = MildFourierEncoder(
                in_channels=3, base_length=1.0, min_length=min_length_norm,
                num_bands=num_bands, spacing="log")
            pos_dim = self.fourier.out_channels
        else:
            self.fourier = None
            pos_dim = 0

        # Token feature = xyz | pc channels beyond xyz | query flag | Fourier(xyz)
        in_dim = 3 + (pc_channels - 3) + 1 + pos_dim
        self.preprocess = MLP(in_dim, hidden_dim * 2, hidden_dim, n_layers=0,
                              res=False, act="gelu")
        self.placeholder = nn.Parameter(
            (1 / hidden_dim) * torch.rand(hidden_dim, dtype=torch.float32))

        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_heads, hidden_dim=hidden_dim, dropout=dropout,
                act="gelu", mlp_ratio=mlp_ratio, out_dim=out_channels,
                slice_num=slice_num, last_layer=(i == n_layers - 1))
            for i in range(n_layers)])

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _tokens(self, xyz, extra, is_query):
        """Build the common token feature vector."""
        B, N, _ = xyz.shape
        flag = torch.full((B, N, 1), 1.0 if is_query else 0.0,
                          device=xyz.device, dtype=xyz.dtype)
        parts = [xyz, extra, flag]
        if self.fourier is not None:
            parts.append(self.fourier(xyz))
        return torch.cat(parts, dim=-1)

    def encode_geometry(self, pc):
        """Subsample the boundary point cloud to the token budget (fixed per case)."""
        n = pc.shape[1]
        if n > self.n_pc_tokens:
            sel = torch.randperm(n, device=pc.device)[:self.n_pc_tokens]
            pc = pc[:, sel]
        return pc

    def decode_query(self, latent, xyz):
        pc = latent
        b_tok = self._tokens(pc[:, :, :3], pc[:, :, 3:], is_query=False)
        q_extra = torch.zeros(xyz.shape[0], xyz.shape[1], self.pc_channels - 3,
                              device=xyz.device, dtype=xyz.dtype)
        q_tok = self._tokens(xyz, q_extra, is_query=True)

        n_q = q_tok.shape[1]
        fx = self.preprocess(torch.cat([b_tok, q_tok], dim=1)) + self.placeholder

        for blk in self.blocks:
            fx = blk(fx)

        return fx[:, -n_q:]

    def forward(self, xyz, pc):
        return self.decode_query(self.encode_geometry(pc), xyz)
