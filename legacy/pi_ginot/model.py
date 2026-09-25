"""
Model definition for PI-GINOT: a physics-bounded Fourier query encoder (Trunk)
fused with the reused PointNet/Perceiver geometry encoder (Branch).

Ported verbatim (behaviour-preserving) from notebook cells 4, 5 and 6, with the
only change being the import source: the geometry encoder now comes from the
local `ginot` package instead of the Kaggle `ginot_test_data_and_configs` package.
"""

import numpy as np
import torch
import torch.nn as nn

from ginot.point_encoding import PointCloudPerceiverChannelsEncoder


class MildFourierEncoder(nn.Module):
    """Band-limited Fourier feature encoder for (x, y, z) queries.

    Frequencies are linearly spaced between a base wavelength and a minimum
    wavelength (`min_length`). The minimum wavelength caps the spatial detail the
    query stream can represent; it is deliberately kept mild to avoid noisy PDE
    derivatives (see the audit note on bandwidth for future tuning).
    """

    def __init__(self, in_channels=3, base_length=1.0, min_length=0.1, num_bands=4,
                 spacing="linear"):
        super().__init__()
        # Angular frequencies (w = 2 * pi / L).
        w_min = 2 * np.pi / base_length
        w_max = 2 * np.pi / min_length

        # `min_length` sets the CEILING (finest resolvable feature); `num_bands`
        # only sets how densely [w_min, w_max] is sampled -- more bands does NOT
        # raise the ceiling.
        #
        # Spacing matters more than band count as the range widens. Linear
        # spacing puts almost every band at high frequency: at min_length=0.0085
        # with 4 bands the wavelengths are 1.0, 0.025, 0.013, 0.0085 -- a total
        # hole across the mid scales where recirculation cells live. Log (octave)
        # spacing is the standard choice for Fourier features and keeps coverage
        # geometric across the whole range.
        if num_bands < 2:
            freqs = torch.tensor([w_max], dtype=torch.float32)
        elif spacing == "log":
            freqs = torch.exp(torch.linspace(float(np.log(w_min)), float(np.log(w_max)),
                                             steps=num_bands))
        elif spacing == "linear":
            freqs = torch.linspace(w_min, w_max, steps=num_bands)
        else:
            raise ValueError(f"spacing must be 'linear' or 'log', got {spacing!r}")

        # Buffer so it moves to the GPU but is not updated by the optimizer.
        self.register_buffer("freqs", freqs)

        # in_channels * num_bands * 2 (sin/cos) channels total.
        self.out_channels = in_channels * num_bands * 2

    def forward(self, x):
        # x: (B, N, 3) -> (B, N, 3, 1); freqs -> (1, 1, 1, num_bands)
        x_expanded = x.unsqueeze(-1)
        freqs_expanded = self.freqs.view(1, 1, 1, -1)

        scaled_x = x_expanded * freqs_expanded  # (B, N, 3, num_bands)

        sin_x = torch.sin(scaled_x)
        cos_x = torch.cos(scaled_x)

        # Flatten spatial-channel and frequency dims -> (B, N, out_channels)
        out = torch.cat([sin_x, cos_x], dim=-1).view(x.shape[0], x.shape[1], -1)
        return out


class Trunk(nn.Module):
    """Spatial physics solver: encodes geometry into a latent, then decodes
    (u, v, w, p) at arbitrary query coordinates via cross-attention."""

    def __init__(self, branch, embed_dim=256, cross_attn_layers=5, num_heads=8,
                 in_channels=3, out_channels=4, min_length_norm=0.1,
                 num_bands=4, freq_spacing="linear"):
        super().__init__()
        self.branch = branch

        # 1. Spatial query encoder. `min_length_norm` caps the finest spatial
        # detail the query stream can represent -- it was originally set to the
        # vent width to keep PDE second derivatives tame. With the PDE dropped
        # that constraint no longer applies, so this is now a tunable.
        self.fourier_enc = MildFourierEncoder(
            in_channels=in_channels,
            base_length=1.0,
            min_length=min_length_norm,
            num_bands=num_bands,
            spacing=freq_spacing,
        )

        # 2. Spatial query encoder (accepts the Fourier output).
        self.Q_encoder = nn.Sequential(
            nn.Linear(self.fourier_enc.out_channels, 2 * embed_dim), nn.SiLU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

        # 3. Point-cloud latent smoother.
        self.latent_encoder = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim), nn.SiLU(),
            nn.Linear(2 * embed_dim, embed_dim), nn.SiLU(),
        )

        # 4. Cross-attention geometry fusion.
        from ginot.transformer import ResidualCrossAttentionBlock
        self.resblocks = nn.ModuleList([
            ResidualCrossAttentionBlock(width=embed_dim, heads=num_heads, dropout=0.0)
            for _ in range(cross_attn_layers)
        ])

        # 5. Physics output projector.
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim), nn.SiLU(),
            nn.Linear(2 * embed_dim, out_channels),
        )

    def encode_geometry(self, pc, sample_ids=None):
        latent = self.branch(pc, sample_ids=sample_ids)
        return self.latent_encoder(latent)

    def decode_query(self, latent, xyt):
        # Transform pure XYZ coordinates into the Fourier feature space.
        xyt_encoded = self.fourier_enc(xyt)

        x = self.Q_encoder(xyt_encoded)

        # Cross-attention with the physics-aware point cloud latent.
        for block in self.resblocks:
            x = block(x, latent)

        x = self.output_proj(x)
        return x.squeeze(-1)

    def forward(self, xyt, pc, sample_ids=None):
        latent = self.encode_geometry(pc, sample_ids)
        return self.decode_query(latent, xyt)


def build_model(branch_args, trunk_args, verbose=True):
    """Assemble the geometry Branch and the physics Trunk into the end-to-end model."""
    branch = PointCloudPerceiverChannelsEncoder(**branch_args)
    if verbose:
        branch_tot = sum(p.numel() for p in branch.parameters())
        branch_train = sum(p.numel() for p in branch.parameters() if p.requires_grad)
        print(f"[Branch] Geo Encoder: {branch_tot:,} total params, {branch_train:,} trainable")

    trunk = Trunk(branch, **trunk_args)
    if verbose:
        model_tot = sum(p.numel() for p in trunk.parameters())
        model_train = sum(p.numel() for p in trunk.parameters() if p.requires_grad)
        print(f"[Total] Assembled PI-GINOT: {model_tot:,} total params, {model_train:,} trainable")

    return trunk
