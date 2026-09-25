"""Geom-DeepONet baseline — point-cloud-branch DeepONet.

Classical DeepONet (Lu et al. 2021) factorizes the operator as a dot product
between a branch network (encodes the input function / geometry) and a trunk
network (encodes the query coordinate). Geom-DeepONet (He et al. 2024) extends it
to varying geometry and additionally *modulates* the trunk with the shape latent
rather than only dot-producting at the end.

Implementation here:
  * branch  = the SAME PointCloudPerceiverChannelsEncoder GINOT uses
    (ginot/point_encoding.py), mean-pooled over its latent tokens to one vector,
    then projected to `n_basis * out_channels` coefficients.
  * trunk   = the SAME MildFourierEncoder + MLP GINOT uses (pi_ginot/model.py),
    producing `n_basis` basis functions per query point.
  * combine = per-output-channel dot product plus bias.
  * `film=True` adds the Geom-DeepONet trunk modulation (FiLM scale/shift from
    the shape latent at each trunk layer).

Sharing the branch and the Fourier trunk with GINOT is deliberate: with `film`
off this model is *GINOT minus cross-attention*, so the gap between them isolates
the value of GINOT's cross-attention decoder rather than confounding it with a
different geometry encoder.
"""

import torch
import torch.nn as nn

from ginot.point_encoding import PointCloudPerceiverChannelsEncoder
from pi_ginot.model import MildFourierEncoder


class FiLMLayer(nn.Module):
    def __init__(self, width, cond_dim):
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.mod = nn.Linear(cond_dim, 2 * width)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.lin(x)
        scale, shift = self.mod(cond).chunk(2, dim=-1)
        return self.act(h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class GeomDeepONet(nn.Module):
    def __init__(self, pc_channels=12, out_channels=3, n_basis=256,
                 trunk_width=256, trunk_layers=4, branch_out_c=256,
                 branch_width=128, branch_latent_d=1024, branch_n_point=1024,
                 branch_radius=0.08, num_bands=8, min_length_norm=0.05,
                 freq_spacing="log", film=True):
        super().__init__()
        self.n_basis = n_basis
        self.out_channels = out_channels
        self.film = film

        self.branch = PointCloudPerceiverChannelsEncoder(
            input_channels=pc_channels, out_c=branch_out_c, width=branch_width,
            latent_d=branch_latent_d, n_point=branch_n_point, radius=branch_radius)

        self.branch_head = nn.Sequential(
            nn.Linear(branch_out_c, 2 * branch_out_c), nn.SiLU(),
            nn.Linear(2 * branch_out_c, n_basis * out_channels),
        )

        self.fourier = MildFourierEncoder(
            in_channels=3, base_length=1.0, min_length=min_length_norm,
            num_bands=num_bands, spacing=freq_spacing)

        self.trunk_in = nn.Sequential(
            nn.Linear(self.fourier.out_channels, trunk_width), nn.SiLU())
        if film:
            self.trunk_layers = nn.ModuleList(
                [FiLMLayer(trunk_width, branch_out_c) for _ in range(trunk_layers)])
        else:
            self.trunk_layers = nn.ModuleList([
                nn.Sequential(nn.Linear(trunk_width, trunk_width), nn.SiLU())
                for _ in range(trunk_layers)])
        self.trunk_out = nn.Linear(trunk_width, n_basis)

        self.bias = nn.Parameter(torch.zeros(out_channels))

    def encode_geometry(self, pc):
        """-> (branch coefficients (1, n_basis, out_ch), shape latent (1, C))."""
        tokens = self.branch(pc)                    # (1, latent_d, out_c)
        shape_vec = tokens.mean(dim=1)              # (1, out_c)
        coeff = self.branch_head(shape_vec)
        coeff = coeff.view(-1, self.n_basis, self.out_channels)
        return coeff, shape_vec

    def decode_query(self, latent, xyz, seed_delta=None, h_delta=None):
        """`seed_delta` / `h_delta` are the hooks `model/lgo.py` uses.

        Both default to None and are then not touched at all, so the arithmetic
        of the plain baseline is UNCHANGED and `runs/short_deeponet` stays
        reproducible by this code. They exist so the hybrid does not have to
        restate this method -- a second copy of the trunk stack is exactly the
        drift that `panels()` paid for three times.

        `seed_delta` (1, M, W) is added to the trunk input, after `trunk_in`:
            the analogue of LGO's `--global_query local`, which lets a
            geometry-relative signal steer the basis functions themselves.
        `h_delta` (1, M, W) is added to the trunk's last hidden state, before
            `trunk_out`: the analogue of LGO adding its local stream into `x`
            before the shared `output_proj`. Because `trunk_out` is linear this
            is exactly basis augmentation,
                trunk_out(h + h_local) = trunk_out(h) + W_out @ h_local,
            with the head SHARED rather than a separate projection -- so it adds
            zero parameters beyond the local branch itself.
        """
        coeff, shape_vec = latent
        h = self.trunk_in(self.fourier(xyz))        # (1, M, W)
        if seed_delta is not None:
            h = h + seed_delta
        for layer in self.trunk_layers:
            h = layer(h, shape_vec) if self.film else layer(h) + h
        if h_delta is not None:
            h = h + h_delta
        basis = self.trunk_out(h)                   # (1, M, n_basis)
        # (1, M, n_basis) x (1, n_basis, out_ch) -> (1, M, out_ch)
        return torch.bmm(basis, coeff) / self.n_basis ** 0.5 + self.bias

    def forward(self, xyz, pc):
        return self.decode_query(self.encode_geometry(pc), xyz)