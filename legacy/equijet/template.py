"""JetTemplate — the shared, vent-local jet field (PLAN_EQUIJET_UPGRADE.md §3.2).

    J(xi, c) = envelope(xi) * MLP( Fourier(xi) , FiLM(c) )

`xi` is the offset from a vent anchor in METRES, so one template is shared by every
vent in every room: moving a vent moves its jet, which is the equivariance the
upgrade is for.

Three properties the rest of the design depends on:

* **Zero-initialized head** — at epoch 0 the template outputs exactly 0, so the
  wrapped model is bitwise identical to the bare trunk. That gives a safe warm
  start and a clean A/B.
* **Finer Fourier bandwidth than the trunk** (`min_length` ~0.08 m vs the trunk's
  0.15 m): jets are the sharp structure, and giving the template its own high
  bandwidth is what lets the trunk stop chasing them.
* **Decay envelope** `exp(-||xi||^2 / 2s^2)` with `s` learnable and clamped — the
  template may only claim the near field, which keeps the correction identifiable
  and prevents the two from double-counting.
"""

import numpy as np
import torch
import torch.nn as nn

from pi_ginot.model import MildFourierEncoder   # imported, not copied


class FiLMBlock(nn.Module):
    """One hidden layer modulated by the vent conditioning vector."""

    def __init__(self, width, cond_dim):
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.mod = nn.Linear(cond_dim, 2 * width)
        self.act = nn.SiLU()

    def forward(self, h, c):
        # h: (K, N, W); c: (K, cond_dim) -> broadcast over the query axis
        gamma, beta = self.mod(c).chunk(2, dim=-1)
        return self.act(self.lin(h) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1))


class JetTemplate(nn.Module):
    def __init__(self, num_bands=6, min_length=0.08, base_length=4.0,
                 hidden=128, depth=4, cond_dim=7, s_init=1.5, s_max=3.0,
                 out_channels=3, radial=False, n_radial=10, radial_max=3.0):
        super().__init__()
        self.s_max = float(s_max)
        self.radial = radial

        # Fourier features on the LOCAL offset, in metres.
        self.fourier = MildFourierEncoder(
            in_channels=3, base_length=base_length, min_length=min_length,
            num_bands=num_bands, spacing="log")

        # Optional radial basis on ||xi||. The Fourier encoder is separable per
        # axis, so a rotationally symmetric jet core has to be assembled out of
        # axis-aligned harmonics — expensive, and it biases level sets towards
        # boxes. Giving the MLP an explicit distance-from-vent channel lets it
        # represent a round core directly. Centres are log-spaced so resolution is
        # finest near the vent, where the jet structure is.
        n_extra = 0
        if radial:
            centres = torch.exp(torch.linspace(float(np.log(0.02)),
                                               float(np.log(radial_max)), n_radial))
            self.register_buffer("rbf_c", centres)
            # Width of each RBF = spacing to its neighbour, so they tile smoothly.
            widths = torch.cat([centres[1:] - centres[:-1],
                                (centres[-1:] - centres[-2:-1])])
            self.register_buffer("rbf_w", widths.clamp_min(1e-3))
            n_extra = n_radial + 1                      # + raw normalized r

        self.inp = nn.Sequential(
            nn.Linear(self.fourier.out_channels + n_extra, hidden), nn.SiLU())
        self.blocks = nn.ModuleList([FiLMBlock(hidden, cond_dim) for _ in range(depth)])
        self.head = nn.Linear(hidden, out_channels)

        # Zero-init head => J == 0 at epoch 0 => wrapper == bare trunk, bitwise.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Envelope scale, stored raw and clamped in forward so it cannot run away
        # to room scale and start competing with the correction.
        self.s_raw = nn.Parameter(torch.tensor(float(s_init)))

    @property
    def s(self):
        return self.s_raw.clamp(0.05, self.s_max)

    def envelope(self, xi):
        r2 = (xi ** 2).sum(dim=-1, keepdim=True)
        return torch.exp(-r2 / (2.0 * self.s ** 2))

    def radial_features(self, xi):
        """Distance-from-vent channels: normalized r plus a bank of RBFs."""
        r = xi.norm(dim=-1, keepdim=True)                       # (K, N, 1)
        rbf = torch.exp(-((r - self.rbf_c) / self.rbf_w) ** 2)  # (K, N, n_radial)
        return torch.cat([r / self.s_max, rbf], dim=-1)

    def forward(self, xi, c):
        """xi: (K, N, 3) metres; c: (K, cond_dim) -> (K, N, out_channels)."""
        feat = self.fourier(xi)
        if self.radial:
            feat = torch.cat([feat, self.radial_features(xi)], dim=-1)
        h = self.inp(feat)
        for blk in self.blocks:
            h = blk(h, c)
        return self.envelope(xi) * self.head(h)
