"""The LGO local branch: relative-coordinate attention over stratified neighbours.

This is the module that replaces SE. SE tripled the prior project's held-out
score by evaluating a jet template in each vent's own frame, so the near field
moves with the vent; it paid for that with hand-clustered vent anchors, a
downward-jet assumption and a fixed 3-vent structure. Here the same symmetry is
architectural: a query sees boundary points only through **offsets in metres**,
so nothing in this branch can read an absolute room coordinate, and the property
holds for any number of vents in any pose on any geometry.

WHY THIS SHOULD FIX GRADIENTS, WHICH IS THE REAL TARGET. The prior project
measured ||grad u||_F = 2.52 in the CFD against 0.84 for its best model: the
learned field is three times too smooth, and no loss term fixed it (supervising
grad u directly made it worse, 0.25). A trunk that reads absolute coordinates
has to *manufacture* a shear layer out of a smooth positional basis. A query
that sees a supply face 12 cm away, with that face's own velocity attached, has
the discontinuity handed to it as an input feature. That is the mechanism this
branch adds, and ||grad u||_F is the metric that should move if the hypothesis
is right.

DESIGN NOTES

* One attention over the CONCATENATED neighbour set, not one per group.
  Stratification is what the *gather* guarantees (data/knn_cache.py): every
  query always sees supply and return points however far away, so the vents that
  cause the flow can never be crowded out by nearby floor. Once that is
  guaranteed, letting a single attention weigh a wall against a vent directly is
  simpler and strictly more expressive than fusing per-group summaries.

* A radial basis on ||dx||, not just a raw scalar. The prior project's one
  intervention that ever moved the occupied zone was radial conditioning
  (occupied-zone R^2 -0.287 -> +0.008, for +1.8% parameters); its own jet
  envelope was a fixed Gaussian in radius. Distance is the single most
  informative thing about a neighbour, and an RBF makes "0.1 m away" and "2 m
  away" linearly separable instead of something the MLP has to carve out of one
  number spanning two decades.

* The output projection is zero-initialised, so at step 0 this branch
  contributes exactly nothing and the model is bit-identical to global-only.
  Any gain has to be earned, and the A/B has no confound. (Same trick the prior
  project used to make SE's comparison honest.)
"""

import math

import torch
import torch.nn as nn

from data.knn_cache import BASE_GROUPS, FEAT_DIM, GROUPS

EPS = 1e-6


def smooth_cutoff(r, n_keep):
    """Raised-cosine window that fades the OUTERMOST neighbours to exactly zero.

    A hard k-nearest-neighbour gather makes the prediction DISCONTINUOUS in the
    query: move x a little, the k-th neighbour is replaced, and that token's
    contribution vanishes abruptly, and a central difference across such a
    boundary diverges as h -> 0 instead of converging. That makes the field non-differentiable exactly where a
    derivative-based construction would need it, and it contributes spurious
    divergence that more training data does not remove.

    The fix is to give a neighbour zero weight BEFORE it leaves the set. Gather
    more neighbours than are used, then weight by

        w = 1                              r <= r_keep
        w = (1 + cos(pi t)) / 2            r_keep < r < r_max,  t = (r-r_keep)/(r_max-r_keep)
        w = 0                              r >= r_max

    `r_keep` and `r_max` are the n_keep-th and last gathered distances. Which
    POINT is n-th changes discontinuously, but the n-th DISTANCE does not, so
    the window moves continuously with the query. Any neighbour at the boundary
    of the gathered set already has w = 0, so entering and leaving cost nothing.
    """
    r_keep = r.gather(1, torch.full((r.shape[0], 1), n_keep - 1,
                                    device=r.device, dtype=torch.long))
    r_max = r.max(dim=1, keepdim=True).values
    span = (r_max - r_keep).clamp_min(EPS)
    t = ((r - r_keep) / span).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


class RadialBasis(nn.Module):
    """Gaussian RBFs on ||dx||, log-spaced from a few cm to a few metres."""

    def __init__(self, n=12, r_min=0.02, r_max=4.0, learnable=False):
        super().__init__()
        centres = torch.logspace(math.log10(r_min), math.log10(r_max), n)
        # Width tracks spacing so the bank tiles the range without gaps.
        widths = torch.cat([centres[1:] - centres[:-1], (centres[-1] - centres[-2]).view(1)])
        if learnable:
            self.centres, self.widths = nn.Parameter(centres), nn.Parameter(widths)
        else:
            self.register_buffer("centres", centres)
            self.register_buffer("widths", widths)
        self.n = n

    def forward(self, r):                      # r: (..., 1)
        z = (r - self.centres) / self.widths.abs().clamp_min(1e-4)
        return torch.exp(-z * z)               # (..., n)


class NeighbourEncoder(nn.Module):
    """One boundary neighbour -> one token, from RELATIVE quantities only.

    Inputs per neighbour: the offset dx = x_boundary - x_query in METRES, the
    boundary point's features (normal, blinded BC velocity, blinded BC
    temperature, leak magnitude) and its group. No absolute coordinate appears
    anywhere, which is what makes the branch translation-equivariant.
    """

    def __init__(self, d_token, n_rbf=12, hidden=64, feat_dim=FEAT_DIM,
                 n_groups=len(BASE_GROUPS), rbf_max=4.0):
        super().__init__()
        self.rbf = RadialBasis(n=n_rbf, r_max=rbf_max)
        # dx(3) + log r(1) + dxhat(3) + n.dxhat(1) + rbf(n) + feat + group one-hot
        in_dim = 3 + 1 + 3 + 1 + n_rbf + feat_dim + n_groups
        # NARROW ON PURPOSE. Every tensor here is per (query, neighbour) pair, so
        # its width is multiplied by N*K -- at 5120 queries and 88 neighbours
        # that is 450k rows, and a 256-wide token costs 460 MB per forward (and
        # the same again held for backward). A boundary point is a far simpler
        # object than a query state: it is a position, a normal, a type and a
        # boundary value. 64 channels carry that comfortably, and the attention
        # projects up to the model width on the way out, where the tensor is
        # per-query again and N*K has collapsed to N.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_token))
        self.norm = nn.LayerNorm(d_token)
        self.in_dim, self.d_token = in_dim, d_token

    def forward(self, dxyz, feat, group_onehot):
        r = dxyz.norm(dim=-1, keepdim=True)                  # (N, K, 1)
        dhat = dxyz / r.clamp_min(EPS)
        normal = feat[..., 0:3]
        n_dot = (normal * dhat).sum(-1, keepdim=True)
        x = torch.cat([dxyz, torch.log(r + 1e-3), dhat, n_dot,
                       self.rbf(r), feat, group_onehot], dim=-1)
        return self.norm(self.mlp(x))


class LocalCrossAttention(nn.Module):
    """Each query attends to its OWN neighbour set. Batched, not global.

    Shapes: q (N, D), tokens (N, K, D), valid (N, K) bool. Cost is O(N*K*D),
    never O(N^2) -- queries do not see each other here.
    """

    def __init__(self, d_model, n_heads=8, dropout=0.0, d_token=None):
        super().__init__()
        d_token = d_token or d_model
        if d_token % n_heads:
            raise ValueError(f"d_token {d_token} must divide by n_heads {n_heads}")
        self.h, self.dk = n_heads, d_token // n_heads
        # Attention runs at the TOKEN width: keys and values are the per-pair
        # tensors, so widening them before the softmax would undo the saving the
        # narrow encoder just made. Only `out` touches the model width.
        self.q_proj = nn.Linear(d_model, d_token)
        self.k_proj = nn.Linear(d_token, d_token)
        self.v_proj = nn.Linear(d_token, d_token)
        self.out = nn.Linear(d_token, d_model)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        # Zero-init: this branch is a no-op at step 0, so the LGO-vs-global-only
        # comparison starts from an identical model and every gain is earned.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, q, tokens, valid=None, soft=None):
        N, K, _ = tokens.shape
        qh = self.q_proj(q).view(N, self.h, 1, self.dk)
        kh = self.k_proj(tokens).view(N, K, self.h, self.dk).transpose(1, 2)
        vh = self.v_proj(tokens).view(N, K, self.h, self.dk).transpose(1, 2)

        att = (qh * kh).sum(-1) / math.sqrt(self.dk)              # (N, h, K)
        alive = None
        if valid is not None:
            alive = valid.any(dim=1)                               # (N,)
            att = att.masked_fill(~valid[:, None, :], float("-inf"))
            # A query whose every neighbour is padding (an empty group, e.g. a
            # case with no leak) would softmax over a row of all -inf and emit
            # NaN, poisoning the whole batch's gradients. Neutralise the row.
            att = torch.where(alive[:, None, None], att, torch.zeros_like(att))
        if soft is not None:
            # Added to the LOGIT, not multiplied after the softmax: a neighbour
            # whose window reaches zero then gets weight exactly zero and the
            # remaining weights renormalise continuously, which is the smooth
            # limit of the hard mask above.
            att = att + torch.log(soft.clamp_min(EPS))[:, None, :]
        w = self.drop(att.softmax(dim=-1))
        out = self.out((w[..., None] * vh).sum(dim=2).reshape(N, -1))
        if alive is not None:
            # Zero the OUTPUT, not the attention weights: zeroing the weights
            # still leaves `self.out`'s bias, so a query with no neighbours at
            # all would emit a constant vector instead of abstaining. Harmless
            # at init (bias starts at 0) and wrong as soon as the bias trains.
            out = out * alive[:, None].to(out.dtype)
        return out


class LocalBranch(nn.Module):
    """NeighbourEncoder + LocalCrossAttention, shared across trunk layers.

    Tokens are encoded ONCE per query batch and reused by every layer: the
    encoding depends only on geometry and boundary conditions, not on the
    evolving query state, so recomputing it per layer would cost L times as much
    for an identical result.
    """

    def __init__(self, d_model, n_heads=8, n_rbf=12, hidden=64, n_layers=1,
                 rbf_max=4.0, d_token=64, groups=BASE_GROUPS, soft_knn=0):
        super().__init__()
        # soft_knn = how many EXTRA neighbours are gathered per group purely to
        # be faded out. 0 keeps the original hard gather byte-for-byte.
        self.soft_knn = int(soft_knn)
        self.d_token = d_token
        self.encoder = NeighbourEncoder(d_token, n_rbf=n_rbf, hidden=hidden,
                                        rbf_max=rbf_max, n_groups=len(groups))
        self.attn = nn.ModuleList(
            [LocalCrossAttention(d_model, n_heads, d_token=d_token)
             for _ in range(n_layers)])
        self.groups = tuple(groups)
        self.n_groups = len(self.groups)

    def encode(self, nb):
        """`nb` = {group: (dxyz, feat, valid)} of torch tensors on one device."""
        dxyz, feat, valid, gid = [], [], [], []
        for i, g in enumerate(self.groups):
            if g not in nb:
                continue
            d, f, v = nb[g]
            dxyz.append(d)
            feat.append(f)
            valid.append(v)
            oh = torch.zeros(d.shape[0], d.shape[1], self.n_groups,
                             device=d.device, dtype=d.dtype)
            oh[..., i] = 1.0
            gid.append(oh)
        if not dxyz:
            raise ValueError("no neighbour groups supplied to LocalBranch")
        soft = None
        if self.soft_knn > 0:
            wins = []
            for d, v in zip(dxyz, valid):
                k_tot = d.shape[1]
                n_keep = max(1, k_tot - self.soft_knn)
                r = d.norm(dim=-1)
                w = smooth_cutoff(r, n_keep)
                wins.append(w * v.to(w.dtype))       # padding stays at zero
            soft = torch.cat(wins, 1)
        return (self.encoder(torch.cat(dxyz, 1), torch.cat(feat, 1), torch.cat(gid, 1)),
                torch.cat(valid, 1), soft)

    def forward(self, q, tokens, valid, layer=0, soft=None):
        return self.attn[layer](q, tokens, valid, soft)


def neighbours_to_torch(nb_np, device, dtype=torch.float32):
    """{group: (dxyz, feat, valid)} numpy -> torch on `device`.

    dxyz stays float32 even under AMP: it carries the metres-scale offsets that
    the equivariance claim and the no-slip mask both rest on.
    """
    return {g: (torch.as_tensor(d, dtype=dtype, device=device),
                torch.as_tensor(f, dtype=dtype, device=device),
                torch.as_tensor(v, dtype=torch.bool, device=device))
            for g, (d, f, v) in nb_np.items()}
