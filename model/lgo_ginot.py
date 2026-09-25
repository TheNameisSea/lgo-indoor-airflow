"""LGO-GINOT -- the local branch on a GINOT cross-attention trunk.

    global : boundary cloud -> Perceiver latent -> cross-attention. Room-scale.
    local  : query -> stratified boundary neighbours, seen only as offsets in
             metres.                 Near-field, equivariant by construction.

`global_query` decides what queries the room latent: `local` (the local stream;
no absolute coordinate enters) or `learned` (one shared query vector). Use
`--model lgo_ginot --global_query local`.

The global path is GINOT reused unmodified from `legacy/`, so this model with the
local branch off is exactly `--model ginot`. The local branch is zero-initialised,
so at step 0 the two are bit-identical. Its radial basis tops out at a few
metres, so room-scale structure has to come from the global path.

Runs whose `args.json` says `"model": "lgo"` and has no `lgo_arch` key were
trained with this architecture (the name `lgo` now means LGO-GDON);
`evaluate.py` resolves them here.

INTERFACE. `encode_geometry(pc)` / `decode_query(latent, xyt)` / `forward`,
matching `legacy/pi_ginot/model.py::Trunk`, so `legacy/eval_thermo.py` scores
this model through the same code path as every baseline. `decode_query` is
handed no case identity, so the case is resolved in `encode_geometry` by a
fingerprint of the boundary cloud and remembered.
"""

import torch
import torch.nn as nn

from data.knn_cache import BASE_GROUPS
from model.boundary_lookup import BoundaryLookup, fingerprint  # noqa: F401
from model.local_branch import LocalBranch

DEFAULT_K = {"supply": 32, "return": 16, "solid": 32, "leak": 8}
# Opt-in coarse room-scale group; 0 keeps a model on BASE_GROUPS.
DEFAULT_K_FAR = 32


class LGOGinot(BoundaryLookup, nn.Module):
    """GINOT trunk + stratified local attention + optional hard no-slip.

    `trunk` is a built `legacy.pi_ginot.model.Trunk`. We do not subclass it: the
    baseline stays byte-identical and reusable, exactly as `hbc/` and `equijet/`
    wrap rather than edit.
    """

    def __init__(self, trunk, coord_min, coord_scale, target_mean, target_std,
                 d_model=256, n_heads=8, k=None, n_local_layers=1, n_rbf=12,
                 local_hidden=128, rbf_max=4.0, use_local=True, use_global=True,
                 use_hbc=False, ell_wall=0.002, chunk_knn=200_000,
                 local_query="learned", global_query="local", soft_knn=0,
                 groups=BASE_GROUPS, far_voxels=None):
        """`local_query` decides whether the local TERM is equivariant.

        "learned" (default, two-stream): the local stream starts at zero, so its
            attention is queried by the projection bias -- a learned constant --
            and then by its own equivariant output. Nothing derived from an
            absolute coordinate ever queries it, so the local term is exactly
            translation-equivariant and is ADDED to the global path. This is the
            decomposition SE used, `u = sum_k J(x - x_k) + R(x, room)`, and it is
            the arrangement whose property can be proven rather than hoped for.

        "global" (interleaved, ablation D6): the local attention is queried by
            the global representation, which carries Fourier(x_query). More
            expressive -- the query can ask "given where I am in the room, what
            matters nearby" -- but the attention weights then move when the room
            is translated, so the equivariance receipt fails for the full model.
            Measured on an untrained model: 3.5e-01 vs 0.0 for "learned".
        """
        super().__init__()
        if local_query not in ("learned", "global"):
            raise ValueError(f"local_query must be 'learned' or 'global', got {local_query!r}")
        if global_query not in ("learned", "local"):
            raise ValueError(f"global_query must be 'learned' or 'local', got "
                             f"{global_query!r}. The 'fourier' setting -- query "
                             f"the latent with Fourier(x_q), an ABSOLUTE "
                             f"position -- has been removed; see below.")
        self.local_query = local_query
        # `global_query` decides WHAT queries the room latent.
        #
        # REMOVED: "fourier", the query being Fourier(x_q), an ABSOLUTE position.
        #     It is the mechanism behind layout memorisation -- a jet at (2,3) m
        #     and the same jet at (6,9) m are two unrelated patterns to be
        #     learned separately -- which is the failure this architecture exists
        #     to avoid. Every run has used "local"; nothing was ever trained on
        #     it. The Fourier encoder itself remains inside the GINOT trunk
        #     because `legacy/equijet` calls `trunk.decode_query` directly and is
        #     a planned comparison arm, and because removing it would change the
        #     state_dict and make every existing checkpoint unloadable for a
        #     path none of them use.
        # "learned": a single learned query vector, shared by every point. The
        #     global path collapses to one room-level context vector added to
        #     every query. The cleanest ablation: it removes position-dependence
        #     from the global path and changes nothing else.
        # "local": the local stream's own output queries the latent. The query
        #     then varies in space but is built ONLY from boundary offsets, so no
        #     absolute coordinate enters it. This is the variant that can still
        #     do spatial work -- "given the geometry around me, what does this
        #     room look like" -- and it requires use_local.
        #
        # Motivation: `lgo_local` (no global path at all, ~188k effective
        # parameters) beat full LGO on held-out layouts, which says the global
        # path hurts. These two settings separate "the room latent hurts" from
        # "reading absolute position hurts".
        self.global_query = global_query
        # Opt-in: rebuild neighbour offsets in torch so d(pred)/d(query) exists.
        # Off by default -- it costs a tensor rebuild per group per forward and
        # nothing in the standard training path differentiates through position.
        self.grad_wrt_query = False
        self.trunk = trunk
        self.use_local = use_local
        # use_global=False is the local-only ablation arm: the query never
        # sees Fourier(x) or the room latent, so the model has to explain the
        # field from boundary offsets alone. Only meaningful with
        # local_query='learned', which does not need a global seed.
        self.use_global = use_global
        self.use_hbc = use_hbc
        self.ell_wall = float(ell_wall)
        # Per-case KD-tree indices, query->metres and the stratified gather all
        # live in BoundaryLookup, shared with model/lgo.py.
        self._init_lookup(k or DEFAULT_K, groups=groups, far_voxels=far_voxels,
                          chunk_knn=chunk_knn)
        self.local = (LocalBranch(d_model, n_heads=n_heads, n_rbf=n_rbf,
                                  hidden=local_hidden, n_layers=n_local_layers,
                                  rbf_max=rbf_max, groups=self.groups,
                                  soft_knn=soft_knn)
                      if use_local else None)
        self.n_local_layers = n_local_layers
        if global_query == "local" and not use_local:
            raise ValueError("global_query='local' needs the local branch to "
                             "produce the query; got use_local=False")
        # One learned query vector, shared by every point, for
        # global_query='learned'. Small init so the global path starts as a
        # near-zero room bias rather than a large constant offset.
        self.global_seed = (nn.Parameter(torch.randn(d_model) * 0.02)
                            if global_query == "learned" else None)

        # Physical-unit constants. Velocity is z-scored, so physical u = 0 is
        # normalized -mu/sigma, not 0: the no-slip mask must be applied in
        # physical units or it enforces the wrong condition.
        self.register_buffer("mu", target_mean[0:3].clone().float())
        self.register_buffer("sigma", target_std[0:3].clone().float())
        self.register_buffer("coord_min", coord_min.clone().float())
        self.register_buffer("coord_scale", coord_scale.clone().float())

        # Debug hook: stash the local stream so its equivariance can be measured
        # directly. Off by default -- at 1.16 M query points a 256-wide stream is
        # ~1.2 GB, and it cannot be recovered from the outputs because
        # `output_proj` is a nonlinear MLP (Linear-SiLU-Linear), so
        # out(local on) - out(local off) is f(x+h) - f(x), not a function of h.
        self.debug_keep_local = False
        self._last_local = None

    # ------------------------------------------------------------------ #
    #  Trunk interface
    # ------------------------------------------------------------------ #
    def encode_geometry(self, pc, sample_ids=None):
        self._active = self._resolve(pc) if (self.use_local or self.use_hbc) else None
        return self.trunk.encode_geometry(pc, sample_ids=sample_ids)

    def decode_query(self, latent, xyt):
        squeeze = xyt.dim() == 2
        if squeeze:
            xyt = xyt.unsqueeze(0)
        B, N, _ = xyt.shape
        if B != 1 and (self.use_local or self.use_hbc):
            raise NotImplementedError(
                "LGO resolves one case per forward pass (batch of rooms is not "
                "supported by the local branch); got batch of {B}.")

        # Metres once, in float64, shared by the local branch and the mask. The
        # KD-tree search and the no-slip mask BOTH need this exact float64 path
        # (see `_metres`), and both are correctly non-differentiable.
        xyz_m = (self._metres(xyt[0], self._active)
                 if (self.use_local or self.use_hbc) else None)
        # A differentiable copy of the same coordinate, used only to rebuild the
        # neighbour offsets. Enabled on demand so the default path is unchanged.
        xq_grad = None
        if self.use_local and self.grad_wrt_query and xyt.requires_grad:
            sc = torch.as_tensor(self._active.coord_scale, dtype=xyt.dtype,
                                 device=xyt.device)
            mn = torch.as_tensor(self._active.coord_min, dtype=xyt.dtype,
                                 device=xyt.device)
            xq_grad = xyt[0] * sc + mn

        d_model = self.trunk.Q_encoder[-1].out_features

        # The local stream runs FIRST when it has to seed the global query
        # (global_query='local'). Otherwise the order is immaterial: the two
        # streams never see each other's output, they are added at the end.
        h_pre = None
        if self.use_local and self.global_query == "local":
            tok_pre, val_pre, soft_pre = self.local.encode(
                self._gather(xyz_m, self._active, xyt.device, xq_grad))
            h_pre = torch.zeros(N, d_model, device=xyt.device, dtype=latent.dtype)
            for l in range(self.n_local_layers):
                h_pre = h_pre + self.local(h_pre, tok_pre, val_pre, layer=l,
                                           soft=soft_pre)

        # --- global path: the unmodified GINOT trunk -------------------- #
        if self.use_global:
            if self.global_query == "learned":
                x = self.global_seed.to(latent.dtype).expand(B, N, -1)
            else:                                   # "local"
                x = h_pre.unsqueeze(0)
            for block in self.trunk.resblocks:
                x = block(x, latent)
        else:
            x = latent.new_zeros(B, N, d_model)

        # --- local path: added into the SAME representation ------------- #
        if self.use_local:
            if h_pre is not None:
                tokens, valid, soft = tok_pre, val_pre, soft_pre
            else:
                tokens, valid, soft = self.local.encode(
                    self._gather(xyz_m, self._active, xyt.device, xq_grad))
            if self.local_query == "learned":
                # Start at zero: the layer-0 query is q_proj's bias (a learned
                # constant) and every later query is the stream's own output, so
                # no absolute coordinate ever reaches the attention. Zero-init on
                # attn.out then keeps this stream exactly 0 at step 0, which is
                # what makes "LGO minus local == GINOT" hold bit-for-bit.
                if h_pre is not None:
                    h = h_pre          # already computed to seed the global query
                else:
                    h = torch.zeros_like(x[0])
                    for l in range(self.n_local_layers):
                        h = h + self.local(h, tokens, valid, layer=l, soft=soft)
                if self.debug_keep_local:
                    self._last_local = h.detach().clone()
                x = x + h.unsqueeze(0)
            else:
                h = x[0]
                for l in range(self.n_local_layers):
                    h = h + self.local(h, tokens, valid, layer=l, soft=soft)
                if self.debug_keep_local:
                    self._last_local = (h - x[0]).detach().clone()
                x = h.unsqueeze(0)

        out = self.trunk.output_proj(x)

        # --- hard no-slip ----------------------------------------------- #
        if self.use_hbc:
            m = self.wall_mask(xyz_m, self._active, xyt.device)
            vel_phys = out[0, :, 0:3] * self.sigma + self.mu
            out = torch.cat([((m * vel_phys) - self.mu) / self.sigma,
                             out[0, :, 3:]], dim=-1).unsqueeze(0)

        return out.squeeze(0) if squeeze else out

    def forward(self, xyt, pc, sample_ids=None):
        return self.decode_query(self.encode_geometry(pc, sample_ids), xyt)


def build_lgo_ginot(pc_channels, coord_min, coord_scale, target_mean, target_std,
              out_channels=5, embed_dim=256, cross_attn_layers=5,
              branch_width=128, branch_latent_d=1024, branch_n_point=1024,
              branch_radius=0.08, min_length_norm=0.017, num_bands=8,
              freq_spacing="log", verbose=True, **lgo_kw):
    """Assemble LGO on top of a freshly built GINOT trunk."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "legacy"))
    from pi_ginot import build_model

    trunk = build_model(
        {"input_channels": pc_channels, "out_c": embed_dim, "width": branch_width,
         "latent_d": branch_latent_d, "n_point": branch_n_point,
         "radius": branch_radius},
        {"in_channels": 3, "out_channels": out_channels, "embed_dim": embed_dim,
         "cross_attn_layers": cross_attn_layers, "min_length_norm": min_length_norm,
         "num_bands": num_bands, "freq_spacing": freq_spacing},
        verbose=verbose)
    model = LGOGinot(trunk, coord_min, coord_scale, target_mean, target_std,
                d_model=embed_dim, **lgo_kw)
    if verbose:
        tot = sum(p.numel() for p in model.parameters())
        loc = sum(p.numel() for p in model.local.parameters()) if model.local else 0
        print(f"[lgo_ginot] {tot:,} params total ({loc:,} in the local branch, "
              f"{tot - loc:,} in GINOT)")
    return model
