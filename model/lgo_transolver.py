"""LGO-TRANSOLVER -- the local branch on a Transolver trunk.

    local  : query -> type-stratified boundary neighbours, seen only as offsets
             in metres -> attention.                              Near-field.
    global : Physics-Attention over boundary + query tokens.      Room-scale.

A wrapper: `baselines/models/transolver.py` is not modified, so `--model
transolver` is exactly the host. Transolver has no hook for an added stream, so
`decode_query` below restates the parent's composition. If
`TransolverBaseline.decode_query` changes, this diverges silently;
`tests/test_lgo_transolver.py` checks that with the branch off this class is
bit-identical to the baseline. Run it after any change to either file.

INJECTION POINT. On GINO and Geom-DeepONet the local stream is added after the
stack, before a shared head (`inject="basis"`). Transolver's output head sits
inside its final block, so here the branch is added to the query tokens before
the first block -- the input-conditioning variant (`inject="seed"`). The head
stays shared and untouched. Describe this arm as seed-style injection; it is not
the same injection point as the other hybrids.
"""

import json

import torch
import torch.nn as nn

from data.knn_cache import BASE_GROUPS, make_groups
from model.boundary_lookup import BoundaryLookup
from model.local_branch import LocalBranch

# Every name that means THIS architecture, defined here and imported by both
# train.py and evaluate.py so the trainer and the evaluator cannot disagree.
LGO_TRANSOLVER_FAMILY = ("lgo_transolver",)
ARCH_KEY = "lgo_arch"
ARCH = "transolver"


class LGOTransolver(BoundaryLookup, nn.Module):
    """`TransolverBaseline` + the LGO local branch, sharing the Transolver head."""

    def __init__(self, transolver, coord_min, coord_scale, target_mean, target_std,
                 k=None, n_heads=8, n_local_layers=1, n_rbf=12, local_hidden=128,
                 rbf_max=4.0, soft_knn=0, groups=BASE_GROUPS,
                 far_voxels=None, chunk_knn=200_000):
        super().__init__()
        self.transolver = transolver
        self.n_local_layers = n_local_layers
        self._init_lookup(k, groups=groups, far_voxels=far_voxels,
                          chunk_knn=chunk_knn)

        # The host's own width -- a projection here would add parameters this arm
        # does not need and break the shared-head equivalence the ablation rests on.
        # `placeholder` IS the (hidden_dim,) parameter added to every token, so
        # its length is the trunk width by construction. `TransolverBaseline`
        # stores no `hidden_dim` attribute, so reading one would be a latent
        # AttributeError waiting for the first non-default config.
        d_model = transolver.placeholder.shape[0]
        self.local = LocalBranch(d_model, n_heads=n_heads, n_rbf=n_rbf,
                                 hidden=local_hidden, n_layers=n_local_layers,
                                 rbf_max=rbf_max, groups=self.groups,
                                 soft_knn=soft_knn)

        self.register_buffer("mu", target_mean[0:3].clone().float())
        self.register_buffer("sigma", target_std[0:3].clone().float())
        self.register_buffer("coord_min", coord_min.clone().float())
        self.register_buffer("coord_scale", coord_scale.clone().float())

        self.use_local = True
        self.use_hbc = False
        self.grad_wrt_query = False

    # ------------------------------------------------------------------ #
    #  Trunk interface -- matches model/lgo_gino.py::LGOGino
    # ------------------------------------------------------------------ #
    def encode_geometry(self, pc, sample_ids=None):
        self._active = self._resolve(pc)
        return self.transolver.encode_geometry(pc)

    def decode_query(self, latent, xyt):
        """⚠ RESTATES TransolverBaseline.decode_query -- see the module docstring.

        The only difference is the two marked lines. Everything else is copied so
        that `use_local=False` reproduces the baseline exactly.
        """
        squeeze = xyt.dim() == 2
        if squeeze:
            xyt = xyt.unsqueeze(0)
        if xyt.shape[0] != 1:
            raise NotImplementedError(
                "LGOTransolver resolves one case per forward pass (the local "
                f"branch does not batch over rooms); got batch of {xyt.shape[0]}.")

        t = self.transolver
        pc = latent
        b_tok = t._tokens(pc[:, :, :3], pc[:, :, 3:], is_query=False)
        q_extra = torch.zeros(xyt.shape[0], xyt.shape[1], t.pc_channels - 3,
                              device=xyt.device, dtype=xyt.dtype)
        q_tok = t._tokens(xyt, q_extra, is_query=True)

        n_q = q_tok.shape[1]
        fx = t.preprocess(torch.cat([b_tok, q_tok], dim=1)) + t.placeholder

        # ---- the local branch, and the ONLY departure from the baseline ----
        if self.use_local:
            # Metres once, in float64, through the INDEX's own arithmetic. See
            # BoundaryLookup._metres -- doing this on the GPU in float32 is the
            # source of a velocity leak through solid walls.
            xyz_m = self._metres(xyt[0], self._active)
            xq_grad = None
            if self.grad_wrt_query and xyt.requires_grad:
                sc = torch.as_tensor(self._active.coord_scale, dtype=xyt.dtype,
                                     device=xyt.device)
                mn = torch.as_tensor(self._active.coord_min, dtype=xyt.dtype,
                                     device=xyt.device)
                xq_grad = xyt[0] * sc + mn
            h_local = self._local_stream(xyz_m, xyt.device, self.n_local_layers,
                                         dtype=fx.dtype, xq_grad=xq_grad)
            # Seed-style: condition the QUERY rows of the trunk input. Boundary
            # rows are left alone -- they are the geometry the branch gathers
            # FROM, and adding a query-conditioned vector to them would leak the
            # query into the geometry encoding.
            fx = torch.cat([fx[:, :-n_q], fx[:, -n_q:] + h_local.unsqueeze(0)], dim=1)
        # --------------------------------------------------------------------

        for blk in t.blocks:
            fx = blk(fx)
        out = fx[:, -n_q:]
        return out.squeeze(0) if squeeze else out

    def forward(self, xyt, pc, sample_ids=None):
        return self.decode_query(self.encode_geometry(pc, sample_ids), xyt)


# ---------------------------------------------------------------------- #
#  ONE construction path, used by BOTH train.py and evaluate.py
# ---------------------------------------------------------------------- #
def build_lgo_transolver(a, ds, device, verbose=True):
    """Assemble the arm from a flat args dict.

    `a` is `vars(args)` during training and the run's own `args.json` at
    evaluation: the two callers MUST build the same object from the same keys or
    a checkpoint is silently rebuilt as a different model.
    """
    from baselines.models import build_baseline

    cfg = a["cfg"] if isinstance(a.get("cfg"), dict) else json.loads(a.get("cfg") or "{}")
    import train_thermo
    # Built through the SAME build_baseline call `--model transolver` uses, with
    # the same cfg, so the Transolver half is byte-identical to the baseline.
    # ⚠ The published Shape-Net Car config is C=256, L=8, heads=8, M=32 and
    # mlp_ratio=2 (their main.py). Our builder defaults mlp_ratio to 4, so the
    # sweep/launcher must pass --cfg '{"mlp_ratio": 2}' for BOTH arms or the pair
    # is matched to each other but not to the published configuration.
    trans = build_baseline("transolver", cfg, pc_channels=ds.pc_channels,
                           out_channels=train_thermo.OUT_CHANNELS,
                           min_length_norm=a.get("min_length_norm", 0.05))

    far_voxels = (a.get("far_voxels") or [0.5]) if a.get("k_solid_far") else None
    groups = make_groups(far_voxels)
    k = dict({"supply": a["k_supply"], "return": a["k_return"],
              "solid": a["k_solid"], "leak": a["k_leak"]},
             **{g: a["k_solid_far"] for g in groups if g.startswith("solid_far")})

    model = LGOTransolver(
        trans, ds.coord_min, ds.coord_scale, ds.target_mean, ds.target_std,
        k=k, groups=groups, far_voxels=far_voxels,
        n_heads=a.get("local_heads", 8),
        n_local_layers=a.get("n_local_layers", 1), n_rbf=a.get("n_rbf", 12),
        local_hidden=a.get("local_hidden", 128), rbf_max=a.get("rbf_max", 4.0),
        soft_knn=a.get("soft_knn", 0),
        chunk_knn=a.get("chunk_knn", 200_000)).to(device)

    if verbose:
        n_loc = sum(p.numel() for p in model.local.parameters())
        n_tot = sum(p.numel() for p in model.parameters())
        print(f"[lgo_transolver] arch=transolver | {n_tot:,} params "
              f"({n_loc:,} local, {n_tot - n_loc:,} Transolver) | k={model.k} | "
              f"soft_knn={model.local.soft_knn} | inject=seed", flush=True)
    return model
