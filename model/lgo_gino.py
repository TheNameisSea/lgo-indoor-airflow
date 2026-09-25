"""LGO-GINO -- the local branch on a GINO host.

    local  : query -> type-stratified boundary neighbours, seen only as offsets
             in metres -> attention.            Near-field, equivariant.
    global : boundary cloud -> input GNO -> FNO on a regular latent grid ->
             output GNO evaluated at the query.  Room-scale.
    out    : projection( h_gno(x) + h_local(x) )

GINO is a graph-kernel integral onto a grid, a spectral convolution on that
grid, and a graph integral back out -- no trunk decoder. The local stream enters
through `GINOBaseline.decode_query(..., h_delta=...)`: it is added to the
per-query state from the output GNO, before the shared `projection` head. With
`h_delta=None` the baseline is unchanged, so this model with the branch off is
exactly `--model gino`, and the branch is zero-initialised.

`projection` is a two-layer channel MLP, so the local stream passes through a
nonlinearity before the output (unlike LGO-GDON, whose head is linear).
"""

import json

import torch
import torch.nn as nn

from data.knn_cache import BASE_GROUPS, make_groups
from model.boundary_lookup import BoundaryLookup
from model.local_branch import LocalBranch

# Every name that means THIS architecture, defined here and imported by both
# train.py and evaluate.py so the trainer and the evaluator cannot disagree --
# a name only one of them recognises silently builds the wrong architecture.
LGO_GINO_FAMILY = ("lgo_gino",)
ARCH_KEY = "lgo_arch"
ARCH = "gino"


class LGOGino(BoundaryLookup, nn.Module):
    """`GINOBaseline` + the LGO local branch, sharing the GINO head.

    Wraps the baseline rather than subclassing or editing it, exactly as
    `model/lgo.py` wraps `GeomDeepONet`: `--model gino` stays loadable and its
    numbers stay reproducible by the file that produced them, so the A/B carries
    no confound.
    """

    def __init__(self, gino, coord_min, coord_scale, target_mean, target_std,
                 k=None, n_heads=8, n_local_layers=1, n_rbf=12, local_hidden=128,
                 rbf_max=4.0, soft_knn=0, groups=BASE_GROUPS,
                 far_voxels=None, chunk_knn=200_000):
        super().__init__()
        self.gino = gino
        self.n_local_layers = n_local_layers
        self._init_lookup(k, groups=groups, far_voxels=far_voxels,
                          chunk_knn=chunk_knn)

        # The host's own width. Carrying a projection instead would add
        # parameters this arm does not need and would break the shared-head
        # equivalence the ablation depends on.
        d_model = gino.net.fno_hidden_channels
        self.local = LocalBranch(d_model, n_heads=n_heads, n_rbf=n_rbf,
                                 hidden=local_hidden, n_layers=n_local_layers,
                                 rbf_max=rbf_max, groups=self.groups,
                                 soft_knn=soft_knn)

        # Physical-unit constants, kept for parity with the other two arms so
        # the same downstream tooling (evaluate.py, physics_gradients.py) reads
        # them without special-casing this one.
        self.register_buffer("mu", target_mean[0:3].clone().float())
        self.register_buffer("sigma", target_std[0:3].clone().float())
        self.register_buffer("coord_min", coord_min.clone().float())
        self.register_buffer("coord_scale", coord_scale.clone().float())

        # evaluate.py gates register_cases on these; physics_gradients.py reads
        # grad_wrt_query. Declared rather than left to duck-typing, because a
        # missing attribute there fails by SKIPPING registration and then
        # raising deep inside decode_query.
        self.use_local = True
        self.use_hbc = False
        self.grad_wrt_query = False

    # ------------------------------------------------------------------ #
    #  Trunk interface -- matches model/lgo.py::LGO
    # ------------------------------------------------------------------ #
    def encode_geometry(self, pc, sample_ids=None):
        self._active = self._resolve(pc)
        return self.gino.encode_geometry(pc)

    def decode_query(self, latent, xyt):
        squeeze = xyt.dim() == 2
        if squeeze:
            xyt = xyt.unsqueeze(0)
        B = xyt.shape[0]
        if B != 1:
            raise NotImplementedError(
                "LGOGino resolves one case per forward pass (the local branch "
                f"does not batch over rooms); got batch of {B}.")

        # Metres once, in float64, through the INDEX's own arithmetic. See
        # BoundaryLookup._metres -- doing this on the GPU in float32 is the
        # 6e-5 m/s wall leak.
        xyz_m = self._metres(xyt[0], self._active)
        xq_grad = None
        if self.grad_wrt_query and xyt.requires_grad:
            sc = torch.as_tensor(self._active.coord_scale, dtype=xyt.dtype,
                                 device=xyt.device)
            mn = torch.as_tensor(self._active.coord_min, dtype=xyt.dtype,
                                 device=xyt.device)
            xq_grad = xyt[0] * sc + mn

        h_local = self._local_stream(xyz_m, xyt.device, self.n_local_layers,
                                     dtype=latent.dtype, xq_grad=xq_grad)
        out = self.gino.decode_query(latent, xyt, h_delta=h_local.unsqueeze(0))
        return out.squeeze(0) if squeeze else out

    def forward(self, xyt, pc, sample_ids=None):
        return self.decode_query(self.encode_geometry(pc, sample_ids), xyt)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Strip neuralop's `_metadata`, which here is nested under `gino.net`.

        `GINOBaseline.load_state_dict` already does this for the bare baseline,
        but wrapping moves the key to `gino.net._metadata` and nn.Module's
        default loader never reaches the baseline's override. Without this,
        every checkpoint this arm writes fails to load with
        "Unexpected key(s): gino.net._metadata" -- the same trap documented in
        `baselines/models/gino.py`, one level deeper.
        """
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.endswith("_metadata")}
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


# ---------------------------------------------------------------------- #
#  ONE construction path, used by BOTH train.py and evaluate.py
# ---------------------------------------------------------------------- #
def build_lgo_gino(a, ds, device, verbose=True):
    """Assemble the arm from a flat args dict.

    `a` is `vars(args)` during training and the run's own `args.json` at
    evaluation, which is why this takes a dict rather than a Namespace: the two
    callers MUST build the same object from the same keys, or a checkpoint is
    silently rebuilt as a different model.
    """
    from baselines.models import build_baseline

    cfg = a["cfg"] if isinstance(a.get("cfg"), dict) else json.loads(a.get("cfg") or "{}")
    import train_thermo
    # No min_length_norm: GINO has no Fourier query encoder. Everything else is
    # computed the way train_thermo.build computes it, so the GINO half is
    # byte-identical to `--model gino` with the same --cfg.
    gino = build_baseline("gino", cfg, pc_channels=ds.pc_channels,
                          out_channels=train_thermo.OUT_CHANNELS)

    far_voxels = (a.get("far_voxels") or [0.5]) if a.get("k_solid_far") else None
    groups = make_groups(far_voxels)
    k = dict({"supply": a["k_supply"], "return": a["k_return"],
              "solid": a["k_solid"], "leak": a["k_leak"]},
             **{g: a["k_solid_far"] for g in groups if g.startswith("solid_far")})

    model = LGOGino(
        gino, ds.coord_min, ds.coord_scale, ds.target_mean, ds.target_std,
        k=k, groups=groups, far_voxels=far_voxels,
        n_heads=a.get("local_heads", 8),
        n_local_layers=a.get("n_local_layers", 1), n_rbf=a.get("n_rbf", 12),
        local_hidden=a.get("local_hidden", 128), rbf_max=a.get("rbf_max", 4.0),
        soft_knn=a.get("soft_knn", 0),
        chunk_knn=a.get("chunk_knn", 200_000)).to(device)

    if verbose:
        n_loc = sum(p.numel() for p in model.local.parameters())
        n_tot = sum(p.numel() for p in model.parameters())
        print(f"[lgo_gino] arch=gino | {n_tot:,} params ({n_loc:,} local, "
              f"{n_tot - n_loc:,} GINO) | k={model.k} | "
              f"soft_knn={model.local.soft_knn} | "
              f"d_model={gino.net.fno_hidden_channels} "
              f"grid={tuple(gino.latent_queries.shape[:3])}", flush=True)
    return model
