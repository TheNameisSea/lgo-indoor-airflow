"""LGO-GDON -- the local branch on a Geom-DeepONet decoder.

    local  : query -> type-stratified boundary neighbours, seen ONLY as offsets
             in METRES -> attention.          Near-field, equivariant.
    global : boundary cloud -> a rank-`n_basis` modal decomposition whose
             coefficients are a function of the LAYOUT.   Room-scale.
    out    : trunk_out(h_global + h_local) . coeff(geometry)

The global half is a Geom-DeepONet decoder (`baselines/models/deeponet.py`,
wrapped not edited). `--model lgo` is a permanent alias. Older runs whose
`args.json` says `"model": "lgo"` without an `lgo_arch` key were trained with
the GINOT-trunk architecture and resolve to `model/lgo_ginot.py` -- see
`evaluate.py`.

WHY THIS DECODER. A GINOT trunk and a plain Geom-DeepONet fail in disjoint
parts of the room: the DeepONet's layout-indexed modal decomposition captures
room-scale recirculation, but its trunk reads an ABSOLUTE coordinate and has no
channel for relative geometry near the jets. The local branch supplies exactly
that. Measurements are in the paper.

TWO INJECTION POINTS, and only the first is the default.

  `basis` (default) -- the local stream is added to the trunk's last hidden
      state, before `trunk_out`. Since `trunk_out` is linear this IS basis
      augmentation, `trunk_out(h + h_local) = trunk_out(h) + W_out @ h_local`,
      with the output head SHARED exactly the way `lgo_ginot` shares
      `output_proj`. No parameters beyond the local branch.
  `seed` -- the local stream is added to the trunk INPUT instead, so the basis
      functions themselves become geometry-relative. This is the analogue of
      `lgo_ginot`'s `--global_query local`, the setting that won there. Opt-in,
      because it puts the whole FiLM stack downstream of the local branch and a
      result then cannot be attributed to the local term alone. UNMEASURED.

Both are exactly zero at step 0 -- `LocalCrossAttention.out` is zero-initialised
-- so this model is BIT-IDENTICAL to the plain `deeponet` baseline at
initialisation and every gain has to be earned. `tests/test_lgo.py` checks that
rather than trusting it.
"""

import json

import torch
import torch.nn as nn

from data.knn_cache import BASE_GROUPS, make_groups
from model.boundary_lookup import BoundaryLookup
from model.local_branch import LocalBranch

INJECT = ("basis", "seed")
# Every name that means THIS architecture. Defined here and imported by both
# train.py and evaluate.py, so the trainer and the evaluator cannot disagree --
# a name only one of them recognises silently builds the wrong architecture.
#
# `deeponet_lgo` is the name this arm was BORN under (2026-09-11, before it was
# promoted). `runs/short_deeponet_lgo` -- the run that is the entire evidence
# for the promotion -- carries it, so the alias is load-bearing, not courtesy.
# `lgo_gdon` is the CANONICAL name as of 2026-09-19: it matches the prose name
# (LGO-GDON) and cannot be misread as "the LGO model" in general, which the bare
# `lgo` invited -- there are three LGO arms and this is only one of them.
#
# `lgo` IS A PERMANENT ALIAS, NOT A DEPRECATION. 32 run directories record
# `"model": "lgo"` in args.json, including every run behind ledger tables D, E,
# F, H, I and J. Dropping the name would make all of them unloadable, and
# rewriting their args.json is forbidden for a separate reason (an old run
# that says `lgo` and lacks `lgo_arch` means the GINOT trunk, and adding the key
# by hand silently re-points it at this architecture). New runs record
# `lgo_gdon`; old runs keep loading exactly as they did.
LGO_FAMILY = ("lgo_gdon", "lgo", "deeponet_lgo")
# Runs written BEFORE the rename say `"model": "lgo"` and mean the GINOT trunk.
# They are distinguished by the ABSENCE of `lgo_arch` in args.json, which
# train.py has recorded on every run since. Verified against all 12 of them.
ARCH_KEY = "lgo_arch"
ARCH = "deeponet"


class LGO(BoundaryLookup, nn.Module):
    """`GeomDeepONet` + the LGO local branch, sharing the DeepONet head.

    We WRAP the baseline rather than subclass or edit it, the same way `LGO`
    wraps a legacy GINOT trunk: `runs/short_deeponet` stays loadable and its
    numbers stay reproducible by the same file that produced them, so
    "hybrid minus local branch" is literally the baseline and the A/B carries no
    confound.
    """

    def __init__(self, deeponet, coord_min, coord_scale, target_mean, target_std,
                 k=None, n_heads=8, n_local_layers=1, n_rbf=12, local_hidden=128,
                 rbf_max=4.0, soft_knn=0, inject="basis", groups=BASE_GROUPS,
                 far_voxels=None, chunk_knn=200_000):
        super().__init__()
        if inject not in INJECT:
            raise ValueError(f"inject must be one of {INJECT}, got {inject!r}")
        self.deeponet = deeponet
        self.inject = inject
        self.n_local_layers = n_local_layers
        self._init_lookup(k, groups=groups, far_voxels=far_voxels,
                          chunk_knn=chunk_knn)

        # The local stream is added to a trunk hidden state, so it must be that
        # width. Carrying a projection instead would add parameters the arm does
        # not need and would break the "shared head" equivalence above.
        d_model = deeponet.trunk_out.in_features
        self.local = LocalBranch(d_model, n_heads=n_heads, n_rbf=n_rbf,
                                 hidden=local_hidden, n_layers=n_local_layers,
                                 rbf_max=rbf_max, groups=self.groups,
                                 soft_knn=soft_knn)

        # Physical-unit constants, kept for parity with LGO so the same
        # downstream tooling (evaluate.py, physics_gradients.py) can read them.
        self.register_buffer("mu", target_mean[0:3].clone().float())
        self.register_buffer("sigma", target_std[0:3].clone().float())
        self.register_buffer("coord_min", coord_min.clone().float())
        self.register_buffer("coord_scale", coord_scale.clone().float())

        # `evaluate.py` gates register_cases on these two, and `physics_
        # physics_gradients.py reads grad_wrt_query. Declared explicitly rather than
        # left to duck-typing: a missing attribute there fails by silently
        # SKIPPING registration, which then raises deep inside decode_query.
        self.use_local = True
        self.use_hbc = False
        self.grad_wrt_query = False

    # ------------------------------------------------------------------ #
    #  Trunk interface -- matches legacy/pi_ginot/model.py::Trunk
    # ------------------------------------------------------------------ #
    def encode_geometry(self, pc, sample_ids=None):
        self._active = self._resolve(pc)
        return self.deeponet.encode_geometry(pc)

    def decode_query(self, latent, xyt):
        squeeze = xyt.dim() == 2
        if squeeze:
            xyt = xyt.unsqueeze(0)
        B = xyt.shape[0]
        if B != 1:
            raise NotImplementedError(
                "LGO resolves one case per forward pass (the local "
                f"branch does not batch over rooms); got batch of {B}.")

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
                                     dtype=latent[0].dtype, xq_grad=xq_grad)
        kw = {("h_delta" if self.inject == "basis" else "seed_delta"):
              h_local.unsqueeze(0)}
        out = self.deeponet.decode_query(latent, xyt, **kw)
        return out.squeeze(0) if squeeze else out

    def forward(self, xyt, pc, sample_ids=None):
        return self.decode_query(self.encode_geometry(pc, sample_ids), xyt)


# ---------------------------------------------------------------------- #
#  ONE construction path, used by BOTH train.py and evaluate.py
# ---------------------------------------------------------------------- #
def build_lgo(a, ds, device, verbose=True):
    """Assemble the arm from a flat args dict.

    `a` is `vars(args)` during training and the run's own `args.json` at
    evaluation, which is why this takes a dict rather than a Namespace: the two
    callers MUST build the same object from the same keys. The paper -- "a
    flag that changes behaviour but not parameter NAMES loads silently into the
    wrong architecture" -- is what a second construction path would reintroduce,
    and `inject` is exactly such a flag (both settings have identical
    state_dicts).
    """
    from baselines.models import build_baseline

    cfg = a["cfg"] if isinstance(a.get("cfg"), dict) else json.loads(a.get("cfg") or "{}")
    # OUT_CHANNELS and min_length_norm are computed the way train_thermo.build
    # computes them, so the DeepONet half is byte-identical to `--model deeponet`.
    import train_thermo
    min_length_norm = a["min_length_m"] / ds.coord_scale.max().item()
    deeponet = build_baseline("deeponet", cfg, pc_channels=ds.pc_channels,
                              out_channels=train_thermo.OUT_CHANNELS,
                              min_length_norm=min_length_norm)

    far_voxels = (a.get("far_voxels") or [0.5]) if a.get("k_solid_far") else None
    groups = make_groups(far_voxels)
    k = dict({"supply": a["k_supply"], "return": a["k_return"],
              "solid": a["k_solid"], "leak": a["k_leak"]},
             **{g: a["k_solid_far"] for g in groups if g.startswith("solid_far")})

    model = LGO(
        deeponet, ds.coord_min, ds.coord_scale, ds.target_mean, ds.target_std,
        k=k, groups=groups, far_voxels=far_voxels,
        n_heads=a.get("local_heads", 8),
        n_local_layers=a.get("n_local_layers", 1), n_rbf=a.get("n_rbf", 12),
        local_hidden=a.get("local_hidden", 128), rbf_max=a.get("rbf_max", 4.0),
        soft_knn=a.get("soft_knn", 0),
        # `inject` travels in --cfg with the rest of the DeepONet knobs, so it
        # lands in args.json and the run is rebuildable from its own output.
        inject=cfg.get("inject", "basis"),
        chunk_knn=a.get("chunk_knn", 200_000)).to(device)

    if verbose:
        n_loc = sum(p.numel() for p in model.local.parameters())
        n_tot = sum(p.numel() for p in model.parameters())
        # PRINT `inject`. It changes behaviour but not parameter names, which is
        # precisely the class of flag that cost three runs on 2026-09-06.
        print(f"[lgo] arch=deeponet | {n_tot:,} params ({n_loc:,} local, "
              f"{n_tot - n_loc:,} DeepONet) | inject={model.inject} | "
              f"k={model.k} | soft_knn={model.local.soft_knn} | "
              f"n_basis={deeponet.n_basis} trunk_w={deeponet.trunk_out.in_features} "
              f"film={deeponet.film}", flush=True)
    return model


def is_legacy_ginot_run(a):
    """True if this `args.json` predates the 2026-09-11 rename.

    `--model lgo` meant the GINOT-trunk architecture until that date, and twelve
    committed runs say exactly that. They are told apart by the ABSENCE of
    `lgo_arch`, which train.py has written on every run since. Anything that
    rebuilds a model from `args.json` MUST go through this -- building the wrong
    one would silently rescore `gap143`, `gap65`
    and both gap-0.58 arms as a different model.
    """
    name = a.get("model", "")
    # Both names were born AFTER the split and have never meant GINOT. The guard
    # matters for `lgo_gdon` specifically: it starts with "lgo", so without this
    # line an `lgo_gdon` run missing `lgo_arch` would be rebuilt as a GINOT trunk
    # -- the silent-wrong-architecture failure this function exists to prevent.
    if name in ("deeponet_lgo", "lgo_gdon"):
        return False
    return name.startswith("lgo") and ARCH_KEY not in a
