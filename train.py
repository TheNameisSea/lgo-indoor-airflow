#!/usr/bin/env python
"""Train LGO (and, unchanged, every baseline) under the prior project's protocol.

This script deliberately does NOT reimplement `train_thermo.main()`. It extends
that function's model factory and re-runs it, so the dataset construction, the
loss, the optimiser and schedule, `args.json`, the validation cadence and the
checkpoint-selection rule stay identical to the baseline **by construction**.

Arms (`--model`):

    lgo_gdon  (alias lgo)        local branch + Geom-DeepONet decoder
                                 (model/lgo.py). Decoder settings go in --cfg,
                                 e.g. '{"inject":"seed"}'. `deeponet_lgo` is an
                                 older alias.
    lgo_ginot, lgo_ginot_hbc     local branch + GINOT trunk (model/lgo_ginot.py)
    lgo_ginot_local,             local path only, no room latent -- the
    lgo_ginot_local_hbc          ablation of the global path
    lgo_gino                     local branch + GINO (model/lgo_gino.py)
    lgo_transolver               local branch + Transolver (model/lgo_transolver.py)

    ginot, ginot_se,             GINOT hosts (legacy/, unmodified)
    ginot_hbc, ginot_se_hbc
    transolver, gino,            baselines, configured via --cfg
    gno, deeponet

Each hybrid with its local branch off is bit-identical to its host.

Runs whose args.json says `"model": "lgo"` without an `lgo_arch` key were
trained with the GINOT-trunk architecture; `evaluate.py` resolves them through
`model.lgo.is_legacy_ginot_run()`. Do not bypass it.

    python -u train.py --model lgo_gdon --out_dir runs/lgo_gdon_s0 \
        --train_dir splits_cfd_gap/train --val_dir splits_cfd_gap/val \
        --lr 1e-3 --epochs 450 --blind_return_velocity

Do not pass --stats_path: normalisation is computed from the run's own training
set. Call the environment's python directly (`conda run` buffers stdout for
hours) and background jobs should be waited on by file, not `pgrep`.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "legacy"))

import train_thermo                                    # noqa: E402  (legacy/)
from data.knn_cache import make_groups                  # noqa: E402
from model.boundary_lookup import BoundaryLookup       # noqa: E402
from model.lgo import ARCH, ARCH_KEY, LGO_FAMILY       # noqa: E402
from model.lgo_gino import LGO_GINO_FAMILY             # noqa: E402
from model.lgo_transolver import LGO_TRANSOLVER_FAMILY   # noqa: E402
from model.lgo_ginot import LGOGinot                   # noqa: E402

# The superseded architecture. `lgo_ginot*` rather than the old `lgo*` names so
# that ONE model name never denotes two architectures at the same time -- the
# rename is the whole reason `is_legacy_ginot_run()` has to exist for old
# args.json files, and repeating that ambiguity going forward would be a choice.
GINOT_LGO_FAMILY = ("lgo_ginot", "lgo_ginot_hbc",
                    "lgo_ginot_local", "lgo_ginot_local_hbc")
_CFG = {}


def add_lgo_args(p):
    g = p.add_argument_group("LGO (local-global operator)")
    g.add_argument("--k_supply", type=int, default=32,
                   help="nearest supply-vent points each query attends to. These "
                        "are ~800 points per case carrying the entire driving "
                        "boundary condition, so they are gathered separately and "
                        "can never be crowded out by nearby floor.")
    g.add_argument("--k_return", type=int, default=16)
    g.add_argument("--k_solid", type=int, default=32)
    g.add_argument("--k_leak", type=int, default=8)
    g.add_argument("--strat_wall_dist", type=float, default=0.0,
                   help="RE-STRATIFY interior points by DISTANCE TO THE BOUNDARY "
                        "instead of by speed (0 = off, the default, which leaves "
                        "every indoor run byte-identical). The parent splits "
                        "interior points into 'stream' and 'background' at a "
                        "hard-coded 0.15 m/s, an indoor-air constant: on "
                        "ShapeNet-Car the median speed is 19.95 m/s so 97.6% of "
                        "points land in 'stream', the background pool collapses "
                        "to 704 points, and the 60/40 sampling degenerates to "
                        "uniform over a volume that is 79% freestream. Set this "
                        "to a distance in the dataset's own length units and the "
                        "near-boundary band becomes 'stream', which is where the "
                        "case-specific signal actually lives.")
    g.add_argument("--k_solid_far", type=int, default=0,
                   help="coarse room-scale solid group. Retired: leave at 0 "
                        "(the default), which keeps the 4-group model.")
    g.add_argument("--far_voxels", type=float, nargs="+", default=None,
                   help="voxel sizes [m] for the coarse room-scale shells, "
                        "coarsest last. One value reproduces the single-shell "
                        "model; several give a HIERARCHY (e.g. 0.4 0.8 1.6), so "
                        "a query sees fine structure nearby and the room's "
                        "outline further out at once -- which is what a "
                        "recirculation cell spanning several metres needs. "
                        "Defaults to [0.5] when --k_solid_far > 0.")
    g.add_argument("--n_local_layers", type=int, default=1)
    g.add_argument("--local_heads", type=int, default=8)
    g.add_argument("--local_hidden", type=int, default=128)
    g.add_argument("--n_rbf", type=int, default=12,
                   help="Gaussian radial basis functions on ||dx||, log-spaced. "
                        "Radial conditioning was the prior project's only "
                        "intervention that ever moved the occupied zone.")
    g.add_argument("--rbf_max", type=float, default=4.0,
                   help="outer radius [m] of the basis. This is what keeps the "
                        "local branch local, so room-scale structure has nowhere "
                        "to live except the global path.")
    g.add_argument("--local_query", default="learned", choices=("learned", "global"),
                   help="'learned' keeps the local TERM exactly equivariant "
                        "(two-stream, added to the global path). 'global' "
                        "interleaves and is measurably not equivariant - "
                        "ablation D6, not a default.")
    g.add_argument("--soft_knn", type=int, default=0,
                   help="extra neighbours gathered per group ONLY to be faded "
                        "out by a raised-cosine window. 0 = the original hard "
                        "k-NN gather, which makes the prediction discontinuous "
                        "in the query: a neighbour swap removes its token "
                        "abruptly, producing measured jumps of ~5e-3 m/s and a "
                        "central difference that DIVERGES as h shrinks. Those "
                        "jumps contribute spurious divergence that more data "
                        "does not remove. 8 is a reasonable starting value.")
    g.add_argument("--global_query", default="local",
                   choices=("learned", "local"),
                   help="what queries the room latent. 'local' (default) lets "
                        "the local stream query it, so the query varies in "
                        "space but is built only from boundary offsets and no "
                        "absolute coordinate enters. 'learned' is one "
                        "shared query vector, so the global path becomes a "
                        "room-level context with no position dependence. "
                        "The absolute-position ('fourier') query was removed: "
                        "it is the mechanism behind layout memorisation and no "
                        "run ever used it.")
    g.add_argument("--normals_k", type=int, default=16)
    g.add_argument("--no_normals", action="store_true",
                   help="ablation D2: drop the estimated surface normals")
    g.add_argument("--chunk_knn", type=int, default=200_000)
    g.add_argument("--augment", nargs="*", default=None,
                   choices=("x", "y", "xy"),
                   help="reflection augmentation, TRAINING SET ONLY. Navier-Stokes "
                        "is invariant under horizontal reflection, so each mirror "
                        "is a physically valid room. Every vent layout in this "
                        "dataset is centred on the same point, so a mirror maps a "
                        "layout ONTO ITSELF -- verified, so no layout can migrate "
                        "between splits and this cannot leak. What it adds is room "
                        "ARRANGEMENTS (up to 4x), which is what far-field "
                        "recirculation is starved of. Note each mirror is a extra "
                        "room per epoch, so scale --epochs down to keep the "
                        "optimiser-step count comparable.")
    g.add_argument("--select_on", default="joint",
                   choices=("joint", "zone_balanced", "occupied",
                            "ashrae", "comfort_balanced"),
                   help="checkpoint criterion. 'joint' (default) is the global "
                        "Taylor score, which is variance-weighted and therefore "
                        "~94% a near-jet score. 'zone_balanced' weights the three "
                        "distance-to-vent zones equally: measured, occupied-zone "
                        "R2 varies by 0.027 across 100 epochs while the global "
                        "score moves 0.0002, so the default selects almost at "
                        "random from that plateau. 'comfort_balanced' splits the "
                        "vote between the near jet and the ASHRAE 55 occupied "
                        "zone (0.10-1.35/1.80 m, set back from walls and "
                        "furniture) -- the region a comfort claim rests on. "
                        "'ashrae' selects on that band alone.")
    g.add_argument("--blind_return_velocity", action="store_true",
                   help="Zero the SOLVED return-vent u/v/w in the encoder's "
                        "point cloud, for train AND val. Returns are pressure "
                        "OUTLETS, so their velocity is an output "
                        "of the same solve the model reproduces, yet it reaches "
                        "the encoder on ~46% of the cloud while pressure, "
                        "off-supply temperature and leak velocity are all "
                        "already blinded. Vent POSITIONS and the CLS_OUTLET "
                        "one-hot are untouched, so the layout signal -- the only "
                        "thing that varies across cases -- is fully preserved. "
                        "Supply u/v/w/T are prescribed constants here, so this "
                        "removes the only solution-derived channel without "
                        "costing the BC-generalization roadmap anything.")
    # STAMPED INTO args.json ON EVERY RUN, and that is its entire job. `--model
    # lgo` denoted the GINOT-trunk architecture before 2026-09-11 and denotes
    # the DeepONet one after, so the model NAME alone can no longer identify an
    # architecture. Its presence is what lets evaluate.py tell a new run from
    # the twelve older ones. Do not remove it, and do not make it conditional.
    g.add_argument("--lgo_arch", default=ARCH,
                   choices=("deeponet", "ginot", "gino"),
                   help=argparse.SUPPRESS)
    return p


def _far_voxels():
    """Voxel sizes for the far shells, or None when the far field is off."""
    if not _CFG.get("k_solid_far"):
        return None
    return _CFG.get("far_voxels") or [0.5]


def _lgo_groups():
    return make_groups(_far_voxels())


def build_with_lgo(args, ds, device):
    """train_thermo.build(), extended with LGO and the superseded GINOT arm."""
    if args.model in LGO_FAMILY:
        from model.lgo import build_lgo
        # The SAME dict shape evaluate.py will later read out of args.json, so
        # there is one construction path and it cannot drift between the two.
        return build_lgo({**vars(args), **_CFG}, ds, device)
    if args.model in LGO_GINO_FAMILY:
        from model.lgo_gino import build_lgo_gino
        return build_lgo_gino({**vars(args), **_CFG}, ds, device)
    if args.model in LGO_TRANSOLVER_FAMILY:
        from model.lgo_transolver import build_lgo_transolver
        return build_lgo_transolver({**vars(args), **_CFG}, ds, device)
    if args.model not in GINOT_LGO_FAMILY:
        return _orig_build(args, ds, device)

    trunk = train_thermo.build(
        argparse.Namespace(**{**vars(args), "model": "ginot"}), ds, device)
    model = LGOGinot(
        trunk, ds.coord_min, ds.coord_scale, ds.target_mean, ds.target_std,
        d_model=args.embed_dim, n_heads=_CFG["local_heads"],
        k=dict({"supply": _CFG["k_supply"], "return": _CFG["k_return"],
                "solid": _CFG["k_solid"], "leak": _CFG["k_leak"]},
               **{g: _CFG["k_solid_far"] for g in _lgo_groups() if g.startswith("solid_far")}),
        groups=_lgo_groups(), far_voxels=_far_voxels(),
        n_local_layers=_CFG["n_local_layers"], n_rbf=_CFG["n_rbf"],
        local_hidden=_CFG["local_hidden"], rbf_max=_CFG["rbf_max"],
        local_query=_CFG["local_query"], global_query=_CFG["global_query"],
        soft_knn=_CFG["soft_knn"],
        use_local=True,
        use_global="local" not in args.model,
        use_hbc=args.model.endswith("hbc"), ell_wall=args.ell_wall,
        chunk_knn=_CFG["chunk_knn"]).to(device)

    n_loc = sum(p.numel() for p in model.local.parameters())
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[{args.model}] {n_tot:,} params ({n_loc:,} local, {n_tot - n_loc:,} "
          f"GINOT) | k={model.k} | local_query={_CFG['local_query']} | "
          f"global={model.use_global} hbc={model.use_hbc}", flush=True)
    return model


def main():
    p = argparse.ArgumentParser(add_help=False)
    add_lgo_args(p)
    extra, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    _CFG.update(vars(extra))

    # `choices` is built from these module globals inside parse_args(), so
    # widening them here is what lets --model lgo through.
    train_thermo.GINOT_FAMILY = (tuple(train_thermo.GINOT_FAMILY) + LGO_FAMILY
                                 + GINOT_LGO_FAMILY + LGO_GINO_FAMILY
                                 + LGO_TRANSOLVER_FAMILY)

    # Re-attach the LGO flags to the parsed namespace so main() records them in
    # args.json: without this the run is unreproducible from its own output.
    orig_parse = train_thermo.parse_args

    def parse_with_extra():
        a = orig_parse()
        for k, v in vars(extra).items():
            setattr(a, k, v)
        # DERIVE the arch marker from the arm rather than trusting the flag's
        # default, so `lgo_ginot` does not record `lgo_arch="deeponet"`. Dispatch
        # does not depend on the VALUE (evaluate.py routes on the name, and on
        # this key's mere presence for pre-rename runs), but a run's args.json is
        # the permanent record of what it was -- storing the wrong architecture
        # there is how the next person gets misled.
        if a.model in GINOT_LGO_FAMILY:
            arch = "ginot"
        elif a.model in LGO_GINO_FAMILY:
            arch = "gino"
        elif a.model in LGO_TRANSOLVER_FAMILY:
            arch = "transolver"
        else:
            arch = ARCH
        setattr(a, ARCH_KEY, arch)
        return a

    train_thermo.parse_args = parse_with_extra
    train_thermo.build = build_with_lgo

    # register_cases() takes the normals switch, which main() cannot know about.
    # Patch it on BoundaryLookup, not on LGO: both LGO and the DeepONet hybrid
    # inherit it from there, and patching the subclass would silently leave the
    # hybrid registering its cases WITHOUT normals -- a difference that trains
    # fine and warns nowhere, since the feature vector keeps its width and the
    # normal channels simply arrive as zeros.
    orig_register = BoundaryLookup.register_cases

    def register(self, dataset, verbose=True, **kw):
        return orig_register(self, dataset, verbose=verbose,
                             normals_k=_CFG["normals_k"],
                             with_normals=not _CFG["no_normals"], **kw)

    BoundaryLookup.register_cases = register

    # Zone-aware selection, by swapping the trainer class main() instantiates.
    # ThermoTrainer's whole selection policy is one line (val_combined = -score),
    # so a subclass that recomputes that key is the entire change; legacy/ is
    # untouched, and select_on='joint' leaves behaviour byte-identical.
    # Reflection augmentation. main() hardcodes augment=[] for the TRAIN dataset
    # and does not pass the kwarg at all for val, so the presence of the keyword
    # is exactly the train/val discriminator -- and val must never be augmented.
    orig_ds = train_thermo.ThermoDataset

    if _CFG.get("blind_return_velocity"):
        # Same operation evaluate.py's probe performs, but applied to BOTH the
        # training and validation datasets so there is no train/test mismatch.
        # Done here rather than in legacy/pi_ginot/dataset.py because legacy/ is
        # vendored and must stay byte-identical.
        _orig_ds_blind = orig_ds

        def _ds_blind(*a, **kw):
            d = _orig_ds_blind(*a, **kw)
            from thermo.dataset import PC_CLS
            n_z = n_tot = 0
            for case in d.cases:
                pc = case.get("pc_full")
                if pc is None or pc.shape[1] < 11:
                    continue
                m = pc[:, PC_CLS].argmax(dim=-1) == 3      # CLS_OUTLET
                pc[m, 8:11] = 0.0                          # u, v, w
                n_z += int(m.sum()); n_tot += len(pc)
            print(f"[blind] return-vent u/v/w zeroed on {n_z:,} of {n_tot:,} "
                  f"point-cloud points ({n_z / max(n_tot, 1):.1%})", flush=True)
            return d

        orig_ds = _ds_blind
        train_thermo.ThermoDataset = _ds_blind

    if _CFG.get("strat_wall_dist", 0.0) > 0:
        # A pure RE-PARTITION of pools the parent already built and normalised --
        # not a reimplementation of _process_single_room. Nothing is recomputed
        # and nothing is resampled, so legacy/ stays byte-identical and this
        # cannot drift from the parent's pooling. Points are simply relabelled:
        # near-boundary -> 'stream' (which __getitem__ samples at 60%),
        # everything else -> 'background'.
        #
        # ⚠ SIZE n_case_pool SO NOTHING IS DISCARDED, or this partition is BIASED.
        # The pools are subsampled BEFORE this runs, under the speed rule, and the
        # two sides are capped differently: stream at int(n_case_pool * 0.4),
        # background at whatever remains. On ShapeNet-Car the speed rule puts
        # 28,794 of 29,498 points in 'stream', so at n_case_pool=30000 all 704
        # background points survive while stream is cut to 12,000 -- and since the
        # slowest points are exactly the near-wall ones, the near band comes out
        # OVER-represented: measured 7.83% against the true 4.7%.
        # The stream cap must clear the interior count: n_case_pool >= N / 0.4.
        # For ShapeNet-Car (29,498 pts/case) use --n_case_pool 75000, which is
        # lossless -- verified 29,498/29,498 retained, near band then measures
        # 4.99% at 0.05 and 21.77% at 0.20, matching the true band fractions.
        _orig_ds_strat = orig_ds
        _thresh = float(_CFG["strat_wall_dist"])

        def _ds_strat(*a, **kw):
            import numpy as _np
            import torch as _torch
            from scipy.spatial import cKDTree as _KD
            d = _orig_ds_strat(*a, **kw)
            cmin, cscale = d.coord_min, d.coord_scale
            n_near = n_tot = 0
            skipped = 0
            for case in d.cases:
                wx = case.get("pool_wall_xyz")
                if wx is None or len(wx) == 0:
                    skipped += 1
                    continue
                xyz = _torch.cat([case["pool_stream_xyz"], case["pool_bg_xyz"]], 0)
                tgt = _torch.cat([case["pool_stream_tgt"], case["pool_bg_tgt"]], 0)
                if len(xyz) == 0:
                    skipped += 1
                    continue
                # Denormalise: coord_scale is PER-AXIS, so normalised space is
                # anisotropic and a distance taken there is not a distance.
                raw = (xyz * cscale + cmin).numpy()
                wraw = (wx * cscale + cmin).numpy()
                dist = _KD(wraw).query(raw)[0]
                near = _torch.from_numpy(_np.asarray(dist < _thresh))
                if int(near.sum()) == 0 or int((~near).sum()) == 0:
                    # One side empty would starve __getitem__'s 60/40 draw.
                    skipped += 1
                    continue
                case["pool_stream_xyz"], case["pool_stream_tgt"] = xyz[near], tgt[near]
                case["pool_bg_xyz"], case["pool_bg_tgt"] = xyz[~near], tgt[~near]
                n_near += int(near.sum()); n_tot += len(near)
            print(f"[strat] re-stratified by wall distance < {_thresh:g}: "
                  f"{n_near:,} of {n_tot:,} interior points are near-boundary "
                  f"({n_near / max(n_tot, 1):.1%}) and now carry the 60% draw"
                  + (f"  [{skipped} case(s) left on the speed rule]" if skipped else ""),
                  flush=True)
            return d

        orig_ds = _ds_strat
        train_thermo.ThermoDataset = _ds_strat

    if _CFG.get("augment"):
        aug = list(_CFG["augment"])

        def _ds(*a, **kw):
            if "augment" in kw:                       # the training set
                kw["augment"] = aug
                print(f"[augment] training rooms mirrored on {aug} "
                      f"(val/test untouched)", flush=True)
            return orig_ds(*a, **kw)

        train_thermo.ThermoDataset = _ds

    orig_trainer = train_thermo.ThermoTrainer
    if _CFG["select_on"] != "joint":
        from thermo_zone import ZoneThermoTrainer
        sel = _CFG["select_on"]

        class _Sel(ZoneThermoTrainer):
            def __init__(self, *a, **kw):
                super().__init__(*a, select_on=sel, **kw)

        train_thermo.ThermoTrainer = _Sel
    try:
        train_thermo.main()
    finally:
        train_thermo.parse_args = orig_parse
        train_thermo.ThermoTrainer = orig_trainer
        train_thermo.ThermoDataset = orig_ds
        BoundaryLookup.register_cases = orig_register


_orig_build = train_thermo.build

if __name__ == "__main__":
    main()
