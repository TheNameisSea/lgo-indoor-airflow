"""Baseline model factory.

Every model exposes the GINOT interface — `encode_geometry(pc)` and
`decode_query(latent, xyz)` — so `baselines/common/{eval,train_loop}.py` drive
them all through one code path.
"""

import torch as _torch

from .deeponet import GeomDeepONet
from .gino import GINOBaseline
from .gno import GNOBaseline
from .transolver import TransolverBaseline

MODELS = ("transolver", "gino", "gno", "deeponet", "ginot")


def build_baseline(name, cfg, pc_channels=12, out_channels=3, min_length_norm=0.05):
    """`cfg` is a dict of model-specific overrides from the CLI / sweep."""
    if name == "transolver":
        # `point_sdf` swaps in the SUBCLASS (baselines/models/transolver_sdf.py),
        # which adds one token channel: distance to the nearest boundary point.
        # transolver.py itself is untouched, so `--model transolver` without the
        # flag is byte-identical to what it always was.
        cls = TransolverBaseline
        if cfg.get("point_sdf", False):
            from .transolver_sdf import TransolverSdfBaseline
            cls = TransolverSdfBaseline
        return cls(
            pc_channels=pc_channels, out_channels=out_channels,
            min_length_norm=min_length_norm,
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=cfg.get("n_layers", 8),
            n_heads=cfg.get("n_heads", 8),
            slice_num=cfg.get("slice_num", 32),
            mlp_ratio=cfg.get("mlp_ratio", 4),
            dropout=cfg.get("dropout", 0.0),
            n_pc_tokens=cfg.get("n_pc_tokens", 4096),
            num_bands=cfg.get("num_bands", 8),
            use_fourier=cfg.get("use_fourier", True),
            eval_chunk=cfg.get("eval_chunk", 5120),
        )

    if name == "gino":
        return GINOBaseline(
            pc_channels=pc_channels, out_channels=out_channels,
            grid_res=cfg.get("grid_res", 24),
            fno_modes=cfg.get("fno_modes", 8),
            fno_hidden=cfg.get("fno_hidden", 64),
            fno_layers=cfg.get("fno_layers", 4),
            in_radius=cfg.get("in_radius", 0.10),
            out_radius=cfg.get("out_radius", 0.08),
            gno_mlp_hidden=cfg.get("gno_mlp_hidden", 64),
            gno_mlp_layers=cfg.get("gno_mlp_layers", 2),
            n_pc_tokens=cfg.get("n_pc_tokens", 4096),
            eval_chunk=cfg.get("eval_chunk", 8192),
            # OFF by default: every recorded gino / lgo_gino number was trained
            # without it, and turning it on changes parameter shapes. See the
            # module docstring in baselines/models/gino.py.
            latent_sdf=cfg.get("latent_sdf", False),
        )

    if name == "gno":
        return GNOBaseline(
            pc_channels=pc_channels, out_channels=out_channels,
            width=cfg.get("width", 128),
            radius=cfg.get("radius", 0.10),
            n_query_layers=cfg.get("n_query_layers", 1),
            query_radius=cfg.get("query_radius", 0.04),
            # mlp_hidden acts on (query, neighbour) PAIRS. Measured worst case over
            # all training cases: 15000 tokens + hidden 128
            # = 14.8 GiB, but + hidden 256 OOMs on a 24 GiB card. Do not raise
            # both together.
            mlp_hidden=cfg.get("mlp_hidden", 128),
            mlp_layers=cfg.get("mlp_layers", 2),
            n_pc_tokens=cfg.get("n_pc_tokens", 15000),
            eval_chunk=cfg.get("eval_chunk", 4096),
            head_hidden=cfg.get("head_hidden", 1024),
            head_layers=cfg.get("head_layers", 3),
        )

    if name == "deeponet":
        return GeomDeepONet(
            pc_channels=pc_channels, out_channels=out_channels,
            min_length_norm=min_length_norm,
            n_basis=cfg.get("n_basis", 512),
            trunk_width=cfg.get("trunk_width", 512),
            trunk_layers=cfg.get("trunk_layers", 6),
            branch_out_c=cfg.get("branch_out_c", 256),
            branch_width=cfg.get("branch_width", 128),
            branch_latent_d=cfg.get("branch_latent_d", 1024),
            branch_n_point=cfg.get("branch_n_point", 1024),
            branch_radius=cfg.get("branch_radius", 0.08),
            num_bands=cfg.get("num_bands", 8),
            film=cfg.get("film", True),
        )

    if name == "ginot":
        # The reference model, built exactly as runs/ginot_pc_2000ep was.
        from pi_ginot import build_model
        branch_args = {
            "input_channels": pc_channels,
            "out_c": cfg.get("embed_dim", 256),
            "width": cfg.get("branch_width", 128),
            "latent_d": cfg.get("branch_latent_d", 1024),
            "n_point": cfg.get("branch_n_point", 1024),
            "radius": cfg.get("branch_radius", 0.08),
        }
        trunk_args = {
            "in_channels": 3,
            "out_channels": out_channels,
            "embed_dim": cfg.get("embed_dim", 256),
            "cross_attn_layers": cfg.get("cross_attn_layers", 5),
            "min_length_norm": min_length_norm,
            "num_bands": cfg.get("num_bands", 8),
            "freq_spacing": cfg.get("freq_spacing", "log"),
        }
        return build_model(branch_args, trunk_args, verbose=False)

    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


# neuralop's BaseModel stores its own init kwargs inside `state_dict()["_metadata"]`,
# and two of them are FUNCTION objects (`gno_channel_mlp_non_linearity`,
# `fno_non_linearity` = torch._C._nn.gelu) plus a CLASS (`fno_conv_module`). Torch
# >= 2.6 loads with weights_only=True by default and refuses those globals, so a
# GINO checkpoint written by this repo cannot be read back by `legacy/eval_thermo.
# load_model` without allowlisting them. They are metadata only -- `load_state_dict`
# never reads `_metadata` for these keys -- but the refusal happens inside
# `torch.load`, before any of that. Allowlist here rather than in `legacy/`, which
# is vendored and md5-verified.
try:
    from neuralop.layers.spectral_convolution import SpectralConv as _SpectralConv
    _torch.serialization.add_safe_globals([_torch._C._nn.gelu, _SpectralConv])
except Exception:                                  # neuralop absent: nothing to allow
    pass
