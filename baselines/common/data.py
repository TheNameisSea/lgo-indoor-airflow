"""Shared dataset construction — identical for GINOT and every baseline.

The protocol is fixed here on purpose: if a baseline ever needs a different point
budget or normalization, it stops being a controlled comparison. `augment` is NOT
exposed — no model in this comparison uses augmentation (the GINOT reference is
runs/ginot_pc_2000ep, which was trained with augment=[]).
"""

import os

from pi_ginot import MultiCase_GINOT_Dataset

# Matches runs/ginot_pc_2000ep/args.json exactly.
# FULL-POOL protocol (runs_final). Every interior point (~1.16 M/case) and every
# boundary point (~228 k/case) is retained. `n_pc` is 150 k, NOT the full 228 k:
# the branch encoder is n_point=1024 centroids x n_sample=8 ball-query neighbours,
# so it consumes at most 8,192 points per forward. Measured ball-query fill:
#   n_pc  15 k -> 4.5/8 slots (56% of balls under-filled)  0.066 s/case,  2.1 GiB
#   n_pc 150 k -> 7.4/8 slots (27% under-filled)           0.19  s/case,  8.0 GiB
#   n_pc 228 k -> 8.0/8 slots ( 5% under-filled)           0.317 s/case, 15.0 GiB
# 150 k buys ~93% of the benefit at ~60% of full cost; past it the median ball
# holds 18 candidates for 8 used slots, i.e. pure redundancy.
PROTOCOL = dict(
    n_case_pool=1_200_000,
    n_boundary_pool=250_000,
    n_pc=15_000,     # ablation winner @2000 ep
    n_collocation_batch=1024,
    n_supervised_batch=4096,
    n_boundary_batch=2048,
    n_inlet_batch=1024,
    n_outlet_batch=1024,
    n_leak_batch=800,
    g=9.81,
    pressure_gauge=True,
    pc_mode="full",
)

# Points supervised per case per epoch = collocation + supervised (GINOT
# supervises both; boundary lambdas are 0 so those points carry no gradient).
N_SUPERVISED_PER_CASE = (PROTOCOL["n_collocation_batch"]
                         + PROTOCOL["n_supervised_batch"])


def build_datasets(train_dir, val_dir, stats_path):
    """Training + validation datasets sharing one normalization.

    `stats_path` is written by the training dataset and re-read by the validation
    dataset, so both use the training z-scores (and every run in the comparison
    uses the same file).
    """
    train_ds = MultiCase_GINOT_Dataset(
        case_dirs=train_dir, saved_stats=None, stats_path=stats_path,
        augment=[], **PROTOCOL)

    val_ds = None
    if val_dir:
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"stats file missing after train build: {stats_path}")
        val_ds = MultiCase_GINOT_Dataset(
            case_dirs=val_dir, saved_stats=stats_path, stats_path=stats_path,
            **PROTOCOL)

    return train_ds, val_ds
