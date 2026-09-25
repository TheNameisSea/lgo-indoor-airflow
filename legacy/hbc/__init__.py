"""Hard wall/inlet boundary-condition upgrade for PI-GINOT (see PLAN_HARD_BC_UPGRADE.md).

Nothing in ginot/, pi_ginot/ or train.py is modified: this package wraps the built
Trunk so the baseline remains byte-identical for A/B comparison.
"""

from .distance import mask, min_dist, nearest_value, phys_coords
from .wrapper import HardBCModel
from .trainer_hbc import HBCTrainer

__all__ = ["HardBCModel", "HBCTrainer", "mask", "min_dist", "nearest_value", "phys_coords"]
