"""
PI-GINOT training package.

Ports the GINOT classes from `ginot-baseline-9-case.ipynb` into importable
modules for a standalone training script (`train.py`).

Data-only: the PDE-residual machinery was removed after three controlled
negatives (see PROJECT_OVERVIEW.md) showed every physics loss hurt.

The heavy geometry encoder is reused unchanged from the local `ginot` package
(`ginot.point_encoding.PointCloudPerceiverChannelsEncoder`).
"""

from .model import MildFourierEncoder, Trunk, build_model
from .losses import (
    wall_bc_loss,
    inlet_bc_loss,
    outlet_bc_loss,
    leak_bc_loss,
    supervised_data_loss,
    center_pressure,
)
from .dataset import MultiCase_GINOT_Dataset
from .trainer import HybridGINOTTrainer

__all__ = [
    "MildFourierEncoder",
    "Trunk",
    "build_model",
    "wall_bc_loss",
    "inlet_bc_loss",
    "outlet_bc_loss",
    "leak_bc_loss",
    "supervised_data_loss",
    "center_pressure",
    "MultiCase_GINOT_Dataset",
    "HybridGINOTTrainer",
]
