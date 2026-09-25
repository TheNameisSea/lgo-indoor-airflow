"""EquiJet: FCU-equivariant jet decomposition on top of PI-GINOT.

See PLAN_EQUIJET_UPGRADE.md. No existing file is modified: this package wraps the
built Trunk, so ginot/, pi_ginot/ and train.py remain the untouched A/B baseline.
Composes with the hbc/ wrapper via EquiJetModel(hbc=...) - see plan section 6.
"""

from .template import JetTemplate
from .vents import build_temperature_table, extract_vents, COND_DIM
from .wrapper import EquiJetModel
from .trainer_eq import EquiJetTrainer

__all__ = ["JetTemplate", "EquiJetModel", "EquiJetTrainer",
           "extract_vents", "build_temperature_table", "COND_DIM"]
