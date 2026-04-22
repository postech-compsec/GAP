"""GAP: Gyroscope Attack Policy — core RL module.

The canonical location for AsymmetricLSTMModule is ``gap.asymmetric_rl_module``.
Existing RLlib checkpoints were pickled when the class lived at the top-level
module path ``asymmetric_rl_module``. We register the alias in sys.modules so
those pickles still resolve after the consolidation.
"""
import sys as _sys
from . import asymmetric_rl_module as _arm

# Backward-compat for pickled checkpoints that reference 'asymmetric_rl_module'.
_sys.modules.setdefault("asymmetric_rl_module", _arm)

from .asymmetric_rl_module import AsymmetricLSTMModule
