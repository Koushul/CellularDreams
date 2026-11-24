"""Tissue simulator plus a kinetic expression engine.

The tissue simulator owns positions, movement, division, and clock time.
The expression engine owns promoter kinetics, regulation, and ligand effects.
Architecture follows PRECOMMIT_TRAVLR_SIM.md: scMultiSim's grid loop cannot
take external positions without being replaced, so the kinetics are a Python
reimplementation of its Beta-Poisson and telegraph updates.
"""

__version__ = "0.1.0"

from .kinetics import beta_poisson_mean, beta_poisson_draw, telegraph_step
from .simulate import Scene, simulate

__all__ = [
    "Scene",
    "simulate",
    "beta_poisson_mean",
    "beta_poisson_draw",
    "telegraph_step",
    "__version__",
]
