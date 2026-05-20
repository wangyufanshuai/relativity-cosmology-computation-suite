"""Einstein-Aether theory and Horava-Lifshitz gravity simulator.

Provides tools for computing aether field configurations, PPN parameters,
modified cosmological equations, and experimental constraint analysis.
"""

from .aether_field import AetherField
from .ppn_parameters import ppn_gamma, ppn_beta, newton_constant_ratio, preferred_frame_params
from .cosmology import modified_friedmann, gw_speed, effective_gravitational_constant
from .constraints import (
    solar_system_constraints,
    gw_speed_constraint,
    parameter_priors,
    viable_parameter_space,
)

__all__ = [
    "AetherField",
    "ppn_gamma",
    "ppn_beta",
    "newton_constant_ratio",
    "preferred_frame_params",
    "modified_friedmann",
    "gw_speed",
    "effective_gravitational_constant",
    "solar_system_constraints",
    "gw_speed_constraint",
    "parameter_priors",
    "viable_parameter_space",
]
