"""Tidal Disruption Event (TDE) simulator."""

from .constants import G, c, M_sun, R_sun, sigma_T, m_p, sigma_SB, k_B
from .params import StellarParams, BlackHoleParams
from .disruption import (
    penetration_factor,
    critical_beta,
    is_full_disruption,
    partial_disruption_fraction,
)
from .observability import energy_spread, is_outside_horizon, maximum_bh_mass_for_tde

# Import the facade module first, then pull in the standalone functions.
# This ordering avoids the module shadowing the tidal_radius function.
from . import tidal_radius as _facade_mod  # noqa: F401

TidalDisruption = _facade_mod.TidalDisruption

from .radii import hill_radius, isco_radius, schwarzschild_radius, tidal_radius

__all__ = [
    # Constants
    "G", "c", "M_sun", "R_sun", "sigma_T", "m_p", "sigma_SB", "k_B",
    # Data classes
    "StellarParams", "BlackHoleParams",
    # Functions
    "tidal_radius", "schwarzschild_radius", "isco_radius", "hill_radius",
    "penetration_factor", "critical_beta", "is_full_disruption",
    "partial_disruption_fraction",
    "energy_spread", "is_outside_horizon", "maximum_bh_mass_for_tde",
    # Facade class
    "TidalDisruption",
]
