"""Freeze-out physics and radiation-era Hubble rate."""

import numpy as np

from .dof import g_star


def freeze_out_temperature(sigma_v=1e-26, m_MeV=100000.0, g_x=2, g_star_val=None):
    """Estimate WIMP freeze-out temperature in MeV.

    Uses x_f = m/T_f ~ 25 approximation.
    """
    x_f = 25.0
    return m_MeV / x_f


def hubble_rate_radiation(T_MeV, g_star_val=None):
    """Hubble rate during radiation domination: H = 1.66 * sqrt(g_*) * T^2 / M_Pl."""
    if g_star_val is None:
        g_star_val = g_star(T_MeV)
    M_Pl_MeV = 1.2209e19 * 1e6
    return 1.66 * np.sqrt(g_star_val) * T_MeV**2 / M_Pl_MeV
