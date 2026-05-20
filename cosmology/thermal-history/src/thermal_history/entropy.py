"""Entropy density and conservation."""

import numpy as np

from .dof import g_star_s


def entropy_density(T_MeV, g_s=None):
    """Entropy density s = (2*pi^2/45) * g_*s * T^3 in MeV^3."""
    if g_s is None:
        g_s = g_star_s(T_MeV)
    return (2.0 * np.pi**2 / 45.0) * g_s * T_MeV**3


def entropy_conservation(T1_MeV, T2_MeV):
    """Ratio s(T1)/s(T2). Should be ~1 for adiabatic expansion."""
    s1 = entropy_density(T1_MeV)
    s2 = entropy_density(T2_MeV)
    return s1 / s2
