"""Temperature-dependent axion mass."""

import numpy as np


def axion_mass_temperature(T, m_a, Lambda_QCD=1e9):
    """Axion mass as function of temperature.

    m_a(T) = m_a * (Lambda_QCD / T)^3.7 for T > Lambda_QCD, else m_a.
    """
    T = np.asarray(T, dtype=float)
    return np.where(T > Lambda_QCD, m_a * (Lambda_QCD / T) ** 3.7, m_a)


def oscillation_temperature(m_a, Lambda_QCD=1e9):
    """Temperature when axion field starts oscillating (m_a(T) ~ 3H(T))."""
    M_Pl_eV = 1.22e28
    return (m_a * Lambda_QCD**3.7 * M_Pl_eV) ** (1.0 / 5.7)
