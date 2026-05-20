"""Axion relic density from misalignment mechanism."""

import numpy as np


def axion_density(m_a, theta_i=1.0, T_gamma=2.7255, h=0.674):
    """Axion relic density Omega_a h^2 from misalignment."""
    m_a_ref = 5e-6
    return 0.12 * theta_i**2 * (m_a / m_a_ref) ** (-7.0 / 6.0)


def axion_mass_from_density(Omega_h2_target=0.12, theta_i=1.0):
    """Infer axion mass from target relic density."""
    m_a_ref = 5e-6
    return m_a_ref * (Omega_h2_target / (0.12 * theta_i**2)) ** (6.0 / 7.0)
