"""Brans-Dicke theory and PPN parameters."""

import numpy as np


# Gravitational constant in SI
G_NEWTON = 6.674e-11  # m^3 kg^-1 s^-2
# Reduced Planck mass in GeV
M_PL = 2.435e18  # GeV


def brans_dicke_action(omega, phi, Lambda=0.0):
    """
    Compute the Brans-Dicke action functional (schematic).

    S = integral [phi*R - omega/phi * (d phi)^2 - V(phi)] sqrt(-g) d^4x

    Parameters
    ----------
    omega : float
        Brans-Dicke coupling parameter.
    phi : float
        Scalar field value (dimensionless, in units of G^{-1}).
    Lambda : float
        Cosmological constant.

    Returns
    -------
    dict
        Dictionary with kinetic_term, coupling, and potential components.
    """
    kinetic_coeff = -omega / phi if phi != 0 else 0.0
    return {
        "kinetic_coefficient": kinetic_coeff,
        "ricci_coupling": phi,
        "cosmological_constant": Lambda,
        "omega": omega,
    }


def ppn_gamma(omega):
    """
    Post-Newtonian parameter gamma for Brans-Dicke theory.

    gamma_BD = (omega + 1) / (omega + 2)

    In GR, gamma = 1. For large omega, gamma -> 1.

    Parameters
    ----------
    omega : float
        Brans-Dicke coupling parameter.

    Returns
    -------
    float
        PPN gamma parameter.
    """
    if omega == -2:
        return np.inf
    return (omega + 1) / (omega + 2)


def ppn_beta(omega):
    """
    Post-Newtonian parameter beta for Brans-Dicke theory.

    beta_BD = 1 (identically in BD theory, since BD is a special case
    of scalar-tensor theories with linear coupling).

    Parameters
    ----------
    omega : float
        Brans-Dicke coupling parameter.

    Returns
    -------
    float
        PPN beta parameter (= 1).
    """
    return 1.0


def cassini_constraint():
    """
    Cassini spacecraft constraint on the Brans-Dicke omega parameter.

    The Cassini mission measured gamma = 1 + (2.1 +/- 2.3)e-5,
    giving the constraint omega > 40000.

    Returns
    -------
    dict
        Dictionary with gamma measurement and omega constraint.
    """
    gamma_measured = 1.0
    gamma_error = 2.3e-5
    omega_min = 40000

    # omega > (1 - gamma_measured + 2*gamma_error) / (2 * (gamma_measured - 2*gamma_error - 1))
    # Simplified: from gamma = (omega+1)/(omega+2) < 1 + 2.3e-5
    # => omega > 1/(2.3e-5) ~ 43478, approximately 40000

    return {
        "gamma_measured": gamma_measured,
        "gamma_error": gamma_error,
        "omega_min": omega_min,
        "constraint": f"omega > {omega_min}",
    }
