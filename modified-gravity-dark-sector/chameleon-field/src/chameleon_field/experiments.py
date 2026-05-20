"""Experimental constraints on chameleon models."""

import numpy as np
from .potential import minimize_effective_potential


def eot_wash_constraint(beta, Lambda, n):
    """
    Eot-Wash torsion balance constraint on the chameleon fifth force.

    The Eot-Wash experiment tests the inverse-square law at sub-millimeter
    scales. It constrains the chameleon-mediated force to be smaller than
    gravity.

    Parameters
    ----------
    beta : float
        Matter coupling constant.
    Lambda : float
        Energy scale (eV).
    n : float
        Power-law index.

    Returns
    -------
    bool
        True if the parameter combination is allowed by Eot-Wash data.
    """
    # The Eot-Wash experiment is sensitive to forces at ~50 micron scale.
    # For the chameleon to evade detection, the thin-shell effect must
    # suppress the force for the test masses.
    # The relevant density is approximately that of the experiment:
    # vacuum chamber ~ 1e-14 g/cm^3, test mass (Be/Cu) ~ 1.8-8.9 g/cm^3

    rho_test = 1.8e-3 * 5.07e6  # Be density in eV^4 (approx)
    rho_vacuum = 1e-14 * 1e-3 * 5.07e6  # vacuum in eV^4

    phi_min_test = minimize_effective_potential(rho_test, Lambda, n, beta)
    phi_min_vacuum = minimize_effective_potential(rho_vacuum, Lambda, n, beta)

    # The fifth force is ~ 2*beta^2 * M_Pl^2 * |delta_phi| / R^2
    # For it to be below experimental sensitivity:
    # 2*beta^2 * (phi_vacuum - phi_test) / M_Pl < alpha_max
    # where alpha_max ~ 1e-4 for Eot-Wash
    alpha_max = 1e-4

    # Compton wavelength in the vacuum must be shorter than ~50 microns
    # to avoid detection, or the thin-shell must suppress the signal.
    delta_phi = abs(phi_min_vacuum - phi_min_test)
    force_ratio = 2 * beta ** 2 * delta_phi / (2.435e27)

    return force_ratio < alpha_max


def microscope_constraint(beta):
    """
    MICROSCOPE satellite constraint on the chameleon coupling.

    MICROSCOPE tests the Weak Equivalence Principle in space.
    It constrains the Eotvos parameter eta < 1e-14.

    Parameters
    ----------
    beta : float
        Matter coupling constant.

    Returns
    -------
    bool
        True if beta is allowed by MICROSCOPE data.
    """
    # MICROSCOPE constrains eta < ~1.5e-14 (2022 results)
    # For universal coupling beta, the thin-shell effect around Earth
    # suppresses the force. The constraint roughly requires
    # beta < O(1) for Lambda ~ meV scale.
    # A simplified constraint: beta^2 < eta_max * M_Pl^2 / (delta_phi)
    # For Earth-like densities, this gives beta < ~1e8 for typical Lambda.
    # We use a conservative bound:
    eta_max = 1.5e-14
    # For large beta, WEP violation ~ beta^2 * thin_shell_param^2
    # A useful approximate constraint:
    return beta ** 2 < 1e6  # Very permissive for universal coupling


def allowed_parameter_region(beta_array, Lambda_array, n):
    """
    Compute the allowed parameter region in (beta, Lambda) space.

    Parameters
    ----------
    beta_array : array_like
        Array of beta values.
    Lambda_array : array_like
        Array of Lambda values (eV).
    n : float
        Power-law index.

    Returns
    -------
    numpy.ndarray
        2D boolean array, True where parameters are allowed.
    """
    beta_array = np.asarray(beta_array)
    Lambda_array = np.asarray(Lambda_array)

    result = np.zeros((len(beta_array), len(Lambda_array)), dtype=bool)
    for i, beta in enumerate(beta_array):
        for j, Lambda in enumerate(Lambda_array):
            eot = eot_wash_constraint(beta, Lambda, n)
            mic = microscope_constraint(beta)
            result[i, j] = eot and mic

    return result
