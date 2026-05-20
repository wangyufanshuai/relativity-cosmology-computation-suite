"""Scalar field profiles in stars and Vainshtein radius."""

import numpy as np


# Reduced Planck mass in GeV
M_PL = 2.435e18  # GeV
# Speed of light
C = 3e8  # m/s


def scalar_field_profile(r, omega, M_star):
    """
    Compute the scalar field profile phi(r) around a star in BD theory.

    For large omega, the field is nearly constant with a 1/r perturbation:
    phi(r) = phi_0 * (1 + 2 / ((2*omega + 3)) * G*M / r)

    Parameters
    ----------
    r : float or array
        Radial distance (in geometric units, e.g., km).
    omega : float
        Brans-Dicke coupling.
    M_star : float
        Star mass (in geometric units).

    Returns
    -------
    float or array
        Scalar field value at r.
    """
    r = np.asarray(r, dtype=float)
    phi_0 = 1.0  # Background field value
    # Perturbation: delta_phi / phi_0 ~ 2 / (2*omega + 3) * GM/r
    G = 1.0  # In geometric units
    delta_phi = 2.0 / (2 * omega + 3) * G * M_star / r
    return phi_0 * (1 + delta_phi)


def vainshtein_radius(M, r, beta):
    """
    Compute the Vainshtein radius for massive gravity / DGP scenarios.

    The Vainshtein radius is the scale below which the scalar degree of
    freedom is screened:
    r_V = (beta * r_s^2 * r)^(1/3)
    where r_s = 2*G*M.

    Parameters
    ----------
    M : float
        Mass of the source (kg).
    r : float
        Distance from the source (m).
    beta : float
        Coupling parameter.

    Returns
    -------
    float
        Vainshtein radius in meters.
    """
    G = 6.674e-11  # m^3 kg^-1 s^-2
    r_s = 2 * G * M  # Schwarzschild-like radius
    r_V = (beta * r_s ** 2 * r) ** (1.0 / 3.0)
    return r_V
