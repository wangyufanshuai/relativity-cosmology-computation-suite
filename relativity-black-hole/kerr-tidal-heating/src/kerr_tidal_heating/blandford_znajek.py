"""Blandford-Znajek mechanism: relativistic jet power from spinning black holes.

The BZ mechanism extracts rotational energy from a Kerr black hole via
magnetic fields threading the horizon, powering relativistic jets.
"""

from __future__ import annotations

import numpy as np


def bz_power(M: float, a: float, B: float) -> float:
    """Blandford-Znajek jet power (spin-averaged / angle-averaged).

    Uses the standard fit formula (Tchekhovskoy et al. 2010, Blandford &
    Znajek 1977):

        P_BZ = kappa / (6 pi) * Phi_B^2 * omega_H^2 * c

    where
        Phi_B = B * pi * r_g^2          (magnetic flux)
        omega_H = a / (2 r_+)           (horizon angular velocity)
        kappa ~ 1                        (efficiency factor)
        r_g = M                         (gravitational radius, G=c=1)

    Simplifying:

        P_BZ = (kappa / 6) * B^2 * M^2 * (a / M)^2 * c

    With kappa = 0.044 (fiducial) and c = 1 in geometric units.

    Parameters
    ----------
    M : float
        Black hole mass (geometric units).
    a : float
        Spin parameter (0 <= a <= M).
    B : float
        Magnetic field strength at the horizon (geometric units).

    Returns
    -------
    float
        BZ power in geometric units.
    """
    if M <= 0:
        return 0.0
    a_star = a / M
    # Fiducial normalisation from Tchekhovskoy+ 2010
    kappa = 0.044
    r_g = M
    return kappa * B**2 * r_g**2 * a_star**2


def bz_power_numerical(M: float, a: float, B: float, theta: float) -> float:
    """Angle-dependent Blandford-Znajek power.

    Includes a simple angular dependence P(theta) = P_0 * f(theta) where
    f(theta) ~ sin^2(theta) for a split-monopole field geometry.

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    B : float
        Magnetic field strength.
    theta : float
        Polar angle (radians).

    Returns
    -------
    float
        Angle-dependent BZ power.
    """
    P0 = bz_power(M, a, B)
    # Split-monopole angular structure
    return P0 * np.sin(theta) ** 2
