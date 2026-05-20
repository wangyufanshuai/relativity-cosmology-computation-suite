"""Modified Friedmann equations and gravitational-wave speed.

Implements the cosmological implications of the Einstein-aether theory,
including the modified Friedmann equation and the speed of
gravitational waves.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ------------------------------------------------------------------
# Gravitational-wave speed
# ------------------------------------------------------------------

def gw_speed(c1: float, c3: float) -> float:
    """Compute the gravitational-wave speed c_T in units of c.

    In Einstein-aether theory the tensor (gravitational-wave) propagation
    speed is:

        c_T^2 = 1 / (1 - c_13)

    where c_13 = c_1 + c_3.  The constraint from GW170817 requires
    c_T = c to within ~10^{-15}, i.e. c_13 ~ 0.

    Parameters
    ----------
    c1, c3 : float
        Coupling constants.

    Returns
    -------
    float
        c_T in units of c (speed of light).
    """
    c13 = c1 + c3
    if abs(1.0 - c13) < 1e-30:
        return np.inf
    return 1.0 / np.sqrt(1.0 - c13)


# ------------------------------------------------------------------
# Friedmann modification factor
# ------------------------------------------------------------------

def friedmann_factor(c1: float, c2: float, c3: float, c4: float) -> float:
    """Compute the Friedmann equation modification factor.

    H^2 = (8 pi G_N / 3) * rho * F

    where F = 1 / (1 - c_13/2 - c_2/4 - c_4/4) approximately.

    This factor arises from the aether field contribution in a
    Friedmann-Robertson-Walker (FRW) background where the aether
    is aligned with the cosmic fluid flow.

    Parameters
    ----------
    c1, c2, c3, c4 : float
        Coupling constants.

    Returns
    -------
    float
        Modification factor F.
    """
    c13 = c1 + c3
    denom = 1.0 - c13 / 2.0 - c2 / 4.0 - c4 / 4.0
    if abs(denom) < 1e-30:
        return np.inf
    return 1.0 / denom


# ------------------------------------------------------------------
# Modified Friedmann equation
# ------------------------------------------------------------------

def modified_friedmann(
    rho: float | NDArray,
    G_N: float = 6.674e-11,
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
    c4: float = 0.0,
) -> float | NDArray:
    """Compute H^2 from the modified Friedmann equation.

    H^2 = (8 pi G_N / 3) * rho * F(c_i)

    where F is the modification factor.  When all c_i = 0 this
    reduces exactly to the standard Friedmann equation.

    Parameters
    ----------
    rho : float or ndarray
        Energy density (SI units: kg m^{-3}).
    G_N : float
        Newton's gravitational constant (default SI value).
    c1, c2, c3, c4 : float
        Aether coupling constants.

    Returns
    -------
    H2 : float or ndarray
        Hubble parameter squared (SI units: s^{-2}).
    """
    F = friedmann_factor(c1, c2, c3, c4)
    return (8.0 * np.pi * G_N / 3.0) * rho * F


# ------------------------------------------------------------------
# Effective gravitational constant for structure growth
# ------------------------------------------------------------------

def effective_gravitational_constant(
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
    c4: float = 0.0,
    G_N: float = 6.674e-11,
) -> float:
    """Compute the effective gravitational constant for structure growth.

    In Einstein-aether cosmology the effective Newton's constant
    governing the growth of matter perturbations is modified:

        G_eff = G_N * (1 + f(c_i))

    where f encodes the scalar-mode contribution from the aether.

    Parameters
    ----------
    c1, c2, c3, c4 : float
        Coupling constants.
    G_N : float
        Newton's constant.

    Returns
    -------
    float
        Effective gravitational constant G_eff.
    """
    c13 = c1 + c3
    c14 = c1 + c4
    c123 = c1 + c2 + c3

    if abs(1.0 - c13) < 1e-15:
        return np.inf

    # Factor from the scalar gravitational potential
    # G_eff / G_N ~ 1 + c_14 / (2 (1 - c_13))  [leading order]
    # plus corrections from the vector and scalar modes
    denom_grav = 2.0 - c14 - c13 * (2.0 + c2) / (1.0 - c13)
    if abs(denom_grav) < 1e-30:
        return np.inf

    ratio = 1.0 - (c14 / 2.0) * (2.0 + c13 + 3.0 * c2) / denom_grav
    if ratio <= 0:
        return np.inf
    return G_N / ratio


# ------------------------------------------------------------------
# Hubble parameter helper
# ------------------------------------------------------------------

def hubble_parameter(
    rho: float | NDArray,
    G_N: float = 6.674e-11,
    c1: float = 0.0,
    c2: float = 0.0,
    c3: float = 0.0,
    c4: float = 0.0,
) -> float | NDArray:
    """Compute the Hubble parameter H from the modified Friedmann equation.

    Parameters
    ----------
    rho : float or ndarray
        Energy density (SI).
    G_N : float
        Newton's constant.
    c1, c2, c3, c4 : float
        Aether coupling constants.

    Returns
    -------
    H : float or ndarray
        Hubble parameter (SI: s^{-1}).
    """
    H2 = modified_friedmann(rho, G_N, c1, c2, c3, c4)
    return np.sqrt(H2)
