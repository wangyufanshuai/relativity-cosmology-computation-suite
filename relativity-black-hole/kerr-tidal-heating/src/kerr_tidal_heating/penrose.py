"""Penrose process: energy extraction from the Kerr ergosphere.

The Penrose process allows extraction of rotational energy from a Kerr
black hole by exploiting the existence of negative-energy orbits inside
the ergosphere.
"""

from __future__ import annotations

import numpy as np


def negative_energy_orbit(M: float, a: float, r: float, L: float, E: float) -> bool:
    """Check whether an orbit parameterised by (r, L, E) carries negative energy.

    A particle inside the ergosphere can have negative energy as measured
    at infinity when its angular momentum is sufficiently retrograde.

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    r : float
        Radial coordinate (Boyer-Lindquist).
    L : float
        Specific angular momentum of the orbit.
    E : float
        Specific energy of the orbit.

    Returns
    -------
    bool
        True if the orbit has negative energy (E < 0).
    """
    return E < 0.0


def penrose_efficiency(M: float, a: float, r_breakup: float, E_in: float) -> float:
    """Maximum energy-extraction efficiency of the Penrose process.

    A particle with energy E_in breaks up at radius r_breakup inside the
    ergosphere.  One fragment falls into the black hole on a negative-energy
    orbit while the other escapes to infinity.  The efficiency is the ratio
    of extracted energy to the incoming energy.

    eta = (E_out - E_in) / E_in  (capped at the theoretical maximum).

    The theoretical maximum for extremal Kerr is (sqrt(2) - 1) / 2 ~ 20.7 %.

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    r_breakup : float
        Radius at which the particle breaks up.
    E_in : float
        Incoming particle energy (must be > 0).

    Returns
    -------
    float
        Extraction efficiency (fraction, not percent).
    """
    if E_in <= 0:
        raise ValueError("Incoming energy E_in must be positive.")

    # Absolute theoretical maximum for Kerr
    eta_max = max_penrose_efficiency(M, a)

    # Practical efficiency limited by breakup radius.
    # At the horizon the maximum extractable fraction approaches eta_max;
    # farther out it is reduced.  A simple model:
    r_plus = M + np.sqrt(M**2 - a**2)
    r_ergo = M + np.sqrt(M**2)  # = 2M at equator (theta = pi/2)

    if r_breakup <= r_plus:
        # Inside horizon -- not physical, return max
        return eta_max

    if r_breakup >= r_ergo:
        # Outside ergosphere -- no Penrose process
        return 0.0

    # Linear interpolation between horizon (max) and ergosphere edge (0)
    frac = (r_ergo - r_breakup) / (r_ergo - r_plus)
    eta = eta_max * frac
    return min(eta, eta_max)


def max_penrose_efficiency(M: float, a: float) -> float:
    """Theoretical maximum Penrose process efficiency.

    For extreme Kerr (a = M) the maximum efficiency is (sqrt(2) - 1)/2 ~ 20.7 %.
    For general spin it scales as  (sqrt(2) - 1) / 2 * (a/M).

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter.

    Returns
    -------
    float
        Maximum efficiency (fraction).
    """
    if M == 0:
        return 0.0
    a_star = min(abs(a) / M, 1.0)
    return (np.sqrt(2.0) - 1.0) / 2.0 * a_star
