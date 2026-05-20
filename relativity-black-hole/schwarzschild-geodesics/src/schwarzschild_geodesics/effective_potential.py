"""
Effective potential analysis for Schwarzschild geodesics.

We work in *geometrized units* throughout this module: G = c = 1.
All lengths (including M) are measured in metres, and the dimensionless
specific energy E and specific angular momentum L_tilde are defined by:

    E       = (1 - r_s/r) dt/dtau          (energy per unit rest-mass)
    L_tilde = r^2 dphi/dtau                (angular momentum per unit rest-mass)

For massive (timelike) particles the effective potential is:
    V_eff(r) = (1 - r_s/r) (1 + L_tilde^2 / r^2)

For massless (null) particles:
    V_eff(r) = (1 - r_s/r) L_tilde^2 / r^2

The radial equation of motion is:
    (dr/dtau)^2 = E^2 - V_eff(r)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from .metric import G, c, schwarzschild_radius


# ---------------------------------------------------------------------------
# Helper: convert SI mass to geometrized length  M_geo = GM/c^2
# ---------------------------------------------------------------------------

def _M_geo(M_si: float) -> float:
    """Geometrized mass  M = GM/c^2  (in metres)."""
    return G * M_si / c**2


# ---------------------------------------------------------------------------
# Effective potentials
# ---------------------------------------------------------------------------

def V_eff_timelike(
    r: float | NDArray[np.floating],
    L_tilde: float,
    M: float,
) -> float | NDArray[np.floating]:
    """Effective potential for massive particles (timelike geodesics).

    V_eff = (1 - r_s/r)(1 + L_tilde^2 / r^2)

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate(s) in metres.
    L_tilde : float
        Specific angular momentum in m^2/s  (dimensionally L = r^2 dphi/dtau).
        Internally we use the dimensionless form L/(GM/c).
    M : float
        Black-hole mass in kg (SI).

    Returns
    -------
    float or ndarray
        V_eff evaluated at r.  Dimensionless.
    """
    rs = schwarzschild_radius(M)
    M_g = _M_geo(M)
    # Normalise L to dimensionless:  l = L / (c * M_g) = L * c / (G M)
    # But the standard form of V_eff in units G=c=1 with M in metres uses
    # L in units of M (= GM/c^2).  So l = L_tilde / (c * M_g).
    l = L_tilde / (c * M_g)
    r_norm = r / M_g                  # r in units of M

    f = 1.0 - 2.0 / r_norm           # 1 - r_s/r  in geometrized units
    V = f * (1.0 + l**2 / r_norm**2)
    return V


def V_eff_null(
    r: float | NDArray[np.floating],
    L_tilde: float,
    M: float,
) -> float | NDArray[np.floating]:
    """Effective potential for photons (null geodesics).

    V_eff = (1 - r_s/r) L^2 / r^2

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate(s) in metres.
    L_tilde : float
        Photon angular momentum parameter (impact parameter proxy).
    M : float
        Black-hole mass in kg (SI).

    Returns
    -------
    float or ndarray
        V_eff evaluated at r.
    """
    rs = schwarzschild_radius(M)
    M_g = _M_geo(M)
    l = L_tilde / (c * M_g)
    r_norm = r / M_g

    f = 1.0 - 2.0 / r_norm
    V = f * l**2 / r_norm**2
    return V


# ---------------------------------------------------------------------------
# Circular orbit utilities
# ---------------------------------------------------------------------------

def circular_orbit_params(
    r: float,
    M: float,
) -> tuple[float, float]:
    """Energy and angular momentum for a circular orbit at radius *r*.

    For Schwarzschild in geometrized units (G = c = 1, mass = M_g metres):

        E = (1 - 2M/r) / sqrt(1 - 3M/r)
        L = sqrt(M r) / sqrt(1 - 3M/r)

    Valid only for r > 3 M_g  (= 1.5 r_s, outside photon sphere).

    Parameters
    ----------
    r : float
        Orbital radius in metres.
    M : float
        Mass in kg (SI).

    Returns
    -------
    tuple[float, float]
        (E, L) — specific energy (dimensionless) and specific angular
        momentum in m^2/s.
    """
    M_g = _M_geo(M)
    r_norm = r / M_g                   # r / M

    if r_norm <= 3.0:
        raise ValueError(
            f"No stable circular orbit at r/M = {r_norm:.4f} "
            f"(must be > 3, i.e. outside photon sphere)."
        )

    denom = np.sqrt(1.0 - 3.0 / r_norm)
    E = (1.0 - 2.0 / r_norm) / denom        # dimensionless specific energy
    L_geo = np.sqrt(r_norm) / denom          # L / M  in geometrized units

    # Convert L back to SI:  L_si = L_geo * M_g * c
    L_si = L_geo * M_g * c
    return E, L_si


def isco_energy_angular(M: float) -> tuple[float, float]:
    """Specific energy and angular momentum at the ISCO.

    For Schwarzschild (r_isco = 6 M_g):
        E_isco = 2 sqrt(2) / 3  ~ 0.94281
        L_isco = 2 sqrt(3) M_g  ~ 3.4641 M_g

    Parameters
    ----------
    M : float
        Mass in kg (SI).

    Returns
    -------
    tuple[float, float]
        (E_isco, L_isco) in SI-compatible units (E dimensionless, L in m^2/s).
    """
    M_g = _M_geo(M)
    E_isco = 2.0 * np.sqrt(2.0) / 3.0           # ~0.9428
    L_isco_geo = 2.0 * np.sqrt(3.0) * M_g       # in metres (geometrized L)
    L_isco_si = L_isco_geo * c                   # convert to m^2/s
    return E_isco, L_isco_si


def find_unstable_circular(
    L_tilde: float,
    M: float,
    particle_type: str = "timelike",
) -> float:
    """Find the *unstable* circular-orbit radius for given L.

    Searches for the local maximum of V_eff inside the stable circular
    orbit region.  For timelike particles this gives the inner (unstable)
    circular orbit; for null particles it gives r = 3 M_g (photon sphere)
    when L is the critical impact parameter.

    Parameters
    ----------
    L_tilde : float
        Specific angular momentum (SI, m^2/s).
    M : float
        Mass in kg (SI).
    particle_type : str
        ``'timelike'`` or ``'null'``.

    Returns
    -------
    float
        Radius of the unstable circular orbit in metres.
    """
    M_g = _M_geo(M)
    l = L_tilde / (c * M_g)

    rs_norm = 2.0   # r_s / M_g = 2

    if particle_type == "null":
        r_lo, r_hi = 2.01 * M_g, 10.0 * M_g
    else:
        # For timelike, the unstable circular orbit is the inner root of
        # dV/dr = 0, which lies between r_s (2 M_g) and the stable orbit.
        # The roots of dV/dr = 0 are at r where l^2(r-3) = r^2 (after
        # simplification of 2r^3 - l^2(r-3)*2 = 0 etc.).
        # For any l^2 > 12, there are two roots: one between (2, 4) and
        # one > 6.  The inner one is the unstable orbit.
        # We search from just above the horizon up to a generous upper bound.
        r_lo, r_hi = 2.01 * M_g, 6.0 * M_g

    def _dV(r_val: float) -> float:
        rn = r_val / M_g
        if particle_type == "null":
            # dV/dr = l^2 * (2(r-3)) / r^4
            return l**2 * 2.0 * (rn - 3.0) / rn**4
        else:
            # dV/dr = d/dr [ (1-2/r)(1 + l^2/r^2) ]
            #       = 2/r^2 (1 + l^2/r^2) - 2 l^2 (1 - 2/r) / r^3
            # Simplified: 2*(r^2 + l^2) / r^4 - 2*l^2*(r-2) / r^4
            #           = 2*(r^2 + l^2 - l^2*r + 2*l^2) / r^4
            #           = 2*(r^2 + 3*l^2 - l^2*r) / r^4
            return 2.0 * (rn**2 + 3.0 * l**2 - l**2 * rn) / rn**4

    try:
        r_unstable = brentq(_dV, r_lo, r_hi, xtol=1e-12 * M_g)
    except ValueError:
        # Widen the search range
        r_lo = 2.001 * M_g
        r_hi = 50.0 * M_g
        r_unstable = brentq(_dV, r_lo, r_hi, xtol=1e-12 * M_g)

    return r_unstable


def classify_orbit(
    E: float,
    L: float,
    M: float,
) -> str:
    """Classify the orbit type from energy and angular momentum.

    Parameters
    ----------
    E : float
        Specific energy (dimensionless, geometrized).
    L : float
        Specific angular momentum in m^2/s (SI).
    M : float
        Mass in kg (SI).

    Returns
    -------
    str
        One of ``'plunge'``, ``'bound'``, ``'deflection'``, ``'capture'``.
    """
    M_g = _M_geo(M)
    l = L / (c * M_g)  # dimensionless angular momentum

    rs_norm = 2.0
    # Maximum of V_eff for timelike particles occurs at
    # the unstable circular orbit radius.  Compute V_eff_max.
    r_unstable = find_unstable_circular(L, M, "timelike")
    V_max = V_eff_timelike(r_unstable, L, M)

    # Circular-orbit energy for given L (stable orbit)
    # For l^2 > 12, there is a stable circular orbit.
    # r_stable satisfies  l^2 = r^2 / (r - 3)   in units M_g
    # => r^2 - l^2 r + 3 l^2 = 0
    if l**2 > 12.0:
        disc = l**4 - 12.0 * l**2
        r_stable = (l**2 + np.sqrt(disc)) / 2.0  # in units M_g
        V_stable = (1.0 - 2.0 / r_stable) * (1.0 + l**2 / r_stable**2)
    else:
        V_stable = None

    # Classification logic
    # Radial equation: (dr/dtau)^2 = E^2 - V_eff
    # So the particle can overcome the potential barrier iff E^2 > V_max
    E2 = E**2

    if E >= 1.0:
        # Unbound: comes from infinity with E >= 1.
        # Particle starts at infinity where V_eff -> 1 and moves inward.
        # If E^2 < V_max the particle bounces off the barrier → deflection.
        # If E^2 > V_max the particle crosses the barrier and falls in → capture.
        if E2 > V_max:
            return "capture"
        else:
            return "deflection"
    else:
        # Bound energy E < 1
        if E2 > V_max:
            return "plunge"  # overcomes the potential barrier, falls in
        else:
            return "bound"  # trapped between turning points
