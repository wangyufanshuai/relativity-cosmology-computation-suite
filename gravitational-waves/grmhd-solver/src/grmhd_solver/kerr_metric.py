"""Kerr spacetime metric functions.

Implements the Kerr metric in Boyer-Lindquist coordinates:
    ds^2 = -(1 - 2Mr/Sigma) dt^2 - (4Mar sin^2(theta)/Sigma) dt dphi
           + (Sigma/Delta) dr^2 + Sigma dtheta^2
           + ((r^2+a^2)^2 - a^2 Delta sin^2(theta))/Sigma * sin^2(theta) dphi^2

where:
    Delta = r^2 - 2Mr + a^2
    Sigma = r^2 + a^2 cos^2(theta)
"""

import numpy as np
from numpy.typing import NDArray


def delta(r: float | NDArray, M: float, a: float) -> float | NDArray:
    """Kerr metric function Delta = r^2 - 2Mr + a^2."""
    return r**2 - 2.0 * M * r + a**2


def sigma(r: float | NDArray, theta: float, a: float) -> float | NDArray:
    """Kerr metric function Sigma = r^2 + a^2 cos^2(theta)."""
    return r**2 + a**2 * np.cos(theta)**2


def boyer_lindquist_radius(r_schwarzschild: float) -> float:
    """Convert Schwarzschild radial coordinate to Boyer-Lindquist (identity for a=0)."""
    return r_schwarzschild


def kerr_metric_coefficients(
    r: float | NDArray,
    theta: float,
    M: float = 1.0,
    a: float = 0.0,
) -> dict:
    """Return the covariant Kerr metric components at (r, theta).

    Parameters
    ----------
    r : float or array
        Boyer-Lindquist radial coordinate.
    theta : float
        Polar angle [rad].
    M : float
        Black hole mass.
    a : float
        Spin parameter (|a| <= M).

    Returns
    -------
    dict with keys 'g_tt', 'g_tphi', 'g_rr', 'g_thth', 'g_phiphi'.
    """
    Sig = sigma(r, theta, a)
    D = delta(r, M, a)

    g_tt = -(1.0 - 2.0 * M * r / Sig)
    g_tphi = -2.0 * M * a * r * np.sin(theta)**2 / Sig
    g_rr = Sig / D
    g_thth = Sig
    A = (r**2 + a**2)**2 - a**2 * D * np.sin(theta)**2
    g_phiphi = A * np.sin(theta)**2 / Sig

    return {
        "g_tt": g_tt,
        "g_tphi": g_tphi,
        "g_rr": g_rr,
        "g_thth": g_thth,
        "g_phiphi": g_phiphi,
    }


def event_horizon(M: float, a: float) -> float:
    """Outer event horizon radius r_+ = M + sqrt(M^2 - a^2)."""
    return M + np.sqrt(M**2 - a**2)


def ergosphere_radius(M: float, a: float, theta: float) -> float:
    """Ergosphere outer boundary r_ergo = M + sqrt(M^2 - a^2 cos^2(theta))."""
    return M + np.sqrt(M**2 - a**2 * np.cos(theta)**2)
