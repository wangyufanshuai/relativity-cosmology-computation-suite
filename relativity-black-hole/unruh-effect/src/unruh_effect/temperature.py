"""
Unruh temperature and thermal spectrum.

A uniformly accelerating observer with proper acceleration *a* perceives
the Minkowski vacuum as a thermal bath at the Unruh temperature

    T_U = a * hbar / (2 * pi * c * k_B)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import C, HBAR, K_B

ArrayLike = float | NDArray[np.floating]


def unruh_temperature(a: ArrayLike) -> ArrayLike:
    """Compute the Unruh temperature for a given proper acceleration.

    Parameters
    ----------
    a : float or array_like
        Proper acceleration [m/s^2].

    Returns
    -------
    T : float or ndarray
        Unruh temperature [K].
    """
    a = np.asarray(a, dtype=float)
    return a * HBAR / (2.0 * np.pi * C * K_B)


def inverse_unruh_temperature(T: ArrayLike) -> ArrayLike:
    """Compute the acceleration needed to reach a given Unruh temperature.

    Parameters
    ----------
    T : float or array_like
        Temperature [K].

    Returns
    -------
    a : float or ndarray
        Proper acceleration [m/s^2].
    """
    T = np.asarray(T, dtype=float)
    return T * 2.0 * np.pi * C * K_B / HBAR


def thermal_spectrum(omega: ArrayLike, T: ArrayLike) -> ArrayLike:
    """Bose-Einstein (Planck) thermal occupation number.

    n(omega) = 1 / (exp(hbar * omega / (k_B * T)) - 1)

    Parameters
    ----------
    omega : float or array_like
        Angular frequency [rad/s].
    T : float or array_like
        Temperature [K].

    Returns
    -------
    n : float or ndarray
        Mean occupation number.  Returns 0 when T == 0.
    """
    omega = np.asarray(omega, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", over="ignore"):
        exponent = HBAR * omega / (K_B * T)
        n = 1.0 / (np.exp(exponent) - 1.0)
    # Handle T=0: n=0
    n = np.where(np.isfinite(n), n, 0.0)
    return n
