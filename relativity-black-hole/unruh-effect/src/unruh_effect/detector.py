"""
Unruh-DeWitt detector model.

An Unruh-DeWitt detector is a pointlike two-level system (energy gap
hbar * omega) coupled linearly to a scalar field.  When carried along a
uniformly accelerated trajectory the detector clicks as if immersed in a
thermal bath at the Unruh temperature.

Key quantities
--------------
* Wightman function  G+(x, x') = <0| phi(x) phi(x') |0>
* Detector transition rate  R(omega) per unit proper time
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from numpy.typing import NDArray

from .constants import C, HBAR, K_B
from .rindler import rindler_to_minkowski
from .temperature import unruh_temperature, thermal_spectrum

ArrayLike = float | NDArray[np.floating]

_EPS = 1e-12  # small regulator to avoid singularities


# ---------------------------------------------------------------------------
# Wightman functions
# ---------------------------------------------------------------------------

def wightman_function_minkowski(
    t1: ArrayLike,
    x1: ArrayLike,
    t2: ArrayLike,
    x2: ArrayLike,
) -> ArrayLike:
    """Massless scalar field Wightman function in (1+1)-D Minkowski space.

    G+(x, x') = -1 / (4 * pi^2) * 1 / ((Delta t - i eps)^2 - (Delta x)^2)

    with an infinitesimal imaginary part in Delta t for time-ordering.

    Parameters
    ----------
    t1, x1 : coordinates of first point
    t2, x2 : coordinates of second point

    Returns
    -------
    G : complex ndarray
    """
    dt = np.asarray(t1, dtype=float) - np.asarray(t2, dtype=float)
    dx = np.asarray(x1, dtype=float) - np.asarray(x2, dtype=float)
    # Add small imaginary part for Feynman prescription
    dt_complex = dt - 1j * _EPS
    denom = dt_complex**2 - dx**2
    return -1.0 / (4.0 * np.pi**2) / denom


def wightman_function_rindler(
    eta1: ArrayLike,
    xi1: ArrayLike,
    eta2: ArrayLike,
    xi2: ArrayLike,
    a: float,
) -> ArrayLike:
    """Wightman function in Rindler coordinates for the right wedge.

    The worldline of a uniformly accelerated observer at fixed xi = 1/a
    is substituted into the Minkowski Wightman function.  The result is
    the pullback expressed in Rindler coordinates.

    Parameters
    ----------
    eta1, xi1 : Rindler coordinates of first point
    eta2, xi2 : Rindler coordinates of second point
    a : proper acceleration parameter

    Returns
    -------
    G : complex ndarray
    """
    t1, x1 = rindler_to_minkowski(eta1, xi1, a)
    t2, x2 = rindler_to_minkowski(eta2, xi2, a)
    return wightman_function_minkowski(t1, x1, t2, x2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _response_analytical(omega: float, T_U: float) -> float:
    """Closed-form transition rate R(omega) = omega/(2pi) * n(omega, T_U).

    Numerically stable for all parameter ranges.
    """
    if omega == 0.0:
        return 0.0
    if T_U <= 0:
        return 0.0
    x = HBAR * abs(omega) / (K_B * T_U)
    if x > 500.0:
        rate = abs(omega) / (2.0 * np.pi) * np.exp(-x)
    else:
        rate = abs(omega) / (2.0 * np.pi) / np.expm1(x)
    return float(rate)


# ---------------------------------------------------------------------------
# Detector response
# ---------------------------------------------------------------------------

def detector_response_rate(
    omega: float,
    a: float,
    method: str = "numerical",
) -> float:
    """Transition rate R(omega) of an Unruh-DeWitt detector.

    For a detector on a uniformly accelerated worldline (proper acceleration
    *a*) the transition rate per unit proper time to first order in the
    coupling constant is

        R(omega) = omega / (2 pi) / (exp(hbar omega / k_B T_U) - 1)

    where T_U = a hbar / (2 pi c k_B) is the Unruh temperature.

    Parameters
    ----------
    omega : float
        Detector energy gap [rad/s].
    a : float
        Proper acceleration [m/s^2].
    method : str
        ``'analytical'`` for closed-form Planckian result,
        ``'numerical'`` for series-sum evaluation (converges to same result).

    Returns
    -------
    R : float
        Transition rate [1/s].
    """
    if a <= 0:
        return 0.0

    T_U = float(unruh_temperature(a))

    if method == "analytical":
        return _response_analytical(omega, T_U)

    # Numerical evaluation via the convergent series representation.
    #
    # The Planck factor can be expanded as a geometric series:
    #
    #   1/(exp(x) - 1) = sum_{n=1}^{inf} exp(-n*x)
    #
    # So  R(omega) = omega/(2pi) * sum_{n=1}^{inf} exp(-n * hbar*omega / (k_B * T_U))
    #
    # with x = hbar*omega/(k_B*T_U) = 2*pi*c*omega/a.
    # The series converges rapidly when x > 0 (which is always the case
    # for omega != 0 and a > 0).

    if omega == 0.0:
        return 0.0

    x = HBAR * abs(omega) / (K_B * T_U)

    if x > 500:
        # Only the first term matters
        return float(abs(omega) / (2.0 * np.pi) * np.exp(-x))

    # Sum the series until terms become negligible
    total = 0.0
    n = 1
    while True:
        term = np.exp(-n * x)
        total += term
        if term < 1e-15 * total or n > 10000:
            break
        n += 1

    return float(abs(omega) / (2.0 * np.pi) * total)


def detector_response_function(
    omega_array: NDArray[np.floating],
    a: float,
) -> NDArray[np.floating]:
    """Compute the detector response function over an array of frequencies.

    Parameters
    ----------
    omega_array : ndarray
        Detector energy gaps [rad/s].
    a : float
        Proper acceleration [m/s^2].

    Returns
    -------
    R_array : ndarray
        Transition rates [1/s].
    """
    omega_array = np.asarray(omega_array, dtype=float)
    return np.array(
        [detector_response_rate(float(w), a) for w in omega_array.flat]
    ).reshape(omega_array.shape)
