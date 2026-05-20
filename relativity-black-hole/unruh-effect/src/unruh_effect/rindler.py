"""
Rindler coordinate transformations and metric.

A uniformly accelerating observer with proper acceleration *a* follows a
hyperbolic worldline in Minkowski spacetime.  The Rindler coordinates
(eta, xi) cover the right Rindler wedge (x > |t|) and are defined by

    t = xi * sinh(eta)
    x = xi * cosh(eta)          (xi > 0)

where eta is the dimensionless Rindler time and xi has units of length.
The proper acceleration of the observer at fixed xi is a_phys = 1 / xi,
so that xi = 1/a corresponds to the observer with proper acceleration a.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayLike = float | NDArray[np.floating]


# ---------------------------------------------------------------------------
# Coordinate transformations
# ---------------------------------------------------------------------------

def minkowski_to_rindler(
    t: ArrayLike,
    x: ArrayLike,
    a: float,
) -> tuple[ArrayLike, ArrayLike]:
    """Convert Minkowski (t, x) to Rindler (eta, xi).

    Parameters
    ----------
    t, x : array_like
        Minkowski coordinates (same shape).  Must satisfy x**2 - t**2 > 0
        (right Rindler wedge, x > 0).
    a : float
        Proper acceleration parameter.  The map is defined so that the
        observer with proper acceleration *a* sits at xi = 1/a.

    Returns
    -------
    eta, xi : same type as inputs
        Rindler time and spatial coordinate.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    xi = np.sqrt(x**2 - t**2)
    eta = np.arctanh(t / x)
    return eta, xi


def rindler_to_minkowski(
    eta: ArrayLike,
    xi: ArrayLike,
    a: float,
) -> tuple[ArrayLike, ArrayLike]:
    """Convert Rindler (eta, xi) to Minkowski (t, x).

    Parameters
    ----------
    eta : array_like
        Dimensionless Rindler time.
    xi : array_like
        Rindler spatial coordinate (must be > 0).
    a : float
        Proper acceleration parameter (not used algebraically but kept for
        a consistent interface).

    Returns
    -------
    t, x : same type as inputs
        Minkowski coordinates.
    """
    eta = np.asarray(eta, dtype=float)
    xi = np.asarray(xi, dtype=float)
    t = xi * np.sinh(eta)
    x = xi * np.cosh(eta)
    return t, x


# ---------------------------------------------------------------------------
# Metric and curvature quantities
# ---------------------------------------------------------------------------

def rindler_metric(xi: ArrayLike, a: float) -> NDArray[np.floating]:
    """Diagonal metric components of the Rindler line element.

    The Rindler metric is

        ds^2 = -(a * xi)^2 d(eta)^2 + d(xi)^2

    (the transverse coordinates y, z are flat and omitted).

    Parameters
    ----------
    xi : array_like
        Rindler spatial coordinate.
    a : float
        Acceleration parameter.

    Returns
    -------
    g : ndarray of shape (*xi.shape, 2, 2)
        Metric tensor g_{mu nu} in the (eta, xi) basis.
    """
    xi = np.asarray(xi, dtype=float)
    scalar = True
    if xi.ndim == 0:
        xi = xi.reshape(1)
        scalar = False

    n = xi.size
    g = np.zeros((n, 2, 2))
    g[:, 0, 0] = -(a * xi) ** 2   # g_{eta eta}
    g[:, 1, 1] = 1.0              # g_{xi xi}

    if scalar:
        return g
    return g.reshape((2, 2))


def rindler_christoffel(a: float) -> NDArray[np.floating]:
    """Non-zero Christoffel symbols for the 2-D Rindler metric.

    With coordinates (x^0, x^1) = (eta, xi) the only non-vanishing
    Christoffel symbols are

        Gamma^0_{01} = Gamma^0_{10} =  1 / xi
        Gamma^1_{00} =  a^2 * xi

    Returns
    -------
    Gamma : dict
        Keys are ``(upper, lower1, lower2)`` tuples of coordinate indices
        (0 = eta, 1 = xi), values are SymPy-like expressions as strings
        plus numerical helpers.  For convenience the full 2x2x2 array is
        also returned as a callable factory.

        The return value is the 2x2x2 array ``Gamma[mu][alpha][beta]``
        given *xi* and *a*.
    """
    # Return a closure that evaluates Christoffels at a given xi.
    def _christoffel(xi: float) -> NDArray[np.floating]:
        G = np.zeros((2, 2, 2))
        G[0, 0, 1] = 1.0 / xi
        G[0, 1, 0] = 1.0 / xi
        G[1, 0, 0] = a**2 * xi
        return G

    return _christoffel
