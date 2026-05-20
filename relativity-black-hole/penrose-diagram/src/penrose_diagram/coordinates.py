"""Coordinate transformations to Penrose-Carter conformal coordinates."""

import numpy as np


def kruskal_to_penrose(U, V):
    """
    Transform Kruskal-Szekeres coordinates (U, V) to Penrose coordinates (T_p, X_p).

    The compactification is:
        T_p = arctan(V) + arctan(U)
        X_p = arctan(V) - arctan(U)

    This maps the infinite Kruskal coordinates to a finite diamond.

    Parameters
    ----------
    U, V : float or ndarray
        Kruskal-Szekeres coordinates.

    Returns
    -------
    T_p, X_p : ndarray
        Penrose diagram coordinates, bounded in [-pi, pi].
    """
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)

    T_p = np.arctan(V) + np.arctan(U)
    X_p = np.arctan(V) - np.arctan(U)

    return T_p, X_p


def tortoise_to_penrose(r_star, t_star):
    """
    Transform tortoise coordinates (r*, t*) to Penrose coordinates.

    Uses the standard compactification:
        T_p = arctan(t_star + r_star) + arctan(t_star - r_star)
        X_p = arctan(t_star + r_star) - arctan(t_star - r_star)

    Parameters
    ----------
    r_star : float or ndarray
        Tortoise radial coordinate.
    t_star : float or ndarray
        Tortoise time coordinate.

    Returns
    -------
    T_p, X_p : ndarray
        Penrose diagram coordinates.
    """
    r_star = np.asarray(r_star, dtype=float)
    t_star = np.asarray(t_star, dtype=float)

    u = t_star + r_star
    v = t_star - r_star

    T_p = np.arctan(v) + np.arctan(u)
    X_p = np.arctan(u) - np.arctan(v)

    return T_p, X_p


def schwarzschild_conformal(r, t, M):
    """
    Transform Schwarzschild coordinates (r, t) to conformal (Penrose) coordinates.

    Uses the tortoise coordinate r* = r + 2M ln|r/(2M) - 1| and then
    compactifies with arctan.

    Parameters
    ----------
    r : float or ndarray
        Schwarzschild radial coordinate (r > 2M for exterior).
    t : float or ndarray
        Schwarzschild time coordinate.
    M : float
        Black hole mass (Schwarzschild radius = 2M).

    Returns
    -------
    T_p, X_p : ndarray
        Penrose diagram coordinates.
    """
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)

    rs = 2.0 * M

    # Tortoise coordinate: r* = r + 2M ln|r/(2M) - 1|
    # Only valid for r > 2M (exterior region)
    ratio = r / rs - 1.0
    ratio = np.where(ratio > 0, ratio, 1e-10)
    r_star = r + rs * np.log(ratio)

    # Null coordinates
    u = t + r_star
    v = t - r_star

    # Compactify
    T_p = np.arctan(v) + np.arctan(u)
    X_p = np.arctan(u) - np.arctan(v)

    return T_p, X_p
