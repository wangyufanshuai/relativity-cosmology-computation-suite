"""Penrose diagram generation for specific spacetimes."""

import numpy as np

from .coordinates import kruskal_to_penrose, schwarzschild_conformal


def penrose_points_schwarzschild(M):
    """
    Generate key boundary points for the Schwarzschild Penrose diagram.

    The Schwarzschild diagram is a diamond shape with:
    - Future/past null infinity at 45-degree lines
    - Singularity at r=0 as a horizontal wavy line at top/bottom
    - Horizon at 45-degree lines through the center

    Parameters
    ----------
    M : float
        Black hole mass.

    Returns
    -------
    dict
        Dictionary of boundary points in (T_p, X_p) coordinates.
    """
    pi = np.pi
    pi2 = pi / 2.0

    # The Schwarzschild Penrose diagram is a square rotated 45 degrees
    # In (T_p, X_p) coordinates, the boundaries are:
    points = {
        # Future timelike infinity
        "i_plus": (pi2, 0.0),
        # Past timelike infinity
        "i_minus": (-pi2, 0.0),
        # Spacelike infinity
        "i_zero": (0.0, pi2),
        # Future null infinity endpoints
        "scri_plus_top": (pi2, 0.0),
        "scri_plus_right": (0.0, pi2),
        # Past null infinity endpoints
        "scri_minus_bottom": (-pi2, 0.0),
        "scri_minus_right": (0.0, pi2),
        # Bifurcation point (center of horizon)
        "bifurcation": (0.0, 0.0),
        # Singularity (future) - top of diamond
        "singularity_future_left": (pi2, 0.0),
        "singularity_future_right": (pi2, 0.0),
        # Singularity (past) - bottom of diamond (white hole)
        "singularity_past_left": (-pi2, 0.0),
        "singularity_past_right": (-pi2, 0.0),
    }

    # Generate the horizon lines (45-degree lines)
    t_horizon = np.linspace(-pi2, pi2, 100)
    horizon_future = {
        "T": t_horizon,
        "X": np.zeros_like(t_horizon) + pi2 * 0.0,  # Simplified
    }

    return points


def penrose_points_desitter(H):
    """
    Generate key boundary points for the de Sitter Penrose diagram.

    De Sitter spacetime has a square Penrose diagram with spacelike
    future and past boundaries.

    Parameters
    ----------
    H : float
        Hubble parameter (related to cosmological constant).

    Returns
    -------
    dict
        Dictionary of boundary points in (T_p, X_p) coordinates.
    """
    pi = np.pi
    pi2 = pi / 2.0

    points = {
        # Future spacelike infinity (top of square)
        "future_infinity_T": pi2,
        "future_infinity_X_range": (-pi2, pi2),
        # Past spacelike infinity (bottom of square)
        "past_infinity_T": -pi2,
        "past_infinity_X_range": (-pi2, pi2),
        # Corners
        "top_right": (pi2, pi2),
        "top_left": (pi2, -pi2),
        "bottom_right": (-pi2, pi2),
        "bottom_left": (-pi2, -pi2),
        # Cosmological horizon
        "horizon_T_range": (-pi2, pi2),
        "horizon_X": pi2,
    }

    return points


def penrose_points_kerr(M, a):
    """
    Generate key boundary points for the Kerr Penrose diagram.

    The Kerr diagram is more complex with an inner horizon (Cauchy)
    and an outer horizon (event), and a ring singularity.

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter (0 <= a <= M).

    Returns
    -------
    dict
        Dictionary of boundary points in (T_p, X_p) coordinates.
    """
    pi = np.pi
    pi2 = pi / 2.0

    # Kerr horizons: r_plus (outer) and r_minus (inner)
    r_plus = M + np.sqrt(M ** 2 - a ** 2) if a <= M else M
    r_minus = M - np.sqrt(M ** 2 - a ** 2) if a <= M else M

    points = {
        # Outer (event) horizon
        "r_plus": r_plus,
        # Inner (Cauchy) horizon
        "r_minus": r_minus,
        # Key diagram points (similar to Schwarzschild but extended)
        "i_plus": (pi2, 0.0),
        "i_minus": (-pi2, 0.0),
        "i_zero": (0.0, pi2),
        "bifurcation_outer": (0.0, 0.0),
        "ring_singularity_r": 0.0,
        "spin_parameter": a,
        # Ergosphere outer boundary
        "r_ergo_equatorial": 2.0 * M,
    }

    return points
