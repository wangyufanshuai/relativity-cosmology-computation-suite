"""Boundary identification for Penrose diagrams."""

import numpy as np


def identify_horizons(metric_func, r_range):
    """
    Identify event horizons as points where g_rr or g_tt changes sign.

    For Schwarzschild: g_tt = -(1 - 2M/r), g_rr = 1/(1 - 2M/r).
    The horizon is at r = 2M where both diverge / vanish.

    Parameters
    ----------
    metric_func : callable
        Function that takes r and returns dict of metric components.
    r_range : tuple (r_min, r_max, n)
        Range to scan for horizons.

    Returns
    -------
    list of float
        Radial coordinates where horizons are detected.
    """
    r_min, r_max, n = r_range
    r = np.linspace(r_min, r_max, int(n))
    dr = r[1] - r[0]

    horizons = []
    for i in range(len(r)):
        try:
            m = metric_func(r[i])
            g_tt = float(np.asarray(m["g_tt"]))
            g_rr = float(np.asarray(m["g_rr"]))

            # Horizon where g_tt changes sign or g_rr diverges
            if i > 0:
                m_prev = metric_func(r[i - 1])
                g_tt_prev = float(np.asarray(m_prev["g_tt"]))

                # g_tt sign change indicates horizon
                if g_tt_prev * g_tt < 0:
                    # Linear interpolation
                    r_horizon = r[i - 1] - g_tt_prev * dr / (g_tt - g_tt_prev)
                    horizons.append(r_horizon)
        except (KeyError, ValueError, ZeroDivisionError):
            continue

    return horizons


def identify_singularity(metric_func):
    """
    Identify curvature singularities by checking where curvature scalars diverge.

    For Schwarzschild, the Kretschner scalar is R_abcd R^abcd = 48 M^2 / r^6,
    which diverges at r = 0.

    Parameters
    ----------
    metric_func : callable
        Function that takes r and returns dict of metric components.

    Returns
    -------
    list of float
        Radial coordinates where singularities are detected (r=0 type).
    """
    # Check near r = 0 for divergence
    singularities = []

    # Test at very small r
    r_test = np.logspace(-6, 0, 100)
    for r in r_test:
        try:
            m = metric_func(r)
            # If metric components are inf or nan, we have a singularity
            for key, val in m.items():
                val = float(np.asarray(val))
                if not np.isfinite(val):
                    if r not in singularities:
                        singularities.append(r)
                    break
        except (ZeroDivisionError, ValueError):
            if r not in singularities:
                singularities.append(r)

    return singularities


def identify_infinities(metric_name):
    """
    Identify the types of infinities for a given spacetime.

    Parameters
    ----------
    metric_name : str
        Name of the spacetime ('schwarzschild', 'desitter', 'kerr', 'minkowski').

    Returns
    -------
    dict
        Dictionary with keys for infinity types and their Penrose locations.
    """
    infinities = {
        "schwarzschild": {
            "scri_plus": "future null infinity (u=pi/2 on X_p axis)",
            "scri_minus": "past null infinity (v=-pi/2 on X_p axis)",
            "i_plus": "future timelike infinity (T_p=pi/2, X_p=0)",
            "i_minus": "past timelike infinity (T_p=-pi/2, X_p=0)",
            "i_zero": "spacelike infinity (T_p=0, X_p=pi/2)",
        },
        "desitter": {
            "scri_plus": "future spacelike infinity",
            "scri_minus": "past spacelike infinity",
        },
        "kerr": {
            "scri_plus": "future null infinity",
            "scri_minus": "past null infinity",
            "i_plus": "future timelike infinity",
            "i_minus": "past timelike infinity",
            "i_zero": "spacelike infinity",
        },
        "minkowski": {
            "scri_plus": "future null infinity",
            "scri_minus": "past null infinity",
            "i_plus": "future timelike infinity",
            "i_minus": "past timelike infinity",
            "i_zero": "spacelike infinity",
        },
    }

    return infinities.get(metric_name, {})
