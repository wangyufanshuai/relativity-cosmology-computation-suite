"""Kerr metric functions: horizon, ergosphere, ISCO, and frame dragging.

All quantities use **geometric units** (G = c = 1) and are expressed in
units of the black hole mass *M*.  The spin parameter *a* has dimensions of
mass (0 <= a <= M).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def kerr_horizon(M: float, a: float) -> float:
    """Outer event horizon of a Kerr black hole.

    Parameters
    ----------
    M : float
        Black hole mass (geometric units).
    a : float
        Spin parameter |J|/M (0 <= a <= M).

    Returns
    -------
    float
        Outer horizon radius r_+ = M + sqrt(M^2 - a^2).
    """
    discriminant = M * M - a * a
    if discriminant < 0:
        raise ValueError(f"No horizon exists for a={a} > M={M} (naked singularity).")
    return M + np.sqrt(discriminant)


def kerr_ergosphere(M: float, a: float, theta: float) -> float:
    """Ergosphere (outer stationary-surface) radius.

    r_ergo = M + sqrt(M^2 - a^2 cos^2(theta))

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter (0 <= a <= M).
    theta : float
        Polar angle in radians.

    Returns
    -------
    float
        Ergosphere boundary radius.
    """
    discriminant = M * M - (a * np.cos(theta)) ** 2
    if discriminant < 0:
        raise ValueError(
            f"No ergosphere for a={a}, theta={theta}: discriminant={discriminant}."
        )
    return M + np.sqrt(discriminant)


def kerr_isco(M: float, a: float, prograde: bool = True) -> float:
    """Innermost stable circular orbit (ISCO) radius for Kerr spacetime.

    Uses the analytic Bardeen, Press & Teukolsky (1972) formula.

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter (0 <= a <= M).
    prograde : bool
        If True, compute prograde ISCO; otherwise retrograde.

    Returns
    -------
    float
        ISCO radius in units of M.
    """
    # Normalised spin
    a_star = a / M if M != 0 else 0.0
    if not -1.0 <= a_star <= 1.0:
        raise ValueError(f"Dimensionless spin a/M = {a_star} out of range [-1, 1].")

    sign = 1.0 if prograde else -1.0

    z1 = 1.0 + (1.0 - a_star**2) ** (1.0 / 3.0) * (
        (1.0 + a_star) ** (1.0 / 3.0) + (1.0 - a_star) ** (1.0 / 3.0)
    )
    z2 = np.sqrt(3.0 * a_star**2 + z1**2)

    r_isco = M * (3.0 + z2 - sign * np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)))
    return r_isco


def frame_dragging_omega(M: float, a: float, r: float, theta: float = np.pi / 2) -> float:
    """Frame-dragging angular velocity (equatorial plane simplification).

    omega = 2 M a r / [(r^2 + a^2)^2 - a^2 Delta sin^2(theta)]

    where Delta = r^2 - 2Mr + a^2.

    At the equator (theta = pi/2) the denominator simplifies to
    (r^2 + a^2)^2 - a^2 Delta.

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    r : float
        Boyer-Lindquist radial coordinate.
    theta : float
        Polar angle (default pi/2, equatorial plane).

    Returns
    -------
    float
        Frame-dragging angular velocity omega (geometric units).
    """
    Delta = r**2 - 2.0 * M * r + a**2
    Sigma = r**2 + (a * np.cos(theta)) ** 2
    A = (r**2 + a**2) ** 2 - a**2 * Delta * np.sin(theta) ** 2
    omega = 2.0 * M * a * r / A
    return omega
