"""Standalone radius functions for tidal disruption physics."""

from .constants import G, c
from .params import BlackHoleParams, StellarParams


def tidal_radius(star: StellarParams, bh: BlackHoleParams) -> float:
    """Classical tidal radius r_t = R_* (M_BH / M_*)^{1/3}.

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.
    bh : BlackHoleParams
        Black hole parameters.

    Returns
    -------
    float
        Tidal radius in cm.
    """
    return star.radius * (bh.mass / star.mass) ** (1.0 / 3.0)


def schwarzschild_radius(bh: BlackHoleParams) -> float:
    """Schwarzschild radius r_S = 2 G M_BH / c^2.

    Parameters
    ----------
    bh : BlackHoleParams
        Black hole parameters.

    Returns
    -------
    float
        Schwarzschild radius in cm.
    """
    return 2.0 * G * bh.mass / c ** 2


def isco_radius(bh: BlackHoleParams) -> float:
    """Inner-most stable circular orbit radius.

    For a non-spinning BH this is 6 GM/c^2 = 3 r_S.
    Uses the exact Kerr (prograde) expression when spin > 0.

    Parameters
    ----------
    bh : BlackHoleParams
        Black hole parameters.

    Returns
    -------
    float
        ISCO radius in cm.
    """
    r_g = G * bh.mass / c ** 2
    a = bh.spin
    z1 = 1.0 + (1.0 - a ** 2) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    z2 = (3.0 * a ** 2 - z1 ** 2) ** 0.5
    r_isco = r_g * (3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)) ** 0.5)
    return r_isco


def hill_radius(
    star: StellarParams, bh: BlackHoleParams, semi_major_axis: float, eccentricity: float
) -> float:
    """Hill radius for a star on an orbit around the BH.

    r_Hill = a (1 - e) (m_star / (3 M_BH))^{1/3}

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.
    bh : BlackHoleParams
        Black hole parameters.
    semi_major_axis : float
        Semi-major axis of the stellar orbit [cm].
    eccentricity : float
        Orbital eccentricity.

    Returns
    -------
    float
        Hill radius in cm.
    """
    return semi_major_axis * (1.0 - eccentricity) * (
        star.mass / (3.0 * bh.mass)
    ) ** (1.0 / 3.0)
