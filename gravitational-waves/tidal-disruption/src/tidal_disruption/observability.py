"""Standalone observability functions for tidal disruption events."""

from .constants import G, c
from .params import BlackHoleParams, StellarParams
from .radii import schwarzschild_radius, tidal_radius


def energy_spread(star: StellarParams, bh: BlackHoleParams) -> float:
    """Specific orbital energy spread across the debris stream.

    Delta E ~ G M_BH R_* / r_t^2

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.
    bh : BlackHoleParams
        Black hole parameters.

    Returns
    -------
    float
        Energy spread [erg/g] (specific energy).
    """
    r_t = tidal_radius(star, bh)
    return G * bh.mass * star.radius / r_t ** 2


def is_outside_horizon(star: StellarParams, bh: BlackHoleParams) -> bool:
    """Check that the tidal radius lies outside the event horizon.

    A TDE is only observable if r_t > r_Schwarzschild (or more
    precisely r_t > r_ISCO). For main-sequence stars this requires
    M_BH < ~10^8 M_sun.

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.
    bh : BlackHoleParams
        Black hole parameters.

    Returns
    -------
    bool
        True if the star is tidally disrupted outside the horizon.
    """
    return tidal_radius(star, bh) > schwarzschild_radius(bh)


def maximum_bh_mass_for_tde(star: StellarParams) -> float:
    """Maximum BH mass [g] for which a TDE is possible (r_t > r_S).

    Solves r_t(M_max) = r_S(M_max) for M_BH, given the stellar
    structure. For a main-sequence star R_* ~ M_*^{0.8} this gives
    M_max ~ 10^8 M_sun.

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.

    Returns
    -------
    float
        Maximum BH mass in grams.
    """
    return (star.radius * c ** 2 / (2.0 * G)) ** 1.5 / star.mass ** 0.5
