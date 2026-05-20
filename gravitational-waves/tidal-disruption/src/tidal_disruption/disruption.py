"""Standalone disruption-criterion functions for tidal disruption events."""

from .constants import M_sun
from .params import BlackHoleParams, StellarParams
from .radii import tidal_radius


def penetration_factor(
    star: StellarParams, bh: BlackHoleParams, r_pericenter: float
) -> float:
    """Dimensionless penetration factor beta = r_t / r_p.

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.
    bh : BlackHoleParams
        Black hole parameters.
    r_pericenter : float
        Pericenter distance of the stellar orbit [cm].

    Returns
    -------
    float
        Penetration factor beta (dimensionless).
    """
    return tidal_radius(star, bh) / r_pericenter


def critical_beta(star: StellarParams) -> float:
    """Critical penetration factor for full disruption.

    For a gamma = 5/3 polytrope beta_crit ~ 1.0-2.0 depending
    on the stellar structure. We adopt beta_crit = 1.85 following
    Mainetti et al. (2017) for a solar-type star.

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.

    Returns
    -------
    float
        Critical beta for full disruption.
    """
    mass_ratio = star.mass / M_sun
    if mass_ratio < 0.5:
        return 2.0
    elif mass_ratio < 1.5:
        return 1.85
    else:
        return 1.0 + 0.85 * (1.5 / mass_ratio)


def is_full_disruption(
    star: StellarParams, bh: BlackHoleParams, r_pericenter: float
) -> bool:
    """Check whether the encounter leads to full disruption.

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.
    bh : BlackHoleParams
        Black hole parameters.
    r_pericenter : float
        Pericenter distance [cm].

    Returns
    -------
    bool
        True if beta > beta_crit (full disruption).
    """
    return penetration_factor(star, bh, r_pericenter) >= critical_beta(star)


def partial_disruption_fraction(
    star: StellarParams, bh: BlackHoleParams, r_pericenter: float
) -> float:
    """Mass fraction stripped during a partial disruption.

    For beta < beta_crit only a fraction of the stellar envelope
    is removed; the core survives.

    Parameters
    ----------
    star : StellarParams
        Stellar parameters.
    bh : BlackHoleParams
        Black hole parameters.
    r_pericenter : float
        Pericenter distance [cm].

    Returns
    -------
    float
        Mass fraction stripped (0 to 1). Returns 1.0 for full disruption.
    """
    beta = penetration_factor(star, bh, r_pericenter)
    beta_crit = critical_beta(star)
    if beta >= beta_crit:
        return 1.0
    if beta <= 0.5:
        return 0.0
    # Approximate mass-loss curve from Guillouchon & Ramirez-Ruiz (2013)
    frac = ((beta - 0.5) / (beta_crit - 0.5)) ** 2.5
    return min(frac, 1.0)
