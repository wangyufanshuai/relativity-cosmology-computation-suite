"""Modified gravity effects in void environments."""

import numpy as np


def void_environment_screening(M_eff, void_radius, rho_mean):
    """Compute the screening factor in a void environment.

    In voids, the low density means suppressed screening, allowing
    modified gravity (fifth force) to operate more freely.

    Parameters
    ----------
    M_eff : float
        Effective mass scale for screening (in solar masses).
    void_radius : float
        Void effective radius (in Mpc/h).
    rho_mean : float
        Mean matter density.

    Returns
    -------
    float
        Screening factor between 0 (fully screened) and 1 (unscreened).
        In voids this is typically close to 1 (unscreened).
    """
    # Vainshtein-like screening: suppressed in overdense environments
    # In voids, the enclosed mass is low -> screening is weak
    if void_radius <= 0:
        return 0.0

    # Enclosed mass estimate
    M_enclosed = (4.0 / 3.0) * np.pi * void_radius**3 * rho_mean * 0.1  # void is ~10% mean

    # Screening ratio: lower ratio means less screening
    # When M_enclosed << M_eff, environment is unscreened
    screening_ratio = M_enclosed / (M_eff + 1e-30)

    # Screening factor: 1 - exp(-ratio) -> approaches 0 for large M_eff/M_enclosed
    screening_factor = 1.0 - np.exp(-screening_ratio)

    return float(np.clip(screening_factor, 0.0, 1.0))


def fifth_force_profile(r, void_radius, M_eff):
    """Compute the fifth force enhancement profile within a void.

    Parameters
    ----------
    r : array_like
        Radial distance(s) from void center.
    void_radius : float
        Void effective radius.
    M_eff : float
        Effective mass/compliness parameter controlling fifth force range.

    Returns
    -------
    array_like
        Enhancement factor (1 + delta_fifth) where delta_fifth is the
        fifth force contribution relative to Newtonian gravity.
    """
    r = np.asarray(r, dtype=float)

    if void_radius <= 0:
        return np.ones_like(r)

    r_norm = r / void_radius

    # Fifth force is enhanced inside voids where screening is suppressed
    # Model: enhancement peaks at center and decays toward edge
    # Using a simple model: delta_fifth ~ (1 - (r/R)^2) * (R/r_s)^2 / (1 + (R/r_s)^2)
    # where r_s = sqrt(M_eff / (4*pi*rho_mean)) is a screening length
    r_s = np.sqrt(M_eff) * 0.01  # simplified screening length

    enhancement = 2.0 * (1.0 - r_norm**2) * np.exp(-r_norm**2)

    # Outside the void, no enhancement
    outside = r_norm > 1.0
    if np.any(outside):
        enhancement[outside] = 0.0

    return 1.0 + enhancement
