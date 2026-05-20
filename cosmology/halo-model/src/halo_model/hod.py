"""Halo Occupation Distribution (HOD) model."""

import numpy as np


def hod_n_galaxies(M, M_min, M1, alpha):
    """Compute the mean number of galaxies in a halo of mass M.

    Uses a simple HOD model with:
    - Central galaxy: step function at M_min
    - Satellite galaxies: power law above M1

    Parameters
    ----------
    M : array_like
        Halo mass(es).
    M_min : float
        Minimum halo mass for hosting a central galaxy.
    M1 : float
        Mass scale for satellite galaxies.
    alpha : float
        Power-law slope for satellite occupation.

    Returns
    -------
    array_like
        Mean number of galaxies <N>(M).
    """
    M = np.asarray(M, dtype=float)

    # Central galaxy: smooth step function
    sigma_logM = 0.3  # scatter in log M_min
    N_cen = 0.5 * (1.0 + scipy_special_erf_approx(np.log10(M / M_min) / sigma_logM))

    # Satellite galaxies: power law above M1
    N_sat = np.where(M > M1, (M / M1) ** alpha, 0.0)

    return N_cen + N_sat


def scipy_special_erf_approx(x):
    """Approximate erf using numpy operations (avoid scipy dependency for this)."""
    # Using the Abramowitz & Stegun approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x**2)
    return sign * y


def hod_occupation(M, params):
    """General HOD occupation function with parameter dict.

    Parameters
    ----------
    M : array_like
        Halo mass(es).
    params : dict
        HOD parameters: 'M_min', 'M1', 'alpha', optionally 'sigma_logM'.

    Returns
    -------
    array_like
        Mean galaxy occupation <N>(M).
    """
    M_min = params.get('M_min', 1e12)
    M1 = params.get('M1', 1e13)
    alpha = params.get('alpha', 1.0)
    sigma_logM = params.get('sigma_logM', 0.3)

    M = np.asarray(M, dtype=float)

    # Central galaxy
    N_cen = 0.5 * (1.0 + scipy_special_erf_approx(np.log10(M / M_min) / sigma_logM))

    # Satellite galaxies
    N_sat = np.where(M > M1, N_cen * (M / M1) ** alpha, 0.0)

    return N_cen + N_sat
