"""Fingers-of-God damping models."""

import numpy as np


def lorentzian_fog(k, mu, sigma_v):
    """Lorentzian (Cauchy) FOG damping factor.

    Parameters
    ----------
    k : array_like
        Wavenumber.
    mu : array_like
        Cosine of angle between wavevector and line of sight.
    sigma_v : float
        Velocity dispersion.

    Returns
    -------
    array_like
        Damping factor 1 / (1 + (k * mu * sigma_v)^2).
    """
    return 1.0 / (1.0 + (k * mu * sigma_v) ** 2)


def gaussian_fog(k, mu, sigma_v):
    """Gaussian FOG damping factor.

    Parameters
    ----------
    k : array_like
        Wavenumber.
    mu : array_like
        Cosine of angle between wavevector and line of sight.
    sigma_v : float
        Velocity dispersion.

    Returns
    -------
    array_like
        Damping factor exp(-(k * mu * sigma_v)^2).
    """
    return np.exp(-(k * mu * sigma_v) ** 2)


def combined_rsd(k, mu, P_real, beta, sigma_v):
    """Combined Kaiser + Lorentzian FOG redshift-space power spectrum.

    Parameters
    ----------
    k : array_like
        Wavenumber.
    mu : array_like
        Cosine of angle between wavevector and line of sight.
    P_real : array_like or float
        Real-space power spectrum.
    beta : float
        Growth-rate parameter.
    sigma_v : float
        Velocity dispersion for FOG damping.

    Returns
    -------
    array_like
        P(k, mu) = (1 + beta*mu^2)^2 * D(k,mu) * P_real(k).
    """
    from .kaiser import kaiser_factor

    kaiser = kaiser_factor(k, mu, beta)
    fog = lorentzian_fog(k, mu, sigma_v)
    return kaiser * fog * P_real
