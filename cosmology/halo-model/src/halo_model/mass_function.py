"""Halo mass functions."""

import numpy as np
from scipy import integrate


def press_schechter(nu):
    """Press-Schechter mass function f(nu).

    Parameters
    ----------
    nu : array_like
        Peak height nu = delta_c / sigma(M).

    Returns
    -------
    array_like
        f(nu) = sqrt(2/pi) * nu * exp(-nu^2/2).
    """
    nu = np.asarray(nu, dtype=float)
    return np.sqrt(2.0 / np.pi) * nu * np.exp(-nu**2 / 2.0)


def sheth_tormen(nu):
    """Sheth-Tormen mass function f(nu).

    Parameters
    ----------
    nu : array_like
        Peak height nu = delta_c / sigma(M).

    Returns
    -------
    array_like
        f(nu) = A * sqrt(2*a/pi) * nu * (1 + (a*nu^2)^(-p)) * exp(-a*nu^2/2).
    """
    nu = np.asarray(nu, dtype=float)
    a = 0.707
    p = 0.3
    A = 0.3222
    return A * np.sqrt(2.0 * a / np.pi) * nu * (1.0 + (a * nu**2) ** (-p)) * np.exp(-a * nu**2 / 2.0)


def halo_mass_function(M_array, sigma_func, rho_mean):
    """Compute dn/dln(M) for an array of masses.

    Parameters
    ----------
    M_array : array_like
        Array of halo masses.
    sigma_func : callable
        Function sigma(M) returning the variance of the density field
        smoothed on scale corresponding to mass M.
    rho_mean : float
        Mean matter density of the universe.

    Returns
    -------
    array_like
        Mass function dn/dlnM = f(nu) * rho_mean / M.
    """
    M_array = np.asarray(M_array, dtype=float)
    sigma = sigma_func(M_array)
    delta_c = 1.686
    nu = delta_c / sigma

    f_nu = sheth_tormen(nu)
    return f_nu * rho_mean / M_array
