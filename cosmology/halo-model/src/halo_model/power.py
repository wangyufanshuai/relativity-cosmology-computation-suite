"""Halo model power spectrum components."""

import numpy as np


def one_halo_term(k, mass_func, density_profile):
    """Compute the 1-halo term of the halo model power spectrum.

    P_1h(k) = integral dM * n(M) * (M/rho_mean)^2 * |u(k|M)|^2

    where u(k|M) is the normalized density profile Fourier transform.

    Parameters
    ----------
    k : array_like
        Wavenumber(s) at which to evaluate.
    mass_func : callable
        Function n(M) returning the halo mass function.
    density_profile : callable
        Function u(k, M) returning the normalized Fourier-space density profile.

    Returns
    -------
    array_like
        1-halo power spectrum at each k.
    """
    k = np.asarray(k, dtype=float)
    if k.ndim == 0:
        k = np.array([k])

    # Integration over mass using simple quadrature
    M_array = np.logspace(10, 15, 500)  # Solar masses
    dlnM = np.log(M_array[1] / M_array[0])

    rho_mean = 2.775e11 * 0.3 * 0.7**2  # Msun/Mpc^3, approximate

    result = np.zeros(len(k))
    for i, ki in enumerate(k):
        n_M = mass_func(M_array)
        u_k = np.array([density_profile(ki, M) for M in M_array])
        integrand = n_M * (M_array / rho_mean) ** 2 * u_k**2 * M_array
        result[i] = np.trapezoid(integrand, M_array)

    return result if len(result) > 1 else result[0]


def two_halo_term(k, bias_func):
    """Compute the 2-halo term of the halo model power spectrum.

    P_2h(k) = P_lin(k) * [integral dM * n(M) * b(M) * u(k|M)]^2

    Parameters
    ----------
    k : array_like
        Wavenumber(s).
    bias_func : callable
        Function returning (bias, u_k, n_M, P_lin) for given k.
        Should return tuple: (bias_weighted_sum, P_linear).

    Returns
    -------
    array_like
        2-halo power spectrum at each k.
    """
    k = np.asarray(k, dtype=float)
    if k.ndim == 0:
        k = np.array([k])

    result = np.zeros(len(k))
    for i, ki in enumerate(k):
        bias_weighted, P_lin = bias_func(ki)
        result[i] = P_lin * bias_weighted**2

    return result if len(result) > 1 else result[0]


def halo_model_Pk(k_array, params):
    """Compute the full halo model power spectrum.

    Parameters
    ----------
    k_array : array_like
        Wavenumber array.
    params : dict
        Dictionary containing:
        - 'mass_func': callable mass function
        - 'density_profile': callable density profile
        - 'bias_func': callable bias function for 2-halo term

    Returns
    -------
    dict
        Dictionary with 'P_1h', 'P_2h', 'P_total' arrays.
    """
    k_array = np.asarray(k_array, dtype=float)

    P_1h = one_halo_term(k_array, params['mass_func'], params['density_profile'])
    P_2h = two_halo_term(k_array, params['bias_func'])

    return {
        'P_1h': P_1h,
        'P_2h': P_2h,
        'P_total': P_1h + P_2h,
    }
