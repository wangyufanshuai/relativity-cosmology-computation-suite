"""Matter power spectrum computation.

Provides primordial power spectrum, linear matter power spectrum using the
EH98 transfer function, and sigma_8 computation with top-hat window function.
"""

import numpy as np
from scipy import integrate

from .constants import (
    A_S_DEFAULT,
    H_DEFAULT,
    K_PIVOT_DEFAULT,
    N_S_DEFAULT,
    OMEGA_B_DEFAULT,
    OMEGA_M_DEFAULT,
    T_CMB_DEFAULT,
)
from .transfer import transfer_EH98


def primordial_power(k, A_s=A_S_DEFAULT, n_s=N_S_DEFAULT, k_pivot=K_PIVOT_DEFAULT):
    """Primordial power spectrum P_primordial(k).

    P_prim(k) = A_s * (k / k_pivot)^(n_s - 1)

    Parameters
    ----------
    k : array_like
        Wavenumber in 1/Mpc.
    A_s : float
        Scalar amplitude.
    n_s : float
        Scalar spectral index.
    k_pivot : float
        Pivot scale in 1/Mpc.

    Returns
    -------
    array_like
        Primordial power spectrum P_prim(k).
    """
    k = np.asarray(k, dtype=float)
    return A_s * (k / k_pivot)**(n_s - 1.0)


def linear_power_spectrum(k, A_s=A_S_DEFAULT, n_s=N_S_DEFAULT, k_pivot=K_PIVOT_DEFAULT,
                          h=H_DEFAULT, Omega_m=OMEGA_M_DEFAULT, Omega_b=OMEGA_B_DEFAULT,
                          T_CMB=T_CMB_DEFAULT):
    """Linear matter power spectrum P(k).

    P(k) = 2 * pi^2 * P_prim(k) * T(k)^2 / k^3

    where P_prim(k) is the primordial power spectrum and T(k) is the
    EH98 no-wiggle transfer function.

    Parameters
    ----------
    k : array_like
        Wavenumber in 1/Mpc.
    A_s : float
        Scalar amplitude.
    n_s : float
        Scalar spectral index.
    k_pivot : float
        Pivot scale in 1/Mpc.
    h : float
        Dimensionless Hubble parameter.
    Omega_m : float
        Matter density parameter.
    Omega_b : float
        Baryon density parameter.
    T_CMB : float
        CMB temperature in Kelvin.

    Returns
    -------
    array_like
        Linear matter power spectrum P(k) in Mpc^3.
    """
    k = np.asarray(k, dtype=float)
    P_prim = primordial_power(k, A_s, n_s, k_pivot)
    T_k = transfer_EH98(k, h, Omega_m, Omega_b, T_CMB)
    return 2.0 * np.pi**2 * P_prim * T_k**2 / k**3


def _tophat_window(kR):
    """Fourier transform of a spherical top-hat window function.

    W(kR) = 3 * (sin(kR) - kR * cos(kR)) / (kR)^3

    Parameters
    ----------
    kR : array_like
        Product of wavenumber and filter radius.

    Returns
    -------
    array_like
        Window function value.
    """
    kR = np.asarray(kR, dtype=float)
    result = np.ones_like(kR)
    mask = kR > 1e-10
    x = kR[mask]
    result[mask] = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    return result


def sigma_8(P_k, k_array, R=8.0):
    """Variance of density fluctuations in a top-hat sphere of radius R.

    sigma_R^2 = (1 / 2*pi^2) * integral k^2 * P(k) * W(kR)^2 dk

    where W(kR) is the Fourier transform of a spherical top-hat of radius R.
    For R=8 Mpc/h, this gives sigma_8.

    Parameters
    ----------
    P_k : array_like
        Power spectrum values at k_array.
    k_array : array_like
        Wavenumber array in 1/Mpc.
    R : float
        Smoothing radius in Mpc/h. Default is 8.0 for sigma_8.

    Returns
    -------
    float
        sigma_R (the rms density fluctuation in the sphere of radius R).
    """
    k_array = np.asarray(k_array, dtype=float)
    P_k = np.asarray(P_k, dtype=float)

    # Window function
    W = _tophat_window(k_array * R)

    # Integrand: k^2 * P(k) * W(kR)^2
    integrand = k_array**2 * P_k * W**2

    # Numerical integration using the trapezoidal rule
    # (we need a well-sampled k array for accuracy)
    integral = np.trapezoid(integrand, k_array)

    return np.sqrt(integral / (2.0 * np.pi**2))
