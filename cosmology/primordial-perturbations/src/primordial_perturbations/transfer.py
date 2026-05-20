"""Transfer function and matter power spectrum.

Implements the Eisenstein & Hu (1998)-like approximate matter-radiation
transfer function and the resulting matter power spectrum.
"""

import numpy as np


def transfer_function(k: np.ndarray, k_eq: float) -> np.ndarray:
    """Approximate matter-radiation transfer function (BBKS-like fit).

    Uses the Bardeen, Bond, Kaiser & Szalay (BBKS) fitting formula:
        T(q) = ln(1 + 2.34q)/(2.34q) * [1 + 3.89q + (16.1q)^2 + (5.46q)^3 + (6.71q)^4]^(-1/4)
    where q = k / (Omega_m * h^2) in units of Mpc^{-1}, but we simplify
    by using q = k / k_eq directly.

    This satisfies:
        T(k -> 0) -> 1  (large scales unaffected)
        T(k -> inf) -> 0  (small scales suppressed)

    Parameters
    ----------
    k : np.ndarray or float
        Wavenumber array [same units as k_eq].
    k_eq : float
        Equality scale (wavenumber at matter-radiation equality).

    Returns
    -------
    np.ndarray
        Transfer function values T(k).
    """
    k = np.asarray(k, dtype=float)
    scalar_input = k.ndim == 0
    k = np.atleast_1d(k)

    q = k / k_eq

    # BBKS-like formula with q = k/k_eq
    T = np.zeros_like(q)

    for i, qi in enumerate(q):
        if qi < 1e-30:
            T[i] = 1.0
        else:
            # BBKS form
            term1 = np.log(1.0 + 2.34 * qi) / (2.34 * qi)
            bracket = (
                1.0
                + 3.89 * qi
                + (16.1 * qi) ** 2
                + (5.46 * qi) ** 3
                + (6.71 * qi) ** 4
            )
            T[i] = term1 * bracket ** (-0.25)

    if scalar_input:
        return T[0]
    return T


def matter_power_spectrum(
    k_array: np.ndarray,
    P_primordial: np.ndarray,
    k_eq: float,
) -> np.ndarray:
    """Compute the linear matter power spectrum.

    P(k) = P_primordial(k) * T(k)^2

    where T(k) is the matter-radiation transfer function.

    Parameters
    ----------
    k_array : np.ndarray
        Wavenumber array.
    P_primordial : np.ndarray
        Primordial power spectrum values.
    k_eq : float
        Wavenumber at matter-radiation equality.

    Returns
    -------
    np.ndarray
        Matter power spectrum P_matter(k).
    """
    T = transfer_function(k_array, k_eq)
    return P_primordial * T**2
