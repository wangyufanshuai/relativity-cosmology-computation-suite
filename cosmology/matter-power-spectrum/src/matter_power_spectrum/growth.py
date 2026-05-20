"""Linear growth factor and growth rate computation.

The growth factor D(a) describes how density perturbations grow in an
expanding universe. In a flat Lambda-CDM cosmology it satisfies:

    D(a) = (5/2) * Omega_m * H(a) / h  *  integral_0^a  da' / (a' * H(a'))^3

where H(a) = H0 * sqrt(Omega_m * a^(-3) + Omega_lambda).

The growth rate f = d ln D / d ln a is approximated by f ~ Omega_m(a)^0.55
(Peebles 1980, Linder 2005).
"""

import numpy as np
from scipy import integrate

from .constants import OMEGA_M_DEFAULT, OMEGA_LAMBDA_DEFAULT


def _H_over_H0(a, Omega_m=OMEGA_M_DEFAULT, Omega_lambda=OMEGA_LAMBDA_DEFAULT):
    """Dimensionless Hubble parameter H(a)/H0.

    Parameters
    ----------
    a : float or array_like
        Scale factor.
    Omega_m : float
        Matter density parameter.
    Omega_lambda : float
        Dark energy density parameter.

    Returns
    -------
    float or array_like
        H(a)/H0 = sqrt(Omega_m * a^{-3} + Omega_lambda)
    """
    return np.sqrt(Omega_m * a**(-3) + Omega_lambda)


def growth_factor(a, Omega_m=OMEGA_M_DEFAULT, Omega_lambda=OMEGA_LAMBDA_DEFAULT):
    """Unnormalized linear growth factor D(a).

    Computes D(a) = H(a) * integral_0^a [da' / (a' * H(a'))^3]
    using numerical integration.

    Parameters
    ----------
    a : float or array_like
        Scale factor (0 < a <= 1, where a=1 is today).
    Omega_m : float
        Matter density parameter.
    Omega_lambda : float
        Dark energy density parameter.

    Returns
    -------
    float or array_like
        Unnormalized growth factor D(a).
    """
    a = np.asarray(a, dtype=float)
    scalar_input = a.ndim == 0
    a = np.atleast_1d(a)

    def _integrand(a_prime):
        """1 / (a' * H(a')/H0)^3"""
        H = _H_over_H0(a_prime, Omega_m, Omega_lambda)
        return 1.0 / (a_prime * H)**3

    results = np.empty_like(a)
    for i, ai in enumerate(a):
        if ai <= 0:
            results[i] = 0.0
        else:
            # Integrate from a small value (near 0) to a
            # The integrand is well-behaved for small a in matter domination
            # where H ~ a^{-3/2}, so integrand ~ a'^{-1} * a'^{9/2} = a'^{7/2}
            a_min = 1e-10
            integral, _ = integrate.quad(_integrand, a_min, ai, limit=200)
            H_a = _H_over_H0(ai, Omega_m, Omega_lambda)
            results[i] = H_a * integral

    if scalar_input:
        return float(results[0])
    return results


def growth_factor_normalized(a, Omega_m=OMEGA_M_DEFAULT, Omega_lambda=OMEGA_LAMBDA_DEFAULT):
    """Normalized linear growth factor D(a) with D(a=1) = 1.

    Parameters
    ----------
    a : float or array_like
        Scale factor (0 < a <= 1, where a=1 is today).
    Omega_m : float
        Matter density parameter.
    Omega_lambda : float
        Dark energy density parameter.

    Returns
    -------
    float or array_like
        Growth factor D(a) normalized to D(a=1) = 1.
    """
    D_unnorm = growth_factor(a, Omega_m, Omega_lambda)
    D_today = growth_factor(1.0, Omega_m, Omega_lambda)
    return D_unnorm / D_today


def growth_rate(a, Omega_m=OMEGA_M_DEFAULT, Omega_lambda=OMEGA_LAMBDA_DEFAULT):
    """Growth rate f = d ln D / d ln a.

    Uses the approximation f ~ Omega_m(a)^0.55 from Peebles (1980)
    and Linder (2005), where Omega_m(a) = Omega_m * a^{-3} / (H(a)/H0)^2.

    Parameters
    ----------
    a : float or array_like
        Scale factor.
    Omega_m : float
        Matter density parameter.
    Omega_lambda : float
        Dark energy density parameter.

    Returns
    -------
    float or array_like
        Growth rate f(a) ~ Omega_m(a)^0.55.
    """
    a = np.asarray(a, dtype=float)

    H_ratio = _H_over_H0(a, Omega_m, Omega_lambda)
    Omega_m_a = Omega_m * a**(-3) / H_ratio**2
    return Omega_m_a**0.55
