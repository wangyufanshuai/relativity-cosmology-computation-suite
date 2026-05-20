"""Kaiser redshift-space distortion model."""

import numpy as np


def kaiser_factor(k, mu, beta):
    """Compute the Kaiser RSD factor (1 + beta * mu^2).

    Parameters
    ----------
    k : array_like
        Wavenumber (not directly used in Kaiser factor, included for API consistency).
    mu : array_like
        Cosine of angle between wavevector and line of sight.
    beta : float
        Growth-rate parameter beta = f / b.

    Returns
    -------
    array_like
        Kaiser factor (1 + beta * mu^2).
    """
    return (1.0 + beta * mu**2) ** 2


def rsd_power_spectrum(k, mu, P_real, beta):
    """Compute the redshift-space distorted power spectrum using Kaiser model.

    Parameters
    ----------
    k : array_like
        Wavenumber.
    mu : array_like
        Cosine of angle between wavevector and line of sight.
    P_real : array_like or float
        Real-space power spectrum evaluated at k.
    beta : float
        Growth-rate parameter.

    Returns
    -------
    array_like
        Redshift-space power spectrum P(k, mu) = (1 + beta*mu^2)^2 * P_real(k).
    """
    return kaiser_factor(k, mu, beta) * P_real


def _legendre_l0(mu):
    """Legendre polynomial l=0: P_0(mu) = 1."""
    return np.ones_like(mu)


def _legendre_l2(mu):
    """Legendre polynomial l=2: P_2(mu) = (3*mu^2 - 1) / 2."""
    return (3.0 * mu**2 - 1.0) / 2.0


def _legendre_l4(mu):
    """Legendre polynomial l=4: P_4(mu) = (35*mu^4 - 30*mu^2 + 3) / 8."""
    return (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0


def multipole_Pk(k, P_real, beta, l):
    """Compute the l-th multipole moment of the redshift-space power spectrum.

    Parameters
    ----------
    k : array_like
        Wavenumber.
    P_real : array_like or float
        Real-space power spectrum at k.
    beta : float
        Growth-rate parameter.
    l : int
        Multipole order (0, 2, or 4).

    Returns
    -------
    array_like
        Multipole moment P_l(k) = (2l+1)/2 * integral P(k,mu) * P_l(mu) dmu.
    """
    mu = np.linspace(-1, 1, 10000)
    dmu = mu[1] - mu[0]

    P_kmu = rsd_power_spectrum(k, mu, P_real, beta)

    legendre_funcs = {0: _legendre_l0, 2: _legendre_l2, 4: _legendre_l4}
    if l not in legendre_funcs:
        raise ValueError(f"Multipole l={l} not supported. Use l=0, 2, or 4.")

    Pl_mu = legendre_funcs[l](mu)
    integral = np.trapezoid(P_kmu * Pl_mu, mu)

    return (2 * l + 1) / 2.0 * integral
