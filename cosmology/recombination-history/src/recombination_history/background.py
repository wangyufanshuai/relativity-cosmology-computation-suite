"""Background cosmology functions for the recombination history calculator.

Provides the Hubble parameter, CMB temperature, and baryon density as
functions of redshift in a flat Lambda-CDM universe.
"""

import numpy as np

from .constants import (
    C,
    G,
    K_B,
    M_H,
    M_P,
    T_CMB0,
    H0_SI,
    RHO_CRIT_FACTOR,
)


def hubble(z, H0, Omega_m, Omega_r, Omega_lambda):
    """Hubble parameter H(z) in units of H0.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_m : float
        Matter density parameter (CDM + baryons).
    Omega_r : float
        Radiation density parameter (photons + neutrinos).
    Omega_lambda : float
        Dark energy (cosmological constant) density parameter.

    Returns
    -------
    float or array_like
        H(z) in s^-1.
    """
    zp1 = np.asarray(z, dtype=float) + 1.0
    H0_si = H0 * 1.0e3 / 3.0856775814913673e22  # km/s/Mpc -> s^-1
    return H0_si * np.sqrt(
        Omega_m * zp1**3 + Omega_r * zp1**4 + Omega_lambda
    )


def temperature(z, T0=T_CMB0):
    """CMB photon temperature at redshift z.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    T0 : float
        CMB temperature today in Kelvin. Defaults to 2.7255 K.

    Returns
    -------
    float or array_like
        T(z) in Kelvin.
    """
    zp1 = np.asarray(z, dtype=float) + 1.0
    return T0 * zp1


def baryon_density(z, Omega_b, h):
    """Baryon number density at redshift z.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    Omega_b : float
        Baryon density parameter.
    h : float
        Reduced Hubble constant (H0 / 100 km/s/Mpc).

    Returns
    -------
    float or array_like
        n_b(z) in m^-3.
    """
    zp1 = np.asarray(z, dtype=float) + 1.0
    # Critical density today: rho_crit = 3 H0^2 / (8 pi G)
    H0_si = 100.0 * h * 1.0e3 / 3.0856775814913673e22
    rho_crit = 3.0 * H0_si**2 / (8.0 * np.pi * G)
    rho_b0 = Omega_b * rho_crit
    rho_b_z = rho_b0 * zp1**3
    return rho_b_z / M_H


def electron_density(z, x_e, Omega_b, h):
    """Free electron number density at redshift z.

    Parameters
    ----------
    z : float or array_like
        Redshift.
    x_e : float or array_like
        Free electron fraction (n_e / n_H).
    Omega_b : float
        Baryon density parameter.
    h : float
        Reduced Hubble constant.

    Returns
    -------
    float or array_like
        n_e(z) in m^-3.
    """
    n_b = baryon_density(z, Omega_b, h)
    return x_e * n_b
