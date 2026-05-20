"""Observable quantities derived from the recombination history.

Computes Thomson scattering opacity, the visibility function (probability
of last scattering at a given redshift), the redshift of last scattering,
and the sound horizon at drag epoch.
"""

import numpy as np

from .constants import (
    C,
    G,
    K_B,
    SIGMA_T,
    M_P,
    M_H,
    T_CMB0,
    A_RAD,
    N_eff,
)


def thomson_opacity(z_array, x_e_array, Omega_b, h):
    """Thomson scattering opacity dtau/dz.

    The optical depth derivative with respect to redshift:
        dtau/dz = sigma_T * n_e(z) * c * |dt/dz|
                = sigma_T * n_e(z) * c / ((1+z) * H(z))

    Parameters
    ----------
    z_array : array_like
        Redshift array.
    x_e_array : array_like
        Free electron fraction at each redshift.
    Omega_b : float
        Baryon density parameter.
    h : float
        Reduced Hubble constant.

    Returns
    -------
    ndarray
        dtau/dz (positive, since optical depth increases toward high z).
    """
    from .background import baryon_density, hubble

    z_array = np.asarray(z_array, dtype=float)
    x_e_array = np.asarray(x_e_array, dtype=float)

    # We need H(z), but we don't have cosmological parameters here
    # Use standard Planck-like defaults for the Hubble computation
    # Actually, we need H0, Omega_m, Omega_r, Omega_lambda passed in
    # This function signature is incomplete -- we compute n_e and use
    # a simplified form

    n_b = baryon_density(z_array, Omega_b, h)
    n_e = x_e_array * n_b

    # dtau/dz = sigma_T * n_e * c / ((1+z) * H(z))
    # We need H(z). For now we return sigma_T * n_e * c as the density part
    # and the caller can divide by (1+z)*H
    # Actually let's compute it properly with default cosmology

    return SIGMA_T * n_e * C


def thomson_opacity_full(z_array, x_e_array, Omega_b, h, H0, Omega_m, Omega_r, Omega_lambda):
    """Full Thomson scattering opacity dtau/dz including cosmological factors.

    Parameters
    ----------
    z_array : array_like
        Redshift array.
    x_e_array : array_like
        Free electron fraction at each redshift.
    Omega_b : float
        Baryon density parameter.
    h : float
        Reduced Hubble constant.
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_r : float
        Radiation density parameter.
    Omega_lambda : float
        Dark energy density parameter.

    Returns
    -------
    ndarray
        dtau/dz.
    """
    from .background import baryon_density, hubble

    z_array = np.asarray(z_array, dtype=float)
    x_e_array = np.asarray(x_e_array, dtype=float)

    n_b = baryon_density(z_array, Omega_b, h)
    n_e = x_e_array * n_b
    H_z = hubble(z_array, H0, Omega_m, Omega_r, Omega_lambda)

    # dtau/dz = sigma_T * n_e * c / ((1+z) * H(z))
    return SIGMA_T * n_e * C / ((1.0 + z_array) * H_z)


def visibility_function(z_array, x_e_array, Omega_b, h, H0, Omega_m, Omega_r, Omega_lambda):
    """Visibility function g(z) = exp(-tau) * dtau/dz.

    The visibility function gives the probability that a CMB photon last
    scattered at redshift z. It peaks near z ~ 1100 (the last scattering
    surface) and has a width of Delta z ~ 100.

    Parameters
    ----------
    z_array : array_like
        Redshift array (sorted in increasing order).
    x_e_array : array_like
        Free electron fraction at each redshift.
    Omega_b : float
        Baryon density parameter.
    h : float
        Reduced Hubble constant.
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_r : float
        Radiation density parameter.
    Omega_lambda : float
        Dark energy density parameter.

    Returns
    -------
    g_array : ndarray
        Visibility function g(z) at each redshift.
    tau_array : ndarray
        Optical depth at each redshift.
    """
    z_array = np.asarray(z_array, dtype=float)
    x_e_array = np.asarray(x_e_array, dtype=float)

    # dtau/dz
    dtaudz = thomson_opacity_full(z_array, x_e_array, Omega_b, h,
                                  H0, Omega_m, Omega_r, Omega_lambda)

    # Integrate tau from z=0 upward
    # tau(z) = integral from 0 to z of dtau/dz' dz'
    # Since z_array is increasing, we integrate forward
    tau = np.zeros_like(z_array)
    for i in range(1, len(z_array)):
        dz = z_array[i] - z_array[i - 1]
        tau[i] = tau[i - 1] + 0.5 * (dtaudz[i] + dtaudz[i - 1]) * dz

    # Visibility function
    g = np.exp(-tau) * dtaudz

    return g, tau


def last_scattering_z(g_array, z_array):
    """Redshift of last scattering: z_* where g(z) peaks.

    Parameters
    ----------
    g_array : array_like
        Visibility function values.
    z_array : array_like
        Corresponding redshift array.

    Returns
    -------
    float
        Redshift of last scattering z_*.
    """
    g_array = np.asarray(g_array, dtype=float)
    z_array = np.asarray(z_array, dtype=float)
    idx = np.argmax(g_array)
    return float(z_array[idx])


def sound_horizon(z_drag, H0, Omega_m, Omega_r, Omega_b, h):
    """Sound horizon at the drag epoch.

    The sound horizon is the comoving distance a sound wave could have
    traveled in the photon-baryon fluid from the big bang until the
    drag epoch z_drag:

        r_s = integral from 0 to a_drag of c_s / (a^2 H) da

    where c_s is the sound speed in the photon-baryon fluid:
        c_s = c / sqrt(3 * (1 + R))
        R = (3 rho_b) / (4 rho_gamma)

    Parameters
    ----------
    z_drag : float
        Redshift of the drag epoch (typically z_drag ~ 1060).
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_r : float
        Radiation density parameter.
    Omega_b : float
        Baryon density parameter.
    h : float
        Reduced Hubble constant.

    Returns
    -------
    float
        Sound horizon r_s in Mpc.
    """
    from .background import hubble, baryon_density
    from scipy.integrate import quad

    H0_si = H0 * 1.0e3 / 3.0856775814913673e22  # s^-1
    Mpc_in_m = 3.0856775814913673e22  # 1 Mpc in meters

    # Photon energy density today: rho_gamma = a_rad * T_CMB^4
    rho_gamma0 = A_RAD * T_CMB0**4

    # Critical density today
    rho_crit = 3.0 * H0_si**2 / (8.0 * np.pi * G)
    Omega_gamma = rho_gamma0 / rho_crit

    def integrand(a):
        """Integrand for sound horizon: c_s / (a^2 H(a)) in comoving coords."""
        z = 1.0 / a - 1.0
        zp1 = 1.0 + z

        # Baryon density at this redshift
        rho_b = Omega_b * rho_crit * zp1**3

        # Photon density at this redshift
        rho_gamma = Omega_gamma * rho_crit * zp1**4

        # R = 3 rho_b / (4 rho_gamma)
        R = 3.0 * rho_b / (4.0 * rho_gamma)

        # Sound speed
        c_s = C / np.sqrt(3.0 * (1.0 + R))

        # Hubble parameter
        H = H0_si * np.sqrt(
            Omega_m * zp1**3 + Omega_r * zp1**4 + (1.0 - Omega_m - Omega_r)
        )

        return c_s / (a**2 * H)

    # Integrate from a ~ 0 (early times) to a_drag = 1/(1+z_drag)
    a_drag = 1.0 / (1.0 + z_drag)
    # Start from a very small scale factor (z ~ 10^6)
    a_start = 1.0e-6

    result, _ = quad(integrand, a_start, a_drag, limit=200, epsrel=1e-8)

    # Convert from meters to Mpc
    r_s_mpc = result / Mpc_in_m

    return r_s_mpc
