"""Background evolution integration for the Friedmann equation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import integrate

from . import constants as const
from .cosmology import Cosmology


class BackgroundResult(NamedTuple):
    """Result of background evolution integration.

    Attributes
    ----------
    t : ndarray
        Cosmic time [Gyr], shape (n_points,).
    a : ndarray
        Scale factor, shape (n_points,).
    z : ndarray
        Redshift, shape (n_points,).
    H : ndarray
        Hubble parameter [km/s/Mpc], shape (n_points,).
    rho_m : ndarray
        Normalized matter density Omega_m * a^{-3}, shape (n_points,).
    rho_r : ndarray
        Normalized radiation density Omega_r * a^{-4}, shape (n_points,).
    rho_de : ndarray
        Normalized dark energy density Omega_DE(a), shape (n_points,).
    """

    t: np.ndarray
    a: np.ndarray
    z: np.ndarray
    H: np.ndarray
    rho_m: np.ndarray
    rho_r: np.ndarray
    rho_de: np.ndarray


def solve_background(
    cosmo: Cosmology,
    z_max: float = 1e10,
    n_points: int = 10000,
) -> BackgroundResult:
    """Integrate the Friedmann equation to obtain a(t).

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    z_max : float
        Maximum redshift to integrate to. Default: 1e10.
    n_points : int
        Number of output points (log-spaced in a). Default: 10000.

    Returns
    -------
    result : BackgroundResult
        Named tuple with arrays (t, a, z, H, rho_m, rho_r, rho_de).
    """
    a_min = 1.0 / (1.0 + z_max)
    a_max = 1.0

    # Use log-spaced scale factors for output
    a_array = np.logspace(np.log10(a_min), np.log10(a_max), n_points)

    # Integrate dt/da = 1/(a * H(a)) from a_min to a
    # We accumulate t(a) by integrating from a_min upward
    def dt_da_func(a):
        return 1.0 / (a * cosmo.H_si(a))

    t_array = np.zeros(n_points)
    t_array[0] = 0.0  # t=0 at a_min (early time approximation)

    # Cumulative integration using Simpson's rule on the log-spaced grid
    for i in range(1, n_points):
        a_lo = a_array[i - 1]
        a_hi = a_array[i]
        # Use quad for each interval for accuracy
        dt, _ = integrate.quad(dt_da_func, a_lo, a_hi)
        t_array[i] = t_array[i - 1] + dt

    # Convert from seconds to Gyr
    sec_per_gyr = 3.1557e16
    t_gyr = t_array / sec_per_gyr

    # Add the time from a=0 to a=a_min (radiation-dominated era approximation)
    # In radiation domination: a ~ t^{1/2}, so t(a) = 1/(2*H(a)*a^2) evaluated at a_min
    # More precisely, t(a_min) ~ integral_0^{a_min} da/(a*H(a))
    if a_min > 0:
        t_early, _ = integrate.quad(dt_da_func, 1e-30, a_min, limit=200)
        t_early_gyr = t_early / sec_per_gyr
        t_gyr = t_gyr + t_early_gyr

    z_array = 1.0 / a_array - 1.0
    H_array = np.array([cosmo.H(a) for a in a_array])
    rho_m_array = cosmo.Omega_m / a_array ** 3
    rho_r_array = cosmo.Omega_r / a_array ** 4
    rho_de_array = np.array([cosmo.rho_de(a) for a in a_array])

    return BackgroundResult(
        t=t_gyr,
        a=a_array,
        z=z_array,
        H=H_array,
        rho_m=rho_m_array,
        rho_r=rho_r_array,
        rho_de=rho_de_array,
    )


def conformal_time(cosmo: Cosmology, a: ArrayLike) -> float:
    """Conformal time eta(a) in Mpc.

    eta = integral_0^a da' / (a'^2 * H(a'))

    Returns eta * c in Mpc (comoving).

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like or float
        Scale factor(s).

    Returns
    -------
    eta : float or ndarray
        Conformal time in Mpc.
    """
    def integrand(a_val):
        return 1.0 / (a_val ** 2 * cosmo.H_si(a_val))

    if np.isscalar(a):
        result, _ = integrate.quad(integrand, 1e-30, a, limit=200)
        return result * const.C / const.MPC_IN_M  # convert s to Mpc via c*eta
    else:
        a = np.asarray(a)
        results = np.empty_like(a)
        for i, ai in enumerate(a.flat):
            val, _ = integrate.quad(integrand, 1e-30, float(ai), limit=200)
            results.flat[i] = val * const.C / const.MPC_IN_M
        return results


@dataclass
class HorizonScales:
    """Key comoving horizon scales in Mpc.

    Attributes
    ----------
    eta_0 : float
        Comoving horizon today [Mpc].
    eta_rec : float
        Comoving horizon at recombination (z=1090) [Mpc].
    eta_eq : float
        Comoving horizon at matter-radiation equality [Mpc].
    z_rec : float
        Recombination redshift.
    z_eq : float
        Matter-radiation equality redshift.
    """

    eta_0: float
    eta_rec: float
    eta_eq: float
    z_rec: float
    z_eq: float


def horizon_scale(cosmo: Cosmology) -> HorizonScales:
    """Compute characteristic comoving horizon scales.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.

    Returns
    -------
    scales : HorizonScales
    """
    # Matter-radiation equality: Omega_m / a^3 = Omega_r / a^4 => a_eq = Omega_r / Omega_m
    a_eq = cosmo.Omega_r / cosmo.Omega_m
    z_eq = 1.0 / a_eq - 1.0

    # Recombination
    z_rec = 1090.0
    a_rec = 1.0 / (1.0 + z_rec)

    # Conformal time at various epochs
    eta_0 = conformal_time(cosmo, 1.0)
    eta_rec = conformal_time(cosmo, a_rec)
    eta_eq = conformal_time(cosmo, a_eq)

    return HorizonScales(
        eta_0=eta_0,
        eta_rec=eta_rec,
        eta_eq=eta_eq,
        z_rec=z_rec,
        z_eq=z_eq,
    )
