"""Neutrino background evolution.

Functions for neutrino temperature, energy density, pressure, equation of
state, and cosmological density parameter.

All computations use natural units where hbar = c = k_B = 1 internally,
converting to physical units only at the interface.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from .constants import (
    C,
    EV_TO_J,
    FD_CONSTANT,
    K_B,
    OMEGA_NU_DENOMINATOR_EV,
    T_NU_OVER_T_GAMMA,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fd_integrand(x: float | np.ndarray) -> float | np.ndarray:
    """Fermi-Dirac distribution integrand x^2 * sqrt(x^2 + y^2) / (e^x + 1).

    Here *x* = p/(k_B T) is the dimensionless momentum and *y* = m c^2/(k_B T)
    is the dimensionless mass parameter.  For the energy density one integrates
    4 pi / (2 pi hbar)^3 * (k_B T)^4 * x^2 sqrt(x^2+y^2) / (e^x+1) dx.
    We factor out the physical prefactors so that this returns the
    dimensionless part.
    """
    return x**2 * np.sqrt(x**2 + 1.0) / (np.exp(x) + 1.0)


def _fd_integrand_energy(x: float | np.ndarray, y: float) -> float | np.ndarray:
    """Fermi-Dirac energy integrand with dimensionless mass y = mc^2/(k_B T).

    Returns  x^2 * sqrt(x^2 + y^2) / (exp(x) + 1).
    """
    return x**2 * np.sqrt(x**2 + y**2) / (np.exp(x) + 1.0)


def _fd_integrand_pressure(x: float | np.ndarray, y: float) -> float | np.ndarray:
    """Fermi-Dirac pressure integrand with dimensionless mass y = mc^2/(k_B T).

    Returns  x^4 / (3 sqrt(x^2 + y^2) * (exp(x) + 1)).
    """
    return x**4 / (3.0 * np.sqrt(x**2 + y**2) * (np.exp(x) + 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def neutrino_temperature(T_cmb: float) -> float:
    """Neutrino temperature after e+e- annihilation.

    Parameters
    ----------
    T_cmb : float
        Photon (CMB) temperature [K].

    Returns
    -------
    float
        Neutrino temperature T_nu = T_cmb * (4/11)^(1/3) [K].
    """
    return T_cmb * T_NU_OVER_T_GAMMA


def neutrino_energy_density(m_nu: float, T_nu: float) -> float:
    """Energy density of a single neutrino species (Fermi-Dirac integral).

    Parameters
    ----------
    m_nu : float
        Neutrino mass [kg].
    T_nu : float
        Neutrino temperature [K].

    Returns
    -------
    float
        Energy density rho_nu [J/m^3].

    Notes
    -----
    rho_nu = (g / 2 pi^2) (k_B T)^4 / (hbar c)^3  *  integral
    with g = 1 (one Weyl fermion) and the integral over the FD spectrum.
    """
    kT = K_B * T_nu
    y = m_nu * C**2 / kT if kT > 0 else np.inf  # dimensionless mass

    hbar = 1.054571817e-34  # J s
    prefactor = 1.0 / (2.0 * np.pi**2) * kT**4 / (hbar * C)**3

    # Upper limit: integrand is negligible beyond ~30 kT
    integral, _ = integrate.quad(_fd_integrand_energy, 0, 50.0, args=(y,))

    return prefactor * integral


def total_neutrino_density(
    m_nu_array: list[float] | np.ndarray,
    T_nu: float,
) -> float:
    """Total neutrino energy density summed over species.

    Parameters
    ----------
    m_nu_array : array_like
        Neutrino masses [kg], one entry per species.
    T_nu : float
        Common neutrino temperature [K].

    Returns
    -------
    float
        Total energy density [J/m^3].
    """
    return sum(neutrino_energy_density(m, T_nu) for m in m_nu_array)


def neutrino_pressure(m_nu: float, T_nu: float) -> float:
    """Pressure of a single neutrino species (Fermi-Dirac integral).

    Parameters
    ----------
    m_nu : float
        Neutrino mass [kg].
    T_nu : float
        Neutrino temperature [K].

    Returns
    -------
    float
        Pressure P_nu [J/m^3].
    """
    kT = K_B * T_nu
    y = m_nu * C**2 / kT if kT > 0 else np.inf

    hbar = 1.054571817e-34
    prefactor = 1.0 / (2.0 * np.pi**2) * kT**4 / (hbar * C)**3

    integral, _ = integrate.quad(_fd_integrand_pressure, 0, 50.0, args=(y,))

    return prefactor * integral


def neutrino_equation_of_state(
    m_nu: float,
    a: float | np.ndarray,
    a_nr: float | None = None,
) -> float | np.ndarray:
    """Neutrino equation of state w_nu(a).

    Transitions from w = 1/3 (relativistic) to w = 0 (non-relativistic)
    around the scale factor a_nr at which k_B T_nu ~ m_nu c^2.

    Parameters
    ----------
    m_nu : float
        Neutrino mass [kg].
    a : float or ndarray
        Scale factor(s).
    a_nr : float or None
        Scale factor of non-relativistic transition.  If None, estimated
        from the mass (assuming T_nu0 ~ 1.95 K).

    Returns
    -------
    float or ndarray
        w_nu(a) = P_nu / rho_nu.
    """
    if a_nr is None:
        # T_nu today ~ 1.95 K;  a_nr ~ T_nu0 / (m_nu c^2 / k_B)
        T_nu0 = neutrino_temperature(2.7255)
        a_nr = K_B * T_nu0 / (m_nu * C**2)

    # Smooth interpolation using the exact FD ratio would require
    # numerical integration at every a.  A standard analytic
    # approximation is:
    #   w(a) = (1/3) / (1 + (a / a_nr)^2)
    # This reproduces w -> 1/3 for a << a_nr and w -> 0 for a >> a_nr.
    a = np.asarray(a, dtype=float)
    return (1.0 / 3.0) / (1.0 + (a / a_nr) ** 2)


def omega_nu(m_nu_sum: float, h: float) -> float:
    """Neutrino density parameter Omega_nu today.

    Parameters
    ----------
    m_nu_sum : float
        Sum of neutrino masses [eV].
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    float
        Omega_nu = Sigma_m_nu / (93.14 eV * h^2).
    """
    return m_nu_sum / (OMEGA_NU_DENOMINATOR_EV * h**2)
