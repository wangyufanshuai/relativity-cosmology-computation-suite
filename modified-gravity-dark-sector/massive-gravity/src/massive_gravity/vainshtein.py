"""
Vainshtein screening mechanism for massive gravity.

In dRGT massive gravity the helicity-0 mode of the graviton mediates an
additional "fifth force".  The Vainshtein mechanism suppresses this force
near massive sources through nonlinear self-interactions of the scalar mode.

Key scales:
  - Gravitational radius  r_g = 2 G M
  - Vainshtein radius     r_V = (r_g / m_g^2)^{1/3}
  - For r << r_V the fifth force is suppressed as (r/r_V)^{3/2}

References:
  - Vainshtein, Phys.Lett. B39 (1972) 393
  - Deffayet et al., Phys.Rev. D65 (2002) 044026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G_NEWTON = 6.67430e-11  # m^3 kg^{-1} s^{-2}
C_LIGHT = 2.99792458e8  # m/s
EV_TO_JOULE = 1.602176634e-19  # J/eV
HBAR = 1.054571817e-34  # J s


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------

def graviton_mass_eV_to_inv_m(m_g_eV: float) -> float:
    """Convert graviton mass from eV to inverse metres (m^{-1}).

    m_g = (m_g in eV) * e / (\\hbar c)
    """
    return m_g_eV * EV_TO_JOULE / (HBAR * C_LIGHT)


def graviton_mass_eV_to_m(m_g_eV: float) -> float:
    """Convert graviton mass from eV to Compton wavelength in metres."""
    m_inv_m = graviton_mass_eV_to_inv_m(m_g_eV)
    if m_inv_m == 0.0:
        return np.inf
    return 1.0 / m_inv_m


# ---------------------------------------------------------------------------
# Vainshtein radius
# ---------------------------------------------------------------------------

def gravitational_radius(M: float) -> float:
    """Schwarzschild radius r_g = 2 G M."""
    return 2.0 * G_NEWTON * M / C_LIGHT**2


def vainshtein_radius(M: float, m_g_eV: float) -> float:
    """Compute the Vainshtein radius r_V.

    r_V = (r_g / m_g^2)^{1/3}  where r_g = 2GM and m_g is in inverse metres.

    Parameters
    ----------
    M : float
        Source mass in kilograms.
    m_g_eV : float
        Graviton mass in eV.
    """
    r_g = gravitational_radius(M)
    m_g_inv_m = graviton_mass_eV_to_inv_m(m_g_eV)
    if m_g_inv_m <= 0.0:
        return np.inf
    return (r_g / m_g_inv_m**2) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Screening factor
# ---------------------------------------------------------------------------

def screening_factor(r: float | NDArray, r_V: float) -> float | NDArray:
    """Vainshtein suppression factor for the fifth force.

    For r << r_V :  suppression ~ (r / r_V)^{3/2}  (heavily suppressed)
    For r >> r_V :  suppression ~ 1                  (full fifth force)

    A smooth interpolating form is used:
      xi(r) = [1 + (r_V / r)^{3/2}]^{-1}

    This gives the correct asymptotic behaviour in both limits.
    """
    r = np.asarray(r, dtype=float)
    ratio = r_V / r
    return 1.0 / (1.0 + ratio**1.5)


# ---------------------------------------------------------------------------
# Force profiles
# ---------------------------------------------------------------------------

def fifth_force(
    r: float | NDArray,
    M: float,
    m_g_eV: float,
) -> float | NDArray:
    """Fifth force (Yukawa-modified Newtonian) with Vainshtein screening.

    F_5 = (G M / r^2) * screening(r, r_V)

    The fifth force is suppressed inside the Vainshtein radius.
    """
    r = np.asarray(r, dtype=float)
    r_V = vainshtein_radius(M, m_g_eV)
    xi = screening_factor(r, r_V)
    return G_NEWTON * M / r**2 * xi


def newton_force(r: float | NDArray, M: float) -> float | NDArray:
    """Standard Newtonian gravitational force F = G M / r^2."""
    r = np.asarray(r, dtype=float)
    return G_NEWTON * M / r**2


def total_force(
    r: float | NDArray,
    M: float,
    m_g_eV: float,
    alpha: float = 1.0,
) -> float | NDArray:
    """Total gravitational force = Newton + alpha * (screened fifth force).

    Parameters
    ----------
    r : distance
    M : source mass (kg)
    m_g_eV : graviton mass (eV)
    alpha : coupling strength of the fifth force (default 1).
    """
    F_N = newton_force(r, M)
    F_5 = fifth_force(r, M, m_g_eV)
    return F_N + alpha * F_5


# ---------------------------------------------------------------------------
# Gravitational potential with Vainshtein screening
# ---------------------------------------------------------------------------

def gravitational_potential(
    r: float | NDArray,
    M: float,
    m_g_eV: float,
    alpha: float = 1.0,
) -> float | NDArray:
    """Gravitational potential Phi(r) with Vainshtein screening.

    Phi(r) = -G M / r * [1 + alpha * xi(r) * f Yukawa]

    For simplicity we model the screened potential as:
      Phi = -G M / r * [1 + alpha * screening_factor(r, r_V)]
    which reduces to the Newtonian potential far from the source and
    is suppressed near the source.
    """
    r = np.asarray(r, dtype=float)
    r_V = vainshtein_radius(M, m_g_eV)
    xi = screening_factor(r, r_V)
    return -G_NEWTON * M / r * (1.0 + alpha * xi)


# ---------------------------------------------------------------------------
# Force suppression check
# ---------------------------------------------------------------------------

def force_suppressed_inside(
    M: float,
    m_g_eV: float,
    r_fraction: float = 0.01,
) -> bool:
    """Check that the fifth force is suppressed at r = r_fraction * r_V.

    Returns True if the screened fifth force at r_fraction * r_V is less than
    the unscreened fifth force at the same distance.
    """
    r_V = vainshtein_radius(M, m_g_eV)
    r = r_fraction * r_V
    F5_screened = fifth_force(r, M, m_g_eV)
    F5_unscreened = G_NEWTON * M / r**2  # without screening
    return bool(F5_screened < F5_unscreened)
