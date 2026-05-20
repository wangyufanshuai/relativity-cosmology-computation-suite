"""Analytical formulae for perihelion precession.

References:
    - Will (2014), "The Confrontation between General Relativity and Experiment"
    - Nobili & Will (1986), "The real value of Mercury's perihelion advance"
    - Park et al. (2017), "Precession of Mercury's Perihelion from MESSENGER data"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .constants import A_MERCURY, C, E_MERCURY, G, J2_SUN, M_SUN, R_SUN


def schwarzschild_precession(
    a: float = A_MERCURY,
    e: float = E_MERCURY,
    M: float = M_SUN,
) -> float:
    """First-order Schwarzschild precession per orbit (radians).

    Δφ = 6πGM / [a(1 - e²)c²]

    Returns arcseconds per century when `as_arcsec_per_century=True`.
    """
    delta_phi = 6.0 * np.pi * G * M / (a * (1.0 - e**2) * C**2)
    return delta_phi


def schwarzschild_2pn(
    a: float = A_MERCURY,
    e: float = E_MERCURY,
    M: float = M_SUN,
) -> float:
    """Second post-Newtonian correction to perihelion precession.

    From the 2PN expansion of the Schwarzschild geodesic:
    Δφ_2PN = (3π/2) * (GM)^2 / [a²(1 - e²)² c⁴] * (10 + 3e²) / (1 - e²)
    """
    gm = G * M
    factor = (3.0 * np.pi / 2.0) * gm**2 / (a**2 * (1.0 - e**2) ** 2 * C**4)
    correction = (10.0 + 3.0 * e**2) / (1.0 - e**2)
    return factor * correction


def quadrupole_precession(
    a: float = A_MERCURY,
    e: float = E_MERCURY,
    J2: float = J2_SUN,
    R: float = R_SUN,
) -> float:
    """Precession due to solar quadrupole moment J2 per orbit (radians).

    Δφ_J2 = 3π J2 R² / [a² (1 - e²)²]
    """
    return 3.0 * np.pi * J2 * R**2 / (a**2 * (1.0 - e**2) ** 2)


def total_analytical(
    a: float = A_MERCURY,
    e: float = E_MERCURY,
    M: float = M_SUN,
    J2: float = J2_SUN,
    R: float = R_SUN,
    include_2pn: bool = True,
    include_J2: bool = True,
) -> dict[str, float]:
    """Compute all analytical contributions to perihelion precession.

    Returns dictionary with:
        - 'schwarzschild_1pn': first-order GR contribution
        - 'schwarzschild_2pn': second-order GR correction
        - 'quadrupole_J2': solar J2 contribution
        - 'total_per_orbit_rad': total per orbit in radians
        - 'total_arcsec_per_century': total in arcsec/century
        - 'orbits_per_century': number of orbits per century
    """
    T_MERCURY_S = 87.969 * 86400.0
    seconds_per_century = 100.0 * 365.25 * 86400.0
    orbits_per_century = seconds_per_century / T_MERCURY_S

    phi_1pn = schwarzschild_precession(a, e, M)
    phi_2pn = schwarzschild_2pn(a, e, M) if include_2pn else 0.0
    phi_J2 = quadrupole_precession(a, e, J2, R) if include_J2 else 0.0

    total_per_orbit = phi_1pn + phi_2pn + phi_J2
    total_per_century_rad = total_per_orbit * orbits_per_century
    rad_to_arcsec = 180.0 * 3600.0 / np.pi
    total_arcsec = total_per_century_rad * rad_to_arcsec

    return {
        "schwarzschild_1pn_rad": phi_1pn,
        "schwarzschild_2pn_rad": phi_2pn,
        "quadrupole_J2_rad": phi_J2,
        "total_per_orbit_rad": total_per_orbit,
        "total_arcsec_per_century": total_arcsec,
        "orbits_per_century": orbits_per_century,
    }


def precession_vs_eccentricity(
    e_range: ArrayLike,
    a: float = A_MERCURY,
    M: float = M_SUN,
) -> np.ndarray:
    """Compute Schwarzschild precession as a function of eccentricity."""
    e_arr = np.asarray(e_range, dtype=float)
    return 6.0 * np.pi * G * M / (a * (1.0 - e_arr**2) * C**2)


def precession_vs_semi_major(
    a_range: ArrayLike,
    e: float = E_MERCURY,
    M: float = M_SUN,
) -> np.ndarray:
    """Compute Schwarzschild precession as a function of semi-major axis."""
    a_arr = np.asarray(a_range, dtype=float)
    return 6.0 * np.pi * G * M / (a_arr * (1.0 - e**2) * C**2)
