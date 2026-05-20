"""Novikov-Thorne thin accretion disk model.

Computes the radial temperature profile and emitted spectrum
of a geometrically thin, optically thick accretion disk around
a Schwarzschild black hole.

References:
    - Novikov & Thorne (1973), in "Black Holes"
    - Page & Thorne (1974), ApJ 191, 499
    - Shakura & Sunyaev (1973), A&A 24, 337
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# Geometric units: G = c = 1, mass parameter M = 1
# ISCO at r = 6M (Schwarzschild)


def disk_temperature_profile(
    r: ArrayLike,
    M_bh: float,
    M_dot: float,
    r_isco: float = 6.0,
    efficiency: float | None = None,
) -> np.ndarray:
    """Novikov-Thorne disk temperature profile T(r).

    The local flux from the disk:
        F(r) = (3 G M M_dot) / (8π σ r³) · f(r)

    where f(r) = 1 - √(r_isco/r) captures the zero-torque inner boundary condition.
    Temperature: σ_SB T⁴ = F → T(r) = (F/σ_SB)^(1/4)

    Parameters
    ----------
    r : radial coordinate in units of GM/c² (array)
    M_bh : black hole mass [kg]
    M_dot : mass accretion rate [kg/s]
    r_isco : ISCO radius in gravitational radii (default 6 for Schwarzschild)
    efficiency : radiative efficiency η (if None, computed from r_isco)

    Returns
    -------
    T : temperature [K] at each radius
    """
    from .constants import G, C, SIGMA_SB

    r = np.asarray(r, dtype=float)
    rs = 2.0 * G * M_bh / C**2  # Schwarzschild radius
    r_m = r * G * M_bh / C**2  # convert from gravitational radii to meters

    # NT flux correction factor
    f_nt = np.where(r > r_isco, 1.0 - np.sqrt(r_isco / r), 0.0)

    # Flux [W/m²]
    F = 3.0 * G * M_bh * M_dot / (8.0 * np.pi * r_m**3) * f_nt

    # Temperature [K]
    T = np.where(F > 0, (F / SIGMA_SB) ** 0.25, 0.0)

    return T


def radiative_efficiency(r_isco: float = 6.0) -> float:
    """Radiative efficiency for Schwarzschild BH.

    η = 1 - E_isco where E_isco = √(1 - 2/(3r_isco)) · ... (Schwarzschild)
    For ISCO = 6M: η = 1 - √(8/9) ≈ 0.0572
    """
    # Specific energy at ISCO for Schwarzschild
    E_isco = np.sqrt(1.0 - 2.0 / (3.0 * r_isco))
    return 1.0 - E_isco


def local_blackbody(
    nu: ArrayLike,
    T: float,
) -> np.ndarray:
    """Planck blackbody spectrum B_ν(T).

    B_ν = 2hν³/c² · 1/(exp(hν/kT) - 1)

    Parameters
    ----------
    nu : frequency [Hz]
    T : temperature [K]

    Returns
    -------
    B_nu : spectral radiance [W/m²/Hz/sr]
    """
    from .constants import H, C, K_B

    nu = np.asarray(nu, dtype=float)
    x = H * nu / (K_B * T)
    x = np.clip(x, 0, 500)  # prevent overflow
    return 2.0 * H * nu**3 / C**2 / (np.exp(x) - 1.0 + 1e-300)


def observed_spectrum(
    nu_observer: ArrayLike,
    M_bh: float,
    M_dot: float,
    inclination: float = 0.0,
    distance: float = 1.0,
    r_in: float = 6.0,
    r_out: float = 1000.0,
    n_rings: int = 200,
) -> np.ndarray:
    """Compute the observed multi-temperature blackbody spectrum.

    Integrates contributions from concentric annuli, accounting for:
    - Doppler boosting (orbital velocity)
    - Gravitational redshift
    - Projection (cos i)

    Parameters
    ----------
    nu_observer : observed frequencies [Hz]
    M_bh : black hole mass [kg]
    M_dot : accretion rate [kg/s]
    inclination : viewing angle from disk normal [radians]
    distance : distance to source [m]
    r_in, r_out : inner/outer disk edge in gravitational radii
    n_rings : number of annuli for integration

    Returns
    -------
    F_nu : observed flux density [W/m²/Hz]
    """
    from .constants import G, C, H, K_B

    nu = np.asarray(nu_observer, dtype=float)
    rs = 2.0 * G * M_bh / C**2
    cos_i = np.cos(inclination)

    # Logarithmically spaced rings
    r_rings = np.logspace(np.log10(r_in), np.log10(r_out), n_rings + 1)
    r_mid = np.sqrt(r_rings[:-1] * r_rings[1:])  # geometric mean
    dr_rings = np.diff(r_rings)

    F_nu = np.zeros_like(nu)

    for idx in range(n_rings):
        r = r_mid[idx]
        dr = dr_rings[idx]

        # Temperature at this ring
        T = disk_temperature_profile(np.array([r]), M_bh, M_dot, r_in)[0]
        if T < 1.0:
            continue

        # Gravitational redshift at this radius
        g_tt = np.sqrt(1.0 - 2.0 / r)
        # Orbital velocity (circular orbit in Schwarzschild)
        v_orb = 1.0 / np.sqrt(r)  # in units of c
        # Doppler factor (simplified: orbital plane)
        doppler = 1.0 / (1.0 + v_orb * np.sin(inclination))

        # Combined redshift factor
        redshift = g_tt * doppler

        # Redshifted frequency in the disk frame
        nu_disk = nu / redshift

        # Local blackbody emission
        B_nu = local_blackbody(nu_disk, T)

        # Area element of ring (proper area)
        r_phys = r * G * M_bh / C**2
        dA = 2.0 * np.pi * r_phys * (dr * G * M_bh / C**2)  # proper area

        # Observed flux from this ring
        F_nu += B_nu * cos_i * dA * redshift**3 / (4.0 * np.pi * distance**2)

    return F_nu


def peak_temperature(M_bh: float, M_dot: float, r_isco: float = 6.0) -> float:
    """Temperature at the peak of the emission profile."""
    from .constants import G, C

    # Peak flux is at r ≈ (49/36) r_isco (Page & Thorne 1974)
    r_peak = (49.0 / 36.0) * r_isco
    T = disk_temperature_profile(np.array([r_peak]), M_bh, M_dot, r_isco)
    return T[0]


def eddington_luminosity(M_bh: float) -> float:
    """Eddington luminosity L_Edd = 4πGM m_p c / σ_T."""
    from .constants import G, C, M_PROTON, SIGMA_T
    return 4.0 * np.pi * G * M_bh * M_PROTON * C / SIGMA_T


def eddington_accretion_rate(M_bh: float, efficiency: float = 0.0572) -> float:
    """Eddington accretion rate M_dot_Edd = L_Edd / (η c²)."""
    from .constants import C
    return eddington_luminosity(M_bh) / (efficiency * C**2)
