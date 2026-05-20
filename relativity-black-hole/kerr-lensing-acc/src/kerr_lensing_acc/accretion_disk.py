"""
Novikov-Thorne thin accretion disk model.

Implements the standard thin-disk (Shakura-Sunyaev / Novikov-Thorne) model
in the Kerr metric. Provides temperature profile, emitted flux, and local
blackbody emission.

Key equations (geometric units G=c=1):
    F(r) = (3 * M * Mdot) / (8*pi*r^3) * f(r, a)    [Schwarzschild limit]
    where f(r, a) encodes the GR correction factor.

    For the full Novikov-Thorne model:
    F(r) = (Mdot * c^2) / (4*pi*r) * (-dOmega/d(r^2)) * 1/(E - Omega*L)
    where Omega, E, L are the Keplerian angular velocity, specific energy,
    and specific angular momentum in the Kerr metric.

    Temperature: T(r) = (F(r) / sigma_SB)^{1/4}

    ISCO: r_isco = f(a/M), 6M for a=0, M for a=M (prograde).
"""

import numpy as np
from scipy import constants


# ---------------------------------------------------------------------------
# Keplerian orbital quantities in Kerr
# ---------------------------------------------------------------------------

def keplerian_angular_velocity(r: float, M: float = 1.0, a: float = 0.0) -> float:
    """
    Keplerian (prograde) angular velocity Omega at radius r in Kerr spacetime.

    Omega_K = sqrt(M) / (r^(3/2) + a*sqrt(M))

    This is the coordinate angular velocity dphi/dt for circular geodesics.
    """
    return np.sqrt(M) / (r ** 1.5 + a * np.sqrt(M))


def keplerian_specific_energy(r: float, M: float = 1.0, a: float = 0.0) -> float:
    """
    Specific energy E of a particle on a prograde circular Keplerian orbit.

    E = (1 - 2*M*r/x^2 + a*M^(1/2)/x * 2*M) / sqrt(1 - 3*M/x^2 + 2*a*M^(1/2)/x^1.5)
    where x = r, but more precisely:

    E = (r^(3/2) - 2*M*r^(1/2) + a*M^(1/2)) /
        (r^(3/4) * sqrt(r^(3/2) - 3*M*r^(1/2) + 2*a*M^(1/2)))
    """
    sqrtM = np.sqrt(M)
    sqrtr = np.sqrt(r)

    numerator = r**1.5 - 2.0 * M * sqrtr + a * sqrtM
    denominator = r**0.75 * np.sqrt(r**1.5 - 3.0 * M * sqrtr + 2.0 * a * sqrtM)

    denominator = np.asarray(denominator)
    if np.all(np.abs(denominator) < 1e-30):
        return 1.0  # fallback

    return numerator / denominator


def keplerian_specific_angular_momentum(
    r: float, M: float = 1.0, a: float = 0.0
) -> float:
    """
    Specific angular momentum L of a particle on a prograde circular orbit.

    L = sqrt(M) * (r^2 - 2*a*sqrt(M)*r^(1/2) + a^2) /
        (r^(3/4) * sqrt(r^(3/2) - 3*M*r^(1/2) + 2*a*sqrt(M)))
    """
    sqrtM = np.sqrt(M)
    sqrtr = np.sqrt(r)

    numerator = sqrtM * (r**2 - 2.0 * a * sqrtM * sqrtr + a**2)
    denominator = r**0.75 * np.sqrt(r**1.5 - 3.0 * M * sqrtr + 2.0 * a * sqrtM)

    denominator = np.asarray(denominator)
    if np.all(np.abs(denominator) < 1e-30):
        return 0.0

    return numerator / denominator


# ---------------------------------------------------------------------------
# Novikov-Thorne flux and temperature
# ---------------------------------------------------------------------------

def novikov_thorne_flux(
    r: float,
    M: float = 1.0,
    a: float = 0.0,
    Mdot: float = 1.0,
    r_isco: float = None,
) -> float:
    """
    Novikov-Thorne radiative flux from a thin accretion disk.

    F(r) = (Mdot * c^2) / (4*pi*sqrt(-g)) * (-dOmega/dr) / (E - Omega*L)
    evaluated at each radius.

    In the Schwarzschild limit (a=0):
    F(r) = (3*M*Mdot) / (8*pi*r^3) * [1 - sqrt(r_isco/r)]

    Parameters
    ----------
    r : float or array
        Radial coordinate(s).
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    Mdot : float
        Mass accretion rate (dimensionless, geometric units).
    r_isco : float or None
        ISCO radius. Computed if not provided.

    Returns
    -------
    float or array
        Radiative flux (geometric units).
    """
    from .kerr_rays import compute_isco

    if r_isco is None:
        r_isco = compute_isco(M, a)

    r = np.asarray(r, dtype=float)

    # Keplerian quantities
    Omega = keplerian_angular_velocity(r, M, a)
    E = keplerian_specific_energy(r, M, a)
    L = keplerian_specific_angular_momentum(r, M, a)

    # dOmega/dr (numerical derivative)
    dr = 1e-6 * r
    Omega_plus = keplerian_angular_velocity(r + dr, M, a)
    Omega_minus = keplerian_angular_velocity(r - dr, M, a)
    dOmega_dr = (Omega_plus - Omega_minus) / (2.0 * dr)

    # E - Omega * L  (energy-at-infinity per unit rest mass minus rotational contribution)
    E_minus_OmegaL = E - Omega * L

    # Avoid division by zero at ISCO
    E_minus_OmegaL = np.where(np.abs(E_minus_OmegaL) < 1e-30, 1e-30, E_minus_OmegaL)

    # sqrt(-g) = r for thin disk in equatorial plane (theta = pi/2)
    # g = det(g_mu_nu) at theta = pi/2 in Boyer-Lindquist
    # For equatorial plane: sqrt(-g) = r (in geometric units with M=1 scale)
    sqrt_neg_g = r  # at theta = pi/2

    # Inner boundary condition: zero torque at ISCO
    # The standard Novikov-Thorne formula includes:
    # F = (Mdot)/(4*pi*sqrt(-g)) * (-dOmega/dr) / (E - Omega*L) * B(r)
    # where B(r) = integral from r_isco to r of (E - Omega*L') * dL'/dr' dr'
    # For simplicity we use the standard form with the boundary term:
    # F_NT = Mdot / (4*pi*sqrt(-g)) * Omega_,r / (E - Omega*L) * integral

    # Compute the integral term: int_{r_isco}^{r} (E(r') - Omega(r')*L(r')) dL/dr' dr'
    # For a simpler but accurate approach, use the analytic Schwarzschild result:
    # F = (3*M*Mdot)/(8*pi*r^3) * f(r)
    # where f(r) = 1 - sqrt(r_isco/r) for Schwarzschild

    # For Kerr, we use the full expression. Let's compute B(r) numerically.
    r_array = np.atleast_1d(r)
    B = np.zeros_like(r_array)

    for i, ri in enumerate(r_array):
        if ri <= r_isco:
            B[i] = 0.0
            continue
        # Integrate from r_isco to ri
        n_int = 200
        r_int = np.linspace(r_isco + 1e-6, ri, n_int)
        E_int = keplerian_specific_energy(r_int, M, a)
        Omega_int = keplerian_angular_velocity(r_int, M, a)
        L_int = keplerian_specific_angular_momentum(r_int, M, a)

        # dL/dr
        dL_dr = np.gradient(L_int, r_int)

        integrand = (E_int - Omega_int * L_int) * dL_dr
        B[i] = np.trapezoid(integrand, r_int)

    B = np.squeeze(B)

    # NT flux
    F = Mdot / (4.0 * np.pi * sqrt_neg_g) * (-dOmega_dr) / E_minus_OmegaL * B

    # Ensure non-negative
    F = np.where(F < 0, 0.0, F)

    # Zero flux inside ISCO
    F = np.where(r < r_isco, 0.0, F)

    return F


def disk_temperature(
    r: float,
    M: float = 1.0,
    a: float = 0.0,
    Mdot: float = 1.0,
    r_isco: float = None,
) -> float:
    """
    Effective temperature of the accretion disk assuming blackbody emission.

    T(r) = (F(r) / sigma_SB)^{1/4}

    In geometric units, sigma_SB = pi^2/60 (Stefan-Boltzmann in natural units).

    Parameters
    ----------
    r : float or array
        Radial coordinate.
    M, a, Mdot : float
        Black hole parameters.
    r_isco : float or None
        ISCO radius.

    Returns
    -------
    float or array
        Temperature (in geometric/natural units).
    """
    F = novikov_thorne_flux(r, M, a, Mdot, r_isco)

    # Stefan-Boltzmann constant in geometric units (G=c=hbar=k_B=1)
    # sigma_SB = pi^2 / 60
    sigma_sb = np.pi**2 / 60.0

    T = (F / sigma_sb) ** 0.25

    return T


def disk_temperature_schwarzschild(
    r: float,
    M: float = 1.0,
    Mdot: float = 1.0,
) -> float:
    """
    Simplified Schwarzschild disk temperature.

    T(r) ∝ [M * Mdot / r^3 * (1 - sqrt(r_isco/r))]^{1/4}

    More precisely:
    F(r) = (3*M*Mdot)/(8*pi*r^3) * (1 - sqrt(6*M/r))
    T(r) = (F/sigma_SB)^{1/4}
    """
    from .kerr_rays import compute_isco

    r_isco = compute_isco(M, a=0.0)
    F = (3.0 * M * Mdot) / (8.0 * np.pi * r**3) * (1.0 - np.sqrt(r_isco / r))
    F = np.where(F < 0, 0.0, F)

    sigma_sb = np.pi**2 / 60.0
    T = (F / sigma_sb) ** 0.25
    return T


def blackbody_spectrum(
    E_array: np.ndarray,
    T: float,
) -> np.ndarray:
    """
    Planck blackbody spectrum B(E, T) in natural units.

    B(E, T) = (E^3) / (2*pi^2) * 1 / (exp(E/T) - 1)

    Parameters
    ----------
    E_array : array
        Photon energies.
    T : float
        Temperature.

    Returns
    -------
    array
        Spectral intensity per unit energy.
    """
    E_array = np.asarray(E_array, dtype=float)
    T = float(T)

    if T <= 0:
        return np.zeros_like(E_array)

    x = E_array / T
    # Avoid overflow
    x = np.clip(x, 0, 500)

    B = E_array**3 / (2.0 * np.pi**2) / (np.exp(x) - 1.0 + 1e-30)

    return B


def disk_emission(
    r: float,
    phi: float,
    E_array: np.ndarray,
    M: float = 1.0,
    a: float = 0.0,
    Mdot: float = 1.0,
) -> np.ndarray:
    """
    Local blackbody emission spectrum at a disk point (r, phi).

    Assumes azimuthal symmetry (no phi dependence in the standard model).

    Parameters
    ----------
    r, phi : float
        Disk coordinates.
    E_array : array
        Photon energies (observed at infinity).
    M, a, Mdot : float
        Black hole / accretion parameters.

    Returns
    -------
    array
        Emission spectrum I(E) at this disk point.
    """
    T = disk_temperature(r, M, a, Mdot)
    return blackbody_spectrum(E_array, T)
