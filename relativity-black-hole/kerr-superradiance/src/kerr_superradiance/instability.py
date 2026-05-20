"""Instability growth rates for superradiant systems.

When a massive scalar field (mass μ) surrounds a Kerr black hole,
superradiant modes with ω < mΩ_H can be trapped by the mass term
(no need for a mirror). This creates a bound state that grows
exponentially — the superradiant instability.

The instability timescale τ ~ 1/ω_I where ω = ω_R + iω_I is the
complex quasi-normal mode frequency. For small a*M*μ:
    ω_I ~ (a/M)^{2l+1} * μ^{4l+5} * M^{4l+4} * C_{lm}

where C_{lm} is a numerical coefficient depending on l, m.

For the Black Hole Bomb (mirror at r_wall), the growth rate is set by
the reflection coefficient and the cavity round-trip time:
    Γ ~ (|R|² - 1) * c / (2 * r_wall)

The scalar cloud profile is a hydrogen-like bound state peaked at
r ~ n²/(μ² M) where n is the principal quantum number.
"""

import numpy as np

from .superradiance import (
    horizon_angular_velocity,
    outer_horizon,
    reflection_coefficient,
    superradiance_condition,
)


def instability_growth_rate(
    mu: float,
    a: float,
    l: int,
    m: int,
    M: float,
    r_wall: float | None = None,
) -> float:
    """Compute the instability growth rate for a massive scalar field.

    For massive scalar fields around Kerr, the superradiant instability
    growth rate can be computed analytically in the small μM limit.
    The dominant mode is l = m = 1.

    The growth rate for the (n, l, m) mode (Detweiler 1980, Zouros & Eardley 1979):
        ω_I ≈ 2 * γ_{nlm} * (r_g * μ)^{4l+5} * (a/M)^{2l+1} * (1/(r_g * M))

    where γ_{nlm} is a numerical coefficient. For l=m=1:
        γ ≈ 48

    For l=m=2:
        γ ≈ ~10^5 (much smaller for same μM)

    When a reflecting wall is present at r_wall (Black Hole Bomb):
        Γ ≈ (|R|² - 1) / (2 * r_wall)

    Parameters
    ----------
    mu : float
        Scalar field mass (in geometric units).
    a : float
        Kerr spin parameter.
    l : int
        Orbital quantum number.
    m : int
        Azimuthal quantum number.
    M : float
        Black hole mass.
    r_wall : float or None
        Mirror radius for BH bomb. If None, uses massive field trapping.

    Returns
    -------
    float
        Growth rate Γ (positive = unstable). In units of 1/M.
    """
    if not superradiance_condition(mu, m, a, M):
        # Not in superradiant regime — no instability
        # (The real part of ω ≈ μ for bound states, so check μ < mΩ_H)
        return 0.0

    if r_wall is not None:
        # Black Hole Bomb: growth rate from cavity amplification
        return black_hole_bomb_growth(a, M, r_wall)

    # Massive scalar field instability (no mirror)
    # Analytic result in the small μM regime
    muM = mu * M

    # Numerical coefficients γ_{nlm} from Detweiler (1980) and
    # subsequent exact calculations (Dolan 2007):
    # n = l + 1 (ground state for given l)
    if l == 1 and m == 1:
        gamma = 48.0
    elif l == 2 and m == 2:
        gamma = 2.0e5
    elif l == 2 and m == 1:
        gamma = 1.0e4
    elif l == 1 and m == 0:
        gamma = 0.0  # m=0: no superradiance
    elif l == 0:
        gamma = 0.0  # l=0, m=0: no superradiance
    else:
        # Generic scaling estimate
        gamma = 48.0 * (2 * l + 1) * np.math.factorial(l + m) / np.math.factorial(l - m + 1)

    if gamma == 0.0:
        return 0.0

    # Growth rate: ω_I = γ * (μM)^{4l+5} * (a/M)^{2l+1} / M
    a_over_M = a / M
    exponent_mu = 4 * l + 5
    exponent_a = 2 * l + 1

    growth = gamma * muM**exponent_mu * a_over_M**exponent_a / M

    return growth


def black_hole_bomb_growth(a: float, M: float, r_wall: float) -> float:
    """Compute the Black Hole Bomb instability growth rate.

    The Press-Teukolsky Black Hole Bomb mechanism: a reflecting mirror
    at radius r_wall traps superradiant modes. The growth rate is:

        Γ = (|R|² - 1) / (2 * r_wall / c)

    where |R|² is the reflection coefficient and 2*r_wall is approximately
    the round-trip time for the wave.

    Parameters
    ----------
    a : float
        Kerr spin parameter.
    M : float
        Black hole mass.
    r_wall : float
        Mirror radius (must be > outer horizon).

    Returns
    -------
    float
        Growth rate Γ in units of 1/M.
    """
    r_plus = outer_horizon(a, M)
    if r_wall <= r_plus:
        return 0.0

    Omega_H = horizon_angular_velocity(a, M)

    # For the dominant mode, use m=1, l=1 at frequency near m*Ω_H
    # The superradiant mode frequency is approximately ω ≈ m*Ω_H/2
    omega = 0.5 * Omega_H  # Characteristic superradiant frequency for m=1

    if not superradiance_condition(omega, 1, a, M):
        return 0.0

    # Reflection coefficient for m=1, l=1
    R_sq = reflection_coefficient(omega, a, 1, 1, M, r_far=r_wall)

    # Growth rate: amplification per round trip
    # Round-trip time ~ 2 * r_wall (in geometric units c=1)
    round_trip_time = 2.0 * r_wall

    growth = (R_sq - 1.0) / round_trip_time

    return max(0.0, growth)


def superradiant_cloud_profile(
    mu: float,
    a: float,
    l: int,
    m: int,
    M: float,
    r_array: np.ndarray,
) -> np.ndarray:
    """Compute the spatial profile of a superradiant scalar cloud.

    For a massive scalar field in the hydrogenic regime (μM << 1),
    the cloud profile is approximately a hydrogen-like wavefunction:

        ψ(r) ~ r^l * L_{n-l-1}^{2l+1}(2μr/(l+1)) * exp(-μr/(l+1))

    where L is the generalized Laguerre polynomial and n = l+1 for the
    ground state. The cloud peaks at r ~ (l+1)²/(μ) in geometric units,
    which for l=m=1 gives r_peak ~ 4/μ = 4/(μM) * M.

    Parameters
    ----------
    mu : float
        Scalar field mass.
    a : float
        Kerr spin parameter.
    l : int
        Orbital quantum number.
    m : int
        Azimuthal quantum number.
    M : float
        Black hole mass.
    r_array : np.ndarray
        Radial coordinates at which to evaluate the profile.

    Returns
    -------
    np.ndarray
        Cloud profile |ψ(r)|² (normalized to peak = 1).
    """
    if mu <= 0:
        return np.zeros_like(r_array)

    n = l + 1  # Principal quantum number for ground state

    # Hydrogenic radial wavefunction (Bohr radius a_0 = n/(μ) in geometric units)
    a_bohr = float(n) / mu  # = (l+1)/μ

    # Generalized Laguerre polynomial L_{n-l-1}^{2l+1}(ρ)
    # For n=l+1, the Laguerre polynomial is L_0^{2l+1}(ρ) = 1
    # So the wavefunction simplifies to:
    # ψ ~ r^l * exp(-r/a_bohr)
    rho = 2.0 * r_array / (n * a_bohr)

    # For n = l+1 (ground state for given l):
    if n - l - 1 == 0:
        # L_0^{2l+1} = 1
        psi = r_array**l * np.exp(-r_array / a_bohr)
    else:
        # Use generalized Laguerre polynomial
        from scipy.special import genlaguerre
        k = n - l - 1
        L_poly = genlaguerre(k, 2 * l + 1)
        psi = r_array**l * L_poly(rho) * np.exp(-rho / 2.0)

    # For Kerr, include a correction factor for frame dragging
    # The cloud profile is slightly modified near the BH, but the
    # hydrogenic approximation is excellent for μM << 1.
    # We cut off the profile inside the horizon.
    r_plus = outer_horizon(a, M)
    mask = r_array > r_plus
    psi = np.where(mask, psi, 0.0)

    # Normalize to peak = 1
    psi_sq = psi**2
    peak = np.max(psi_sq)
    if peak > 0:
        psi_sq = psi_sq / peak

    return psi_sq
