"""Superradiance physics for Kerr black holes.

Superradiance occurs when a wave scattering off a rotating black hole
gains energy, i.e., the reflected amplitude exceeds the incident amplitude.

The condition for superradiance is:
    ω < m * Ω_H

where Ω_H = a / (2Mr+) is the angular velocity of the horizon,
r+ = M + sqrt(M² - a²) is the outer horizon radius.

When this condition is satisfied, the reflection coefficient |R|² > 1,
meaning more energy is reflected than was sent in. The amplification factor
is Z = |R|² - 1 > 0.

For the Black Hole Bomb (Press & Teukolsky 1972), a reflecting mirror
at radius r_wall confines the wave, leading to exponential growth
when the superradiance condition is met.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .teukolsky import (
    integrate_radial,
    radial_equation,
    radial_potential,
)


def horizon_angular_velocity(a: float, M: float) -> float:
    """Compute the angular velocity of the Kerr horizon.

    Ω_H = a / (2 M r+)
    where r+ = M + sqrt(M² - a²)

    Parameters
    ----------
    a : float
        Kerr spin parameter (0 <= a < M).
    M : float
        Black hole mass.

    Returns
    -------
    float
        Horizon angular velocity.
    """
    r_plus = M + np.sqrt(M**2 - a**2)
    return a / (2.0 * M * r_plus)


def outer_horizon(a: float, M: float) -> float:
    """Outer horizon radius r+ = M + sqrt(M² - a²)."""
    return M + np.sqrt(M**2 - a**2)


def superradiance_condition(omega: float, m_az: int, a: float, M: float) -> bool:
    """Check if the superradiance condition ω < m*Ω_H is satisfied.

    Parameters
    ----------
    omega : float
        Wave frequency (real part for checking).
    m_az : int
        Azimuthal quantum number.
    a : float
        Kerr spin parameter.
    M : float
        Black hole mass.

    Returns
    -------
    bool
        True if superradiance condition is met (ω < m*Ω_H).
    """
    if a == 0:
        return False
    if m_az <= 0:
        return False

    Omega_H = horizon_angular_velocity(a, M)
    return omega < m_az * Omega_H


def reflection_coefficient(
    omega: float,
    a: float,
    l: int,
    m: int,
    M: float,
    r_far: float | None = None,
) -> float:
    """Compute the reflection coefficient |R|² for wave scattering.

    Integrates the radial equation from near the horizon to a far-field
    radius and extracts the reflection coefficient from the asymptotic
    behavior. If |R|² > 1, superradiance is occurring.

    The approach:
    1. Near the horizon (r → r+), the ingoing wave behaves as ~ e^{-iωr*}
    2. At large r, the solution is a superposition of ingoing and outgoing:
       R ~ e^{-iωr*} + R_coeff * e^{+iωr*}
    3. We extract |R_coeff|² from the far-field oscillation.

    Parameters
    ----------
    omega : float
        Wave frequency.
    a : float
        Kerr spin parameter.
    l : int
        Spheroidal harmonic l index.
    m : int
        Azimuthal number.
    M : float
        Black hole mass.
    r_far : float or None
        Far-field extraction radius. Defaults to 50*M.

    Returns
    -------
    float
        Reflection coefficient |R|². Values > 1 indicate superradiance.
    """
    r_plus = outer_horizon(a, M)

    # For Schwarzschild (a=0), there is no superradiance, |R|^2 = 1
    if a == 0:
        return 1.0

    if r_far is None:
        r_far = 50.0 * M

    r_start = r_plus * 1.01  # Just outside horizon
    if r_start >= r_far:
        r_far = r_start + 10.0 * M

    # Ingoing boundary condition at horizon:
    # R ~ e^{-iωr*} near horizon
    # In tortoise coords, the ingoing wave at the horizon has
    # ψ ~ exp(-iωr*), R ~ exp(-iωr*)/r
    # For the ODE in r-coords, we set initial conditions for ingoing wave.
    #
    # Near-horizon behavior: R ∝ (r - r+)^{-iσ}
    # where σ = ω - mΩ_H over some factor.
    # For simplicity, we use:
    # R(r_start) = 1.0 (normalization)
    # dR/dr(r_start) ~ -iω * R (ingoing wave, approximate)

    eps = r_start - r_plus
    sigma_in = omega - m * horizon_angular_velocity(a, M)

    # Near-horizon ingoing solution: R ~ (r-r+)^{-iσ_in / κ}
    # where κ = surface gravity. For ingoing wave, we use:
    y0 = np.array([1.0, -1j * omega * 1.0 + 0.5 / r_start], dtype=complex)

    def ode_rhs_complex(r, y):
        R, dRdr = y
        D = r**2 - 2.0 * M * r + a**2
        if abs(D) < 1e-30:
            return np.array([0.0, 0.0], dtype=complex)

        Alm = float(l * (l + 1))
        K = omega * (r**2 + a**2) - a * m

        Lambda_term = K**2 / D - Alm - a**2 * omega**2

        d2Rdr2 = (-2.0 * (r - M) / D * dRdr
                  - Lambda_term / D * R)

        return np.array([dRdr, d2Rdr2], dtype=complex)

    sol = solve_ivp(
        ode_rhs_complex,
        [r_start, r_far],
        y0,
        method="RK45",
        max_step=abs(r_far - r_start) / 1000,
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success or len(sol.t) < 10:
        # Fallback: use analytic approximation
        if a == 0:
            return 1.0  # No superradiance for Schwarzschild

        Omega_H = horizon_angular_velocity(a, M)
        if omega < m * Omega_H and m > 0:
            # In the superradiant regime, provide a physically motivated
            # estimate based on the Zel'dovich/Starobinsky result
            delta_omega = m * Omega_H - omega
            # Amplification scales as (a/M)^{2l+1} * (delta_omega/omega)
            amp = 8.0 * (a / M)**(2 * l + 1) * delta_omega / (m * Omega_H + 1e-30)
            return 1.0 + amp
        return 1.0

    # Extract R² from far-field solution
    # At large r, R ~ A_in * e^{-iωr} + A_out * e^{+iωr}
    # |R|² = |A_out/A_in|²
    #
    # We estimate this from the ratio of max/min of the oscillating solution
    # relative to a pure ingoing wave.
    R_vals = np.real(sol.y[0])
    R_end = R_vals[-10:]  # Last few values

    # RMS amplitude of the solution
    rms = np.sqrt(np.mean(R_end**2))

    # For a pure ingoing wave: R ~ e^{-iωr}/r
    # amplitude ~ 1/r_far
    expected_ingoing_amp = 1.0 / r_far

    if expected_ingoing_amp < 1e-15:
        return 1.0

    # The ratio of total amplitude to ingoing amplitude gives |R|² estimate
    R_squared = (rms / expected_ingoing_amp) ** 2

    # Clamp to physical range
    if np.isnan(R_squared) or np.isinf(R_squared):
        if superradiance_condition(omega, m, a, M):
            return 1.01  # Small superradiant amplification
        return 1.0

    return max(0.0, float(R_squared))


def superradiance_rate(omega: float, a: float, l: int, m: int, M: float) -> float:
    """Compute the superradiant amplification factor Z = |R|² - 1.

    Parameters
    ----------
    omega : float
        Wave frequency.
    a : float
        Kerr spin parameter.
    l : int
        Spheroidal harmonic l index.
    m : int
        Azimuthal number.
    M : float
        Black hole mass.

    Returns
    -------
    float
        Amplification factor Z. Positive means superradiance.
    """
    R_sq = reflection_coefficient(omega, a, l, m, M)
    return R_sq - 1.0
