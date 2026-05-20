"""
Teukolsky equation machinery for Kerr black hole perturbations.

Provides the effective radial potential, angular eigenvalue computation
via continued fractions, and horizon frequency calculation.
"""

from __future__ import annotations

import numpy as np
import mpmath


# ---------------------------------------------------------------------------
# Radial Teukolsky potential
# ---------------------------------------------------------------------------

def teukolsky_radial_potential(
    r: float | complex,
    a: float,
    M: float,
    omega: complex,
    l: int,
    m: int,
    s: int = -2,
) -> complex:
    """
    Effective potential for spin-*s* perturbations in the Kerr geometry.

    The Teukolsky radial equation (in Boyer-Lindquist coordinates) is

        Δ d²R/dr*² + (ω²r² - 2amωr + a²m²)Δ/(r² + a²)² R = V_eff R

    We return the effective potential *V_eff* as a function of Boyer-Lindquist
    radius *r*, with the spin parameter *a* (0 ≤ a < M), mass *M*,
    frequency ω, spheroidal harmonic indices (l, m), and spin weight s.

    Parameters
    ----------
    r : float or complex
        Boyer-Lindquist radial coordinate (in units of M).
    a : float
        Kerr spin parameter (a = J/M, in units of M).
    M : float
        Black hole mass (set to 1 for geometric units).
    omega : complex
        Mode frequency.
    l, m : int
        Spheroidal harmonic indices.
    s : int
        Spin weight of the field (0 scalar, ±1 EM, ±2 gravitational).

    Returns
    -------
    complex
        Value of the effective potential at radius *r*.
    """
    r = mpmath.mpc(r)
    omega = mpmath.mpc(omega)
    a = mpmath.mpf(a)
    M = mpmath.mpf(M)

    # Delta = r^2 - 2Mr + a^2
    Delta = r ** 2 - 2 * M * r + a ** 2

    # Horizons
    r_plus = M + mpmath.sqrt(M ** 2 - a ** 2)

    # Angular eigenvalue (separation constant)
    A_lm = teukolsky_angular_eigenvalue(float(a), complex(omega), l, m, s)
    A_lm = mpmath.mpc(A_lm)

    # Potassium K = (r^2 + a^2) omega - a m
    K = (r ** 2 + a ** 2) * omega - a * m

    # The effective potential in the Sasaki-Nakamura / Teukolsky form
    # Using the form from Teukolsky (1973), Eq. (3.7) for spin-s:
    # V_eff includes the angular eigenvalue and the coupling terms.

    # For the radial Teukolsky equation the effective potential is:
    # V = [K^2 - i s K Delta'] / Delta + (i s Delta' - Delta'')/2
    #     + (s(s+1) - A_lm) * Delta / (r^2 + a^2)
    #     + 2 i s (2 omega r - a m) / (r^2 + a^2)
    # where ' = d/dr

    Delta_prime = 2 * r - 2 * M
    Delta_double_prime = mpmath.mpf(2)

    rho2 = r ** 2 + a ** 2

    V = (
        (K ** 2 - 1j * s * K * Delta_prime) / Delta
        + (1j * s * Delta_prime - Delta_double_prime) / 2
        + (s * (s + 1) - A_lm) * Delta / rho2
        + 2j * s * (2 * omega * r - a * m) / rho2
    )

    return complex(V)


# ---------------------------------------------------------------------------
# Angular separation constant
# ---------------------------------------------------------------------------

def teukolsky_angular_eigenvalue(
    a: float,
    omega: complex,
    l: int,
    m: int,
    s: int = -2,
) -> complex:
    """
    Angular separation constant A_lm for the spin-weighted spheroidal
    harmonic equation (angular Teukolsky equation).

    Uses the continued-fraction method to solve the three-term recurrence
    for the angular eigenvalue.  For *a* = 0 this reduces to the exact
    result  A = l(l+1) - s(s+1).

    Parameters
    ----------
    a : float
        Kerr spin parameter (units of M).
    omega : complex
        Mode frequency (geometric units).
    l, m : int
        Spheroidal harmonic indices.
    s : int
        Spin weight.

    Returns
    -------
    complex
        Separation constant A_{lm}.
    """
    # Exact Schwarzschild limit
    if abs(a) < 1e-15 and abs(omega) < 1e-15:
        return complex(l * (l + 1) - s * (s + 1))

    # For small a*omega we use the series expansion:
    # A = l(l+1) - s(s+1) + sum_{k=1}^{N} f_k (a*omega)^k
    # The coefficients f_k are known analytically for low orders.

    # Use the continued-fraction approach with mpmath for robustness.
    c_omega = mpmath.mpc(omega)
    a_mp = mpmath.mpf(a)
    s_mp = mpmath.mpf(s)
    l_mp = mpmath.mpf(l)
    m_mp = mpmath.mpf(m)

    # The angular eigenvalue satisfies a three-term recurrence.
    # We use the perturbative expansion up to (a*omega)^4 which is
    # accurate for moderate spin (see Seidel 1989, Berti et al. 2009).

    a_omega = a_mp * c_omega

    # Base eigenvalue (Schwarzschild)
    A0 = l_mp * (l_mp + 1) - s_mp * (s_mp + 1)

    # First-order correction
    f1 = -2 * m_mp * s_mp

    # Second-order correction
    # f2 depends on l, m, s in a known way
    k1 = mpmath.mpf(1) if l != abs(m) else mpmath.mpf(0)
    if abs(l_mp - abs(m_mp)) > 0:
        f2 = (
            -2 * l_mp * (l_mp + 1) * s_mp ** 2
            + 2 * s_mp ** 2
            - 2 * m_mp ** 2 * s_mp ** 2 / (l_mp * (l_mp + 1))
        )
    else:
        f2 = -2 * l_mp * (l_mp + 1) * s_mp ** 2 + 2 * s_mp ** 2

    # Third-order correction (simplified)
    f3 = 0
    if abs(l_mp - abs(m_mp)) > 0:
        f3 = (
            -2 * m_mp * s_mp
            * (l_mp * (l_mp + 1) - s_mp ** 2 + m_mp ** 2)
        ) / 3

    A = A0 + f1 * a_omega + f2 * a_omega ** 2 + f3 * a_omega ** 3

    # For more accuracy, iterate using the continued fraction
    # (Leaver's method for the angular problem).
    # We do a fixed number of CF iterations for stability.
    _A = complex(A)
    _A = _refine_angular_eigenvalue(a, omega, _A, l, m, s, n_iter=20)

    return _A


def _refine_angular_eigenvalue(
    a: float,
    omega: complex,
    A_guess: complex,
    l: int,
    m: int,
    s: int,
    n_iter: int = 20,
) -> complex:
    """
    Refine the angular eigenvalue using Newton-Raphson on the
    continued-fraction condition for the angular recurrence.

    The three-term recurrence coefficients for the angular equation are:
        alpha_n A_n + beta_n A_n + gamma_n A_{n-1} = 0
    The CF converges when the minimal solution condition is satisfied.
    """
    a_mp = mpmath.mpf(a)
    c_omega = mpmath.mpc(omega)
    s_mp = mpmath.mpf(s)
    l_mp = mpmath.mpf(l)
    m_mp = mpmath.mpf(m)

    # Set mpmath precision
    mpmath.mp.dps = 30

    A = mpmath.mpc(A_guess)

    c = a_mp * c_omega  # c = a * omega

    for _ in range(n_iter):
        # Compute the continued fraction residual
        # Recurrence coefficients (Seidel 1989, Eqs. 2.6-2.8):
        # alpha_n * b_{n+1} + beta_n * b_n + gamma_n * b_{n-1} = 0
        # where the angular eigenvalue A appears linearly in beta_n.

        # Build the CF from the bottom up
        try:
            residual = _angular_cf_residual(c, A, l_mp, m_mp, s_mp, n_max=80)
            # Numerical derivative
            dA = mpmath.mpf("1e-12") * (1 + abs(A))
            residual_dA = _angular_cf_residual(
                c, A + dA, l_mp, m_mp, s_mp, n_max=80
            )
            deriv = (residual_dA - residual) / dA
            if abs(deriv) < 1e-50:
                break
            A = A - residual / deriv
        except Exception:
            break

    mpmath.mp.dps = 15  # reset
    return complex(A)


def _angular_cf_residual(
    c: mpmath.mpc,
    A: mpmath.mpc,
    l: mpmath.mpf,
    m: mpmath.mpf,
    s: mpmath.mpf,
    n_max: int = 80,
) -> mpmath.mpc:
    """
    Compute the continued-fraction residual for the angular eigenvalue.

    The recurrence for the angular expansion coefficients has A appearing
    in the beta_n term.  The CF must equal zero for a valid eigenvalue.
    """

    def _alpha(n: int) -> mpmath.mpc:
        # n is the recurrence index (n >= 0)
        return -c ** 2

    def _beta(n: int) -> mpmath.mpc:
        # n >= 0
        nn = mpmath.mpf(n)
        return (
            A
            + s_mp * (s_mp + 1)
            - c ** 2
            - 2 * c * m_mp * s_mp / ((nn + l_mp + s_mp + 1) * (nn + l_mp - s_mp + 1) + m_mp ** 2 - c ** 2)
            + (nn + l_mp + s_mp + 1) * (nn + l_mp - s_mp + 1)
            - m_mp ** 2
        )

    def _gamma(n: int) -> mpmath.mpc:
        nn = mpmath.mpf(n)
        val = (nn + l_mp + s_mp) * (nn + l_mp - s_mp) - m_mp ** 2
        if abs(val) < 1e-50:
            return mpmath.mpf(0)
        return (
            -2 * c * m_mp * s_mp
            * val
            / ((nn + l_mp + s_mp + 1) * (nn + l_mp - s_mp + 1) + m_mp ** 2 - c ** 2)
        )

    # Evaluate the continued fraction K = beta_0 + alpha_0 / (beta_1 + ...)
    # using Lentz's method, starting from n = n_max and working backward.
    # The residual is the value of the CF; it should be zero at the eigenvalue.

    # Simplified: evaluate the infinite CF  beta_0 - gamma_1/(beta_1 - gamma_2/(beta_2 - ...))
    # using the forward recurrence and minimal solution.

    # Use a simple backward evaluation of the CF
    f = _beta(n_max)
    for n in range(n_max - 1, -1, -1):
        b_n = _beta(n)
        a_n = _gamma(n + 1)  # coupling from next level
        f = b_n - a_n / f

    return f


# ---------------------------------------------------------------------------
# Horizon frequency
# ---------------------------------------------------------------------------

def horizon_frequency(a: float, M: float, omega: complex, m: int) -> complex:
    """
    Compute the co-rotating frame frequency at the horizon.

    Returns  omega - m * Omega_H  where  Omega_H = a / (2 M r_+)
    is the angular velocity of the event horizon and  r_+ = M + sqrt(M^2 - a^2).

    Parameters
    ----------
    a : float
        Kerr spin parameter.
    M : float
        Black hole mass.
    omega : complex
        Mode frequency.
    m : int
        Azimuthal number.

    Returns
    -------
    complex
        The quantity omega - m * Omega_H.
    """
    a = mpmath.mpf(a)
    M = mpmath.mpf(M)
    omega = mpmath.mpc(omega)

    r_plus = M + mpmath.sqrt(M ** 2 - a ** 2)
    Omega_H = a / (2 * M * r_plus)

    return complex(omega - m * Omega_H)
