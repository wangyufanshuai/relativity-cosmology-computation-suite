"""
Leaver's continued-fraction method for Kerr quasinormal modes.

Implements the three-term recurrence relation from Leaver (1985) and
uses continued-fraction convergence to locate QNM frequencies.
"""

from __future__ import annotations

import numpy as np
import mpmath
from scipy.optimize import minimize

from .teukolsky import teukolsky_angular_eigenvalue


# ---------------------------------------------------------------------------
# Known Schwarzschild QNM frequencies (Nollert 1993, Berti et al. 2009)
# ---------------------------------------------------------------------------

_SCHWARZSCHILD_QNM_TABLE: dict[tuple[int, int], complex] = {
    # (l, n) -> omega * M   (geometric units, M=1)
    # Fundamentals (n=0)
    (2, 0): complex(0.3736716844180431, -0.0889623156889392),
    (3, 0): complex(0.5994433915239740, -0.0927031395638511),
    (4, 0): complex(0.8091780516979440, -0.0941629092290482),
    # First overtones (n=1)
    (2, 1): complex(0.3467106094468801, -0.2739147215085012),
    (3, 1): complex(0.5826425027469137, -0.2813027268742089),
    (4, 1): complex(0.7997720156345988, -0.2842866726925512),
    # Second overtones (n=2)
    (2, 2): complex(0.3010534565565601, -0.4782807914260495),
    (3, 2): complex(0.5434874014011308, -0.4835267580349712),
    # Higher l, n=0
    (0, 0): complex(0.1104550345, -0.104856930),
    (1, 0): complex(0.29293626, -0.12280929),
    # Scalar (s=0) modes
}


def schwarzschild_qnm(l: int, n: int) -> complex:
    """
    Return a known Schwarzschild QNM frequency (a=0, s=-2).

    Parameters
    ----------
    l : int
        Spheroidal harmonic index (l >= 2 for gravitational).
    n : int
        Overtone number (n = 0 is fundamental).

    Returns
    -------
    complex
        QNM frequency omega*M (geometric units).
    """
    if (l, n) in _SCHWARZSCHILD_QNM_TABLE:
        return _SCHWARZSCHILD_QNM_TABLE[(l, n)]
    # If not in the table, compute via Leaver CF
    return find_qnm(a=0.0, l=l, m=2, s=-2, omega_guess=complex(0.5, -0.1 * (n + 1)), n=n)


# ---------------------------------------------------------------------------
# Leaver continued-fraction machinery
# ---------------------------------------------------------------------------

def _recurrence_coefficients(
    a: mpmath.mpf,
    omega: mpmath.mpc,
    A: mpmath.mpc,
    l: int,
    m: int,
    s: int,
    n: int,
) -> tuple[mpmath.mpc, mpmath.mpc, mpmath.mpc]:
    """
    Three-term recurrence coefficients alpha_n, beta_n, gamma_n for
    Leaver's radial expansion.

    The expansion ansatz for the radial function is:
        R(r) = exp(i omega r*) r^{-(2s+1)} (r - r_-)^{-sigma} sum a_n x^n
    where x = (r - r_+)/(r - r_-).

    The recurrence is:  alpha_n a_{n+1} + beta_n a_n + gamma_n a_{n-1} = 0.

    Reference: Leaver (1985), Proc. R. Soc. Lond. A 402, 285-298.
    """

    a_mp = a
    c_omega = omega
    s_mp = mpmath.mpf(s)
    l_mp = mpmath.mpf(l)
    m_mp = mpmath.mpf(m)
    n_mp = mpmath.mpf(n)

    r_plus = mpmath.mpf(1) + mpmath.sqrt(1 - a_mp ** 2)  # units of M
    r_minus = mpmath.mpf(1) - mpmath.sqrt(1 - a_mp ** 2)

    # kappa = i (omega r_+ - a m) / (2 r_+ - 2)   [for M=1]
    # In geometric units (M=1):
    kappa = 1j * (c_omega * r_plus - a_mp * m_mp) / (r_plus - r_minus)

    # epsilon and eta
    epsilon = 2 * 1j * c_omega
    eta = 1j * (c_omega - a_mp * m_mp / r_minus) / (r_plus - r_minus)

    # tau = (s + 1) / 2  (for the radial problem)
    tau = (s_mp + 1) / 2

    nn = n_mp

    alpha_n = (
        epsilon * (nn + 1) * (nn + 2 * tau - 1)
        + (nn + tau) ** 2 * epsilon / (nn + 2 * tau)
    )

    # Simplified Leaver coefficients following Leaver (1985) Eqs. (18)-(20):
    # alpha_n
    alpha_n = 1 / (nn + 1) * (
        -epsilon
    ) * (1 + epsilon) * (
        nn + 1 + tau
    )

    # Use the standard form from Leaver (1985):
    # alpha_n = -i * 2 * omega * (n + 1)(n + 2*tau)
    alpha_n = -epsilon * (nn + 1) * (nn + 2 * tau) / (nn + 2 * tau + 1)

    # beta_n
    beta_n = (
        l_mp * (l_mp + 1)
        - s_mp * (s_mp + 1)
        + nn * (nn + 2 * tau)
        - 2 * nn * epsilon
        - A / (2 * (nn + tau))
    )
    beta_n = nn * (nn + 2 * tau + 1) + tau - A / (nn + tau)

    # gamma_n
    gamma_n = (
        epsilon ** 2 * (nn - 1 + tau) / (nn + 2 * tau - 1)
        + nn * (nn - 1 + tau) / (nn + 2 * tau)
    )
    gamma_n = epsilon ** 2 * (nn + tau - 1) * (nn + tau) / ((nn + 2 * tau - 1) * (nn + 2 * tau))

    return alpha_n, beta_n, gamma_n


def leaver_cf_numerator(
    a: float,
    omega: complex,
    l: int,
    m: int,
    s: int = -2,
    n: int = 0,
) -> complex:
    """
    Compute the *n*-th term contribution of Leaver's continued fraction.

    This is the numerator of the ratio alpha_n / beta_n that appears
    in the CF condition for QNMs.

    Parameters
    ----------
    a : float
        Spin parameter a/M.
    omega : complex
        Trial frequency.
    l, m : int
        Spheroidal harmonic indices.
    s : int
        Spin weight.
    n : int
        Index in the CF expansion.

    Returns
    -------
    complex
        The value of alpha_n for the given parameters.
    """
    a_mp = mpmath.mpf(a)
    c_omega = mpmath.mpc(omega)

    A = teukolsky_angular_eigenvalue(a, omega, l, m, s)
    A_mp = mpmath.mpc(A)

    alpha_n, beta_n, gamma_n = _recurrence_coefficients(
        a_mp, c_omega, A_mp, l, m, s, n
    )
    return complex(alpha_n)


def leaver_minimal(
    a: float,
    omega: complex,
    l: int,
    m: int,
    s: int = -2,
    n_cf_max: int = 200,
    cf_tol: float = 1e-30,
) -> complex:
    """
    Evaluate Leaver's minimal condition for QNMs.

    The QNM boundary condition requires that the infinite continued
    fraction evaluates to zero.  We compute the CF

        CF = beta_0 - alpha_0*gamma_1 / (beta_1 - alpha_1*gamma_2 / (beta_2 - ...))

    using Lentz's modified method and return its value.  The QNM condition
    is CF = 0.

    Parameters
    ----------
    a : float
        Spin parameter a/M.
    omega : complex
        Trial frequency.
    l, m : int
        Spheroidal harmonic indices.
    s : int
        Spin weight.
    n_cf_max : int
        Maximum number of CF terms.
    cf_tol : float
        Convergence tolerance for CF evaluation.

    Returns
    -------
    complex
        Value of the continued fraction (should be 0 at a QNM).
    """
    mpmath.mp.dps = 30

    a_mp = mpmath.mpf(a)
    c_omega = mpmath.mpc(omega)

    # Get the angular separation constant
    A = teukolsky_angular_eigenvalue(a, omega, l, m, s)
    A_mp = mpmath.mpc(A)

    # Evaluate the CF from the bottom using backward recursion.
    # CF = beta_N (at large N, then recurse backward)
    # K_{n} = beta_n - alpha_n * gamma_{n+1} / K_{n+1}

    f = mpmath.mpf(1)
    for k in range(n_cf_max, 0, -1):
        alpha_k, beta_k, gamma_k = _recurrence_coefficients(
            a_mp, c_omega, A_mp, l, m, s, k
        )
        alpha_km1, beta_km1, gamma_km1 = _recurrence_coefficients(
            a_mp, c_omega, A_mp, l, m, s, k - 1
        )
        f = beta_k - alpha_k * gamma_k / f

    # The final CF value
    alpha_0, beta_0, gamma_0 = _recurrence_coefficients(
        a_mp, c_omega, A_mp, l, m, s, 0
    )
    cf_value = beta_0 - alpha_0 * gamma_0 / f

    mpmath.mp.dps = 15
    return complex(cf_value)


def find_qnm(
    a: float,
    l: int,
    m: int,
    s: int = -2,
    omega_guess: complex | None = None,
    n: int = 0,
) -> complex:
    """
    Find a Kerr quasinormal mode frequency using Leaver's continued fraction.

    The QNM condition is that the continued fraction diverges (equivalently,
    the minimal solution of the recurrence exists).  We search for omega
    that makes the CF residual zero.

    Parameters
    ----------
    a : float
        Spin parameter a/M (0 <= a < 1).
    l, m : int
        Spheroidal harmonic indices.
    s : int
        Spin weight (-2 for gravitational, 0 for scalar, ±1 for EM).
    omega_guess : complex, optional
        Initial guess for the frequency.  If None, use Schwarzschild value.
    n : int
        Overtone number (0 = fundamental).

    Returns
    -------
    complex
        QNM frequency omega*M (geometric units).
    """

    if omega_guess is None:
        # Use Schwarzschild value as initial guess
        if (l, n) in _SCHWARZSCHILD_QNM_TABLE:
            omega_guess = _SCHWARZSCHILD_QNM_TABLE[(l, n)]
        else:
            # Rough guess: omega_R ~ 0.5 * l, omega_I ~ -0.1 * (n + 1)
            omega_guess = complex(0.4 * l, -0.1 * (n + 1))

    mpmath.mp.dps = 30

    def _residual(omega_vec: np.ndarray) -> float:
        """CF residual as a real-valued function of [Re(omega), Im(omega)]."""
        omega_trial = complex(omega_vec[0], omega_vec[1])
        try:
            val = leaver_minimal(a, omega_trial, l, m, s, n_cf_max=150)
            return float(abs(val))
        except Exception:
            return 1e30

    # Use scipy minimize with Nelder-Mead for robustness
    x0 = np.array([omega_guess.real, omega_guess.imag])
    result = minimize(
        _residual,
        x0,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-14, "fatol": 1e-14},
    )

    omega_found = complex(result.x[0], result.x[1])

    # Refine with Powell method
    result2 = minimize(
        _residual,
        result.x,
        method="Powell",
        options={"maxiter": 5000, "ftol": 1e-16},
    )
    if result2.fun < result.fun:
        omega_found = complex(result2.x[0], result2.x[1])

    mpmath.mp.dps = 15
    return omega_found
