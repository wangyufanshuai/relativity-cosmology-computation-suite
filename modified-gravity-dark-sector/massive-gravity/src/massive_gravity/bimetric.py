"""
Bimetric (bigravity) theory with two dynamical metrics g_{\\mu\\nu} and f_{\\mu\\nu}.

In Hassan-Rosen bimetric theory both metrics carry their own Einstein-Hilbert term
and are coupled through the dRGT interaction potential.  On a cosmological
background we take:

  g-metric : flat FRW   ds^2_g = -dt^2 + a_g(t)^2 d\\vec{x}^2
  f-metric : de Sitter   ds^2_f = -dt^2 + a_f(t)^2 d\\vec{x}^2

and solve the coupled Friedmann equations for the two scale factors.

References:
  - Hassan, Rosen, JHEP 1202 (2012) 126
  - von Strauss et al., JCAP 1203 (2012) 042
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
H0_SI = 2.184e-18  # Hubble constant ~ 67.4 km/s/Mpc in s^{-1}

# ---------------------------------------------------------------------------
# Cosmological background: FRW g + de Sitter f
# ---------------------------------------------------------------------------

def frw_metric(a: float) -> NDArray:
    """Flat FRW metric with scale factor *a* in conformal-time diag form.

    ds^2 = -dt^2 + a^2 (dx^2 + dy^2 + dz^2)  (reduced to comoving form).
    Returns the spatial part as a 4x4 matrix in (t, x, y, z) coordinates.
    """
    return np.diag([-1.0, a**2, a**2, a**2])


def desitter_metric(H_f: float, t: float) -> NDArray:
    """de Sitter metric with Hubble rate H_f at coordinate time t.

    ds^2 = -dt^2 + exp(2 H_f t) (dx^2 + dy^2 + dz^2).
    """
    a_f = np.exp(H_f * t)
    return np.diag([-1.0, a_f**2, a_f**2, a_f**2])


# ---------------------------------------------------------------------------
# Coupled Friedmann equations
# ---------------------------------------------------------------------------

def friedmann_rhs(
    t: float,
    y: NDArray,
    H0: float,
    Omega_m: float,
    Omega_rad: float,
    beta1: float,
    kappa: float,
    m_g: float,
    rho_f: float,
) -> NDArray:
    """Right-hand side of the coupled Friedmann ODE system.

    State vector  y = [a_g, \\dot{a}_g].

    The g-metric Friedmann equation (with massive/bimetric coupling):
      3 H_g^2 = 8\\pi G \\rho + \\rho_{mg}(a_g, a_f)

    The effective massive-gravity density contribution is modeled as
      \\rho_{mg} = m_g^2 H0^2 [\\beta_1 + \\kappa (a_g / a_f - 1)]

    The f-metric scale factor is fixed to de Sitter: a_f = e^{\\sqrt{\\rho_f/3} t}.
    """
    a_g, a_g_dot = y

    # Prevent a_g from reaching zero
    a_g = max(a_g, 1e-30)

    H_g = a_g_dot / a_g

    # f-metric scale factor (de Sitter)
    H_f = np.sqrt(max(rho_f / 3.0, 0.0))
    a_f = np.exp(H_f * t)

    # Matter + radiation density  \\rho \\propto \\Omega_m / a^3 + \\Omega_rad / a^4
    rho_matter = Omega_m * H0**2 / a_g**3
    rho_radiation = Omega_rad * H0**2 / a_g**4

    # Massive gravity contribution to the Friedmann equation
    ratio = a_g / a_f
    rho_mg = m_g**2 * H0**2 * (beta1 + kappa * (ratio - 1.0))

    # Friedmann acceleration equation: \\ddot{a}/a = -4\\pi G/3 (\\rho + 3p)
    # For dust p=0 and radiation p=\\rho/3
    rho_total = rho_matter + rho_radiation + rho_mg
    # \\ddot{a}/a = -H0^2/2 [\\Omega_m/a^3 + 2\\Omega_rad/a^4] + mg correction
    a_g_ddot = a_g * (
        -0.5 * H0**2 * (Omega_m / a_g**3 + 2.0 * Omega_rad / a_g**4)
        + (1.0 / 3.0) * m_g**2 * H0**2 * kappa * H_f * ratio
    )

    return np.array([a_g_dot, a_g_ddot])


def solve_bimetric_cosmology(
    t_span: tuple[float, float] = (1e-6, 1.0),
    H0: float = H0_SI,
    Omega_m: float = 0.3,
    Omega_rad: float = 9e-5,
    beta1: float = 1.0,
    kappa: float = 0.5,
    m_g: float = 1.0,
    rho_f: float = 1e-3,
    n_points: int = 200,
) -> dict:
    """Solve the coupled bimetric Friedmann equations.

    Returns a dictionary with keys 't', 'a_g', 'a_f', 'H_g', 'y_ratio'.
    """
    a_init = 1e-4
    a_dot_init = H0 * a_init  # Start near matter-dominated Hubble flow

    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        friedmann_rhs,
        t_span,
        [a_init, a_dot_init],
        args=(H0, Omega_m, Omega_rad, beta1, kappa, m_g, rho_f),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    a_g = sol.y[0]
    a_g_dot = sol.y[1]
    H_g = a_g_dot / a_g

    # f-metric scale factor (de Sitter)
    H_f = np.sqrt(max(rho_f / 3.0, 0.0))
    a_f = np.exp(H_f * sol.t)

    y_ratio = a_g / a_f

    return {
        "t": sol.t,
        "a_g": a_g,
        "a_f": a_f,
        "H_g": H_g,
        "y_ratio": y_ratio,
    }


# ---------------------------------------------------------------------------
# Scale-factor ratio y = a_g / a_f
# ---------------------------------------------------------------------------

def compute_y_ratio(a_g: NDArray, t: NDArray, rho_f: float) -> NDArray:
    """Compute the ratio y = a_g / a_f as a function of time.

    The f-metric scale factor is a_f = e^{H_f t} with H_f = sqrt(rho_f/3).
    """
    H_f = np.sqrt(max(rho_f / 3.0, 0.0))
    a_f = np.exp(H_f * t)
    return a_g / a_f


def y_ratio_finite(
    a_g: NDArray,
    t: NDArray,
    rho_f: float,
) -> bool:
    """Check that the ratio y = a_g/a_f is finite for all times."""
    y = compute_y_ratio(a_g, t, rho_f)
    return bool(np.all(np.isfinite(y)))
