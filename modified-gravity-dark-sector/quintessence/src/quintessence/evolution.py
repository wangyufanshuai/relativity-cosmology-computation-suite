"""Quintessence field evolution.

Solves the coupled Friedmann + Klein-Gordon equations for a scalar field
in an expanding FRW background containing matter and radiation.

We work in dimensionless units normalised to the critical density today.
The scale factor *a* is the independent variable (a = 1 today).

Conventions
-----------
- H^2 = H0^2 [ (1/2)(d_phi)^2 + V_norm + Omega_m0 * a^{-3} + Omega_r0 * a^{-4} ]
- The Klein-Gordon equation is written as a first-order autonomous system:
      d(phi)/d(ln a)  = psi
      d(psi)/d(ln a)  = -3 psi - dV_norm/dphi / (H/H0) * (3/2) * Omega_DE0^{-1}
  where we use the Friedmann constraint to compute H/H0 at each step.

For numerical convenience we express everything in terms of the Hubble-normalised
energy densities.  We set H0 = 1 so that all quantities are dimensionless ratios.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .constants import EIGHT_PI_G


def _total_energy_density(
    phi_dot: float,
    V_val: float,
    Omega_m0: float,
    Omega_r0: float,
    a: float,
) -> float:
    """Total dimensionless energy density (normalised to rho_crit,0).

    rho_total / rho_crit,0 = (1/2) phi_dot^2 + V + Omega_m0 * a^{-3} + Omega_r0 * a^{-4}

    Here phi_dot and V are already in units of rho_crit,0 (dimensionless).
    """
    rho_de = 0.5 * phi_dot**2 + V_val
    rho_m = Omega_m0 * a ** (-3)
    rho_r = Omega_r0 * a ** (-4)
    return rho_de + rho_m + rho_r


def friedmann_H(
    phi: float,
    phi_dot: float,
    V_func: Callable[[float], float],
    Omega_m0: float,
    Omega_r0: float,
    a: float,
) -> float:
    """Compute the Hubble parameter H(a) from the Friedmann equation.

    Parameters
    ----------
    phi : float
        Scalar field value.
    phi_dot : float
        Time derivative of phi (d phi / d t), in units of sqrt(rho_crit,0).
    V_func : callable
        Potential function V(phi), in units of rho_crit,0.
    Omega_m0 : float
        Matter density fraction today.
    Omega_r0 : float
        Radiation density fraction today.
    a : float
        Scale factor.

    Returns
    -------
    float
        Hubble parameter H in units of H0 (= 1 in our convention).
    """
    V_val = V_func.V(phi)
    rho_total = _total_energy_density(phi_dot, V_val, Omega_m0, Omega_r0, a)
    return np.sqrt(rho_total)


def kg_rhs(
    a: float,
    y: NDArray,
    V_func: Callable,
    Omega_m0: float,
    Omega_r0: float,
) -> NDArray:
    """Right-hand side of the Klein-Gordon autonomous system.

    We use ln(a) as the independent variable so that
        d(phi)/d(ln a) = psi
        d(psi)/d(ln a) = -3 psi  -  V'(phi) / H^2

    The Friedmann equation supplies H^2 at each step.  In our convention
    H^2 = E(a) where E is the dimensionless Hubble function.

    Parameters
    ----------
    a : float
        Scale factor (used as integration variable internally via log).
    y : array_like, shape (2,)
        State vector [phi, psi] where psi = d(phi)/d(ln a).
    V_func : callable
        Potential with a ``dV`` method (or duck-type callable).
    Omega_m0, Omega_r0 : float
        Density parameters.

    Returns
    -------
    dyda : NDArray, shape (2,)
        Derivatives d(phi)/da and d(psi)/da.
    """
    phi, psi = y
    V_val = V_func.V(phi)
    dV_val = V_func.dV(phi)

    # Hubble function squared (dimensionless)
    E2 = _total_energy_density(psi, V_val, Omega_m0, Omega_r0, a)
    if E2 <= 0:
        E2 = 1e-30  # safety floor

    # d(phi)/da = psi / a
    # d(psi)/da = [-3*psi - dV/H^2] / a
    dphi_da = psi / a
    dpsi_da = (-3.0 * psi - dV_val / E2) / a

    return np.array([dphi_da, dpsi_da])


def evolve_quintessence(
    V_func: Callable,
    phi0: float,
    phi_dot0: float,
    Omega_m0: float,
    Omega_r0: float,
    a_range: tuple[float, float] = (1e-3, 1.0),
    n_points: int = 500,
) -> dict:
    """Evolve the quintessence field from a_start to a_end.

    Parameters
    ----------
    V_func : callable
        Potential object with V(phi) and dV(phi) methods.
    phi0 : float
        Initial field value at a_start.
    phi_dot0 : float
        Initial d(phi)/d(ln a) at a_start.
    Omega_m0 : float
        Present-day matter density parameter.
    Omega_r0 : float
        Present-day radiation density parameter.
    a_range : tuple (a_start, a_end)
        Scale factor range.  Defaults to a = 1e-3 .. 1.0.
    n_points : int
        Number of output points.

    Returns
    -------
    dict with keys:
        'a' : NDArray  - scale factor
        'phi' : NDArray  - field value
        'phi_dot' : NDArray  - d(phi)/d(ln a)
        'H' : NDArray  - Hubble parameter (in H0 units)
        'w' : NDArray  - equation-of-state parameter
        'Omega_DE' : NDArray  - dark energy density fraction
    """
    a_start, a_end = a_range
    a_eval = np.linspace(a_start, a_end, n_points)

    y0 = np.array([phi0, phi_dot0])

    sol = solve_ivp(
        kg_rhs,
        (a_start, a_end),
        y0,
        args=(V_func, Omega_m0, Omega_r0),
        t_eval=a_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        max_step=(a_end - a_start) / 50,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    a = sol.t
    phi = sol.y[0]
    phi_dot = sol.y[1]  # this is d(phi)/d(ln a)

    # Compute derived quantities at each point
    w_arr = np.empty_like(phi)
    H_arr = np.empty_like(phi)
    Omega_DE_arr = np.empty_like(phi)

    for i in range(len(a)):
        V_val = V_func.V(phi[i])
        kinetic = 0.5 * phi_dot[i] ** 2
        rho_de = kinetic + V_val

        # Equation of state
        if rho_de > 0:
            w_arr[i] = (kinetic - V_val) / rho_de
        else:
            w_arr[i] = -1.0

        # Hubble
        E2 = _total_energy_density(phi_dot[i], V_val, Omega_m0, Omega_r0, a[i])
        H_arr[i] = np.sqrt(max(E2, 0.0))

        # Omega_DE
        total = E2
        Omega_DE_arr[i] = rho_de / total if total > 0 else 0.0

    return {
        "a": a,
        "phi": phi,
        "phi_dot": phi_dot,
        "H": H_arr,
        "w": w_arr,
        "Omega_DE": Omega_DE_arr,
    }


def equation_of_state(
    phi: float | NDArray,
    phi_dot: float | NDArray,
    V_func: Callable,
) -> float | NDArray:
    """Compute the dark energy equation of state w = p / rho.

    w = (K - V) / (K + V)  where K = (1/2) phi_dot^2.

    Parameters
    ----------
    phi : float or NDArray
        Field value(s).
    phi_dot : float or NDArray
        d(phi)/d(ln a) value(s).
    V_func : callable
        Potential function.

    Returns
    -------
    float or NDArray
        Equation of state w.
    """
    V_val = V_func.V(phi)
    kinetic = 0.5 * np.asarray(phi_dot) ** 2
    rho = kinetic + V_val
    # Guard against division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(rho > 0, (kinetic - V_val) / rho, -1.0)
    return float(w) if np.ndim(w) == 0 else w
