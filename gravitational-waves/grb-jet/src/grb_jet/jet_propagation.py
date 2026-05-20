"""Relativistic jet propagation through a stellar envelope.

Implements a simplified 1-D analytic/semi-analytic model for a
relativistic jet boring through a massive-star progenitor, including:

* Ram-pressure balance at the jet head
* Cocoon formation from shocked jet and stellar material
* Jet breakout condition
* Cocoon collimation pressure

All in units where c = 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Stellar-envelope density profile
# ---------------------------------------------------------------------------

def envelope_density(r: NDArray, rho_0: float = 1e-3,
                     r_0: float = 1e9, alpha: float = 2.0) -> NDArray:
    """Power-law stellar envelope density.

    rho(r) = rho_0 * (r / r_0)^(-alpha)

    Default gives a typical Wolf-Rayet-like envelope.
    """
    return rho_0 * np.power(np.maximum(r, 1.0) / r_0, -alpha)


# ---------------------------------------------------------------------------
# Jet head dynamics (ram-pressure balance)
# ---------------------------------------------------------------------------

def jet_head_velocity(v_j: float, rho_a: float,
                      rho_j: float) -> float:
    """Jet head velocity from ram-pressure balance.

    v_h = v_j / (1 + sqrt(rho_a / rho_j))

    Parameters
    ----------
    v_j : jet bulk velocity (must be < 1).
    rho_a : ambient (stellar envelope) density at the head.
    rho_j : jet density.

    Returns
    -------
    v_h : head velocity (< 1).
    """
    ratio = np.sqrt(rho_a / max(rho_j, 1e-30))
    v_h = v_j / (1.0 + ratio)
    # Ensure sub-luminal
    return min(v_h, 1.0 - 1e-12)


def jet_head_position(t: float, v_j: float, rho_a_func,
                       rho_j: float, R_star: float,
                       dt_int: float = 0.01) -> float:
    """Integrate jet head position until breakout or time *t*.

    Parameters
    ----------
    t : maximum propagation time.
    v_j : jet velocity.
    rho_a_func : callable rho_a(r) returning ambient density.
    rho_j : jet density.
    R_star : stellar radius.
    dt_int : internal integration time step.

    Returns
    -------
    r_h : head position at time t (capped at R_star).
    """
    r_h = 1.0  # start slightly off-center to avoid r=0 singularity
    t_curr = 0.0
    while t_curr < t and r_h < R_star:
        rho_a = rho_a_func(r_h)
        v_h = jet_head_velocity(v_j, rho_a, rho_j)
        r_h += v_h * dt_int
        t_curr += dt_int
    return min(r_h, R_star)


def breakout_time(v_j: float, rho_a_func, rho_j: float,
                  R_star: float, dt_int: float = 0.01) -> float:
    """Time for the jet to break out of the star.

    Returns
    -------
    t_bo : breakout time. np.inf if the jet fails to break out within
           a generous upper limit.
    """
    r_h = 1.0  # start slightly off-center to avoid r=0 singularity
    t_curr = 0.0
    t_max = 10.0 * R_star  # generous upper bound
    while r_h < R_star and t_curr < t_max:
        rho_a = rho_a_func(r_h)
        v_h = jet_head_velocity(v_j, rho_a, rho_j)
        r_h += v_h * dt_int
        t_curr += dt_int
    if r_h < R_star:
        return np.inf
    return t_curr


# ---------------------------------------------------------------------------
# Cocoon
# ---------------------------------------------------------------------------

def cocoon_pressure(L_j: float, r_h: float, v_h: float,
                     Omega_j: float) -> float:
    """Estimate cocoon pressure.

    P_c ~ L_j / (4 pi r_h^2 v_h Omega_j)

    Parameters
    ----------
    L_j : jet luminosity.
    r_h : jet head radius.
    v_h : jet head velocity.
    Omega_j : jet solid angle (steradians).
    """
    denom = 4.0 * np.pi * max(r_h, 1.0)**2 * max(v_h, 1e-10) * max(Omega_j, 1e-10)
    return L_j / denom


def cocoon_energy(L_j: float, t: float, eta_heat: float = 0.5) -> float:
    """Energy deposited into the cocoon.

    E_c = eta_heat * L_j * t
    """
    return eta_heat * L_j * t


def collimation_radius(P_c: float, L_j: float, Gamma_j: float) -> float:
    """Jet collimation (recollimation) radius from cocoon pressure.

    r_coll ~ sqrt(L_j / (4 pi P_c)) / Gamma_j
    """
    return np.sqrt(L_j / (4.0 * np.pi * max(P_c, 1e-30))) / max(Gamma_j, 1.0)


# ---------------------------------------------------------------------------
# Convenience: full propagation summary
# ---------------------------------------------------------------------------

def propagate(v_j: float, rho_j: float, L_j: float,
              R_star: float, theta_j: float,
              rho_0: float = 1e-3, r_0: float = 1e9,
              alpha: float = 2.0,
              t_max: float = 100.0,
              dt_int: float = 0.01) -> dict:
    """Run the jet propagation model and return summary diagnostics.

    Parameters
    ----------
    v_j : jet velocity (< 1).
    rho_j : jet density.
    L_j : jet luminosity.
    R_star : stellar radius.
    theta_j : half-opening angle (rad).
    rho_0, r_0, alpha : envelope density profile parameters.
    t_max : maximum propagation time.
    dt_int : integration timestep.

    Returns
    -------
    dict with keys: r_h, v_h, t_bo, P_cocoon, E_cocoon, r_coll, broken_out.
    """
    Omega_j = 2.0 * np.pi * (1.0 - np.cos(theta_j))

    def rho_a(r):
        return envelope_density(np.atleast_1d(r), rho_0, r_0, alpha)[0]

    t_bo = breakout_time(v_j, rho_a, rho_j, R_star, dt_int)
    broken_out = t_bo < t_max
    t_eval = min(t_bo, t_max) if broken_out else t_max

    r_h = jet_head_position(t_eval, v_j, rho_a, rho_j, R_star, dt_int)
    rho_a_head = rho_a(r_h)
    v_h = jet_head_velocity(v_j, rho_a_head, rho_j)

    P_c = cocoon_pressure(L_j, r_h, v_h, Omega_j)
    E_c = cocoon_energy(L_j, t_eval)
    Gamma_j = 1.0 / np.sqrt(1.0 - v_j**2)
    r_coll = collimation_radius(P_c, L_j, Gamma_j)

    return {
        "r_h": r_h,
        "v_h": v_h,
        "t_bo": t_bo,
        "P_cocoon": P_c,
        "E_cocoon": E_c,
        "r_coll": r_coll,
        "broken_out": broken_out,
    }
