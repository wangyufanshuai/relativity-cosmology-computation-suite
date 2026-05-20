"""Special relativistic hydrodynamics in 1D.

Implements conservative-to-primitive variable recovery, the HLL Riemann
solver, and characteristic speed computation for a relativistic ideal-gas
EOS.

Coordinate conventions
----------------------
* Units where *c* = 1.
* Adiabatic index ``gamma_ad`` (written Gamma in formulae).
* 1-D planar or spherical geometry (selected per-flux).

Conservative variables
    D  = rho * W          (lab-frame density)
    S  = rho * h * W^2 * v  (momentum density)
    tau = rho * h * W^2 - p - D  (energy - rest-mass)

Primitive variables
    rho  (rest-frame density)
    v    (coordinate velocity, |v| < 1)
    p    (thermal pressure)

Equation of state
    p = (gamma_ad - 1) * rho * epsilon
    h  = 1 + epsilon + p / rho
    W  = 1 / sqrt(1 - v^2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Equation-of-state helpers
# ---------------------------------------------------------------------------

def enthalpy(rho: NDArray, p: NDArray, gamma_ad: float) -> NDArray:
    """Specific enthalpy *h* = 1 + eps + p/rho."""
    eps = p / ((gamma_ad - 1.0) * rho)
    return 1.0 + eps + p / rho


def lorentz(v: NDArray) -> NDArray:
    """Lorentz factor W = 1/sqrt(1 - v^2)."""
    return 1.0 / np.sqrt(1.0 - v * v)


def sound_speed(rho: NDArray, p: NDArray, gamma_ad: float) -> NDArray:
    """Relativistic sound speed c_s.

    c_s^2 = gamma_ad * p / (rho * h)
    """
    h = enthalpy(rho, p, gamma_ad)
    cs2 = gamma_ad * p / (rho * h)
    # Clamp for numerical safety
    cs2 = np.clip(cs2, 0.0, 1.0 - 1e-12)
    return np.sqrt(cs2)


# ---------------------------------------------------------------------------
# Conservative <-> Primitive conversions
# ---------------------------------------------------------------------------

def prim_to_cons(rho: NDArray, v: NDArray, p: NDArray,
                 gamma_ad: float) -> tuple[NDArray, NDArray, NDArray]:
    """Convert primitive (rho, v, p) -> conservative (D, S, tau)."""
    W = lorentz(v)
    h = enthalpy(rho, p, gamma_ad)
    D = rho * W
    S = rho * h * W * W * v
    tau = rho * h * W * W - p - D
    return D, S, tau


def cons_to_prim(D: NDArray, S: NDArray, tau: NDArray,
                 gamma_ad: float, tol: float = 1e-12,
                 max_iter: int = 200) -> tuple[NDArray, NDArray, NDArray]:
    """Recover primitive variables from conservative variables.

    Uses a Newton-Raphson iteration on the pressure, following the
    standard approach (e.g. Noble et al. 2006, simplified for 1-D).

    Returns (rho, v, p).
    """
    # Initial guess for p -- simple estimate
    p = np.maximum((gamma_ad - 1.0) * tau / (gamma_ad + 1e-14), 1e-14)
    p = np.asarray(p, dtype=np.float64)

    D = np.asarray(D, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    for _ in range(max_iter):
        # Solve for v given current p:
        #   v = S / (tau + p + D)
        # then rho = D / W, then update p from EOS.
        denom = tau + p + D
        v = S / denom
        # Ensure |v| < 1
        v = np.clip(v, -1.0 + 1e-14, 1.0 - 1e-14)
        W = lorentz(v)
        rho = D / W
        rho = np.maximum(rho, 1e-14)
        eps = p / ((gamma_ad - 1.0) * rho)
        h = 1.0 + eps + p / rho
        # Residual: p_new from EOS minus current p
        # We want f(p) = p - (gamma_ad - 1) * rho * epsilon = 0
        # But rho*epsilon = (tau + D*(1-W) + p*(1-W^2)) / W^2
        # Simpler: just recompute p from the current rho and epsilon
        p_new = (gamma_ad - 1.0) * rho * eps
        p_new = np.maximum(p_new, 1e-14)
        dp = p_new - p
        p = p_new
        if np.max(np.abs(dp)) < tol:
            break

    # Final values
    denom = tau + p + D
    v = S / denom
    v = np.clip(v, -1.0 + 1e-14, 1.0 - 1e-14)
    W = lorentz(v)
    rho = D / W
    rho = np.maximum(rho, 1e-14)
    return rho, v, p


# ---------------------------------------------------------------------------
# Flux functions
# ---------------------------------------------------------------------------

def flux(D: NDArray, S: NDArray, tau: NDArray,
         p: NDArray, v: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Physical flux F(U) for the relativistic Euler equations.

    F(D)   = D * v
    F(S)   = S * v + p
    F(tau) = S - D * v
    """
    fD = D * v
    fS = S * v + p
    fT = S - D * v
    return fD, fS, fT


# ---------------------------------------------------------------------------
# HLL Riemann solver
# ---------------------------------------------------------------------------

def char_speeds(v: NDArray, rho: NDArray, p: NDArray,
                gamma_ad: float) -> tuple[NDArray, NDArray]:
    """Left- and right-going characteristic signal speeds.

    Uses the relativistic velocity addition of the sound speed:
        lambda_{pm} = (v pm c_s) / (1 pm v * c_s)
    """
    cs = sound_speed(rho, p, gamma_ad)
    lam_p = (v + cs) / (1.0 + v * cs)
    lam_m = (v - cs) / (1.0 - v * cs)
    return lam_p, lam_m


def hll_flux(rhoL: NDArray, vL: NDArray, pL: NDArray,
             rhoR: NDArray, vR: NDArray, pR: NDArray,
             gamma_ad: float) -> tuple[NDArray, NDArray, NDArray]:
    """HLL flux at each cell interface.

    F_HLL = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L))
            / (S_R - S_L)

    where S_L = min(lambda-_L, lambda-_R), S_R = max(lambda+_L, lambda+_R).
    """
    # Conservative states
    DL, SL, tauL = prim_to_cons(rhoL, vL, pL, gamma_ad)
    DR, SR, tauR = prim_to_cons(rhoR, vR, pR, gamma_ad)

    # Fluxes
    fDL, fSL, fTL = flux(DL, SL, tauL, pL, vL)
    fDR, fSR, fTR = flux(DR, SR, tauR, pR, vR)

    # Signal speeds
    lam_pL, lam_mL = char_speeds(vL, rhoL, pL, gamma_ad)
    lam_pR, lam_mR = char_speeds(vR, rhoR, pR, gamma_ad)

    S_L = np.minimum(lam_mL, lam_mR)
    S_R = np.maximum(lam_pL, lam_pR)

    # Avoid division by zero
    denom = S_R - S_L
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

    # HLL flux for each conserved variable
    fD = (S_R * fDL - S_L * fDR + S_L * S_R * (DR - DL)) / denom
    fS = (S_R * fSL - S_L * fSR + S_L * S_R * (SR - SL)) / denom
    fT = (S_R * fTL - S_L * fTR + S_L * S_R * (tauR - tauL)) / denom

    return fD, fS, fT


# ---------------------------------------------------------------------------
# 1-D finite-volume update (single step)
# ---------------------------------------------------------------------------

def fv_step(rho: NDArray, v: NDArray, p: NDArray,
            dx: float, dt: float, gamma_ad: float,
            geometry: str = "planar") -> tuple[NDArray, NDArray, NDArray]:
    """One forward-Euler finite-volume step.

    Parameters
    ----------
    rho, v, p : cell-centred primitive arrays of length *N*.
    dx, dt : grid spacing and time step (CFL must be satisfied externally).
    gamma_ad : adiabatic index.
    geometry : ``"planar"`` or ``"spherical"``.

    Returns
    -------
    Updated (rho, v, p) arrays.
    """
    N = rho.shape[0]

    # Ghost cells (2 on each side, outflow BC)
    rho_e = np.pad(rho, 2, mode="edge")
    v_e = np.pad(v, 2, mode="edge")
    p_e = np.pad(p, 2, mode="edge")

    # Interface fluxes  (N+1 interfaces for N cells)
    fD_arr = np.zeros(N + 1)
    fS_arr = np.zeros(N + 1)
    fT_arr = np.zeros(N + 1)

    for i in range(N + 1):
        iL = i + 1  # left cell index in extended array (ghost offset)
        iR = i + 2
        fd, fs, ft = hll_flux(
            np.atleast_1d(rho_e[iL]), np.atleast_1d(v_e[iL]),
            np.atleast_1d(p_e[iL]),
            np.atleast_1d(rho_e[iR]), np.atleast_1d(v_e[iR]),
            np.atleast_1d(p_e[iR]),
            gamma_ad,
        )
        fD_arr[i] = fd[0]
        fS_arr[i] = fs[0]
        fT_arr[i] = ft[0]

    # Conservative update
    D, S, tau = prim_to_cons(rho, v, p, gamma_ad)

    if geometry == "spherical":
        # Geometric source terms for spherical coords
        r = (np.arange(N) + 0.5) * dx
        r = np.maximum(r, 0.5 * dx)
        D_new = D - dt / dx * (fD_arr[1:] - fD_arr[:-1]) - dt * 2.0 * D * v / r
        S_new = S - dt / dx * (fS_arr[1:] - fS_arr[:-1]) - dt * 2.0 * S * v / r
        tau_new = tau - dt / dx * (fT_arr[1:] - fT_arr[:-1]) - dt * 2.0 * (tau + p) * v / r
    else:
        D_new = D - dt / dx * (fD_arr[1:] - fD_arr[:-1])
        S_new = S - dt / dx * (fS_arr[1:] - fS_arr[:-1])
        tau_new = tau - dt / dx * (fT_arr[1:] - fT_arr[:-1])

    rho_new, v_new, p_new = cons_to_prim(D_new, S_new, tau_new, gamma_ad)
    return rho_new, v_new, p_new
