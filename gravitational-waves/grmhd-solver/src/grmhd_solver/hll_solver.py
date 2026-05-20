"""HLL Riemann solver for relativistic MHD.

Implements the Harten-Lax-van Leer (HLL) approximate Riemann solver
for the equations of special relativistic MHD in 1-D.

Conservative variables (relativistic MHD):
    D   = rho * W                  (lab-frame density)
    S_i = (rho h + b^2) W^2 v_i - b_i b_0  (momentum)
    tau = (rho h + b^2) W^2 - (p + b^2/2) - D  (energy - rest-mass)

For simplicity (pure hydro, no magnetic field), this reduces to:
    D   = rho W
    S   = rho h W^2 v
    tau = rho h W^2 - p - D

Units: c = 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lorentz(v: NDArray) -> NDArray:
    """Lorentz factor W = 1/sqrt(1 - v^2)."""
    return 1.0 / np.sqrt(1.0 - v * v)


def enthalpy(rho: NDArray, p: NDArray, gamma_ad: float) -> NDArray:
    """Specific enthalpy h = 1 + eps + p/rho."""
    eps = p / ((gamma_ad - 1.0) * rho)
    return 1.0 + eps + p / rho


def sound_speed(rho: NDArray, p: NDArray, gamma_ad: float) -> NDArray:
    """Relativistic sound speed c_s = sqrt(gamma_ad * p / (rho * h))."""
    h = enthalpy(rho, p, gamma_ad)
    cs2 = gamma_ad * p / (rho * h)
    cs2 = np.clip(cs2, 0.0, 1.0 - 1e-12)
    return np.sqrt(cs2)


def prim_to_cons(
    rho: NDArray, v: NDArray, p: NDArray, gamma_ad: float
) -> tuple[NDArray, NDArray, NDArray]:
    """Convert primitive (rho, v, p) -> conservative (D, S, tau)."""
    W = lorentz(v)
    h = enthalpy(rho, p, gamma_ad)
    D = rho * W
    S = rho * h * W * W * v
    tau = rho * h * W * W - p - D
    return D, S, tau


def cons_to_prim(
    D: NDArray, S: NDArray, tau: NDArray,
    gamma_ad: float, tol: float = 1e-12, max_iter: int = 200,
) -> tuple[NDArray, NDArray, NDArray]:
    """Recover primitive variables from conservative via Newton-Raphson on pressure."""
    p = np.maximum((gamma_ad - 1.0) * tau / (gamma_ad + 1e-14), 1e-14)
    p = np.asarray(p, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)

    for _ in range(max_iter):
        denom = tau + p + D
        v = S / denom
        v = np.clip(v, -1.0 + 1e-14, 1.0 - 1e-14)
        W = lorentz(v)
        rho = np.maximum(D / W, 1e-14)
        eps = p / ((gamma_ad - 1.0) * rho)
        p_new = np.maximum((gamma_ad - 1.0) * rho * eps, 1e-14)
        dp = p_new - p
        p = p_new
        if np.max(np.abs(dp)) < tol:
            break

    denom = tau + p + D
    v = np.clip(S / denom, -1.0 + 1e-14, 1.0 - 1e-14)
    W = lorentz(v)
    rho = np.maximum(D / W, 1e-14)
    return rho, v, p


def flux(
    D: NDArray, S: NDArray, tau: NDArray, p: NDArray, v: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Physical flux F(U) for the relativistic Euler equations."""
    return D * v, S * v + p, S - D * v


def char_speeds(
    v: NDArray, rho: NDArray, p: NDArray, gamma_ad: float,
) -> tuple[NDArray, NDArray]:
    """Left and right characteristic signal speeds using relativistic velocity addition."""
    cs = sound_speed(rho, p, gamma_ad)
    lam_p = (v + cs) / (1.0 + v * cs)
    lam_m = (v - cs) / (1.0 - v * cs)
    return lam_p, lam_m


def hll_flux(
    rhoL: NDArray, vL: NDArray, pL: NDArray,
    rhoR: NDArray, vR: NDArray, pR: NDArray,
    gamma_ad: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """HLL flux at each cell interface.

    F_HLL = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)
    """
    DL, SL, tauL = prim_to_cons(rhoL, vL, pL, gamma_ad)
    DR, SR, tauR = prim_to_cons(rhoR, vR, pR, gamma_ad)

    fDL, fSL, fTL = flux(DL, SL, tauL, pL, vL)
    fDR, fSR, fTR = flux(DR, SR, tauR, pR, vR)

    lam_pL, lam_mL = char_speeds(vL, rhoL, pL, gamma_ad)
    lam_pR, lam_mR = char_speeds(vR, rhoR, pR, gamma_ad)

    S_L = np.minimum(lam_mL, lam_mR)
    S_R = np.maximum(lam_pL, lam_pR)

    denom = S_R - S_L
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)

    fD = (S_R * fDL - S_L * fDR + S_L * S_R * (DR - DL)) / denom
    fS = (S_R * fSL - S_L * fSR + S_L * S_R * (SR - SL)) / denom
    fT = (S_R * fTL - S_L * fTR + S_L * S_R * (tauR - tauL)) / denom

    return fD, fS, fT
