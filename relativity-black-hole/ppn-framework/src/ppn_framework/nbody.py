"""Einstein-Infeld-Hoffmann (EIH) N-body equations of motion.

Implements the post-Newtonian N-body equations for a system of
N gravitating bodies. The 1PN equations include:

    a_i = a_i^Newton + a_i^1PN

where the 1PN correction includes velocity-dependent terms,
nonlinear gravitational potentials, and cross-coupling.

References:
    - Will (1993), "Theory and Experiment in Gravitational Physics"
    - Damour & Schäfer (1988), "GRT and binary pulsars"
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from .constants import G, C
from .metric import PPNParameters


def eih_acceleration(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    ppn: PPNParameters | None = None,
    pn_order: int = 1,
) -> np.ndarray:
    """Compute Einstein-Infeld-Hoffmann accelerations for N bodies.

    Parameters
    ----------
    positions : (N, 3) array of positions [m]
    velocities : (N, 3) array of velocities [m/s]
    masses : (N,) array of masses [kg]
    ppn : PPN parameters (default GR)
    pn_order : 1 for 1PN, 2 for 2PN (partial)

    Returns
    -------
    accelerations : (N, 3) array [m/s²]
    """
    if ppn is None:
        ppn = PPNParameters()

    N = len(masses)
    gamma = ppn.gamma
    beta = ppn.beta
    acc = np.zeros_like(positions)

    for i in range(N):
        ri = positions[i]
        vi = velocities[i]

        for j in range(N):
            if i == j:
                continue
            rj = positions[j]
            vj = velocities[j]

            rij = ri - rj
            r_ij = np.linalg.norm(rij)
            n_ij = rij / r_ij
            v_ij = vi - vj

            # Newtonian acceleration
            a_newton = -G * masses[j] / r_ij**2 * n_ij

            if pn_order >= 1:
                # 1PN correction terms (Will 1993, Eq. 4.1)
                vi2 = np.dot(vi, vi)
                vj2 = np.dot(vj, vj)
                vi_vj = np.dot(vi, vj)
                nij_vij = np.dot(n_ij, v_ij)
                U_over_c2 = 0.0
                for k in range(N):
                    if k == i:
                        continue
                    rk = np.linalg.norm(ri - positions[k])
                    if rk > 0:
                        U_over_c2 += G * masses[k] / (rk * C**2)

                a_1pn = a_newton / C**2 * (
                    # Velocity terms
                    -vi2
                    - 2.0 * vj2
                    + 4.0 * vi_vj
                    + 1.5 * nij_vij**2
                    # Potential terms
                    + (4.0 * gamma + 4.0) * G * masses[j] / r_ij
                    + (2.0 * gamma + 2.0) * U_over_c2 * C**2
                    # Beta term
                    - (2.0 * beta - 1.0) * U_over_c2 * C**2
                )

                # Additional velocity-dependent 1PN term
                a_1pn_vel = G * masses[j] / (r_ij**2 * C**2) * (
                    n_ij * (4.0 * vi_vj - 3.0 * vj2) * 0.5
                    + v_ij * (4.0 + 2.0 * gamma) * nij_vij
                )

                acc[i] += a_newton + a_1pn + a_1pn_vel
            else:
                acc[i] += a_newton

    return acc


def _eih_rhs(
    t: float,
    y: np.ndarray,
    masses: np.ndarray,
    ppn: PPNParameters,
    pn_order: int,
) -> np.ndarray:
    """RHS for the EIH N-body problem. State = [x1,v1,x2,v2,...]."""
    N = len(masses)
    positions = y[: 3 * N].reshape(N, 3)
    velocities = y[3 * N :].reshape(N, 3)

    acc = eih_acceleration(positions, velocities, masses, ppn, pn_order)

    return np.concatenate([velocities.flatten(), acc.flatten()])


def integrate_nbody(
    masses: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    t_span: tuple[float, float],
    ppn: PPNParameters | None = None,
    pn_order: int = 1,
    method: str = "DOP853",
    rtol: float = 1e-12,
    atol: float = 1e-14,
    t_eval: ArrayLike | None = None,
) -> dict:
    """Integrate the post-Newtonian N-body system.

    Parameters
    ----------
    masses : (N,) masses [kg]
    positions : (N, 3) initial positions [m]
    velocities : (N, 3) initial velocities [m/s]
    t_span : integration time interval [s]
    ppn : PPN parameters
    pn_order : PN order (1 or 2)

    Returns
    -------
    dict with 't', 'positions', 'velocities', 'accelerations'
    """
    if ppn is None:
        ppn = PPNParameters()

    N = len(masses)
    y0 = np.concatenate([positions.flatten(), velocities.flatten()])

    sol = solve_ivp(
        _eih_rhs,
        t_span,
        y0,
        args=(masses, ppn, pn_order),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"N-body integration failed: {sol.message}")

    positions_out = sol.y[: 3 * N, :].T.reshape(-1, N, 3)
    velocities_out = sol.y[3 * N :, :].T.reshape(-1, N, 3)

    return {
        "t": sol.t,
        "positions": positions_out,
        "velocities": velocities_out,
    }
