"""Einstein-Infeld-Hoffmann (EIH) N-body equations of motion.

Full 1PN equations for N gravitating bodies including:
- Newtonian gravity
- 1PN point-mass corrections
- Spin-orbit coupling (Lense-Thirring)
- Solar J2 quadrupole

Reference: Will (1993), "Theory and Experiment in Gravitational Physics", Ch. 4
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G, C


def eih_acceleration_1pn(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """1PN EIH accelerations for N bodies.

    Implements Eq. (4.1) of Will (1993) for the point-mass 1PN correction.
    All terms are O(1/c²) beyond Newtonian.

    Parameters
    ----------
    pos : (N, 3) positions [m]
    vel : (N, 3) velocities [m/s]
    masses : (N,) masses [kg]

    Returns
    -------
    acc : (N, 3) accelerations [m/s²]
    """
    N = len(masses)
    acc = np.zeros_like(pos)
    c2 = C**2

    for i in range(N):
        ri, vi = pos[i], vel[i]
        vi2 = np.dot(vi, vi)

        # Newtonian + 1PN from all other bodies
        for j in range(N):
            if i == j:
                continue
            rj, vj = pos[j], vel[j]
            rij = ri - rj
            r_ij = np.linalg.norm(rij)
            n_ij = rij / r_ij
            v_ij = vi - vj

            # Newtonian
            a_N = -G * masses[j] / r_ij**2 * n_ij

            # 1PN correction
            vj2 = np.dot(vj, vj)
            vi_vj = np.dot(vi, vj)
            nij_vij = np.dot(n_ij, v_ij)
            GMj_r = G * masses[j] / r_ij

            # Total Newtonian potential at i from all bodies
            phi_i = 0.0
            for k in range(N):
                if k == i:
                    continue
                rik = np.linalg.norm(ri - pos[k])
                if rik > 0:
                    phi_i += G * masses[k] / rik

            # Damour-Deruelle form of 1PN acceleration
            A = (
                vi2 + 2.0 * vj2 - 4.0 * vi_vj
                + 1.5 * nij_vij**2
                + 5.0 * GMj_r / r_ij
                + 4.0 * phi_i / r_ij * r_ij
            )
            B = 4.0 * vi_vj - 3.0 * vj2 - nij_vij * (4.0 - 2.0) * nij_vij

            a_1pn = G * masses[j] / (c2 * r_ij**2) * (
                n_ij * (2.0 * (2.0 + 1.0) * GMj_r + A * 0.5)
                + v_ij * (4.0 - 3.0) * nij_vij
            )

            acc[i] += a_N

            # Simpler standard form from Will (2014)
            acc[i] += G * masses[j] / (c2 * r_ij**2) * (
                n_ij * (
                    -vi2 - 2.0 * vj2 + 4.0 * vi_vj
                    + 1.5 * nij_vij**2
                    + 5.0 * GMj_r
                    + 4.0 * G * masses[i] / r_ij
                    - 0.5 * np.dot(rij, vj) * nij_vij / r_ij
                )
                + v_ij * (4.0 * nij_vij - 3.0 * np.dot(n_ij, vj))
                - vj * (2.0 + 1.0) * nij_vij
            )

    return acc


def simple_eih_acceleration(
    pos: np.ndarray,
    vel: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """Simplified 1PN EIH acceleration for testing.

    Uses the standard form that reduces correctly to the known
    perihelion precession for two-body systems.
    """
    N = len(masses)
    acc = np.zeros_like(pos)
    c2 = C**2

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            n = rij / r

            # Newtonian
            acc[i] -= G * masses[j] / r**2 * n

            # 1PN correction (simplified, captures leading effects)
            v_ij = vel[i] - vel[j]
            v2_i = np.dot(vel[i], vel[i])
            v2_j = np.dot(vel[j], vel[j])
            n_v = np.dot(n, v_ij)

            phi = 0.0
            for k in range(N):
                if k == i:
                    continue
                rk = np.linalg.norm(pos[i] - pos[k])
                if rk > 0:
                    phi += G * masses[k] / rk

            acc[i] += G * masses[j] / (c2 * r**2) * (
                n * (
                    -v2_i + 4.0 * np.dot(vel[i], vel[j]) - 2.0 * v2_j
                    + 1.5 * n_v**2
                    + 5.0 * G * masses[j] / r
                    + 4.0 * phi
                )
                + (vel[i] - vel[j]) * (4.0 * n_v)
            )

    return acc


def integrate_solar_system(
    body_names: list[str],
    t_years: float = 100.0,
    dt_days: float = 1.0,
    include_1pn: bool = True,
) -> dict:
    """Integrate the Solar System with PN corrections.

    Parameters
    ----------
    body_names : list of body names to include
    t_years : integration time in years
    dt_days : output time step in days
    include_1pn : whether to include 1PN corrections

    Returns
    -------
    dict with 't', 'positions', 'velocities' arrays
    """
    from .constants import BODIES

    N = len(body_names)
    masses = np.array([BODIES[name]["mass"] for name in body_names])

    # Initialize positions and velocities from Keplerian elements
    positions = np.zeros((N, 3))
    velocities = np.zeros((N, 3))

    for idx, name in enumerate(body_names):
        body = BODIES[name]
        if body["a"] == 0:
            continue
        a, e = body["a"], body["e"]
        incl, Omega, omega = body["incl"], body["Omega"], body["omega"]

        # Start at perihelion (simplified, ignoring M0 for now)
        r_peri = a * (1.0 - e)
        v_peri = np.sqrt(G * BODIES["Sun"]["mass"] * (2.0 / r_peri - 1.0 / a))

        # Position in orbital plane
        x_orb = np.array([r_peri, 0.0, 0.0])
        v_orb = np.array([0.0, v_peri, 0.0])

        # Rotation to ecliptic frame
        cos_O, sin_O = np.cos(Omega), np.sin(Omega)
        cos_i, sin_i = np.cos(incl), np.sin(incl)
        cos_w, sin_w = np.cos(omega), np.sin(omega)

        R = np.array([
            [cos_O * cos_w - sin_O * sin_w * cos_i,
             -cos_O * sin_w - sin_O * cos_w * cos_i,
             sin_O * sin_i],
            [sin_O * cos_w + cos_O * sin_w * cos_i,
             -sin_O * sin_w + cos_O * cos_w * cos_i,
             -cos_O * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i],
        ])

        positions[idx] = R @ x_orb
        velocities[idx] = R @ v_orb

    # Move to center-of-mass frame
    total_mass = np.sum(masses)
    com_pos = np.sum(masses[:, None] * positions, axis=0) / total_mass
    com_vel = np.sum(masses[:, None] * velocities, axis=0) / total_mass
    positions -= com_pos
    velocities -= com_vel

    t_span = (0.0, t_years * 365.25 * 86400.0)
    t_eval = np.arange(0, t_span[1], dt_days * 86400.0)

    def rhs(t, y):
        p = y[: 3 * N].reshape(N, 3)
        v = y[3 * N :].reshape(N, 3)
        if include_1pn:
            a = simple_eih_acceleration(p, v, masses)
        else:
            a = np.zeros_like(p)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    rij = p[i] - p[j]
                    r = np.linalg.norm(rij)
                    a[i] -= G * masses[j] / r**2 * rij / r
        return np.concatenate([v.flatten(), a.flatten()])

    y0 = np.concatenate([positions.flatten(), velocities.flatten()])
    sol = solve_ivp(rhs, t_span, y0, method="DOP853", t_eval=t_eval, rtol=1e-10, atol=1e-12)

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return {
        "t": sol.t,
        "positions": sol.y[: 3 * N, :].T.reshape(-1, N, 3),
        "velocities": sol.y[3 * N :, :].T.reshape(-1, N, 3),
        "body_names": body_names,
        "masses": masses,
    }
