"""Post-processing: extract orbital elements and precession from simulation."""

from __future__ import annotations

import numpy as np

from .constants import G, C


def orbital_elements(
    pos: np.ndarray,
    vel: np.ndarray,
    M_central: float,
) -> dict:
    """Compute osculating Keplerian elements from state vectors.

    Parameters
    ----------
    pos : (3,) position [m]
    vel : (3,) velocity [m/s]
    M_central : central mass [kg]

    Returns
    -------
    dict with a, e, i, Omega, omega, true_anomaly, r, v
    """
    r_vec = pos
    v_vec = vel
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    mu = G * M_central

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    # Node vector
    z_hat = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(z_hat, h_vec)
    n = np.linalg.norm(n_vec)

    # Eccentricity vector
    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    # Semi-major axis
    energy = v**2 / 2.0 - mu / r
    if abs(e - 1.0) > 1e-6:
        a = -mu / (2.0 * energy)
    else:
        a = np.inf

    # Inclination
    incl = np.arccos(np.clip(h_vec[2] / h, -1, 1))

    # Longitude of ascending node
    if n > 1e-20:
        Omega = np.arccos(np.clip(n_vec[0] / n, -1, 1))
        if n_vec[1] < 0:
            Omega = 2.0 * np.pi - Omega
    else:
        Omega = 0.0

    # Argument of perihelion
    if n > 1e-20 and e > 1e-15:
        omega = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0:
            omega = 2.0 * np.pi - omega
    else:
        omega = 0.0

    # True anomaly
    if e > 1e-15:
        nu = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            nu = 2.0 * np.pi - nu
    else:
        nu = 0.0

    return {
        "a": a,
        "e": e,
        "incl": incl,
        "Omega": Omega,
        "omega": omega,
        "nu": nu,
        "r": r,
        "v": v,
        "energy": energy,
        "h": h,
    }


def extract_precession(
    positions: np.ndarray,
    velocities: np.ndarray,
    body_idx: int,
    M_central: float,
) -> dict:
    """Extract perihelion precession rate from time series of orbital elements.

    Parameters
    ----------
    positions : (T, N, 3) positions over time
    velocities : (T, N, 3) velocities over time
    body_idx : index of body to analyze
    M_central : central mass [kg]

    Returns
    -------
    dict with precession rates and omega time series
    """
    T = positions.shape[0]
    omegas = np.zeros(T)

    for t in range(T):
        elements = orbital_elements(positions[t, body_idx], velocities[t, body_idx], M_central)
        omegas[t] = elements["omega"]

    # Unwrap omega to handle 2π crossings
    omegas_unwrapped = np.unwrap(omegas)

    # Fit linear trend
    t_arr = np.arange(T)
    coeffs = np.polyfit(t_arr, omegas_unwrapped, 1)
    domega_dt = coeffs[0]  # rad per time step

    return {
        "omega_series": omegas_unwrapped,
        "domega_dt": domega_dt,
        "total_precession": omegas_unwrapped[-1] - omegas_unwrapped[0],
    }
