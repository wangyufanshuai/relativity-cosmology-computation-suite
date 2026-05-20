"""Numerical integration of Schwarzschild geodesics for perihelion precession.

Uses the Binet equation formulation:
    u'' + u = GM/h² + 3GM·u²/c²

where u = 1/r, h = specific angular momentum, and prime = d/dφ.
The 3GM·u²/c² term is the GR correction that causes perihelion advance.

Perihelion detection uses dense ODE output + Brent's root finding for
maximum precision (~10⁻¹⁰ relative accuracy).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .constants import A_MERCURY, C, E_MERCURY, G, M_SUN


def schwarzschild_radius(M: float = M_SUN) -> float:
    """Schwarzschild radius r_s = 2GM/c²."""
    return 2.0 * G * M / C**2


def effective_potential(r: ArrayLike, L_tilde: float, rs: float) -> ArrayLike:
    """Effective potential V_eff(r) = (1 - rs/r)(1 + L_tilde²/r²)."""
    r = np.asarray(r, dtype=float)
    f = 1.0 - rs / r
    return f * (1.0 + L_tilde**2 / r**2)


def orbital_params_to_conserved(
    a: float = A_MERCURY,
    e: float = E_MERCURY,
    M: float = M_SUN,
) -> tuple[float, float, float, float]:
    """Convert Keplerian orbital elements to conserved quantities.

    Returns (E_tilde, L_tilde, r_perihelion, r_aphelion).
    """
    r_p = a * (1.0 - e)
    r_a = a * (1.0 + e)
    E_tilde = 1.0 - G * M / (2.0 * a * C**2)
    L_tilde = np.sqrt(G * M * a * (1.0 - e**2)) / C
    return E_tilde, L_tilde, r_p, r_a


def _binet_rhs(phi: float, y: np.ndarray, GM: float, h: float) -> np.ndarray:
    """RHS of the Binet equation for Schwarzschild geodesic.

    u'' + u = GM/h² + 3GM·u²/c²
    State: y = [u, u'] where u = 1/r.
    """
    u, up = y
    upp = GM / h**2 - u + 3.0 * GM * u**2 / C**2
    return np.array([up, upp])


def _find_perihelion_in_range(
    sol: object, phi_lo: float, phi_hi: float, n_probe: int = 10000
) -> float | None:
    """Find a perihelion (u maximum = up zero crossing pos→neg) in [phi_lo, phi_hi].

    Uses dense_output from solve_ivp and Brent's method for sub-interval accuracy.
    """
    phi_test = np.linspace(phi_lo, phi_hi, n_probe)
    up_test = np.array([sol.sol(p)[1] for p in phi_test])

    for i in range(1, len(up_test)):
        if up_test[i - 1] > 0 and up_test[i] < 0:
            return brentq(lambda p: sol.sol(p)[1], phi_test[i - 1], phi_test[i])
    return None


def integrate_orbit(
    a: float = A_MERCURY,
    e: float = E_MERCURY,
    M: float = M_SUN,
    n_orbits: int = 10,
    method: str = "DOP853",
    rtol: float = 1e-13,
    atol: float = 1e-15,
) -> dict:
    """Integrate Mercury's orbit using the Binet equation.

    Parameters
    ----------
    a : semi-major axis (m)
    e : eccentricity
    M : central mass (kg)
    n_orbits : number of orbits to integrate
    method : ODE solver
    rtol, atol : tolerances

    Returns
    -------
    dict with precession measurements and orbital data.
    """
    GM = G * M
    h = np.sqrt(GM * a * (1.0 - e**2))
    p = a * (1.0 - e**2)
    r_p = a * (1.0 - e)

    # Initial conditions at perihelion
    u0 = (1.0 + e) / p  # = 1/r_p
    up0 = 0.0

    phi_max = n_orbits * 2.0 * np.pi * 1.01  # slight overshoot

    sol = solve_ivp(
        _binet_rhs,
        [0.0, phi_max],
        np.array([u0, up0]),
        args=(GM, h),
        method=method,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=0.01,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    # Sample orbit for plotting
    phi_plot = np.linspace(0, n_orbits * 2.0 * np.pi, n_orbits * 5000 + 1)
    u_plot = sol.sol(phi_plot)[0]
    r_plot = 1.0 / u_plot

    # Find perihelion passages using dense output + Brent
    # phi=0 is the first perihelion
    perihelion_phis = [0.0]
    for orbit_idx in range(1, n_orbits + 1):
        # Search in a window around 2π·n
        phi_center = orbit_idx * 2.0 * np.pi
        phi_lo = phi_center - 0.01  # 10 mrad window
        phi_hi = phi_center + 0.01
        phi_peri = _find_perihelion_in_range(sol, phi_lo, phi_hi)
        if phi_peri is not None:
            perihelion_phis.append(phi_peri)

    perihelion_phis = np.array(perihelion_phis)

    result: dict = {
        "phi": phi_plot,
        "r": r_plot,
        "u": u_plot,
        "perihelion_phis": perihelion_phis,
    }

    # Compute precession
    if len(perihelion_phis) >= 2:
        advances = np.diff(perihelion_phis)
        precessions = advances - 2.0 * np.pi
        mean_precession = np.mean(precessions)

        T_orbit = 2.0 * np.pi * np.sqrt(a**3 / GM)
        seconds_per_century = 100.0 * 365.25 * 86400.0
        orbits_per_century = seconds_per_century / T_orbit
        precession_arcsec = mean_precession * orbits_per_century * (180.0 * 3600.0 / np.pi)

        result["precession_per_orbit_rad"] = mean_precession
        result["precession_per_orbit_std"] = np.std(precessions) if len(precessions) > 1 else 0.0
        result["precession_arcsec_per_century"] = precession_arcsec
    else:
        result["precession_per_orbit_rad"] = 0.0
        result["precession_per_orbit_std"] = 0.0
        result["precession_arcsec_per_century"] = 0.0

    analytical = 6.0 * np.pi * GM / (a * (1.0 - e**2) * C**2)
    result["analytical_precession_rad"] = analytical

    return result
