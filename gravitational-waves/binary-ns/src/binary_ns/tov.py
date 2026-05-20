"""
Tolman-Oppenheimer-Volkoff (TOV) equation solver.

Solves the TOV structure equations for a static, spherically symmetric
neutron star given an equation of state.

Geometrized units (G = c = 1), lengths in km.

TOV equations
-------------
dm/dr      = 4 pi r^2 epsilon
dP/dr      = - (epsilon + P)(m + 4 pi r^3 P) / [r(r - 2m)]
dPhi/dr    = (m + 4 pi r^3 P) / [r(r - 2m)]

Boundary conditions:
    m(0) = 0, P(0) = P_c, Phi(R) surface matched to exterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .eos import BaseEOS

# Physical constants in geometrized units (km)
G = 1.0  # geometrized
C = 1.0  # geometrized
M_SUN_KM = 1.476625038  # 1 Msun in km  (G Msun / c^2)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TOVResult:
    """Output from a single TOV integration."""

    radius_km: float           # stellar radius R
    mass_msun: float           # gravitational mass in Msun
    r: np.ndarray              # radial grid  [km]
    m: np.ndarray              # enclosed mass [km]
    P: np.ndarray              # pressure      [km^-2]
    epsilon: np.ndarray        # energy density [km^-2]
    central_pressure: float    # P_c


@dataclass
class MassRadiusCurve:
    """Mass-radius relation from a sequence of TOV solutions."""

    masses_msun: np.ndarray    # gravitational mass [Msun]
    radii_km: np.ndarray       # stellar radius [km]
    central_pressures: np.ndarray  # P_c [km^-2]

    @property
    def max_mass(self) -> float:
        return float(np.max(self.masses_msun))

    @property
    def max_mass_radius(self) -> float:
        idx = int(np.argmax(self.masses_msun))
        return float(self.radii_km[idx])

# ---------------------------------------------------------------------------
# Single star
# ---------------------------------------------------------------------------

def solve_tov(eos: BaseEOS,
              P_c: float,
              r_max: float = 30.0,
              dr: float = 0.005,
              P_min_frac: float = 1e-10) -> TOVResult:
    """Integrate the TOV equations for a given central pressure.

    Parameters
    ----------
    eos : BaseEOS
        Equation of state.
    P_c : float
        Central pressure [km^-2].
    r_max : float
        Maximum integration radius [km].
    dr : float
        Radial step size [km].
    P_min_frac : float
        Stop when P drops below P_min_frac * P_c.

    Returns
    -------
    TOVResult
    """
    eps_c = float(eos.epsilon_from_pressure(np.array([P_c]))[0])
    P_min = P_min_frac * P_c

    def rhs(r, y):
        m_val, P_val = y
        if r < 1e-12:
            return [0.0, 0.0]
        eps_val = float(eos.epsilon_from_pressure(np.array([max(P_val, 0.0)]))[0])
        denom = r * (r - 2.0 * m_val)
        if abs(denom) < 1e-30:
            return [0.0, 0.0]
        dmdr = 4.0 * np.pi * r**2 * eps_val
        dPdr = -(eps_val + P_val) * (m_val + 4.0 * np.pi * r**3 * P_val) / denom
        return [dmdr, dPdr]

    def surface_event(r, y):
        """P = P_min triggers termination."""
        return y[1] - P_min
    surface_event.terminal = True
    surface_event.direction = -1

    # Taylor expansion near r=0 to avoid 0/0:
    #   m ~ (4/3) pi eps_c r^3
    #   P ~ P_c - (2/3) pi (eps_c + P_c)(eps_c + 3P_c) r^2
    r0 = dr
    m0 = (4.0 / 3.0) * np.pi * eps_c * r0**3
    P0 = P_c - (2.0 / 3.0) * np.pi * (eps_c + P_c) * (eps_c + 3.0 * P_c) * r0**2
    P0 = max(P0, P_min * 0.5)

    sol = solve_ivp(rhs, (r0, r_max), [m0, P0],
                    events=surface_event,
                    max_step=dr, rtol=1e-8, atol=1e-10,
                    dense_output=True)

    r_arr = sol.t
    m_arr = sol.y[0]
    P_arr = np.maximum(sol.y[1], 0.0)

    # Append r=0 point
    r_arr = np.concatenate([[0.0], r_arr])
    m_arr = np.concatenate([[0.0], m_arr])
    P_arr = np.concatenate([[P_c], P_arr])

    eps_arr = eos.epsilon_from_pressure(P_arr)

    R = float(r_arr[-1])
    M = float(m_arr[-1])

    return TOVResult(
        radius_km=R,
        mass_msun=M / M_SUN_KM,
        r=r_arr,
        m=m_arr,
        P=P_arr,
        epsilon=eps_arr,
        central_pressure=P_c,
    )

# ---------------------------------------------------------------------------
# Mass-radius curve
# ---------------------------------------------------------------------------

def mass_radius_curve(eos: BaseEOS,
                      log_P_c_min: float = -6.0,
                      log_P_c_max: float = 0.5,
                      n_points: int = 80,
                      **tov_kw) -> MassRadiusCurve:
    """Compute the mass-radius relation by varying central pressure.

    Parameters
    ----------
    eos : BaseEOS
    log_P_c_min, log_P_c_max : float
        log10 of the central-pressure range [km^-2].
    n_points : int
        Number of central pressures to sample.

    Returns
    -------
    MassRadiusCurve
    """
    log_Pc = np.linspace(log_P_c_min, log_P_c_max, n_points)
    P_c_arr = 10.0 ** log_Pc

    masses = []
    radii = []
    for P_c in P_c_arr:
        try:
            res = solve_tov(eos, P_c, **tov_kw)
            masses.append(res.mass_msun)
            radii.append(res.radius_km)
        except Exception:
            continue

    masses = np.array(masses)
    radii = np.array(radii)
    Pc_valid = P_c_arr[: len(masses)]

    return MassRadiusCurve(
        masses_msun=masses,
        radii_km=radii,
        central_pressures=Pc_valid,
    )
