"""
Tidal deformability and Love numbers for neutron stars.

Computes the quadrupolar tidal Love number k_2 and the dimensionless
tidal deformability Lambda for a given TOV solution and EOS.

Geometrized units (G = c = 1), lengths in km.

Physics
-------
* The tidal Love number k_2 is obtained by integrating the second-order
  (l=2) perturbation equation (H function) alongside the TOV equations.
* Lambda = (2/3) k_2 (R / (M))^5        (C = M/R is compactness)
* For a binary (m1, m2) the combined dimensionless tidal deformability is

  Lambda_tilde = (16/13) * [(m1 + 12*m2)*m1^4*Lambda1
                            + (m2 + 12*m1)*m2^4*Lambda2] / (m1+m2)^5

References
----------
Hinderer (2008) ApJ 677 1216; Flanagan & Hinderer (2008) PRL 100 1108;
Damour & Nagar (2009) PRD 80 084035.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .eos import BaseEOS
from .tov import solve_tov, TOVResult, M_SUN_KM

# ---------------------------------------------------------------------------
# Tidal Love number k_2
# ---------------------------------------------------------------------------

def love_number_k2(tov: TOVResult, eos: BaseEOS) -> float:
    """Compute the tidal Love number k_2 for a given TOV star.

    This integrates the perturbation variable H alongside the TOV equations
    using the method described in Hinderer (2008) and Damour & Nagar (2009).

    Parameters
    ----------
    tov : TOVResult
        A solved TOV profile (must contain r, m, P, epsilon arrays).
    eos : BaseEOS
        Equation of state used for the TOV solution.

    Returns
    -------
    k2 : float
        Tidal Love number (dimensionless).
    """
    R = tov.radius_km
    M = tov.mass_msun * M_SUN_KM  # back to km

    if R < 1e-6 or M < 1e-12:
        return 0.0

    C = M / R  # compactness

    # We need to co-integrate the TOV + perturbation equations.
    # The perturbation equation for H (l=2) is:
    #
    #   H'' + H' * [dPhi/dr + (2/r)] - H * [4 pi (dP/deps + 1)(eps+P)/r(r-2m)
    #              + 4 pi (eps+P)/(r-2m) * (5*eps + 8*P)/r + 6/r^2
    #              - 4 (dPhi/dr)^2]
    #   = 0
    #
    # where dPhi/dr = (m + 4 pi r^3 P) / (r(r - 2m)).
    #
    # Rather than re-integrating from scratch, we use the semi-analytic
    # formula for k_2 that requires only stellar-surface quantities:
    #
    #   k_2 = (8 C^5 / 5) * (1 - 2C)^2 [2 + 2C(y_R - 1) - y_R]
    #         / { 2C [6 - 3y_R + 3C(5y_R - 8)]
    #             + 4C^3 [13 - 11y_R + C(3y_R - 2) + 2C^2(1+y_R)]
    #             + 3(1-2C)^2 [2 - y_R + 2C(y_R-1)] ln(1-2C) }
    #
    # where y_R = y(R) and y = r * H'/H.

    # To get y_R, we integrate y alongside the TOV structure.
    # The ODE for y (Hinderer 2008, eq. 9):
    #
    #   dy/dr = -y^2/r - y [ (dPhi/dr) + 1/r ] + r * S(r)
    #
    # where S(r) encodes the source terms:
    #   S = 4 pi (dP/deps + 1)(eps+P) / [r(r-2m)]
    #       - 4 pi (5 eps + 8 P) / (r-2m)
    #       + 6/r^2
    #       - 4 (dPhi/dr)^2

    P_c = tov.central_pressure
    eps_c = float(eos.epsilon_from_pressure(np.array([P_c]))[0])

    def rhs(r, state):
        m_val, P_val, y_val = state
        if r < 1e-12:
            return [0.0, 0.0, 2.0]

        eps_val = float(eos.epsilon_from_pressure(np.array([max(P_val, 0.0)]))[0])
        denom = r * (r - 2.0 * m_val)
        if abs(denom) < 1e-30:
            return [0.0, 0.0, 2.0]

        # TOV
        dmdr = 4.0 * np.pi * r**2 * eps_val
        dPdr = -(eps_val + P_val) * (m_val + 4.0 * np.pi * r**3 * P_val) / denom
        dPhi_dr = (m_val + 4.0 * np.pi * r**3 * P_val) / denom

        # dP/deps via EOS sound speed
        eps_arr = np.array([eps_val])
        cs2 = float(eos.sound_speed_squared(eps_arr)[0])

        # Perturbation y
        S = (4.0 * np.pi * (cs2 + 1.0) * (eps_val + P_val) / denom
             - 4.0 * np.pi * (5.0 * eps_val + 8.0 * P_val) / (r - 2.0 * m_val)
             + 6.0 / r**2
             - 4.0 * dPhi_dr**2)

        dy_dr = -y_val**2 / r - y_val * (dPhi_dr + 1.0 / r) + r * S

        return [dmdr, dPdr, dy_dr]

    def surface_event(r, state):
        return state[1] - 1e-10 * P_c
    surface_event.terminal = True
    surface_event.direction = -1

    dr = 0.005
    r0 = dr
    m0 = (4.0 / 3.0) * np.pi * eps_c * r0**3
    P0 = P_c - (2.0 / 3.0) * np.pi * (eps_c + P_c) * (eps_c + 3.0 * P_c) * r0**2
    P0 = max(P0, 0.0)

    # Near r=0, y -> 2  (regular solution)
    y0 = 2.0

    sol = solve_ivp(rhs, (r0, 30.0), [m0, P0, y0],
                    events=surface_event,
                    max_step=dr, rtol=1e-8, atol=1e-10)

    if sol.t[-1] < 0.5:
        return 0.0

    y_R = float(sol.y[2, -1])
    R_actual = float(sol.t[-1])
    M_actual = float(sol.y[0, -1])
    C_actual = M_actual / R_actual

    # Hinderer k2 formula
    k2 = _k2_from_y_C(y_R, C_actual)
    return float(np.clip(k2, 0.0, 1.0))


def _k2_from_y_C(y_R: float, C: float) -> float:
    """Compute k_2 from surface y-value and compactness C = M/R."""
    if C >= 0.5 or C <= 0.0:
        return 0.0
    num = (8.0 / 5.0) * C**5 * (1.0 - 2.0 * C)**2 * (
        2.0 + 2.0 * C * (y_R - 1.0) - y_R
    )
    denom = (
        2.0 * C * (6.0 - 3.0 * y_R + 3.0 * C * (5.0 * y_R - 8.0))
        + 4.0 * C**3 * (13.0 - 11.0 * y_R + C * (3.0 * y_R - 2.0)
                         + 2.0 * C**2 * (1.0 + y_R))
        + 3.0 * (1.0 - 2.0 * C)**2 * (2.0 - y_R + 2.0 * C * (y_R - 1.0))
          * np.log(1.0 - 2.0 * C)
    )
    if abs(denom) < 1e-30:
        return 0.0
    return num / denom

# ---------------------------------------------------------------------------
# Tidal deformability
# ---------------------------------------------------------------------------

@dataclass
class TidalResult:
    """Tidal properties of a single neutron star."""

    k2: float             # Love number
    lambda_: float        # dimensionless tidal deformability
    mass_msun: float
    radius_km: float
    compactness: float    # C = M/R


def compute_tidal(eos: BaseEOS, P_c: float, **tov_kw) -> TidalResult:
    """Compute tidal Love number and deformability for a star with central
    pressure *P_c*.

    Parameters
    ----------
    eos : BaseEOS
    P_c : float   Central pressure [km^-2].

    Returns
    -------
    TidalResult
    """
    tov = solve_tov(eos, P_c, **tov_kw)
    k2 = love_number_k2(tov, eos)

    M_km = tov.mass_msun * M_SUN_KM
    R_km = tov.radius_km
    C = M_km / R_km if R_km > 0 else 0.0

    # Lambda = (2/3) k2 * C^{-5}
    if C > 0 and k2 > 0:
        lam = (2.0 / 3.0) * k2 * (1.0 / C) ** 5
    else:
        lam = 0.0

    return TidalResult(
        k2=k2,
        lambda_=lam,
        mass_msun=tov.mass_msun,
        radius_km=tov.radius_km,
        compactness=C,
    )

# ---------------------------------------------------------------------------
# Binary tidal deformability
# ---------------------------------------------------------------------------

def combined_tidal_deformability(m1: float, m2: float,
                                 lam1: float, lam2: float) -> float:
    """Combined dimensionless tidal deformability for a binary.

    Lambda_tilde = (16/13) * [(m1 + 12*m2)*m1^4*Lambda1
                              + (m2 + 12*m1)*m2^4*Lambda2] / (m1+m2)^5

    All masses in Msun.
    """
    M = m1 + m2
    if M <= 0:
        return 0.0
    lt = (16.0 / 13.0) * (
        (m1 + 12.0 * m2) * m1**4 * lam1
        + (m2 + 12.0 * m1) * m2**4 * lam2
    ) / M**5
    return float(lt)


def binary_love_relation(m1: float, m2: float,
                         lam1: float) -> float:
    """Approximate Lambda2 from Lambda1 using the binary Love relation.

    Lambda1 / Lambda2 ~ (m2 / m1)^6  (leading-order scaling).
    """
    if m1 <= 0 or lam1 <= 0:
        return 0.0
    return lam1 * (m1 / m2) ** 6
