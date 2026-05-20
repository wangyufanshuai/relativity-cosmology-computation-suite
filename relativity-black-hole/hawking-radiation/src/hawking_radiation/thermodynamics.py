"""Black hole thermodynamics and Hawking radiation.

All formulas in SI units unless noted.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G, C, HBAR, K_B, M_SUN


def hawking_temperature(M: float) -> float:
    """Hawking temperature T_H = ℏc³/(8πGMk_B) [K].

    Inversely proportional to mass: smaller BH = hotter.
    """
    return HBAR * C**3 / (8.0 * np.pi * G * M * K_B)


def bh_entropy(M: float) -> float:
    """Bekenstein-Hawking entropy S = 4πGM²/(ℏc) in units of k_B."""
    return 4.0 * np.pi * G * M**2 / (HBAR * C)


def bh_heat_capacity(M: float) -> float:
    """Specific heat C = -8πG²M²k_B/(ℏc³) [J/K].

    Always negative: black holes are thermodynamically unstable.
    """
    return -8.0 * np.pi * G**2 * M**2 * K_B / (HBAR * C**3)


def bh_luminosity(M: float) -> float:
    """Luminosity from Hawking radiation L = ℏc⁶/(15360πG²M²) [W].

    Assumes Stefan-Boltzmann with greybody factor ≈ 1.
    """
    return HBAR * C**6 / (15360.0 * np.pi * G**2 * M**2)


def bh_lifetime(M: float) -> float:
    """Evaporation lifetime t ≈ 5120πG²M³/(ℏc⁴) [s]."""
    return 5120.0 * np.pi * G**2 * M**3 / (HBAR * C**4)


def emission_spectrum(omega: float, M: float, greybody: float = 1.0) -> float:
    """Hawking emission rate dN/(dω dt) for bosons.

    Γ(ω)/(2π) · 1/(exp(ℏω/(k_B T_H)) - 1)
    """
    T = hawking_temperature(M)
    x = HBAR * omega / (K_B * T)
    x = min(x, 500)  # prevent overflow
    return greybody / (2.0 * np.pi) * 1.0 / (np.exp(x) - 1.0 + 1e-300)


def fermion_emission(omega: float, M: float, greybody: float = 1.0) -> float:
    """Hawking emission rate for fermions (Fermi-Dirac)."""
    T = hawking_temperature(M)
    x = HBAR * omega / (K_B * T)
    x = min(x, 500)
    return greybody / (2.0 * np.pi) * 1.0 / (np.exp(x) + 1.0)


def integrate_evaporation(M_initial: float, t_end: float, n_points: int = 1000) -> dict:
    """Integrate BH evaporation dM/dt = -L(M)/c².

    Returns dict with 't', 'M', 'T', 'S' arrays.
    """
    def rhs(t, M_arr):
        M = M_arr[0]
        if M < 1e-10:  # effectively evaporated
            return [0.0]
        L = bh_luminosity(M)
        return [-L / C**2]

    # Don't integrate past complete evaporation
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, n_points)

    sol = solve_ivp(rhs, t_span, [M_initial], t_eval=t_eval,
                    method="RK45", rtol=1e-10, atol=1e-10)

    M_arr = sol.y[0]
    M_arr = np.maximum(M_arr, 0)  # clamp

    return {
        "t": sol.t,
        "M": M_arr,
        "T": np.array([hawking_temperature(m) if m > 0 else np.inf for m in M_arr]),
        "S": np.array([bh_entropy(m) if m > 0 else 0 for m in M_arr]),
    }


def compute_page_curve(M_initial: float, n_points: int = 200) -> dict:
    """Compute Page curve: S_rad + S_BH vs time.

    The Page curve shows:
    - S_BH decreases monotonically (BH shrinks)
    - S_rad increases initially (radiation accumulates)
    - Total entropy peaks at Page time when S_rad ≈ S_BH
    """
    t_evap = bh_lifetime(M_initial)
    t_array = np.linspace(0, t_evap * 0.99, n_points)

    result = integrate_evaporation(M_initial, t_array[-1], n_points)

    S_bh = result["S"]
    S_initial = S_bh[0]

    # Radiation entropy: approximate as S_rad ≈ S_initial - S_bh(t)
    # (unitarity assumption: total entropy conserved)
    S_rad = S_initial - S_bh

    # Page time: when S_rad = S_bh
    page_idx = np.argmin(np.abs(S_rad - S_bh))

    return {
        "t": result["t"],
        "S_bh": S_bh,
        "S_rad": S_rad,
        "S_total": S_rad + S_bh,
        "page_time": t_array[page_idx] if page_idx < len(t_array) else t_array[-1],
        "page_fraction": t_array[page_idx] / t_evap if page_idx < len(t_array) else 0.5,
    }
