"""IMR (Inspiral-Merger-Ringdown) waveform model."""

from __future__ import annotations

import numpy as np
from .constants import G, C, M_SUN
from .orbital import chirp_mass, symmetric_mass_ratio


def final_mass(m1: float, m2: float) -> float:
    """Final BH mass after merger (fits from numerical relativity)."""
    M = m1 + m2
    eta = symmetric_mass_ratio(m1, m2)
    # Fit from Healy et al. (2014)
    return M * (1.0 - 0.0580 * eta - 0.0463 * eta**2)


def final_spin(m1: float, m2: float) -> float:
    """Final BH dimensionless spin after merger."""
    eta = symmetric_mass_ratio(m1, m2)
    # Fit from Healy et al. (2014) for non-spinning binaries
    return 0.6858 * (4 * eta)**0.5 + 0.1868 * (4 * eta)


def ringdown_frequency(m1: float, m2: float) -> float:
    """Ringdown frequency (Hz) for the l=2, m=2 mode.

    f_RD = c³(1 - 0.63(1-η)^0.3) / (2πGM_f)
    """
    eta = symmetric_mass_ratio(m1, m2)
    M_f = final_mass(m1, m2)
    omega_22 = (1.0 - 0.63 * (1.0 - eta)**0.3)
    return omega_22 * C**3 / (2.0 * np.pi * G * M_f)


def ringdown_damping(m1: float, m2: float) -> float:
    """Ringdown damping time (s).

    τ = 2(1-η)^0.45 / ω_I where ω_I from QNM fits.
    """
    eta = symmetric_mass_ratio(m1, m2)
    M_f = final_mass(m1, m2)
    # Damping from Echeverria (1989) fit
    f_rd = ringdown_frequency(m1, m2)
    Q = 2.0 * (1.0 - eta)**(-0.45)
    return Q / (np.pi * f_rd)


def imr_waveform(
    m1: float,
    m2: float,
    distance: float = 1.0e25,
    f_low: float = 20.0,
    dt: float = 1.0 / 4096.0,
) -> dict:
    """Generate IMR waveform (inspiral + merger + ringdown).

    Uses a simplified IMRPhenomD-like model in the frequency domain.
    """
    from .taylor import taylor_t4

    Mc = chirp_mass(m1, m2)
    eta = symmetric_mass_ratio(m1, m2)
    M_f = final_mass(m1, m2)
    f_rd = ringdown_frequency(m1, m2)
    tau_rd = ringdown_damping(m1, m2)

    # Inspiral phase using TaylorT4
    f_merger = 0.0186 / (G * (m1 + m2) / C**3)  # rough merger frequency
    inspiral = taylor_t4(Mc, eta, f_low, min(f_merger, f_rd * 0.8), dt, distance=distance)

    # Ringdown: damped sinusoid
    t_end_inspiral = inspiral["t"][-1] if len(inspiral["t"]) > 0 else 0
    phi_end = inspiral["phase"][-1] if len(inspiral["phase"]) > 0 else 0
    h_end = inspiral["h_plus"][-1] if len(inspiral["h_plus"]) > 0 else 0

    n_ringdown = int(10 * tau_rd / dt)
    t_ring = np.arange(n_ringdown) * dt
    omega_rd = 2.0 * np.pi * f_rd

    h_ring_plus = h_end * np.exp(-t_ring / tau_rd) * np.cos(omega_rd * t_ring + phi_end)
    h_ring_cross = h_end * np.exp(-t_ring / tau_rd) * np.sin(omega_rd * t_ring + phi_end)

    # Concatenate
    t_full = np.concatenate([inspiral["t"], t_end_inspiral + t_ring[1:]])
    h_plus = np.concatenate([inspiral["h_plus"], h_ring_plus[1:]])
    h_cross = np.concatenate([inspiral["h_cross"], h_ring_cross[1:]])
    f_full = np.concatenate([inspiral["frequency"], np.full(len(t_ring) - 1, f_rd)])

    return {
        "t": t_full,
        "h_plus": h_plus,
        "h_cross": h_cross,
        "frequency": f_full,
        "phase": np.cumsum(2 * np.pi * f_full * dt),
    }
