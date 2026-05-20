"""Taylor-family post-Newtonian time- and frequency-domain waveforms.

Implements TaylorT1, TaylorT2, and TaylorT4 approximants.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .constants import C, G, MPC
from .orbital import chirp_mass, symmetric_mass_ratio, schwarzschild_isco_freq


def _amp_scale(Mc: float, distance: float) -> float:
    """Leading-order GW amplitude prefactor."""
    return (4.0 * G * Mc / C ** 2) ** (5.0 / 4.0) * (G * Mc / C ** 3) ** (-1.0 / 4.0) / distance


def _dfdt(Mc: float, eta: float, f: float, pn_order: float = 3.5) -> float:
    """PN-corrected df/dt for the inspiral.

    Parameters
    ----------
    Mc : float
        Chirp mass [kg].
    eta : float
        Symmetric mass ratio.
    f : float
        GW frequency [Hz].
    pn_order : float
        PN order.

    Returns
    -------
    float
        df/dt [Hz/s].
    """
    # Leading order
    v = (np.pi * G * Mc * f / C ** 3) ** (1.0 / 3.0)
    x = v * v  # PN expansion parameter

    lo = (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * (G * Mc / C ** 3) ** (5.0 / 3.0) * f ** (11.0 / 3.0)

    # Sum PN corrections
    correction = 1.0
    if pn_order >= 0.5:
        correction += (743.0 / 336.0 + 11.0 * eta / 4.0) * x
    if pn_order >= 1.0:
        correction += (-32.0 * np.pi) * x ** 1.5
    if pn_order >= 1.5:
        correction += (34103.0 / 18144.0
                       + 13661.0 * eta / 2016.0
                       + 59.0 * eta ** 2 / 18.0) * x ** 2
    if pn_order >= 2.0:
        correction += (-(4159.0 / 672.0) * np.pi
                       - (189.0 / 8.0) * np.pi * eta) * x ** 2.5
    if pn_order >= 2.5:
        correction += (16447322263.0 / 139708800.0
                       - 1712.0 * np.pi ** 2 / 105.0
                       + (451.0 * np.pi ** 2 / 48.0
                          - 56198689.0 / 217728.0) * eta
                       + 541.0 * eta ** 2 / 896.0
                       - 5605.0 * eta ** 3 / 2592.0) * x ** 3
    if pn_order >= 3.0:
        correction += (-(4415.0 / 4032.0) * np.pi
                       - (358675.0 / 6048.0) * np.pi * eta
                       + (91495.0 / 1512.0) * np.pi * eta ** 2) * x ** 3.5

    return lo * correction


# ---------------------------------------------------------------------------
# TaylorT1 — ODE-based time-domain waveform
# ---------------------------------------------------------------------------

def taylor_t1(
    Mc: float,
    eta: float,
    f_low: float,
    f_high: float,
    dt: float,
    pn_order: float = 3.5,
    distance: float = 1.0 * MPC,
) -> dict[str, NDArray[np.floating]]:
    """TaylorT1 time-domain waveform.

    Solves the coupled ODEs df/dt and dphi/dt = 2 pi f numerically.

    Parameters
    ----------
    Mc : float
        Chirp mass [kg].
    eta : float
        Symmetric mass ratio.
    f_low : float
        Starting GW frequency [Hz].
    f_high : float
        Ending GW frequency [Hz].
    dt : float
        Time step [s].
    pn_order : float
        PN order.
    distance : float
        Luminosity distance [m].

    Returns
    -------
    dict
        Keys: 't', 'h_plus', 'h_cross', 'frequency', 'phase'.
    """
    # Integrate backwards from coalescence so we use positive times
    # Instead, we integrate forward and reverse at the end.
    def rhs(t, y):
        f = y[0]
        if f <= 0 or f >= f_high:
            return [0.0, 0.0]
        dfdt_val = _dfdt(Mc, eta, f, pn_order)
        dphidt = 2.0 * np.pi * f
        return [dfdt_val, dphidt]

    def event_f_high(t, y):
        return y[0] - f_high
    event_f_high.terminal = True

    # Estimate total duration (leading order)
    Mc_dim = G * Mc / C ** 3
    t_chirp = (5.0 / 256.0) * Mc_dim ** (-5.0 / 3.0) * (np.pi * f_low) ** (-8.0 / 3.0)

    sol = solve_ivp(
        rhs,
        [0, 2.0 * t_chirp],
        [f_low, 0.0],
        max_step=dt,
        dense_output=True,
        events=event_f_high,
        rtol=1e-10,
        atol=1e-12,
    )

    t_arr = np.arange(0, sol.t[-1], dt)
    y_arr = sol.sol(t_arr)
    freq = y_arr[0]
    phase = y_arr[1]

    amp = _amp_scale(Mc, distance) * (np.pi * f_low) ** (2.0 / 3.0) * (freq / f_low) ** (2.0 / 3.0)

    h_plus = amp * np.cos(phase)
    h_cross = amp * np.sin(phase)

    # Make time go from -t to 0
    t_arr = t_arr - t_arr[-1]

    return {
        "t": t_arr,
        "h_plus": h_plus,
        "h_cross": h_cross,
        "frequency": freq,
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# TaylorT2 — stationary-phase frequency-domain waveform
# ---------------------------------------------------------------------------

def taylor_t2(
    Mc: float,
    eta: float,
    f_low: float,
    f_high: float,
    df: float,
    pn_order: float = 3.5,
) -> dict[str, NDArray[np.floating]]:
    """TaylorT2 frequency-domain waveform via stationary-phase approximation.

    Parameters
    ----------
    Mc : float
        Chirp mass [kg].
    eta : float
        Symmetric mass ratio.
    f_low : float
        Starting GW frequency [Hz].
    f_high : float
        Ending GW frequency [Hz].
    df : float
        Frequency resolution [Hz].
    pn_order : float
        PN order.

    Returns
    -------
    dict
        Keys: 't' (frequencies), 'h_plus', 'h_cross', 'frequency', 'phase'.
    """
    freq = np.arange(f_low, f_high, df)
    if len(freq) == 0:
        freq = np.array([f_low])

    Mc_dim = G * Mc / C ** 3
    v = (np.pi * Mc_dim * freq) ** (1.0 / 3.0)

    # Leading-order amplitude
    amp = (5.0 / 24.0) ** 0.5 * (np.pi ** (-2.0 / 3.0)) * Mc_dim ** (5.0 / 6.0) * freq ** (-7.0 / 6.0)

    # Phase in SPA: Psi(f) = 2 pi f t_c - phi_c - pi/4 + 3/(128 eta v^5) * (sum PN corrections)
    phase = 3.0 / (128.0 * eta * v ** 5)

    # Add PN phase corrections
    x = v ** 2
    if pn_order >= 0.5:
        phase += (3715.0 / 756.0 + 55.0 * eta / 9.0) / (128.0 * eta) * (1.0 / v) * x
    if pn_order >= 1.0:
        phase += (-16.0 * np.pi) / (128.0 * eta) * (1.0 / v) * x ** 1.5

    h_plus = amp * np.cos(phase)
    h_cross = amp * np.sin(phase)

    return {
        "t": freq,  # frequency axis
        "h_plus": h_plus,
        "h_cross": h_cross,
        "frequency": freq,
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# TaylorT4 — closed-form time-domain waveform
# ---------------------------------------------------------------------------

def taylor_t4(
    Mc: float,
    eta: float,
    f_low: float,
    f_high: float,
    dt: float,
    pn_order: float = 3.5,
    distance: float = 1.0 * MPC,
) -> dict[str, NDArray[np.floating]]:
    """TaylorT4 closed-form time-domain waveform.

    The frequency evolution is computed analytically using the
    leading-order expression for f(t), with PN corrections applied
    as a time-dependent factor.

    Parameters
    ----------
    Mc : float
        Chirp mass [kg].
    eta : float
        Symmetric mass ratio.
    f_low : float
        Starting GW frequency [Hz].
    f_high : float
        Ending GW frequency [Hz].
    dt : float
        Time step [s].
    pn_order : float
        PN order.
    distance : float
        Luminosity distance [m].

    Returns
    -------
    dict
        Keys: 't', 'h_plus', 'h_cross', 'frequency', 'phase'.
    """
    Mc_dim = G * Mc / C ** 3

    # Time to coalescence from f_low (leading order)
    tau0 = (5.0 / 256.0) * (Mc_dim * np.pi * f_low) ** (-5.0 / 3.0) / (np.pi * f_low)

    # More precise: t_chirp = 5/(256 * pi^{8/3} * Mc_dim^{5/3} * f_low^{8/3})
    t_chirp = 5.0 / (256.0 * np.pi ** (8.0 / 3.0) * Mc_dim ** (5.0 / 3.0) * f_low ** (8.0 / 3.0))

    # Build time array
    t = np.arange(-t_chirp, 0, dt)
    if len(t) == 0:
        t = np.array([-t_chirp])

    # Leading-order f(t) = f_low * (1 - t/tau)^{-3/8}
    # where tau = 5/(256) * (pi * Mc_dim * f_low)^{-5/3} / f_low
    tau = t_chirp

    # tau - t_to_merge > 0 always since t < 0
    time_to_merge = -t  # positive

    # Apply PN correction factor to the time-to-coalescence
    x_low = (np.pi * Mc_dim * f_low) ** (2.0 / 3.0)
    pn_time_correction = 1.0
    if pn_order >= 0.5:
        pn_time_correction += (743.0 / 336.0 + 11.0 * eta / 4.0) * x_low / 4.0

    tau_eff = tau * pn_time_correction

    bracket = 1.0 - time_to_merge / tau_eff
    bracket = np.maximum(bracket, 1e-30)

    freq = f_low * bracket ** (-3.0 / 8.0)

    # Clip to f_high
    mask = freq <= f_high
    if not np.all(mask):
        idx = np.searchsorted(mask, False)
        freq = freq[:idx]
        t = t[:idx]

    # Phase: integral of 2 pi f dt
    phase = np.cumsum(2.0 * np.pi * freq * dt)

    # Amplitude
    amp = _amp_scale(Mc, distance) * (np.pi * freq) ** (2.0 / 3.0)

    h_plus = amp * np.cos(phase)
    h_cross = amp * np.sin(phase)

    return {
        "t": t,
        "h_plus": h_plus,
        "h_cross": h_cross,
        "frequency": freq,
        "phase": phase,
    }
