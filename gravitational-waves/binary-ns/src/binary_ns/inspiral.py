"""
Inspiral waveform model for binary neutron star coalescence.

Produces a frequency-domain waveform h(f) with point-particle (TaylorT4-like)
inspiral and leading-order tidal corrections at 5PN order.

References
----------
Flanagan & Hinderer (2008) PRL 100 1108;
Damour, Nagar & Villain (2012) PRD 85 123007.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G = 6.67430e-11          # m^3 kg^-1 s^-2
C = 2.99792458e8         # m s^-1
M_SUN = 1.98892e30       # kg
PC = 3.085677581e16      # m
MPC = 1e6 * PC           # Mpc in m

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _mc(m1: float, m2: float) -> float:
    """Chirp mass [kg] from component masses [kg]."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

def _mc_msun(m1_msun: float, m2_msun: float) -> float:
    """Chirp mass [Msun] from component masses [Msun]."""
    return _mc(m1_msun, m2_msun)  # same formula, different units

def _mtot(m1: float, m2: float) -> float:
    return m1 + m2

def _eta(m1: float, m2: float) -> float:
    """Symmetric mass ratio."""
    M = m1 + m2
    return m1 * m2 / M**2

# ---------------------------------------------------------------------------
# Frequency evolution
# ---------------------------------------------------------------------------

def dfdt_point_particle(f: np.ndarray, Mc: float) -> np.ndarray:
    """Point-particle frequency evolution (leading order, Newtonian).

    df/dt = (96/5) pi^{8/3} (G Mc / c^3)^{5/3} f^{11/3}

    Parameters
    ----------
    f : array   Frequency [Hz].
    Mc : float   Chirp mass [kg].

    Returns
    -------
    dfdt : array   [Hz s^-1]
    """
    Mc_geo = G * Mc / C**3  # chirp mass in seconds
    return (96.0 / 5.0) * np.pi ** (8.0 / 3.0) * Mc_geo ** (5.0 / 3.0) * f ** (11.0 / 3.0)


def dfdt_with_tidal(f: np.ndarray, Mc: float, Mtot: float,
                    lam_tilde: float) -> np.ndarray:
    """Frequency evolution including leading-order tidal correction.

    The tidal correction enters as an additive term at 5PN order:

    df/dt = df/dt|_PP * [ 1 + (12/13) * (2 * pi * M * f)^{10/3}
                          * 39 * lam_tilde / 2 ]

    where M = Mtot * G / c^3  (total mass in seconds).

    Parameters
    ----------
    f : array      Frequency [Hz].
    Mc : float      Chirp mass [kg].
    Mtot : float    Total mass [kg].
    lam_tilde : float  Combined dimensionless tidal deformability.

    Returns
    -------
    dfdt : array   [Hz s^-1]
    """
    dfdt_pp = dfdt_point_particle(f, Mc)
    Mtot_s = G * Mtot / C**3  # total mass in seconds
    v = (np.pi * Mtot_s * f) ** (5.0 / 3.0)
    # Tidal factor speeds up inspiral (positive correction to df/dt)
    tidal_factor = 1.0 + (12.0 / 13.0) * 39.0 * lam_tilde * v**2
    return dfdt_pp * tidal_factor

# ---------------------------------------------------------------------------
# Phasing
# ---------------------------------------------------------------------------

def psi_point_particle(f: np.ndarray, Mc: float, eta: float,
                       f_ref: float = 20.0) -> np.ndarray:
    """Point-particle phase in the stationary-phase approximation (0PN).

    Psi(f) = 2 pi f t_c - phi_c
             + (3/128)(pi Mc f)^{-5/3} / eta

    Here we set t_c = phi_c = 0 (coalescence time and phase) and measure
    phase relative to f_ref so the result is well-behaved.

    Parameters
    ----------
    f : array      Frequency grid [Hz].
    Mc : float      Chirp mass [kg].
    eta : float     Symmetric mass ratio.
    f_ref : float   Reference frequency [Hz].

    Returns
    -------
    Psi : array     Phase [radians].
    """
    Mc_s = G * Mc / C**3
    x = np.pi * Mc_s * f
    x_ref = np.pi * Mc_s * f_ref
    psi = (3.0 / 128.0) / eta * (x**(-5.0 / 3.0) - x_ref**(-5.0 / 3.0))
    return psi


def psi_tidal_correction(f: np.ndarray, Mtot: float,
                         lam_tilde: float, eta: float) -> np.ndarray:
    """Leading-order tidal phase correction (5PN).

    Psi_tidal = -(39/2) * lam_tilde * (pi M f)^{5/3} / eta

    where M = G Mtot / c^3.

    Parameters
    ----------
    f : array       [Hz].
    Mtot : float     Total mass [kg].
    lam_tilde : float  Combined dimensionless tidal deformability.
    eta : float      Symmetric mass ratio.

    Returns
    -------
    dPsi : array    Tidal phase contribution [radians].
    """
    M_s = G * Mtot / C**3
    v = (np.pi * M_s * f) ** (5.0 / 3.0)
    return -(39.0 / 2.0) * lam_tilde * v / eta


def psi_total(f: np.ndarray, m1_kg: float, m2_kg: float,
              lam_tilde: float, f_ref: float = 20.0) -> np.ndarray:
    """Total phase = point-particle + tidal.

    Parameters
    ----------
    f : array       [Hz]
    m1_kg, m2_kg : float  Component masses [kg].
    lam_tilde : float      Combined tidal deformability.
    f_ref : float          Reference frequency [Hz].

    Returns
    -------
    Psi : array [radians]
    """
    Mc = _mc(m1_kg, m2_kg)
    Mtot = _mtot(m1_kg, m2_kg)
    eta = _eta(m1_kg, m2_kg)
    psi_pp = psi_point_particle(f, Mc, eta, f_ref)
    psi_tid = psi_tidal_correction(f, Mtot, lam_tilde, eta)
    return psi_pp + psi_tid

# ---------------------------------------------------------------------------
# Full waveform
# ---------------------------------------------------------------------------

@dataclass
class WaveformResult:
    """Frequency-domain inspiral waveform."""

    f: np.ndarray           # frequency grid [Hz]
    h_plus: np.ndarray      # h_+(f)  (complex)
    h_cross: np.ndarray     # h_x(f)  (complex)
    amplitude: np.ndarray   # |h(f)|  real
    phase: np.ndarray       # Psi(f)  real
    time: np.ndarray        # approximate time-to-coalescence [s]


def inspiral_waveform(m1_msun: float, m2_msun: float,
                      lam_tilde: float,
                      f_min: float = 20.0,
                      f_max: float = 2000.0,
                      df: float = 0.125,
                      f_ref: float = 20.0,
                      distance_mpc: float = 40.0) -> WaveformResult:
    """Generate the frequency-domain inspiral waveform with tidal corrections.

    Parameters
    ----------
    m1_msun, m2_msun : float
        Component masses [Msun].
    lam_tilde : float
        Combined dimensionless tidal deformability.
    f_min, f_max : float
        Frequency band [Hz].
    df : float
        Frequency resolution [Hz].
    f_ref : float
        Reference frequency [Hz].
    distance_mpc : float
        Luminosity distance [Mpc].

    Returns
    -------
    WaveformResult
    """
    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    Mc = _mc(m1, m2)
    Mtot = _mtot(m1, m2)
    eta_val = _eta(m1, m2)

    f = np.arange(f_min, f_max + df, df)
    f = f[f > 0]

    # Phase
    phase = psi_total(f, m1, m2, lam_tilde, f_ref)

    # Amplitude  (leading-order SPA)
    # A(f) = (1/dL) * sqrt(5/24) * pi^{-2/3} * (G Mc / c^3)^{5/6} * f^{-7/6}
    d_l = distance_mpc * MPC
    Mc_s = G * Mc / C**3
    amp = (1.0 / d_l) * np.sqrt(5.0 / 24.0) * np.pi ** (-2.0 / 3.0) \
          * Mc_s ** (5.0 / 6.0) * f ** (-7.0 / 6.0)

    h = amp * np.exp(-1j * phase)
    h_plus = 0.5 * (1 + np.cos(0.0)**2) * h  # iota=0 approx
    h_cross = 1j * np.cos(0.0) * h

    # Approximate time to coalescence  t(f) = integral 1/(df/dt) df
    dfdt = dfdt_with_tidal(f, Mc, Mtot, lam_tilde)
    dt_df = 1.0 / dfdt
    t = np.zeros_like(f)
    t[1:] = cumulative_trapezoid(dt_df, f)
    t = t[-1] - t  # time *to* coalescence

    return WaveformResult(
        f=f,
        h_plus=h_plus,
        h_cross=h_cross,
        amplitude=amp,
        phase=phase,
        time=t,
    )

# ---------------------------------------------------------------------------
# SNR estimate
# ---------------------------------------------------------------------------

def _aLIGO_psd(f: np.ndarray) -> np.ndarray:
    """Approximate Advanced LIGO noise PSD (zero-detuned, high-power).

    Simplified analytic fit from literature:
    S_n(f) = S_0 * [ (f0/f)^4 + 2(1 + (f/f0)^2) / (1 + (f/f0)^2) ]

    We use a simpler standard approximation.
    """
    f0 = 215.0  # Hz
    S0 = 1e-49   # Hz^-1
    x = f / f0
    # Simplified: diverges at low f, flat at high f
    psd = S0 * (x**(-4.14) - 5.0 * x**(-2) + 111.0 * (1.0 - x**2 + 0.5 * x**4) / (1.0 + 0.5 * x**2))
    return np.maximum(psd, 1e-52)


def estimate_snr(wf: WaveformResult, psd_func=None) -> float:
    """Compute SNR for a waveform given a detector PSD.

    SNR^2 = 4 * int |h(f)|^2 / S_n(f) df

    Parameters
    ----------
    wf : WaveformResult
    psd_func : callable(f) -> S_n(f), optional
        Defaults to approximate aLIGO PSD.

    Returns
    -------
    snr : float
    """
    if psd_func is None:
        psd_func = _aLIGO_psd

    f = wf.f
    h = wf.amplitude
    S = psd_func(f)
    df = f[1] - f[0] if len(f) > 1 else 1.0
    integrand = h**2 / S
    snr_sq = 4.0 * np.trapezoid(integrand, f)
    return float(np.sqrt(max(snr_sq, 0.0)))
