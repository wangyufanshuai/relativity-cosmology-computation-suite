"""Energy density of primordial gravitational waves Omega_GW(f).

Computes the stochastic gravitational-wave background energy density
spectrum as a function of frequency, including:

- Inflationary tensor power spectrum (scale-invariant or power-law)
- Transfer function for sub-horizon evolution through radiation/matter eras
- First-order phase transition contributions (bubble collisions, sound waves,
  MHD turbulence)

References
----------
- Smith, Caldwell, No, arXiv:1905.09557
- Turner, "Detectability of inflation-produced gravitational waves"
  PRD 55 (1997) R435
- Caprini et al., "Science with the space-based interferometer eLISA"
  JCAP 04 (2016) 001, arXiv:1512.06239
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# Planck mass in GeV
_M_PL = 1.2209e19  # GeV

# Reduced Planck mass in GeV
_M_PL_RED = _M_PL / np.sqrt(8.0 * np.pi)

# Speed of light in m/s
_C = 2.998e8

# Boltzmann constant in GeV/K
_K_B = 8.617e-14  # GeV/K

# Current CMB temperature in K
_T_CMB = 2.7255

# Effective relativistic degrees of freedom today (photons + 3 massless neutrinos)
_G_STAR_0 = 3.36

# g_* at neutrino decoupling (eV scale): ~3.36
# g_* at QCD transition (~200 MeV): ~60
# g_* at electroweak scale (~100 GeV): ~106.75
_G_STAR_SM = 106.75


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------

def transfer_function(x: np.ndarray) -> np.ndarray:
    """GW transfer function T(x) where x = f / f_c (horizon crossing ratio).

    T(x) = 1 for x >> 1 (sub-horizon, free propagation)
    T(x) ~ 3 j_1(x) / x for x << 1 (super-horizon, suppressed)

    Analytic approximation from Turner (1997):
        T(x) = [1 + (0.4 x)^2]^{1/2} * [1 + (2 x / 3)^3]^{-1/6}
        * (3 j_1(x) / x)   for x < 1
        T(x) = 1             for x >= 1

    A simpler and numerically stable form combining both regimes:
        T(x)^2 = Omega_mat(x) * [3 j_1(x) / x]^2 + Omega_rad(x)

    We use a widely-adopted smooth interpolation.

    Parameters
    ----------
    x : np.ndarray
        Dimensionless frequency ratio f / f_c, where f_c is the frequency
        corresponding to horizon crossing. x >> 1 means well inside the
        horizon at the time of production.

    Returns
    -------
    np.ndarray
        Transfer function values, same shape as x.
    """
    x = np.asarray(x, dtype=float)

    # Spherical Bessel function j_1(x) = (sin x / x^2) - (cos x / x)
    # 3 j_1(x) / x = 3 (sin x / x - cos x) / x^2
    # Avoid division by zero
    result = np.ones_like(x)

    # For small x: 3 j_1(x)/x ~ 1 - x^2/10 + ...
    # We use the full expression where safe
    mask = np.abs(x) > 1e-10
    x_safe = np.where(mask, x, 1.0)  # placeholder for x=0

    j1_times_3_over_x = np.where(
        mask,
        3.0 * (np.sin(x_safe) / x_safe - np.cos(x_safe)) / x_safe**2,
        1.0,  # limit as x->0 is 1
    )

    # Smooth interpolation following Turner (1997) / Smith et al.
    # T(x) = sqrt(1 + c1 x^2) * (c2 / (1 + c3 x^(3/2))) * 3 j_1(x)/x
    # Optimised to match numerical results:
    c1 = 0.4
    c3 = 2.0 / 3.0

    T = (
        np.sqrt(1.0 + (c1 * x_safe) ** 2)
        * (1.0 + (c3 * x_safe) ** 3) ** (-1.0 / 6.0)
        * j1_times_3_over_x
    )

    # For x >= 1 the transfer function saturates to ~1
    result = np.where(x >= 1.0, 1.0, T)
    result = np.where(np.abs(x) < 1e-10, 1.0, result)

    return result


# ---------------------------------------------------------------------------
# Main Omega_GW(f) for inflationary background
# ---------------------------------------------------------------------------

def omega_gw(
    f: np.ndarray,
    r: float = 0.01,
    n_t: float = 0.0,
    A_s: float = 2.1e-9,
    k_pivot: float = 0.05,
    Omega_r: float = 9.1e-5,
    H0: float = 67.4,
) -> np.ndarray:
    """Energy density of primordial GW background Omega_GW(f).

    Omega_GW(f) = (1/12) r A_s Omega_r (f/f_c)^{n_t} T^2(f/f_c)

    where f_c is the characteristic frequency at which modes re-enter
    the horizon during the radiation era.

    Parameters
    ----------
    f : np.ndarray
        Frequency array in Hz.
    r : float
        Tensor-to-scalar ratio. Planck 2018 bound: r < 0.036 (95% CL).
    n_t : float
        Tensor spectral index. Consistency relation: n_t = -r/8.
    A_s : float
        Scalar power spectrum amplitude at pivot. Planck: ~2.1e-9.
    k_pivot : float
        Pivot scale in Mpc^{-1}. Planck standard: 0.05.
    Omega_r : float
        Radiation density parameter today (photons + neutrinos).
        Includes 3 massless neutrino species: ~9.1e-5 (for H0=67.4).
    H0 : float
        Hubble constant in km/s/Mpc.

    Returns
    -------
    np.ndarray
        Omega_GW(f) array, same shape as f.

    Notes
    -----
    The present-day energy density fraction is:
        Omega_GW(f) = (pi^2 / 12) * P_t(f) * (g_*/g_*0) * (T_*/T_0)^4 * T^2(f/f_c)

    Simplifying with the radiation-dominated transfer function and
    assuming standard thermal history (g_* ~ 106.75 at production,
    T_* ~ 10^15 GeV), the result reduces to the formula above with
    f_c ~ k_pivot * c / (2 pi) ~ 7.7e-17 Hz.

    For the scale-invariant case (n_t = 0):
        Omega_GW ~ 10^{-15} * r  (nearly flat spectrum)
    """
    f = np.asarray(f, dtype=float)

    # Tensor amplitude
    A_t = r * A_s

    # Characteristic frequency: mode that enters the horizon at matter-radiation
    # equality. f_c ~ k_pivot * c / (2 pi) in Hz
    # k_pivot = 0.05 Mpc^{-1} => f_pivot = k_pivot * c / (2 pi) ~ 3.7e-18 Hz
    # More precisely, using H0 to convert:
    #   f_c = k_pivot * c / (2 pi a_0) = k_pivot * c * H0 / (2 pi)
    # with H0 in s^{-1}: H0_SI = H0 * 1000 / (3.0857e22) ~ 2.19e-18 s^{-1}
    H0_si = H0 * 1e3 / 3.0857e22  # Convert km/s/Mpc to s^{-1}
    # Mpc in metres: 3.0857e22 m
    Mpc_m = 3.0857e22
    f_pivot = k_pivot * _C / (2.0 * np.pi * Mpc_m)  # Hz

    # Ratio x = f / f_c
    # For modes re-entering during radiation domination, f_c is roughly
    # the pivot frequency. The transfer function handles the suppression.
    x = f / f_pivot

    # Omega_GW = (1/12) P_t(k) Omega_r T^2(x)
    P_t = A_t * (f / f_pivot) ** n_t  # Power-law power spectrum in f-space
    Ogw = (1.0 / 12.0) * P_t * Omega_r * transfer_function(x) ** 2

    return Ogw


# ---------------------------------------------------------------------------
# Observable frequency range
# ---------------------------------------------------------------------------

def frequency_range(H0: float = 67.4) -> Tuple[float, float]:
    """Observable frequency range of primordial gravitational waves.

    Returns
    -------
    f_min : float
        Minimum observable frequency ~10^{-18} Hz, set by the present
        Hubble horizon. Modes with longer wavelengths have not re-entered.
    f_max : float
        Maximum frequency ~10^{9} Hz, set by the reheating temperature.
        Modes with f > f_max were never in causal contact during inflation
        (they correspond to wavelengths smaller than the Hubble radius at
        the end of inflation).

    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc.

    Notes
    -----
    f_min = H0 / (2 pi) ~ 3.5e-19 Hz
    f_max depends on the reheating temperature T_reh:
        f_max ~ (g_* / 106.75)^{1/6} * (T_reh / 10^15 GeV) * 10^9 Hz
    For T_reh ~ 10^15 GeV (GUT scale): f_max ~ 10^9 Hz.
    """
    # Hubble frequency today
    H0_si = H0 * 1e3 / 3.0857e22  # s^{-1}
    f_min = H0_si / (2.0 * np.pi)  # ~3.5e-19 Hz

    # Upper bound from reheating: f_max = (a_end / a_0) * H_end / (2 pi)
    # For GUT-scale inflation T_reh ~ 10^15 GeV:
    # f_max ~ 10^9 Hz (very rough; exact value depends on inflation model)
    T_reh = 1e15  # GeV, GUT-scale reheating temperature
    g_star_reh = 106.75  # SM value at high T

    # H_end ~ 8 pi^2 M_Pl^2 V^{1/2} / (3 M_Pl^2) ...
    # Simplified: f_max ~ (T_reh / T_CMB) * (g_*0 / g_*reh)^{1/3} * H0/(2pi)
    # A more standard estimate:
    # f_max = 1.1e9 * (T_reh / 10^{15} GeV) * (g_star_reh / 106.75)^{1/6} Hz
    f_max = 1.1e9 * (T_reh / 1e15) * (g_star_reh / 106.75) ** (1.0 / 6.0)

    return f_min, f_max


# ---------------------------------------------------------------------------
# Omega_GW from a general power spectrum
# ---------------------------------------------------------------------------

def omega_gw_from_power_spectrum(
    k: np.ndarray,
    P_t: np.ndarray,
    Omega_r: float = 9.1e-5,
    k_pivot: float = 0.05,
    H0: float = 67.4,
) -> np.ndarray:
    """Compute Omega_GW from a general tensor power spectrum (not power-law).

    Omega_GW(k) = (1/12) P_t(k) Omega_r T^2(k / k_pivot)

    Parameters
    ----------
    k : np.ndarray
        Wavenumber array in Mpc^{-1}.
    P_t : np.ndarray
        Tensor power spectrum P_t(k), same shape as k. This can be the
        output of tensor_mukhanov_sasaki or any other calculation.
    Omega_r : float
        Radiation density parameter today.
    k_pivot : float
        Pivot scale in Mpc^{-1} used to define the transfer function
        reference frequency.
    H0 : float
        Hubble constant in km/s/Mpc.

    Returns
    -------
    np.ndarray
        Omega_GW(k) array, same shape as k.
    """
    k = np.asarray(k, dtype=float)
    P_t = np.asarray(P_t, dtype=float)

    Mpc_m = 3.0857e22
    f_pivot = k_pivot * _C / (2.0 * np.pi * Mpc_m)

    # Convert k to frequency: f = k * c / (2 pi)
    f = k * _C / (2.0 * np.pi * Mpc_m)
    x = f / f_pivot

    Ogw = (1.0 / 12.0) * P_t * Omega_r * transfer_function(x) ** 2
    return Ogw


# ---------------------------------------------------------------------------
# First-order phase transition contribution
# ---------------------------------------------------------------------------

def first_order_phase_transition(
    f: np.ndarray,
    alpha: float = 0.1,
    beta_over_H: float = 100.0,
    T_star: float = 100.0,
    v_w: float = 0.95,
) -> np.ndarray:
    """Omega_GW from a first-order phase transition (e.g., electroweak).

    Combined contribution from three mechanisms:
    1. Bubble wall collisions (envelopes approximation)
    2. Sound waves in the plasma
    3. Magnetohydrodynamic (MHD) turbulence

    The sound-wave contribution typically dominates.

    Parameters
    ----------
    f : np.ndarray
        Frequency array in Hz.
    alpha : float
        Phase transition strength parameter: ratio of latent heat to
        radiation energy density. alpha ~ 0.1 is a "moderately strong"
        transition; alpha >> 1 is very strong.
    beta_over_H : float
        Inverse phase transition duration in Hubble units. beta/H ~ 100
        means the transition completes in ~1% of a Hubble time.
        Smaller values mean slower (stronger) transitions.
    T_star : float
        Transition temperature in GeV. For the electroweak phase transition
        T_star ~ 100 GeV. For the QCD transition T_star ~ 0.15 GeV.
    v_w : float
        Bubble wall velocity in units of c. v_w = 1 is the ultra-relativistic
        limit; v_w ~ 0.95 is typical for a detonation.

    Returns
    -------
    np.ndarray
        Omega_GW(f) from the phase transition, same shape as f.

    Notes
    -----
    Peak frequency (redshifted to today):
        f_peak ~ 2.6e-3 * (beta_over_H / 100) * (T_star / 100 GeV)
                 * (g_* / 100)^{1/6} Hz

    For the electroweak transition (T ~ 100 GeV): f_peak ~ mHz,
    coinciding with the LISA sensitivity band.
    """
    f = np.asarray(f, dtype=float)

    # Effective relativistic degrees of freedom at T_star
    # Approximate interpolation of SM g_*(T)
    if T_star > 100.0:  # Above EW scale
        g_star = 106.75
    elif T_star > 1.0:  # Between EW and QCD scales
        g_star = 80.0
    elif T_star > 0.15:  # Around QCD scale
        g_star = 30.0
    else:  # Below QCD scale
        g_star = 10.0

    # Peak frequency today (redshifted from production)
    # f_peak = (1/2 pi) * (beta/a_0) = (beta/H) * (a_star/a_0) * H / (2 pi)
    # a_star/a_0 accounts for redshift
    # f_peak ~ 2.6e-3 * (beta/H / 100) * (T_star / 100 GeV) * (g_*/100)^{1/6} Hz
    f_peak = (
        2.6e-3
        * (beta_over_H / 100.0)
        * (T_star / 100.0)
        * (g_star / 100.0) ** (1.0 / 6.0)
    )

    # Hubble rate at transition (radiation domination)
    # H_star^2 = (8 pi^3 / 90) g_* T^4 / M_Pl^2
    H_star_si = (
        np.sqrt(8.0 * np.pi**3 / 90.0)
        * np.sqrt(g_star)
        * T_star**2
        / _M_PL**2
    )

    # Ratio f/f_peak for spectral shape
    x = f / f_peak

    # --- Bubble collisions (envelopes approximation) ---
    # E_n = 1 for envelope, K = kinetic energy fraction
    # K = alpha / (1 + alpha) * v_w^3 / (0.24 + v_w^3) * (v_w^3 * 0.715)
    kappa_b = (
        alpha / (1.0 + alpha)
        * v_w**6 * 0.715
        / (0.24 + v_w**3)
    )

    # Amplitude: Omega_GW,bubble ~ 1.67e-5 * K^2 * (H/beta)^2
    Omega_bubble_amp = (
        1.67e-5
        * kappa_b**2
        * (1.0 / beta_over_H) ** 2
        * (100.0 / g_star) ** (1.0 / 3.0)
    )

    # Spectral shape: broken power law
    # S_b(x) = (x / p)^3 / (1 - x/p)^3 * exp(-c x)
    # Simplified: S_b ~ (x)^3 / (1 + x)^4 * exp(-x)
    S_bubble = x**3 / (1.0 + x) ** 4 * np.exp(-x)

    # --- Sound waves (typically dominant) ---
    # Amplitude: Omega_GW,sw ~ 2.65e-6 * K * (H/beta) * (c_s / v_w)
    # c_s = 1/sqrt(3) is the sound speed in the plasma
    c_s = 1.0 / np.sqrt(3.0)
    kappa_sw = (
        alpha / (1.0 + alpha)
        * c_s**2 * v_w**3 * 0.515
        / (0.24 + c_s**2 * v_w**3)
    )

    Omega_sw_amp = (
        2.65e-6
        * kappa_sw
        * (1.0 / beta_over_H)
        * (100.0 / g_star) ** (1.0 / 3.0)
    )

    # Sound wave spectral shape: peaked
    # S_sw ~ (x)^3 / (1 + x)^3 * (1 + (x/c1)^n)^{-1/n}
    c1_sw = 3.0
    n_sw = 4.0
    S_sw = x**3 / (1.0 + x) ** 3 * (1.0 + (x / c1_sw) ** n_sw) ** (-1.0 / n_sw)

    # --- MHD turbulence ---
    # Typically subdominant, amplitude ~ 10% of sound waves
    Omega_turb_amp = 0.1 * Omega_sw_amp

    # Turbulence spectral shape: broader peak
    S_turb = (
        x**3
        / (1.0 + x) ** (11.0 / 3.0)
        * (1.0 + (x / 2.0) ** 2) ** (-1.0)
    )

    # Total: sum of three contributions
    Omega_gw_total = (
        Omega_bubble_amp * S_bubble
        + Omega_sw_amp * S_sw
        + Omega_turb_amp * S_turb
    )

    # Avoid spurious values at f = 0
    Omega_gw_total = np.where(f > 0, Omega_gw_total, 0.0)

    return Omega_gw_total
