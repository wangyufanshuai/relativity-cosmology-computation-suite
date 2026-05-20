"""Afterglow synchrotron radiation model.

External-shock model for a relativistic blast wave decelerating in a
circumburst medium (ISM or stellar wind).  Implements:

* Blandford-McKee self-similar deceleration dynamics
* Synchrotron break frequencies (nu_m, nu_c, F_nu,max)
* Flux density in each spectral regime
* Closure relations connecting temporal and spectral indices

All quantities in CGS unless noted.  Electron index *p* > 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Physical constants (CGS)
c_cgs: float = 2.99792458e10       # cm/s
m_e: float = 9.10938356e-28        # g
m_p: float = 1.6726219e-24         # g
q_e: float = 4.80320425e-10        # esu
sigma_T: float = 6.6524587e-25     # cm^2

# -----------------------------------------------------------------------
# Blast-wave dynamics
# -----------------------------------------------------------------------


def blast_radius(t: NDArray, E: float, n: float,
                 regime: str = "adiabatic") -> NDArray:
    """Sedov-Taylor / Blandford-McKee blast-wave radius.

    Adiabatic:   R ~ (E t^2 / n)^{1/5}    ->  R ~ t^{2/5}
    Radiative:   R ~ (E t / n)^{2/7}       ->  R ~ t^{4/9} (approx)
    """
    t = np.asarray(t, dtype=np.float64)
    if regime == "adiabatic":
        # R = (17 E t^2 / (4 pi n m_p))^{1/5}
        return np.power(17.0 * E * t**2 / (4.0 * np.pi * n * m_p + 1e-30),
                        0.2)
    else:
        # radiative (approximate)
        return np.power(E * t / (n * m_p + 1e-30), 2.0 / 7.0)


def lorentz_factor(t: NDArray, E: float, n: float,
                   medium: str = "ISM") -> NDArray:
    """Bulk Lorentz factor evolution.

    ISM:   Gamma(t) ~ (E / (n m_p c^5))^{1/8} t^{-3/8}
    Wind:  Gamma(t) ~ t^{-1/4}
    """
    t = np.asarray(t, dtype=np.float64)
    t = np.maximum(t, 1e-30)
    if medium == "ISM":
        # Gamma ~ (17 E / (4 pi n m_p c^5))^{1/8} t^{-3/8}
        prefactor = np.power(
            17.0 * E / (4.0 * np.pi * n * m_p * c_cgs**5 + 1e-30),
            0.125,
        )
        return prefactor * np.power(t, -0.375)
    else:
        # Wind: Gamma ~ t^{-1/4}
        return np.power(t, -0.25)


# -----------------------------------------------------------------------
# Synchrotron break frequencies & peak flux
# -----------------------------------------------------------------------


def break_frequencies(t: NDArray, E: float, n: float,
                      epsilon_B: float, epsilon_e: float,
                      p: float = 2.3,
                      medium: str = "ISM") -> tuple[NDArray, NDArray, NDArray]:
    """Synchrotron break frequencies and peak flux.

    Returns (nu_m, nu_c, F_nu_max) in units of (Hz, Hz, mJy).

    Parameters
    ----------
    t : observer time (s).
    E : isotropic-equivalent blast energy (erg).
    n : circumburst number density (cm^{-3}).
    epsilon_B : fraction of shock energy in B-field.
    epsilon_e : fraction of shock energy in electrons.
    p : electron energy index (N(gamma) ~ gamma^{-p}).
    medium : ``"ISM"`` or ``"wind"``.
    """
    t = np.asarray(t, dtype=np.float64)
    t = np.maximum(t, 1e-30)

    Gamma = lorentz_factor(t, E, n, medium)

    # Post-shock magnetic field  B = sqrt(32 pi epsilon_B n m_p c^2) * Gamma
    B = np.sqrt(32.0 * np.pi * epsilon_B * n * m_p * c_cgs**2) * Gamma

    # Minimum electron Lorentz factor
    gamma_m = epsilon_e * (p - 2.0) / (p - 1.0) * (m_p / m_e) * (Gamma - 1.0)
    gamma_m = np.maximum(gamma_m, 1.0)

    # Cooling Lorentz factor
    gamma_c = 6.0 * np.pi * m_e * c_cgs / (sigma_T * B**2 * t + 1e-30)
    gamma_c = np.maximum(gamma_c, 1.0)

    # nu_m = (3/4pi) gamma_m^2 (q_e B) / (m_e c)
    nu_m = 0.75 / np.pi * gamma_m**2 * q_e * B / (m_e * c_cgs)

    # nu_c = (3/4pi) gamma_c^2 (q_e B) / (m_e c)
    nu_c = 0.75 / np.pi * gamma_c**2 * q_e * B / (m_e * c_cgs)

    # Peak flux  F_nu,max ~ (sqrt(3) e^3 B Gamma N_e) / (4 pi m_e c^2 d_L^2)
    # Use d_L = 1e28 cm (~ 3 Gpc) as default luminosity distance
    d_L = 1e28
    R = blast_radius(t, E, n, "adiabatic")
    N_e = (4.0 / 3.0) * np.pi * R**3 * n
    F_max = (np.sqrt(3.0) * q_e**3 * B * Gamma * N_e
             / (4.0 * np.pi * m_e * c_cgs**2 * d_L**2))
    # Convert erg/s/cm^2/Hz -> mJy:  1 erg/s/cm^2/Hz = 1e26 Jy = 1e29 mJy
    F_max_mJy = F_max * 1e29

    return nu_m, nu_c, F_max_mJy


# -----------------------------------------------------------------------
# Synchrotron spectrum
# -----------------------------------------------------------------------


def synchrotron_flux(nu: NDArray, t: NDArray, E: float, n: float,
                     epsilon_B: float, epsilon_e: float,
                     p: float = 2.3,
                     medium: str = "ISM") -> NDArray:
    """Synchrotron flux density F_nu(t) at frequency nu.

    Handles both slow-cooling (nu_m < nu_c) and fast-cooling (nu_c < nu_m)
    regimes with the standard broken power-law spectrum.

    Returns flux in mJy.
    """
    nu = np.asarray(nu, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    nu_m, nu_c, F_max = break_frequencies(t, E, n, epsilon_B, epsilon_e,
                                           p, medium)

    # Determine slow vs fast cooling
    slow = nu_m < nu_c

    F = np.zeros_like(nu)

    # --- Slow cooling (nu_m < nu_c) ---
    mask_s = slow
    nu_m_s = nu_m[mask_s] if np.ndim(nu_m) == 0 else nu_m
    nu_c_s = nu_c[mask_s] if np.ndim(nu_c) == 0 else nu_c
    F_max_s = F_max[mask_s] if np.ndim(F_max) == 0 else F_max

    # nu < nu_m: F ~ F_max (nu/nu_m)^{1/3}
    m1 = mask_s & (nu < nu_m_s)
    F[m1] = F_max_s * np.power(nu[m1] / nu_m_s, 1.0 / 3.0)

    # nu_m < nu < nu_c: F ~ F_max (nu/nu_m)^{-(p-1)/2}
    m2 = mask_s & (nu >= nu_m_s) & (nu < nu_c_s)
    F[m2] = F_max_s * np.power(nu[m2] / nu_m_s, -(p - 1.0) / 2.0)

    # nu > nu_c: F ~ F_max (nu_c/nu_m)^{-(p-1)/2} (nu/nu_c)^{-p/2}
    m3 = mask_s & (nu >= nu_c_s)
    F[m3] = (F_max_s * np.power(nu_c_s / nu_m_s, -(p - 1.0) / 2.0)
             * np.power(nu[m3] / nu_c_s, -p / 2.0))

    # --- Fast cooling (nu_c < nu_m) ---
    mask_f = ~slow
    nu_m_f = nu_m[mask_f] if np.ndim(nu_m) == 0 else nu_m
    nu_c_f = nu_c[mask_f] if np.ndim(nu_c) == 0 else nu_c
    F_max_f = F_max[mask_f] if np.ndim(F_max) == 0 else F_max

    # nu < nu_c: F ~ F_max (nu/nu_c)^{1/3}
    m4 = mask_f & (nu < nu_c_f)
    F[m4] = F_max_f * np.power(nu[m4] / nu_c_f, 1.0 / 3.0)

    # nu_c < nu < nu_m: F ~ F_max (nu/nu_c)^{-1/2}
    m5 = mask_f & (nu >= nu_c_f) & (nu < nu_m_f)
    F[m5] = F_max_f * np.power(nu[m5] / nu_c_f, -0.5)

    # nu > nu_m: F ~ F_max (nu_m/nu_c)^{-1/2} (nu/nu_m)^{-p/2}
    m6 = mask_f & (nu >= nu_m_f)
    F[m6] = (F_max_f * np.power(nu_m_f / nu_c_f, -0.5)
             * np.power(nu[m6] / nu_m_f, -p / 2.0))

    return np.maximum(F, 0.0)


# -----------------------------------------------------------------------
# Closure relations
# -----------------------------------------------------------------------


def closure_relation(alpha: float, beta: float,
                     regime: str = "ISM_slow",
                     p: float = 2.3) -> float:
    """Return the expected alpha given beta (or vice-versa) from closure.

    Convention: F_nu ~ t^{-alpha} nu^{-beta}.

    Returns the expected alpha for the given beta and regime.

    Regimes:
        ISM_slow_<case>, ISM_fast_<case>, wind_slow_<case>, wind_fast_<case>
        case: below_nu_m, between, above_nu_c
    """
    # Standard closure relations (ISM, slow cooling)
    closures = {
        # ISM, slow cooling
        "ISM_slow_below_nu_m": (beta, 0.5 * beta + 0.0),          # beta=1/3 -> alpha=1/2*beta
        "ISM_slow_between": (beta, 1.5 * beta + 0.5),              # beta=(p-1)/2 -> alpha=3(p-1)/4
        "ISM_slow_above_nu_c": (beta, 1.5 * beta),                 # beta=p/2 -> alpha=3p/4
        # ISM, fast cooling
        "ISM_fast_below_nu_c": (beta, 0.5 * beta),
        "ISM_fast_between": (beta, 1.5 * beta - 0.5),              # beta=1/2 -> alpha=1/4
        "ISM_fast_above_nu_m": (beta, 1.5 * beta),
        # Wind, slow cooling
        "wind_slow_below_nu_m": (beta, 0.5 * beta - 0.5),
        "wind_slow_between": (beta, 1.5 * beta + 0.0),
        "wind_slow_above_nu_c": (beta, 1.5 * beta - 0.5),
    }
    if regime in closures:
        _, alpha_expected = closures[regime]
        return alpha_expected
    return alpha
