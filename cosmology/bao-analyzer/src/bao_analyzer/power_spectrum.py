"""
Power spectrum computation with BAO wiggles.

Implements the Eisenstein & Hu (1998) no-wiggle transfer function,
linear matter power spectrum, and wiggle/no-wiggle decomposition.

References
----------
Eisenstein, D. J., & Hu, W. (1998). "Baryonic Features in the Matter
Transfer Function". ApJ, 496, 605. arXiv:astro-ph/9709112.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from .utils import get_cosmo, logspace_k, omega_gamma, DEFAULT_COSMO


# ---------------------------------------------------------------------------
# Eisenstein-Hu (1998) no-wiggle transfer function
# ---------------------------------------------------------------------------

def eisenstein_hu_transfer_nw(k, cosmo=None):
    """Eisenstein-Hu (1998) no-wiggle (smooth) transfer function T_nw(k).

    This is the CDM+baryon transfer function without BAO oscillations,
    from equations (26)-(29) of EH98.

    Parameters
    ----------
    k : ndarray
        Wavenumber in h/Mpc (comoving).
    cosmo : dict, optional
        Cosmological parameters. Uses defaults if None.

    Returns
    -------
    ndarray
        Transfer function T_nw(k), normalized so T(0) -> 1.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    h = cosmo["h"]
    Om = cosmo["Omega_m"]
    Ob = cosmo["Omega_b"]
    T_CMB = cosmo["T_CMB"]

    Omh2 = Om * h ** 2
    Obh2 = Ob * h ** 2
    f_b = Ob / Om  # baryon fraction
    f_c = 1.0 - f_b  # CDM fraction

    # Photon density parameter
    Og = omega_gamma(cosmo)
    # Equilibrium redshift: matter-radiation equality
    # zeq = Om / Og - 1
    zeq = 2.5e4 * Omh2 * (T_CMB / 2.7255) ** (-4)
    # keq in h/Mpc
    keq = 0.0746 * Omh2 * (T_CMB / 2.7255) ** (-2)  # Mpc^{-1} * h

    # Sound horizon (approximate, EH98 eq 6)
    s = 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2 ** 0.75)  # Mpc

    # Alpha_gamma: effective scale for small-k suppression
    alpha_gamma = (
        1.0
        - 0.328 * np.log(431 * Omh2) * f_b
        + 0.38 * np.log(22.3 * Omh2) * f_b ** 2
    )

    # Gamma_eff (shape parameter)
    Gamma_eff = Om * h * (alpha_gamma + (1 - alpha_gamma) /
                          (1 + (0.43 * k * s) ** 4))

    # q = k / (Gamma * h) * (T_CMB/2.7)^2
    # More precisely: q = k * Theta^{-2} / Gamma_eff, where Theta = T_CMB/2.7
    Theta = T_CMB / 2.7
    q = k * Theta ** 2 / (Gamma_eff * h)

    # No-wiggle transfer function (EH98 eq. 29)
    L = np.log(2 * np.e + 1.8 * q)
    C = 14.1 + 731.0 / (1 + 62.5 * q)
    T_nw = L / (L + C * q ** 2)

    return T_nw


# ---------------------------------------------------------------------------
# Eisenstein-Hu (1998) full transfer function (with wiggles)
# ---------------------------------------------------------------------------

def _sound_horizon_eh(cosmo):
    """Sound horizon from EH98 eq (6), in Mpc (not Mpc/h)."""
    h = cosmo["h"]
    Omh2 = cosmo["Omega_m"] * h ** 2
    Obh2 = cosmo["Omega_b"] * h ** 2
    return 44.5 * np.log(9.83 / Omh2) / np.sqrt(1 + 10 * Obh2 ** 0.75)


def eisenstein_hu_transfer(k, cosmo=None):
    """Eisenstein-Hu (1998) full transfer function with BAO wiggles.

    Uses the zero-baryon transfer function T_0 and the baryon correction
    from EH98 eqs (15)-(25).

    Parameters
    ----------
    k : ndarray
        Wavenumber in h/Mpc.
    cosmo : dict, optional
        Cosmological parameters.

    Returns
    -------
    ndarray
        Transfer function T(k).
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    h = cosmo["h"]
    Om = cosmo["Omega_m"]
    Ob = cosmo["Omega_b"]
    T_CMB = cosmo["T_CMB"]

    Omh2 = Om * h ** 2
    Obh2 = Ob * h ** 2
    f_b = Ob / Om
    f_c = 1.0 - f_b

    Theta = T_CMB / 2.7

    # Equality wavenumber (Mpc^{-1})
    keq = 0.0746 * Omh2 * Theta ** (-2)
    # Sound horizon
    s = _sound_horizon_eh(cosmo)

    # Silk damping scale
    k_Silk = 1.6 * Obh2 ** 0.52 * Omh2 ** 0.73 * (1 + (10.4 * Omh2) ** (-0.95))

    # Redshift of drag epoch (EH98 eq 4)
    b1 = 0.313 * Omh2 ** (-0.419) * (1 + 0.607 * Omh2 ** 0.674)
    b2 = 0.238 * Omh2 ** 0.223
    z_d = 1291 * Omh2 ** 0.251 / (1 + 0.659 * Omh2 ** 0.828) * (
        1 + b1 * Obh2 ** b2
    )

    # R at drag epoch
    R_d = 31.5 * Obh2 * Theta ** (-4) * (1000.0 / z_d)

    # Silk damping exponent
    # Using the simplified form from EH98
    k_s = k * h  # k in Mpc^{-1}

    # q = k / keq
    q = k_s / keq

    # Zero-baryon transfer function T_0 (EH98 eq 15)
    T_0 = _transfer_zero_baryon(q)

    # Baryon CDM transfer function (EH98 eq 17-18)
    # Beta term
    beta_node = 8.41 * Omh2 ** 0.435
    beta_silk = 0.44 * (1 + 0.209 * Omh2 ** 0.694) * (1 + 0.258 * f_b)

    # Node term: phase shift for baryons
    s_eff = s / ((1 + (beta_node / (k_s * s)) ** 3) ** (1.0 / 3))

    # Oscillatory part
    j_osc = np.sin(k_s * s_eff) / (k_s * s_eff + 1e-30)

    # Silk damping envelope
    # Approximate the damping envelope
    G = np.exp(-(k_s / k_Silk) ** 1.4)

    # Full transfer function
    # T(k) = f_c * T_0(q) + f_b * envelope * oscillation
    T_baryon = G * j_osc

    # Approximate the full transfer function
    # A simple effective approach: interpolate between smooth and wiggly
    T = T_0 * (1.0 - f_b * (1.0 - T_baryon / (T_0 + 1e-30)))

    # Ensure T is well-behaved
    T = np.where(np.isfinite(T), T, T_0)

    return T


def _transfer_zero_baryon(q):
    """Zero-baryon transfer function T_0(q) from EH98 eq (15).

    Parameters
    ----------
    q : ndarray
        Dimensionless wavenumber k/keq.

    Returns
    -------
    ndarray
        T_0(q).
    """
    C = 14.2 + 731.0 / (1 + 62.5 * q)
    L = np.log(np.e + 1.8 * q)

    T0 = L / (L + C * q ** 2)

    return T0


# ---------------------------------------------------------------------------
# Power spectra
# ---------------------------------------------------------------------------

def linear_power_spectrum(k, cosmo=None, normalize_sigma8=True):
    """Linear matter power spectrum P(k) at z=0.

    P(k) = A_s * (k / k_pivot)^(n_s - 1) * T(k)^2 * (2 pi^2 / k^3)

    Parameters
    ----------
    k : ndarray
        Wavenumber in h/Mpc.
    cosmo : dict, optional
        Cosmological parameters.
    normalize_sigma8 : bool
        If True, normalize to sigma8.

    Returns
    -------
    ndarray
        P(k) in (Mpc/h)^3.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]
    k_pivot = cosmo["k_pivot"]

    k_safe = np.where(k > 0, k, 1e-30)
    T = eisenstein_hu_transfer(k, cosmo)

    # Primordial spectrum times transfer function squared
    pk = A_s * (k_safe / k_pivot) ** (n_s - 1) * T ** 2 * (2 * np.pi ** 2 / k_safe ** 3)

    if normalize_sigma8:
        pk = _normalize_to_sigma8(k, pk, cosmo)

    return pk


def no_wiggle_power_spectrum(k, cosmo=None, normalize_sigma8=True):
    """No-wiggle (smooth) power spectrum P_nw(k) at z=0.

    Same primordial shape but with the smooth transfer function T_nw(k).

    Parameters
    ----------
    k : ndarray
        Wavenumber in h/Mpc.
    cosmo : dict, optional
        Cosmological parameters.
    normalize_sigma8 : bool
        If True, normalize to sigma8.

    Returns
    -------
    ndarray
        P_nw(k) in (Mpc/h)^3.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]
    k_pivot = cosmo["k_pivot"]

    k_safe = np.where(k > 0, k, 1e-30)
    T_nw = eisenstein_hu_transfer_nw(k, cosmo)

    pk_nw = A_s * (k_safe / k_pivot) ** (n_s - 1) * T_nw ** 2 * (2 * np.pi ** 2 / k_safe ** 3)

    if normalize_sigma8:
        pk_nw = _normalize_to_sigma8(k, pk_nw, cosmo)

    return pk_nw


def wiggle_power_spectrum(k, cosmo=None, normalize_sigma8=True):
    """Wiggle-only (BAO) power spectrum P_wig(k) = P(k) - P_nw(k).

    Parameters
    ----------
    k : ndarray
        Wavenumber in h/Mpc.
    cosmo : dict, optional
        Cosmological parameters.
    normalize_sigma8 : bool
        If True, normalize to sigma8.

    Returns
    -------
    ndarray
        P_wig(k) in (Mpc/h)^3.
    """
    pk = linear_power_spectrum(k, cosmo, normalize_sigma8)
    pk_nw = no_wiggle_power_spectrum(k, cosmo, normalize_sigma8)
    return pk - pk_nw


# ---------------------------------------------------------------------------
# sigma8 normalization
# ---------------------------------------------------------------------------

def _sigma8_norm_factor(k, pk, cosmo):
    """Compute the factor needed to normalize P(k) to sigma8."""
    sigma8_target = cosmo["sigma8"]

    # Compute sigma(R=8 Mpc/h)
    R = 8.0  # Mpc/h
    kr = k * R
    # Window function W(kR) = 3 * (sin(kR) - kR*cos(kR)) / (kR)^3
    W = 3.0 * (np.sin(kr) - kr * np.cos(kr)) / (kr ** 3 + 1e-30)
    # Handle kr -> 0: W -> 1
    W = np.where(kr < 1e-6, 1.0, W)

    integrand = k ** 2 * pk * W ** 2
    sigma8_sq = np.trapezoid(integrand, k) / (2 * np.pi ** 2)

    sigma8_current = np.sqrt(np.abs(sigma8_sq))
    if sigma8_current == 0:
        return 1.0

    return (sigma8_target / sigma8_current) ** 2


def _normalize_to_sigma8(k, pk, cosmo):
    """Normalize power spectrum amplitude to match sigma8."""
    factor = _sigma8_norm_factor(k, pk, cosmo)
    return pk * factor
