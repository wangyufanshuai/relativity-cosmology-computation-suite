"""
Utility functions for BAO analysis.

Provides cosmological parameter management, unit conversions,
Hankel transform, and helper routines.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# ---------------------------------------------------------------------------
# Fiducial cosmology (Planck 2018 TT,TE,EE+lowE+lensing)
# ---------------------------------------------------------------------------
DEFAULT_COSMO = {
    "H0": 67.4,           # km/s/Mpc
    "Omega_m": 0.315,
    "Omega_b": 0.049,
    "h": 0.674,
    "n_s": 0.965,
    "sigma8": 0.811,
    "A_s": 2.1e-9,        # scalar amplitude (approximate)
    "k_pivot": 0.05,      # Mpc^{-1}
    "T_CMB": 2.7255,      # K
    "N_eff": 3.046,       # effective number of relativistic species
}


def get_cosmo(override=None):
    """Return cosmology dict, optionally overriding defaults.

    Parameters
    ----------
    override : dict, optional
        Keys to override in DEFAULT_COSMO.

    Returns
    -------
    dict
        Cosmological parameters.
    """
    cosmo = dict(DEFAULT_COSMO)
    if override is not None:
        cosmo.update(override)
    return cosmo


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def hubble_distance(cosmo=None):
    """Hubble distance c/H0 in Mpc."""
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    # c in km/s
    c = 2.99792458e5
    return c / cosmo["H0"]


def omega_gamma(cosmo=None):
    """Photon density parameter Omega_gamma at z=0."""
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    # rho_gamma = pi^2/15 * T_CMB^4 / (h^2 c^2) in SI
    # Omega_gamma = rho_gamma / rho_crit
    # rho_crit = 3 H0^2 / (8 pi G)
    # Using the standard formula:
    # Omega_gamma = 2.469e-5 * (T_CMB/2.7255)^4 / h^2
    h = cosmo["h"]
    T = cosmo["T_CMB"]
    return 2.469e-5 * (T / 2.7255) ** 4 / h ** 2


def sound_horizon(cosmo=None):
    """Sound horizon r_s at the drag epoch in Mpc (fitting formula).

    Uses the approximation from Eisenstein & Hu (1998) eq. (6).
    Returns r_d in Mpc (not Mpc/h).
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    h = cosmo["h"]
    Ob = cosmo["Omega_b"]
    Om = cosmo["Omega_m"]
    Obh2 = Ob * h ** 2
    Omh2 = Om * h ** 2
    T_CMB = cosmo["T_CMB"]

    # Eisenstein & Hu (1998) fitting formula for r_d
    b1 = 0.313 * Omh2 ** (-0.419) * (1 + 0.607 * Omh2 ** 0.674)
    b2 = 0.238 * Omh2 ** 0.223
    z_d = 1291 * Omh2 ** 0.251 / (1 + 0.659 * Omh2 ** 0.828) * (
        1 + b1 * Obh2 ** b2
    )

    # Sound horizon integral approximation
    # R = 3/4 * rho_b / rho_gamma at drag epoch
    R_d = 31.5 * Obh2 * (T_CMB / 2.7255) ** (-4) * (1000.0 / (1 + z_d))

    # c_s / c = 1 / sqrt(1 + R)
    # r_d = integral of c_s / H from z_d to infinity
    # Using the approximation:
    r_d = (
        2.0
        / (3.0 * (1 + R_d) ** 0.5)
        * (1.0 / np.sqrt(6.0 / R_d + 1.0))
        * np.log(
            (np.sqrt(1 + R_d) + np.sqrt(R_d + 6.0 * Omh2 * (1 + z_d) / (Obh2 * 1000.0)))
            / (1.0 + np.sqrt(1 + 6.0 * Omh2 * (1 + z_d) / (Obh2 * 1000.0)))
        )
        * hubble_distance(cosmo)
    )

    # Simpler and more robust: use standard approximation
    # r_d ~ 55.28 * (Obh2/0.022)^(-0.4) * (Omh2/0.14)^(-0.15)  (in Mpc)
    # Actually let's use the full EH98 formula properly
    # We'll use the simpler approximation for reliability
    r_d_simple = (
        44.5
        * np.log(9.83 / Omh2)
        / np.sqrt(1 + 10 * Obh2 ** 0.75)
    )  # in Mpc

    return r_d_simple


def sound_horizon_h(cosmo=None):
    """Sound horizon in Mpc/h units."""
    if cosmo is None:
        cosmo = DEFAULT_COSMO
    return sound_horizon(cosmo) * cosmo["h"]


# ---------------------------------------------------------------------------
# Hankel transform via log-spaced FFT
# ---------------------------------------------------------------------------

def hankel_transform(k, pk, r, ell=0):
    """Compute Hankel transform: xi_ell(r) from P_ell(k).

    Uses the spherical Bessel transform:
        xi_ell(r) = (-1)^(ell/2) / (2 pi^2) int k^2 dk j_ell(kr) P(k)

    For ell=0 (monopole):
        xi_0(r) = 1/(2 pi^2) int k^2 dk sin(kr)/(kr) P(k)

    Parameters
    ----------
    k : ndarray
        Wavenumber array in h/Mpc.
    pk : ndarray
        Power spectrum in (Mpc/h)^3.
    r : ndarray
        Separation array in Mpc/h.
    ell : int
        Multipole order (0, 2, or 4).

    Returns
    -------
    ndarray
        Correlation function xi_ell(r).
    """
    from scipy.special import spherical_jn

    xi = np.zeros_like(r, dtype=float)
    dk = np.gradient(k)

    for i, ri in enumerate(r):
        kr = k * ri
        if ell == 0:
            # j0(kr) = sin(kr)/(kr), with sin(kr)/kr -> 1 as kr -> 0
            integrand = k ** 2 * pk * np.sinc(kr / np.pi)  # np.sinc(x) = sin(pi*x)/(pi*x)
            # So np.sinc(kr/pi) = sin(kr)/(kr)
        else:
            integrand = k ** 2 * pk * spherical_jn(ell, kr)
        xi[i] = np.trapezoid(integrand, k)

    xi /= 2.0 * np.pi ** 2

    if ell in (2, 4):
        xi *= (-1) ** (ell // 2)

    return xi


# ---------------------------------------------------------------------------
# Smoothing & interpolation helpers
# ---------------------------------------------------------------------------

def smooth_spline(x, y, s=None):
    """Smooth interpolation using cubic spline.

    Parameters
    ----------
    x, y : ndarray
        Data points.
    s : float, optional
        Smoothing factor.

    Returns
    -------
    callable
        Spline function.
    """
    return InterpolatedUnivariateSpline(x, y, s=s)


def logspace_k(nk=500, k_min=1e-4, k_max=10.0):
    """Return log-spaced wavenumber array in h/Mpc."""
    return np.logspace(np.log10(k_min), np.log10(k_max), nk)


def linspace_s(ns=200, s_min=1.0, s_max=300.0):
    """Return linearly-spaced separation array in Mpc/h."""
    return np.linspace(s_min, s_max, ns)
