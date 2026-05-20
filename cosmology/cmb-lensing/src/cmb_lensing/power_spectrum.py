"""CMB lensing power spectrum C_L^{phi phi} theory predictions."""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Cosmological helpers
# ---------------------------------------------------------------------------

def _hubble_distance(h: float) -> float:
    """Hubble distance c / H0 in Mpc."""
    return 2997.9 / h  # Mpc


def _comoving_distance(z: float, Omega_m: float = 0.3, Omega_b: float = 0.05,
                       h: float = 0.674, n_points: int = 2000) -> float:
    """Comoving distance chi(z) via numerical trapezoidal integration.

    Flat LCDM:
        chi(z) = c/H0 * integral_0^z dz' / E(z')
        E(z) = sqrt(Omega_m (1+z)^3 + Omega_Lambda)
    """
    Omega_lambda = 1.0 - Omega_m
    z_arr = np.linspace(0, z, n_points)
    Ez = np.sqrt(Omega_m * (1 + z_arr) ** 3 + Omega_lambda)
    dz = z_arr[1] - z_arr[0] if n_points > 1 else z
    chi = _hubble_distance(h) * np.trapezoid(1.0 / Ez, dx=dz)
    return chi


def _comoving_distance_vec(z_arr, Omega_m, h, n_points=2000):
    """Vectorised comoving distance for an array of redshifts."""
    return np.array([_comoving_distance(z, Omega_m, h=h, n_points=n_points)
                     for z in z_arr])


def _eisenstein_hu_transfer(k: np.ndarray, Omega_m: float = 0.3,
                            Omega_b: float = 0.05, h: float = 0.674) -> np.ndarray:
    """Simplified Eisenstein & Hu (1998) zero-baryon transfer function T(k).

    Parameters
    ----------
    k : np.ndarray
        Wavenumber in h/Mpc.

    Returns
    -------
    np.ndarray
        Transfer function T(k), normalised to 1 at k -> 0.
    """
    k = np.asarray(k, dtype=np.float64)
    Omega_m_h2 = Omega_m * h**2
    Theta_CMB = 2.7255 / 2.7
    k_eq = 7.46e-2 * Omega_m_h2 * Theta_CMB**(-2)  # h/Mpc

    q = k / (13.41 * k_eq)
    T0 = np.log(1.0 + 2.34 * q) / (2.34 * q) * (
        1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    ) ** (-0.25)
    return T0


def _growth_factor(z: float, Omega_m: float = 0.3) -> float:
    """Linear growth factor D(z) normalised to D(0)=1 (Carroll+ 1992)."""
    Omega_lambda = 1.0 - Omega_m
    a = 1.0 / (1.0 + z)
    E2 = Omega_m / a**3 + Omega_lambda
    Omega_m_z = (Omega_m / a**3) / E2
    Omega_lambda_z = Omega_lambda / E2

    def _D(Om, Ol):
        return (5.0 / 2.0) * Om / (
            Om ** (4.0 / 7.0) - Ol + (1.0 + Om / 2.0) * (1.0 + Ol / 70.0)
        )

    return _D(Omega_m_z, Omega_lambda_z) / _D(Omega_m, Omega_lambda)


def _precompute_matter_pk(Omega_m: float = 0.3, Omega_b: float = 0.05,
                          h: float = 0.674, sigma8: float = 0.81,
                          n_s: float = 0.965, n_k: int = 4000):
    """Pre-compute a log-spaced P(k) table at z=0 (for interpolation)."""
    k_tab = np.logspace(-4, 1, n_k)
    T_tab = _eisenstein_hu_transfer(k_tab, Omega_m, Omega_b, h)
    P_tab = k_tab**n_s * T_tab**2  # un-normalised shape

    # sigma8 normalisation
    R = 8.0
    x = k_tab * R
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x**3
    s8_sq = np.trapezoid(k_tab**2 * P_tab * W**2, x=k_tab) / (2.0 * np.pi**2)
    A_norm = sigma8**2 / s8_sq if s8_sq > 0 else 1.0
    P_tab *= A_norm
    return k_tab, P_tab


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lensing_potential_power_spectrum(
    L: np.ndarray,
    z_source: float = 1100.0,
    Omega_m: float = 0.3,
    Omega_b: float = 0.05,
    h: float = 0.674,
    sigma8: float = 0.81,
) -> np.ndarray:
    """Theoretical lensing-potential power spectrum C_L^{phi phi}.

    Uses the Limber approximation with the Eisenstein-Hu transfer function
    and Carroll+ growth factors.  The integrand is evaluated on a fixed grid
    in chi and then interpolated onto the required k values, making the
    computation fully vectorised.

    Parameters
    ----------
    L : np.ndarray
        Multipole array.
    z_source : float
        Source redshift (CMB ~ 1100).
    Omega_m, Omega_b, h, sigma8 : float
        Cosmological parameters.

    Returns
    -------
    np.ndarray
        C_L^{phi phi} in units of rad^4.
    """
    L = np.asarray(L, dtype=np.float64)
    L_safe = np.maximum(L, 2.0)

    chi_source = _comoving_distance(z_source, Omega_m, Omega_b, h)

    # Pre-compute P(k) table at z=0 for fast interpolation
    k_tab, Pk_tab = _precompute_matter_pk(Omega_m, Omega_b, h, sigma8)
    log_k_tab = np.log(k_tab)
    log_Pk_tab = np.log(np.maximum(Pk_tab, 1e-300))

    # Integration grid in comoving distance
    n_chi = 200
    chi_arr = np.linspace(0.01 * chi_source, 0.999 * chi_source, n_chi)
    dchi = chi_arr[1] - chi_arr[0]

    # Window function  W(chi)
    W_arr = lensing_window_function(chi_arr, chi_source)  # (n_chi,)

    # Growth factor D(z(chi)) at each chi — approximate z from chi
    # For flat LCDM:  chi(z) is monotonic, invert via Newton or precompute
    # We use a simple numerical inversion
    z_grid = np.linspace(0.01, 5.0, 500)
    chi_grid = _comoving_distance_vec(z_grid, Omega_m, h=h, n_points=500)
    # extend to high z
    z_high = np.linspace(5.0, z_source * 0.999, 200)
    if len(z_high) > 0:
        chi_high = _comoving_distance_vec(z_high, Omega_m, h=h, n_points=300)
        z_grid = np.concatenate([z_grid, z_high])
        chi_grid = np.concatenate([chi_grid, chi_high])

    z_of_chi = np.interp(chi_arr, chi_grid, z_grid)
    D_arr = np.array([_growth_factor(z, Omega_m) for z in z_of_chi])

    # For each chi, compute k = L / chi  for all L simultaneously
    # Shape: (n_chi, n_L)
    k_h = L_safe[np.newaxis, :] / chi_arr[:, np.newaxis] * h  # h/Mpc

    # Interpolate log P(k) at z=0 and scale by D(z)^2
    log_kh_flat = np.log(np.maximum(k_h.ravel(), 1e-10))
    log_Pk_flat = np.interp(log_kh_flat, log_k_tab, log_Pk_tab)
    Pk_z0 = np.exp(log_Pk_flat).reshape(k_h.shape)
    Pk = Pk_z0 * (D_arr[:, np.newaxis] ** 2) / h**3  # Mpc^3

    # Integrand:  W^2 * P(k) / chi^2
    integrand = (W_arr[:, np.newaxis] ** 2) * Pk / (chi_arr[:, np.newaxis] ** 2)

    # Trapezoidal integration over chi
    C_L = 16.0 * np.pi**2 / L_safe**4 * np.trapezoid(integrand, dx=dchi, axis=0)

    return C_L


def lensed_cmb_power_spectrum(
    ell: np.ndarray,
    C_ell_unlensed: np.ndarray,
    C_ell_phi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute the lensed CMB power spectrum from the unlensed spectrum.

    The dominant effect is smoothing of the acoustic peaks:

        C_ell^{lensed}(l) ≈ Gaussian convolution of C_unlensed with width
        sigma_ell ~ 1 / alpha_rms.

    Parameters
    ----------
    ell : np.ndarray
        Multipole array.
    C_ell_unlensed : np.ndarray
        Unlensed TT power spectrum (same length as *ell*).
    C_ell_phi : np.ndarray or None
        Lensing potential power spectrum.  Computed from theory if ``None``.

    Returns
    -------
    np.ndarray
        Lensed power spectrum.
    """
    ell = np.asarray(ell, dtype=np.float64)
    C_u = np.asarray(C_ell_unlensed, dtype=np.float64)

    if C_ell_phi is None:
        C_ell_phi = lensing_potential_power_spectrum(ell)

    # RMS deflection angle  <alpha^2> = (1/2pi) int dL L^3 C_L^{phiphi}
    dL = np.gradient(ell)
    dL = np.where(dL > 0, dL, 1.0)
    alpha_sq = np.sum(ell**3 * C_ell_phi * dL) / (2.0 * np.pi)

    if alpha_sq > 0:
        sigma_ell = 1.0 / np.sqrt(alpha_sq)
    else:
        sigma_ell = 1e6

    # Gaussian convolution (vectorised)
    # C_lensed(l) = sum_l' C_u(l') exp(-(l-l')^2/(2 s^2)) / Z
    delta = ell[:, np.newaxis] - ell[np.newaxis, :]  # (n_ell, n_ell)
    kernel = np.exp(-delta**2 / (2.0 * sigma_ell**2))
    Z = kernel.sum(axis=1)
    Z = np.where(Z > 0, Z, 1.0)
    C_lensed = (kernel @ C_u) / Z

    return C_lensed


def lensing_window_function(chi: np.ndarray, chi_source: float) -> np.ndarray:
    """CMB lensing window function.

    W(chi) = 2 * (1 - chi / chi_source)   for chi < chi_source, else 0.

    Parameters
    ----------
    chi : np.ndarray
        Comoving distance(s) [Mpc].
    chi_source : float
        Comoving distance to the source plane [Mpc].

    Returns
    -------
    np.ndarray
        Window function values.
    """
    chi = np.asarray(chi, dtype=np.float64)
    return np.where(chi < chi_source, 2.0 * (1.0 - chi / chi_source), 0.0)


def rms_deflection(
    C_ell_phi: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    l_min: int = 2,
    l_max: int = 5000,
) -> float:
    """RMS CMB lensing deflection angle in arcminutes.

    <alpha^2> = integral d^2 L / (4 pi) L^2 C_L^{phi phi}
              = (1 / 2 pi) integral dL L^3 C_L^{phi phi}

    Typical value ~ 2.7 arcmin for Planck cosmology.

    Parameters
    ----------
    C_ell_phi : np.ndarray or None
        Lensing potential power spectrum.  Computed from theory if ``None``.
    L : np.ndarray or None
        Multipoles corresponding to *C_ell_phi*.
    l_min, l_max : int
        Multipole bounds (used when computing C_L from theory).

    Returns
    -------
    float
        RMS deflection in arcminutes.
    """
    if C_ell_phi is None or L is None:
        L = np.arange(l_min, l_max + 1, dtype=np.float64)
        C_ell_phi = lensing_potential_power_spectrum(L)
    else:
        L = np.asarray(L, dtype=np.float64)
        C_ell_phi = np.asarray(C_ell_phi, dtype=np.float64)

    dL = np.gradient(L)
    dL = np.where(dL > 0, dL, 1.0)
    alpha_sq_rad = np.sum(L**3 * C_ell_phi * dL) / (2.0 * np.pi)

    rad_to_arcmin = 180.0 * 60.0 / np.pi
    return float(np.sqrt(alpha_sq_rad) * rad_to_arcmin)
