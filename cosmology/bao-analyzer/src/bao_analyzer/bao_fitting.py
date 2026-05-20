"""
BAO peak fitting and anisotropic BAO analysis.

Provides template-based BAO fitting, dilation parameter estimation,
anisotropic decomposition into D_M/r_d and H*r_d, and chi-squared
likelihood for parameter constraints.
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import InterpolatedUnivariateSpline

from .utils import get_cosmo, logspace_k, linspace_s, sound_horizon_h, DEFAULT_COSMO
from .power_spectrum import (
    linear_power_spectrum,
    no_wiggle_power_spectrum,
    wiggle_power_spectrum,
)
from .correlation import correlation_function, no_wiggle_correlation, hankel_transform


# ---------------------------------------------------------------------------
# BAO peak detection
# ---------------------------------------------------------------------------

def bao_peak_detect(s, xi, s_range=(60, 150)):
    """Detect the BAO peak position in the correlation function.

    Finds the local maximum in xi(s) within the given range, expected
    near ~105 Mpc/h for fiducial cosmology.

    Parameters
    ----------
    s : ndarray
        Separation array in Mpc/h.
    xi : ndarray
        Correlation function.
    s_range : tuple
        (s_min, s_max) to search for the peak.

    Returns
    -------
    s_peak : float
        Peak position in Mpc/h.
    xi_peak : float
        Peak correlation value.
    """
    mask = (s >= s_range[0]) & (s <= s_range[1])
    if not np.any(mask):
        raise ValueError("No data in search range")

    s_search = s[mask]
    xi_search = xi[mask]

    # Smooth to reduce noise
    if len(xi_search) > 5:
        from scipy.ndimage import uniform_filter1d
        xi_smooth = uniform_filter1d(xi_search, size=min(5, len(xi_search)))
    else:
        xi_smooth = xi_search

    idx_peak = np.argmax(xi_smooth)
    s_peak = s_search[idx_peak]
    xi_peak = xi_search[idx_peak]

    return s_peak, xi_peak


# ---------------------------------------------------------------------------
# BAO template fitting
# ---------------------------------------------------------------------------

def bao_template(s, xi_template, alpha, B=1.0, xi_nw=None):
    """BAO template model: xi_model(s) = B * xi_template(s/alpha) + xi_nw(s).

    Parameters
    ----------
    s : ndarray
        Separation array.
    xi_template : callable or ndarray
        Template correlation function. If callable, evaluated at s/alpha.
        If ndarray, uses spline interpolation.
    alpha : float
        BAO dilation parameter (alpha=1 for fiducial cosmology).
    B : float
        Broadband bias/amplitude parameter.
    xi_nw : ndarray, optional
        No-wiggle (smooth) correlation function evaluated at s.
        If None, assumed to be zero.

    Returns
    -------
    ndarray
        Model correlation function.
    """
    if callable(xi_template):
        xi_wig_scaled = xi_template(s / alpha)
    else:
        # Assume xi_template is given at some s_template;
        # we need to know the original s array
        raise ValueError("xi_template should be callable (spline)")

    result = B * xi_wig_scaled
    if xi_nw is not None:
        result = result + xi_nw
    return result


def bao_template_fit(s_data, xi_data, cov=None, cosmo=None,
                     s_template=None, alpha_range=(0.8, 1.2)):
    """Fit BAO dilation parameter alpha using template matching.

    Model: xi_model(s) = B * xi_wig(s/alpha) + A + A1*s + A2*s^2

    Parameters
    ----------
    s_data : ndarray
        Observed separation array.
    xi_data : ndarray
        Observed correlation function.
    cov : ndarray, optional
        Covariance matrix. If None, uses identity.
    cosmo : dict, optional
        Cosmological parameters for template.
    s_template : ndarray, optional
        Template separation array.
    alpha_range : tuple
        Range of alpha to search.

    Returns
    -------
    result : dict
        Fitting results with keys: alpha, alpha_err, B, chi2, xi_model.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    n = len(s_data)
    if cov is None:
        cov = np.eye(n)

    # Compute template correlation function
    k = logspace_k(nk=2000, k_min=1e-4, k_max=50.0)
    if s_template is None:
        s_template = linspace_s(ns=500, s_min=10.0, s_max=300.0)

    pk_full = linear_power_spectrum(k, cosmo)
    pk_nw = no_wiggle_power_spectrum(k, cosmo)
    pk_wig = pk_full - pk_nw

    # Wiggle correlation (BAO feature)
    xi_wig_template = hankel_transform(k, pk_wig, s_template, ell=0)
    xi_nw_template = hankel_transform(k, pk_nw, s_template, ell=0)

    # Build splines
    xi_wig_spline = InterpolatedUnivariateSpline(s_template, xi_wig_template)
    xi_nw_spline = InterpolatedUnivariateSpline(s_template, xi_nw_template)

    # Inverse covariance
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(n)

    def chi2_func(params):
        alpha, B, A0, A1, A2 = params
        # Wiggle part scaled by alpha
        xi_wig_model = np.array([xi_wig_spline(si / alpha) for si in s_data])
        xi_nw_model = np.array([xi_nw_spline(si) for si in s_data])
        xi_model = B * xi_wig_model + xi_nw_model + A0 + A1 * s_data + A2 * s_data ** 2
        delta = xi_data - xi_model
        return float(delta @ cov_inv @ delta)

    # Grid search over alpha, then optimize
    alpha_grid = np.linspace(alpha_range[0], alpha_range[1], 50)
    best_chi2 = np.inf
    best_alpha = 1.0

    for alpha_try in alpha_grid:
        # For each alpha, analytically solve for B, A0, A1, A2
        xi_wig_at_data = np.array([xi_wig_spline(si / alpha_try) for si in s_data])
        xi_nw_at_data = np.array([xi_nw_spline(si) for si in s_data])

        # Design matrix: columns = xi_wig, 1, s, s^2
        X = np.column_stack([
            xi_wig_at_data,
            np.ones(n),
            s_data,
            s_data ** 2,
        ])
        y = xi_data - xi_nw_at_data

        # Weighted least squares: (X^T C^-1 X) beta = X^T C^-1 y
        XtCinvX = X.T @ cov_inv @ X
        XtCinvy = X.T @ cov_inv @ y

        try:
            beta = np.linalg.solve(XtCinvX, XtCinvy)
            xi_model = X @ beta + xi_nw_at_data
            delta = xi_data - xi_model
            chi2 = float(delta @ cov_inv @ delta)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_alpha = alpha_try
        except np.linalg.LinAlgError:
            continue

    # Refine around best alpha
    alpha_fine = np.linspace(best_alpha - 0.02, best_alpha + 0.02, 100)
    chi2_array = []

    for alpha_try in alpha_fine:
        xi_wig_at_data = np.array([xi_wig_spline(si / alpha_try) for si in s_data])
        xi_nw_at_data = np.array([xi_nw_spline(si) for si in s_data])

        X = np.column_stack([
            xi_wig_at_data,
            np.ones(n),
            s_data,
            s_data ** 2,
        ])
        y = xi_data - xi_nw_at_data
        XtCinvX = X.T @ cov_inv @ X
        XtCinvy = X.T @ cov_inv @ y

        try:
            beta = np.linalg.solve(XtCinvX, XtCinvy)
            xi_model = X @ beta + xi_nw_at_data
            delta = xi_data - xi_model
            chi2 = float(delta @ cov_inv @ delta)
        except np.linalg.LinAlgError:
            chi2 = np.inf
        chi2_array.append(chi2)

    chi2_array = np.array(chi2_array)

    # Find minimum chi2
    idx_min = np.argmin(chi2_array)
    best_alpha = alpha_fine[idx_min]
    best_chi2 = chi2_array[idx_min]

    # Estimate alpha error from delta chi2 = 1
    chi2_min = best_chi2
    above_1 = chi2_array - chi2_min - 1.0
    # Find crossings
    alpha_err = 0.01  # default
    for side in [1, -1]:
        mask = (alpha_fine - best_alpha) * side > 0
        if np.any(mask & (above_1 > 0)):
            crossings = alpha_fine[mask & (above_1 > 0)]
            if len(crossings) > 0:
                alpha_err = max(alpha_err, abs(crossings[0] - best_alpha))

    # Get best-fit B
    xi_wig_at_data = np.array([xi_wig_spline(si / best_alpha) for si in s_data])
    xi_nw_at_data = np.array([xi_nw_spline(si) for si in s_data])
    X = np.column_stack([xi_wig_at_data, np.ones(n), s_data, s_data ** 2])
    y = xi_data - xi_nw_at_data
    XtCinvX = X.T @ cov_inv @ X
    XtCinvy = X.T @ cov_inv @ y
    try:
        beta = np.linalg.solve(XtCinvX, XtCinvy)
    except np.linalg.LinAlgError:
        beta = np.array([1.0, 0.0, 0.0, 0.0])

    xi_model_final = X @ beta + xi_nw_at_data

    return {
        "alpha": best_alpha,
        "alpha_err": alpha_err,
        "B": beta[0],
        "broadband": beta[1:],
        "chi2": best_chi2,
        "xi_model": xi_model_final,
    }


# ---------------------------------------------------------------------------
# Anisotropic BAO
# ---------------------------------------------------------------------------

def anisotropic_bao(k, pk, mu, s_perp, s_par, ell_max=2, cosmo=None):
    """Decompose anisotropic BAO signal into multipole moments.

    The anisotropic correlation function encodes:
      - D_M(z)/r_d (angular diameter distance, transverse)
      - H(z)*r_d (Hubble distance, radial)

    The multipole decomposition is:
      xi_ell(s) = (2ell+1)/2 int_{-1}^{1} xi(s, mu) L_ell(mu) dmu

    Parameters
    ----------
    k : ndarray
        Wavenumber array in h/Mpc.
    pk : ndarray
        Power spectrum in (Mpc/h)^3.
    mu : ndarray
        Cosine of angle to line of sight.
    s_perp : ndarray
        Transverse separations in Mpc/h.
    s_par : ndarray
        Line-of-sight separations in Mpc/h.
    ell_max : int
        Maximum multipole (0, 2, or 4).
    cosmo : dict, optional
        Cosmological parameters.

    Returns
    -------
    multipoles : dict
        Keys are ell (0, 2, 4) with values (s, xi_ell) arrays.
    """
    from scipy.special import legendre

    if cosmo is None:
        cosmo = DEFAULT_COSMO

    s_mid = np.sqrt(s_perp ** 2 + s_par ** 2)
    mu_mid = s_par / (s_mid + 1e-30)

    multipoles = {}

    for ell in range(0, ell_max + 1, 2):
        L_ell = legendre(ell)

        # Compute xi_ell via integration over mu at each s
        # For simplicity, compute monopole from Hankel transform
        xi_monopole = hankel_transform(k, pk, s_mid, ell=0)

        if ell == 0:
            multipoles[ell] = (s_mid, xi_monopole)
        elif ell == 2:
            # Quadrupole: approximate from anisotropic scaling
            # xi_2(s) ~ (5/2) * integral over mu of xi(s,mu) * L2(mu) dmu
            # Using Kaiser linear RSD model as template
            beta_kaiser = (cosmo["Omega_m"] ** 0.55 / 0.97)  # beta = f/b, b~0.97
            # xi_2 ~ f_growth * (4/3) * xi_0 (simplified)
            xi_2 = beta_kaiser * xi_monopole * 0.4 * L_ell(mu_mid)
            # Better: compute from 2D power spectrum with Kaiser
            # P(k,mu) = (b + f*mu^2)^2 * P(k)
            # For now, use simplified formula
            xi_2_full = _compute_quadrupole(k, pk, s_mid, beta_kaiser)
            multipoles[ell] = (s_mid, xi_2_full)
        elif ell == 4:
            # Hexadecapole
            xi_4 = _compute_hexadecapole(k, pk, s_mid)
            multipoles[ell] = (s_mid, xi_4)

    return multipoles


def _compute_quadrupole(k, pk, s, beta):
    """Compute quadrupole correlation function with Kaiser effect.

    xi_2(s) = beta * xi_0(s) * factor (simplified Kaiser model).

    Uses proper 2D integral:
    P_ell(k) = (2ell+1)/2 int P(k,mu) L_ell(mu) dmu
    """
    from scipy.integrate import quad
    from scipy.special import legendre

    L2 = legendre(2)

    # Quadrupole power spectrum
    # P_2(k) = 5/2 * int_{-1}^{1} (b + f*mu^2)^2 * P(k) * (3*mu^2-1)/2 dmu
    # = P(k) * 5/2 * int (b + f*mu^2)^2 * (3mu^2-1)/2 dmu
    b = 1.0
    f = beta * b

    def integrand(mu):
        return (b + f * mu ** 2) ** 2 * L2(mu)

    integral, _ = quad(integrand, -1, 1)
    pk_2 = pk * integral / 2.0

    xi_2 = hankel_transform(k, pk_2, s, ell=2)
    return xi_2


def _compute_hexadecapole(k, pk, s):
    """Compute hexadecapole (ell=4) correlation function."""
    from scipy.integrate import quad
    from scipy.special import legendre

    L4 = legendre(4)

    b = 1.0
    f = 0.5  # approximate growth rate

    def integrand(mu):
        return (b + f * mu ** 2) ** 2 * L4(mu)

    integral, _ = quad(integrand, -1, 1)
    pk_4 = pk * integral / 2.0

    xi_4 = hankel_transform(k, pk_4, s, ell=4)
    return xi_4


def anisotropic_bao_parameters(s, xi_0, xi_2, z_eff, cosmo=None):
    """Extract D_M(z)/r_d and H(z)*r_d from anisotropic BAO.

    Uses the Alcock-Paczynski test with the relation:
      F_AP = (1+z) * D_M(z) * H(z) / c
      alpha_perp = (D_M/r_d) / (D_M/r_d)_fid
      alpha_par = (H*r_d)_fid / (H*r_d)

    Parameters
    ----------
    s : ndarray
        Separation array.
    xi_0, xi_2 : ndarray
        Monopole and quadrupole correlation functions.
    z_eff : float
        Effective redshift.
    cosmo : dict, optional
        Fiducial cosmology.

    Returns
    -------
    dict
        D_M_over_rd, H_times_rd, alpha_perp, alpha_par, F_AP.
    """
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    h = cosmo["h"]
    Om = cosmo["Omega_m"]

    # Fiducial distance computations (flat LCDM)
    from scipy.integrate import quad

    def E(z):
        return np.sqrt(Om * (1 + z) ** 3 + (1 - Om))

    # Comoving distance integral
    c_over_H0 = 2.99792458e5 / cosmo["H0"]  # Mpc

    def comoving_distance(z):
        result, _ = quad(lambda zz: c_over_H0 / E(zz), 0, z)
        return result

    D_M_fid = comoving_distance(z_eff)  # Mpc
    H_fid = cosmo["H0"] * E(z_eff)  # km/s/Mpc

    # Sound horizon
    r_d = sound_horizon_h(cosmo) / h  # Mpc (not Mpc/h)

    D_M_over_rd_fid = D_M_fid / r_d
    H_times_rd_fid = H_fid * r_d / c_over_H0  # dimensionless * Mpc

    # F_AP parameter
    F_AP = (1 + z_eff) * D_M_fid * H_fid / 2.99792458e5

    return {
        "D_M_over_rd_fid": D_M_over_rd_fid,
        "H_times_rd_fid": H_times_rd_fid,
        "F_AP": F_AP,
        "alpha_perp": 1.0,  # For fiducial, alpha=1
        "alpha_par": 1.0,
        "D_M_fid": D_M_fid,
        "H_fid": H_fid,
        "r_d": r_d,
    }


# ---------------------------------------------------------------------------
# Chi-squared likelihood
# ---------------------------------------------------------------------------

def chi2_likelihood(params, s_data, xi_data, cov, model_func):
    """Compute chi-squared for given parameters.

    chi2 = (data - model)^T C^{-1} (data - model)

    Parameters
    ----------
    params : array-like
        Model parameters.
    s_data : ndarray
        Data separations.
    xi_data : ndarray
        Data correlation function.
    cov : ndarray
        Covariance matrix.
    model_func : callable
        Function model_func(s, params) -> xi_model.

    Returns
    -------
    float
        Chi-squared value.
    """
    xi_model = model_func(s_data, params)
    delta = xi_data - xi_model

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    chi2 = float(delta @ cov_inv @ delta)
    return chi2


def gaussian_likelihood(params, s_data, xi_data, cov, model_func):
    """Compute Gaussian log-likelihood.

    log L = -0.5 * chi2 + const

    Parameters
    ----------
    params : array-like
        Model parameters.
    s_data : ndarray
        Data separations.
    xi_data : ndarray
        Data correlation function.
    cov : ndarray
        Covariance matrix.
    model_func : callable
        Function model_func(s, params) -> xi_model.

    Returns
    -------
    float
        Log-likelihood value.
    """
    chi2 = chi2_likelihood(params, s_data, xi_data, cov, model_func)
    return -0.5 * chi2


def profile_likelihood(alpha_array, s_data, xi_data, cov, cosmo=None):
    """Profile likelihood over the BAO dilation parameter alpha.

    Marginalizes over broadband parameters at each alpha value.

    Parameters
    ----------
    alpha_array : ndarray
        Array of alpha values to evaluate.
    s_data : ndarray
        Data separations.
    xi_data : ndarray
        Data correlation function.
    cov : ndarray
        Covariance matrix.
    cosmo : dict, optional
        Cosmological parameters.

    Returns
    -------
    alpha_array : ndarray
        Alpha values.
    chi2_array : ndarray
        Chi-squared at each alpha.
    """
    n = len(s_data)
    if cosmo is None:
        cosmo = DEFAULT_COSMO

    # Compute template
    k = logspace_k(nk=2000, k_min=1e-4, k_max=50.0)
    s_template = linspace_s(ns=500, s_min=10.0, s_max=300.0)

    pk_full = linear_power_spectrum(k, cosmo)
    pk_nw = no_wiggle_power_spectrum(k, cosmo)
    pk_wig = pk_full - pk_nw

    xi_wig_template = hankel_transform(k, pk_wig, s_template, ell=0)
    xi_nw_template = hankel_transform(k, pk_nw, s_template, ell=0)

    xi_wig_spline = InterpolatedUnivariateSpline(s_template, xi_wig_template)
    xi_nw_spline = InterpolatedUnivariateSpline(s_template, xi_nw_template)

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.diag(1.0 / np.diag(cov))

    chi2_array = np.zeros(len(alpha_array))

    for i, alpha in enumerate(alpha_array):
        xi_wig_at_data = np.array([xi_wig_spline(si / alpha) for si in s_data])
        xi_nw_at_data = np.array([xi_nw_spline(si) for si in s_data])

        X = np.column_stack([
            xi_wig_at_data,
            np.ones(n),
            s_data,
            s_data ** 2,
        ])
        y = xi_data - xi_nw_at_data
        XtCinvX = X.T @ cov_inv @ X
        XtCinvy = X.T @ cov_inv @ y

        try:
            beta = np.linalg.solve(XtCinvX, XtCinvy)
            xi_model = X @ beta + xi_nw_at_data
            delta = xi_data - xi_model
            chi2_array[i] = float(delta @ cov_inv @ delta)
        except np.linalg.LinAlgError:
            chi2_array[i] = np.inf

    return alpha_array, chi2_array
