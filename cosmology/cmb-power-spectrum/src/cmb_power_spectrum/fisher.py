"""Fisher matrix forecasting for CMB parameter constraints.

F_{ij} = sum_l (2l+1) f_sky / (C_l + N_l)^2
         * dC_l/dtheta_i * dC_l/dtheta_j

Numerical derivatives via central differences.
"""

import numpy as np
from .theory_cls import compute_cl_tt, fiducial_params, params_to_vector, vector_to_params


def _noise_cl(ells, delta_T=50.0, theta_fwhm=7.0):
    """Instrumental noise power spectrum.

    N_l = (delta_T / T_CMB)^2 * exp(l(l+1) * theta_fwhm^2 / (8 ln 2))

    Parameters
    ----------
    ells : ndarray
        Multipole moments.
    delta_T : float
        Noise level in uK-arcmin (default 50 uK-arcmin, WMAP-like).
    theta_fwhm : float
        Beam FWHM in arcmin (default 7 arcmin).

    Returns
    -------
    N_l : ndarray
        Noise power spectrum in uK^2.
    """
    T_CMB = 2.7255e6  # in uK
    theta_rad = np.deg2rad(theta_fwhm / 60.0)
    nl = (delta_T / T_CMB) ** 2 * np.exp(
        ells * (ells + 1) * theta_rad ** 2 / (8.0 * np.log(2.0))
    )
    return nl


def _numerical_derivative(params, param_index, epsilon_frac=0.01, lmax=2500):
    """Compute dC_l/dtheta_i using central differences.

    Uses a relative step size of epsilon_frac * |theta_i| with a minimum
    floor to handle parameters near zero.
    """
    vec = params_to_vector(params)

    # Step size: fractional with floor
    abs_val = np.abs(vec[param_index])
    if abs_val > 0:
        eps = epsilon_frac * abs_val
    else:
        eps = 1e-4

    vec_plus = vec.copy()
    vec_minus = vec.copy()
    vec_plus[param_index] += eps
    vec_minus[param_index] -= eps

    params_plus = vector_to_params(vec_plus)
    params_minus = vector_to_params(vec_minus)

    _, cl_plus = compute_cl_tt(params_plus, lmax=lmax)
    _, cl_minus = compute_cl_tt(params_minus, lmax=lmax)

    return (cl_plus - cl_minus) / (2.0 * eps)


def fisher_matrix(params=None, lmax=2500, f_sky=1.0,
                  delta_T=50.0, theta_fwhm=7.0):
    """Compute the Fisher information matrix.

    Parameters
    ----------
    params : dict, optional
        Cosmological parameters. Defaults to fiducial_params.
    lmax : int
        Maximum multipole.
    f_sky : float
        Sky fraction observed (0 < f_sky <= 1).
    delta_T : float
        Noise level in uK-arcmin.
    theta_fwhm : float
        Beam FWHM in arcmin.

    Returns
    -------
    F : ndarray, shape (6, 6)
        Fisher matrix in the order:
        (omega_b, omega_c, theta_s, tau, ln10^10 A_s, n_s)
    param_names : list of str
        Parameter names in order.
    """
    if params is None:
        params = fiducial_params.copy()

    n_params = 6
    ells, cl_fid = compute_cl_tt(params, lmax=lmax)
    noise = _noise_cl(ells, delta_T, theta_fwhm)

    cl_total = cl_fid + noise
    prefactor = (2.0 * ells + 1.0) * f_sky / cl_total ** 2

    # Compute derivatives
    derivs = []
    for i in range(n_params):
        dcl = _numerical_derivative(params, i, lmax=lmax)
        derivs.append(dcl)

    # Build Fisher matrix
    F = np.zeros((n_params, n_params))
    for i in range(n_params):
        for j in range(i, n_params):
            F[i, j] = np.sum(prefactor * derivs[i] * derivs[j])
            F[j, i] = F[i, j]

    return F, ["omega_b", "omega_c", "theta_s", "tau", "ln10As", "n_s"]


def fisher_errors(F):
    """Compute marginalised 1-sigma errors from the Fisher matrix.

    Parameters
    ----------
    F : ndarray, shape (n, n)
        Fisher matrix.

    Returns
    -------
    sigma : ndarray, shape (n,)
        1-sigma marginalised errors: sigma_i = sqrt((F^{-1})_{ii}).
    """
    cov = np.linalg.inv(F)
    return np.sqrt(np.diag(cov))
