"""Gaussian likelihood for the CMB TT power spectrum.

ln L = -1/2 * sum_l (2l+1) f_sky
       * [C_l^{data} / C_l^{theory} + ln(C_l^{theory}) - ln(C_l^{data})]

This is the exact Gaussian likelihood for the angular power spectrum
under the approximation that each C_l is inverse-Gamma distributed,
which becomes the expression above for the sufficient statistic.
"""

import numpy as np
from .theory_cls import compute_cl_tt, fiducial_params
from .fisher import _noise_cl


def generate_mock_data(params=None, lmax=2500, f_sky=1.0,
                       delta_T=50.0, theta_fwhm=7.0, seed=42):
    """Generate a mock C_l^{TT} dataset with cosmic variance and noise.

    Parameters
    ----------
    params : dict, optional
        Cosmological parameters for the signal. Defaults to fiducial.
    lmax : int
        Maximum multipole.
    f_sky : float
        Sky fraction.
    delta_T : float
        Instrumental noise in uK-arcmin.
    theta_fwhm : float
        Beam FWHM in arcmin.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ells : ndarray
        Multipole moments.
    cl_data : ndarray
        Mock observed C_l (signal + cosmic variance realisation + noise).
    cl_signal : ndarray
        Underlying signal C_l.
    noise : ndarray
        Noise power spectrum N_l.
    """
    if params is None:
        params = fiducial_params.copy()

    rng = np.random.default_rng(seed)

    ells, cl_signal = compute_cl_tt(params, lmax=lmax)
    noise = _noise_cl(ells, delta_T, theta_fwhm)

    cl_total = cl_signal + noise

    # Cosmic variance: variance of C_l estimate is 2/(2l+1) * C_l^2 / f_sky
    # Each observed C_l is drawn from an inverse-gamma distribution,
    # approximated here as Gaussian for large (2l+1)*f_sky
    variance = 2.0 / ((2.0 * ells + 1.0) * f_sky) * cl_total ** 2
    cl_data = cl_total + rng.normal(0, np.sqrt(variance))

    # Ensure positivity of mock data
    cl_data = np.maximum(cl_data, 1e-30)

    return ells, cl_data, cl_signal, noise


def gaussian_log_likelihood(params, ells, cl_data, f_sky=1.0,
                            delta_T=50.0, theta_fwhm=7.0):
    """Compute the Gaussian log-likelihood for C_l^{TT}.

    Parameters
    ----------
    params : dict
        Cosmological parameters for the theory spectrum.
    ells : ndarray
        Multipole moments of the data.
    cl_data : ndarray
        Observed C_l values.
    f_sky : float
        Sky fraction.
    delta_T : float
        Noise level in uK-arcmin.
    theta_fwhm : float
        Beam FWHM in arcmin.

    Returns
    -------
    logL : float
        Log-likelihood value.
    """
    lmax = int(ells[-1])
    _, cl_theory = compute_cl_tt(params, lmax=lmax)

    # Truncate to match data length
    cl_theory = cl_theory[: len(ells)]

    noise = _noise_cl(ells, delta_T, theta_fwhm)
    cl_theory_total = cl_theory + noise

    # Log-likelihood for inverse-Gamma distributed C_l:
    # ln L = -1/2 sum_l (2l+1) f_sky [C_l^data/C_l^theory_total
    #         + ln(C_l^theory_total) - ln(C_l^data)]
    # (up to constants that don't depend on parameters)
    ratio = cl_data / cl_theory_total
    log_ratio = np.log(cl_theory_total) - np.log(cl_data)

    logL = -0.5 * np.sum((2.0 * ells + 1.0) * f_sky * (ratio + log_ratio))

    return logL
