"""Planck 2018 best-fit parameters and Fisher forecast tools."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .cosmology import Cosmology
from .background import solve_background


def planck18_params() -> Cosmology:
    """Return a Cosmology with Planck 2018 TT,TE,EE+lowE+lensing best-fit parameters.

    Reference: Planck 2018 results VI. Table 2 (base_lcdm)

    Returns
    -------
    cosmo : Cosmology
        Planck18 best-fit flat Lambda-CDM cosmology.
    """
    # Planck 2018 best-fit: Omega_b h^2 = 0.02237, Omega_cdm h^2 = 0.1200
    # h = 0.6736 => Omega_m = (0.02237 + 0.1200) / 0.6736^2 = 0.3133
    # Omega_Lambda = 1 - Omega_m (flat)
    # tau = 0.0544, ln(10^10 A_s) = 3.044, n_s = 0.9649
    # We use the rounded H0=67.4 from the problem spec for consistency
    return Cosmology(
        H0=67.4,
        Omega_m=0.315,
        Omega_r=0.0,        # computed from T_CMB + N_eff
        Omega_lambda=0.685,
        Omega_k=None,       # flat: set from sum
        w0=-1.0,
        wa=0.0,
        N_eff=3.046,
        T_cmb=2.7255,
    )


def planck18_derived() -> dict:
    """Return Planck 2018 derived cosmological parameters.

    Returns
    -------
    derived : dict
        Dictionary with keys:
        - age: Age of universe [Gyr]
        - sigma8: RMS density fluctuation on 8 Mpc/h scale
        - r_drag: Sound horizon at drag epoch [Mpc]
        - z_star: Redshift of last scattering
        - z_drag: Redshift of baryon drag
        - theta_star: Angular sound horizon scale [deg]
        - Omega_m: Matter density parameter
        - Omega_lambda: Dark energy density parameter
        - Omega_r: Radiation density parameter
        - Omega_b: Baryon density parameter
        - h: H0/100
    """
    cosmo = planck18_params()
    h = cosmo.H0 / 100.0

    # Derived parameters from Planck 2018
    age = cosmo.age()
    r_drag = cosmo.sound_horizon(z_drag=1060.0)

    derived = {
        "age": age,
        "sigma8": 0.8111,
        "r_drag": r_drag,
        "z_star": 1089.92,
        "z_drag": 1060.0,
        "theta_star": 0.010410,  # degrees, not rad
        "Omega_m": cosmo.Omega_m,
        "Omega_lambda": cosmo.Omega_lambda,
        "Omega_r": cosmo.Omega_r,
        "Omega_b": 0.0224 / h ** 2,
        "h": h,
    }
    return derived


def fisher_matrix(
    cosmo: Cosmology,
    param_names: Sequence[str] = ("H0", "Omega_m", "w0", "wa"),
    sigma_H0: float = 0.5,
    sigma_omega_m: float = 0.007,
    sigma_w0: float | None = None,
    sigma_wa: float | None = None,
    z_survey: NDArray | None = None,
    n_modes: int = 1000,
) -> dict:
    """Compute a simplified Fisher matrix forecast for a DESI-like BAO survey.

    This uses the BAO distance ratios as observables and computes partial
    derivatives of d_L(z) and H(z) with respect to cosmological parameters.

    Parameters
    ----------
    cosmo : Cosmology
        Fiducial cosmology.
    param_names : sequence of str
        Parameters to vary. Default: ("H0", "Omega_m", "w0", "wa").
    sigma_H0 : float
        Prior uncertainty on H0 [km/s/Mpc]. Default: 0.5.
    sigma_omega_m : float
        Prior uncertainty on Omega_m. Default: 0.007.
    sigma_w0 : float or None
        Prior uncertainty on w0. If None, no prior applied.
    sigma_wa : float or None
        Prior uncertainty on wa. If None, no prior applied.
    z_survey : ndarray or None
        Redshift bins for the survey. Default: DESI-like bins.
    n_modes : int
        Number of BAO modes measured per redshift bin. Default: 1000.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - fisher: Fisher matrix (n_params x n_params)
        - covariance: Inverse Fisher matrix (parameter covariances)
        - param_names: Parameter names
        - errors: 1-sigma forecast errors on each parameter
    """
    if z_survey is None:
        z_survey = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.1])

    param_names = list(param_names)
    n_params = len(param_names)
    n_z = len(z_survey)

    # Step size for numerical derivatives
    step = {
        "H0": 0.1,
        "Omega_m": 0.001,
        "Omega_lambda": 0.001,
        "w0": 0.01,
        "wa": 0.01,
        "Omega_k": 0.001,
    }

    # Observables: d_L(z) ratios and H(z) at each redshift
    # Partial derivatives via finite differences
    def make_cosmo(delta_param, delta_val):
        """Create a perturbed cosmology."""
        kwargs = {
            "H0": cosmo.H0,
            "Omega_m": cosmo.Omega_m,
            "Omega_lambda": cosmo.Omega_lambda,
            "w0": cosmo.w0,
            "wa": cosmo.wa,
            "Omega_k": cosmo.Omega_k,
            "N_eff": cosmo.N_eff,
            "T_cmb": cosmo.T_cmb,
        }
        # When varying Omega_m, adjust Omega_lambda to keep flatness
        kwargs[delta_param] = kwargs.get(delta_param, 0) + delta_val
        if delta_param == "Omega_m":
            kwargs["Omega_lambda"] = cosmo.Omega_lambda - delta_val
        return Cosmology(**kwargs)

    # Compute derivatives of d_L(z) and H(z) w.r.t. each parameter
    # Shape: (2 * n_z, n_params) — d_L at each z, then H at each z
    derivs = np.zeros((2 * n_z, n_params))

    fid_dL = np.array([cosmo.luminosity_distance(z) for z in z_survey])
    fid_H = np.array([cosmo.H(1.0 / (1.0 + z)) for z in z_survey])

    for j, pname in enumerate(param_names):
        dp = step.get(pname, 0.01)
        cosmo_p = make_cosmo(pname, dp)
        cosmo_m = make_cosmo(pname, -dp)

        dL_p = np.array([cosmo_p.luminosity_distance(z) for z in z_survey])
        dL_m = np.array([cosmo_m.luminosity_distance(z) for z in z_survey])
        H_p = np.array([cosmo_p.H(1.0 / (1.0 + z)) for z in z_survey])
        H_m = np.array([cosmo_m.H(1.0 / (1.0 + z)) for z in z_survey])

        derivs[:n_z, j] = (dL_p - dL_m) / (2.0 * dp)
        derivs[n_z:, j] = (H_p - H_m) / (2.0 * dp)

    # Noise: fractional BAO measurement ~ 1/sqrt(N_modes) * distance
    sigma_dL = fid_dL * 0.01  # ~1% distance measurement
    sigma_H = fid_H * 0.02    # ~2% H(z) measurement

    # Fisher matrix from BAO observations: F = sum_k (dO/dp)_k (dO/dp)_k^T / sigma_k^2
    F = np.zeros((n_params, n_params))
    for i in range(n_z):
        # d_L contribution
        dL_deriv = derivs[i, :]
        F += np.outer(dL_deriv, dL_deriv) / sigma_dL[i] ** 2
        # H(z) contribution
        H_deriv = derivs[n_z + i, :]
        F += np.outer(H_deriv, H_deriv) / sigma_H[i] ** 2

    # Add priors
    priors = {
        "H0": sigma_H0,
        "Omega_m": sigma_omega_m,
        "w0": sigma_w0,
        "wa": sigma_wa,
    }
    for j, pname in enumerate(param_names):
        prior_sigma = priors.get(pname)
        if prior_sigma is not None:
            F[j, j] += 1.0 / prior_sigma ** 2

    # Invert to get covariance
    try:
        cov = np.linalg.inv(F)
        errors = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        cov = np.full((n_params, n_params), np.nan)
        errors = np.full(n_params, np.nan)

    return {
        "fisher": F,
        "covariance": cov,
        "param_names": param_names,
        "errors": dict(zip(param_names, errors)),
    }
