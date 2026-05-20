"""Chi-squared fitting for supernova cosmology."""

import numpy as np

from .distances import distance_modulus


def chi_squared(z_obs, mu_obs, mu_err, Omega_m=0.315, Omega_Lambda=0.685,
                w=-1.0, H0=67.4):
    """Compute chi-squared for SN cosmology fit."""
    z_obs = np.asarray(z_obs)
    mu_obs = np.asarray(mu_obs)
    mu_err = np.asarray(mu_err)

    mu_theory = np.array([
        distance_modulus(z, Omega_m, Omega_Lambda, w, H0)
        for z in z_obs
    ])
    return np.sum(((mu_obs - mu_theory) / mu_err) ** 2)
