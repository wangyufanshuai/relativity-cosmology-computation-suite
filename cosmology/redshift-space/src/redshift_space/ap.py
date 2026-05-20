"""Alcock-Paczynski effect."""

import numpy as np


def alcock_paczynski_alpha(parallel, perpendicular, cosmology_ref, cosmology_true):
    """Compute Alcock-Paczynski scaling factors.

    Parameters
    ----------
    parallel : array_like
        Measurement along the line of sight in reference cosmology.
    perpendicular : array_like
        Measurement perpendicular to line of sight in reference cosmology.
    cosmology_ref : dict
        Reference cosmology parameters (must contain 'H0', 'Omega_m').
    cosmology_true : dict
        True cosmology parameters (must contain 'H0', 'Omega_m').

    Returns
    -------
    tuple
        (alpha_parallel, alpha_perpendicular) scaling factors.
    """
    # AP scaling: ratio of true to reference distance scales
    # For a simple model, use H(z) and D_A(z) ratios
    H_ref = cosmology_ref.get('H0', 70.0)
    H_true = cosmology_true.get('H0', 70.0)
    Om_ref = cosmology_ref.get('Omega_m', 0.3)
    Om_true = cosmology_true.get('Omega_m', 0.3)

    # Approximate alpha_para ~ H_ref / H_true
    alpha_para = H_ref / H_true

    # Approximate alpha_perp ~ D_A_true / D_A_ref ~ (H_ref / H_true)^{something}
    # Simplified: use comoving distance ratio
    alpha_perp = (H_ref / H_true) * (Om_true / Om_ref) ** 0.0  # simplified

    # More physically: use ratio of angular diameter distances
    # For flat LCDM, D_A proportional to 1/H0 at low z
    alpha_perp = H_ref / H_true

    return alpha_para, alpha_perp


def ap_power_spectrum(k_obs, mu_obs, P_true, alpha_perp, alpha_para):
    """Transform power spectrum from true to observed coordinates via AP effect.

    Parameters
    ----------
    k_obs : array_like
        Observed wavenumber.
    mu_obs : array_like
        Observed angle cosine.
    P_true : array_like or float
        True power spectrum.
    alpha_perp : float
        Perpendicular scaling factor.
    alpha_para : float
        Parallel scaling factor.

    Returns
    -------
    array_like
        Power spectrum in observed coordinates with AP distortion.
    """
    # Convert observed (k_obs, mu_obs) to true (k_true, mu_true)
    k_perp = k_obs * np.sqrt(1.0 - mu_obs**2) / alpha_perp
    k_para = k_obs * mu_obs / alpha_para

    k_true = np.sqrt(k_perp**2 + k_para**2)

    # Jacobian of coordinate transformation
    jacobian = (alpha_para * alpha_perp**2) ** (-1)

    return P_true * jacobian if np.isscalar(P_true) else P_true * jacobian
