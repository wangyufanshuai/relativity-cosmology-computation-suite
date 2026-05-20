"""Metropolis-Hastings MCMC sampler for CMB parameter estimation.

Uses a Gaussian proposal distribution. Convergence assessed via
the Gelman-Rubin R-hat diagnostic.

Parameters: (omega_b, omega_c, theta_s, tau, ln10^10 A_s, n_s)
"""

import numpy as np
from .theory_cls import fiducial_params, params_to_vector, vector_to_params
from .likelihood import gaussian_log_likelihood


def _log_posterior(param_vec, ells, cl_data, f_sky=1.0,
                   delta_T=50.0, theta_fwhm=7.0):
    """Compute log-posterior (log-likelihood + flat priors).

    Flat priors within physically reasonable bounds.
    """
    # Flat prior bounds
    bounds = [
        (0.005, 0.05),     # omega_b
        (0.05, 0.25),      # omega_c
        (0.005, 0.02),     # theta_s
        (0.01, 0.20),      # tau
        (2.0, 4.0),        # ln10^10 A_s
        (0.8, 1.1),        # n_s
    ]

    for i, (lo, hi) in enumerate(bounds):
        if param_vec[i] < lo or param_vec[i] > hi:
            return -np.inf

    params = vector_to_params(param_vec)
    logL = gaussian_log_likelihood(
        params, ells, cl_data, f_sky=f_sky,
        delta_T=delta_T, theta_fwhm=theta_fwhm
    )
    return logL


def gelman_rubin(chains):
    """Compute the Gelman-Rubin R-hat diagnostic.

    Parameters
    ----------
    chains : list of ndarray
        Each element is an ndarray of shape (n_samples, n_params).

    Returns
    -------
    R_hat : ndarray, shape (n_params,)
        R-hat values for each parameter. Should be < 1.1 for convergence.
    """
    m = len(chains)
    n = chains[0].shape[0]
    n_params = chains[0].shape[1]

    # Chain means
    chain_means = np.array([c.mean(axis=0) for c in chains])
    overall_mean = chain_means.mean(axis=0)

    # Between-chain variance
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2, axis=0)

    # Within-chain variance
    W = np.mean([np.var(c, axis=0, ddof=1) for c in chains], axis=0)

    # Pooled variance estimate
    var_hat = (1 - 1.0 / n) * W + (1.0 / n) * B

    R_hat = np.sqrt(var_hat / W)
    return R_hat


def run_mcmc(ells, cl_data, n_steps=5000, n_walkers=4, f_sky=1.0,
             delta_T=50.0, theta_fwhm=7.0, seed=42,
             proposal_scale=None):
    """Run Metropolis-Hastings MCMC chains.

    Parameters
    ----------
    ells : ndarray
        Multipole moments of the data.
    cl_data : ndarray
        Observed C_l values.
    n_steps : int
        Number of steps per chain.
    n_walkers : int
        Number of independent chains.
    f_sky : float
        Sky fraction.
    delta_T : float
        Noise level in uK-arcmin.
    theta_fwhm : float
        Beam FWHM in arcmin.
    seed : int
        Random seed.
    proposal_scale : ndarray or None
        Diagonal proposal covariance scale. If None, uses reasonable defaults.

    Returns
    -------
    chains : list of ndarray
        Each chain has shape (n_steps, 6).
    acceptance_rates : list of float
        Acceptance rate for each chain.
    R_hat : ndarray
        Gelman-Rubin R-hat for each parameter.
    """
    rng = np.random.default_rng(seed)

    # Default proposal scales (~2.5% of fiducial)
    fid_vec = params_to_vector(fiducial_params)
    if proposal_scale is None:
        proposal_scale = np.array([
            0.0001,   # omega_b
            0.0005,   # omega_c
            0.00005,  # theta_s
            0.005,    # tau
            0.005,    # ln10As
            0.003,    # n_s
        ])

    proposal_cov = np.diag(proposal_scale ** 2)

    # Initialise walkers near fiducial with small scatter
    init_scatter = proposal_scale * 2.0

    chains = []
    acceptance_rates = []

    for w in range(n_walkers):
        # Starting point: perturb fiducial
        current = fid_vec + rng.normal(0, init_scatter)
        current_lp = _log_posterior(
            current, ells, cl_data, f_sky, delta_T, theta_fwhm
        )

        # If starting point is bad, fall back to fiducial
        if not np.isfinite(current_lp):
            current = fid_vec.copy()
            current_lp = _log_posterior(
                current, ells, cl_data, f_sky, delta_T, theta_fwhm
            )

        chain = np.zeros((n_steps, len(fid_vec)))
        accepted = 0

        for step in range(n_steps):
            # Propose
            proposal = current + rng.multivariate_normal(
                np.zeros(len(fid_vec)), proposal_cov
            )
            proposal_lp = _log_posterior(
                proposal, ells, cl_data, f_sky, delta_T, theta_fwhm
            )

            # Accept/reject
            log_alpha = proposal_lp - current_lp
            if np.log(rng.uniform()) < log_alpha:
                current = proposal
                current_lp = proposal_lp
                accepted += 1

            chain[step] = current

        chains.append(chain)
        acceptance_rates.append(accepted / n_steps)

    # Gelman-Rubin diagnostic
    # Use second half of each chain (burn-in removal)
    burn_in = n_steps // 2
    trimmed_chains = [c[burn_in:] for c in chains]
    R_hat = gelman_rubin(trimmed_chains)

    return chains, acceptance_rates, R_hat
