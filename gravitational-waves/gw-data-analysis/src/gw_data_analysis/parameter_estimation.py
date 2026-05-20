"""Bayesian parameter estimation for gravitational wave signals.

Implements whitening, likelihood evaluation, and a simple MCMC sampler
for posterior inference on source parameters.
"""

import numpy as np

from .matched_filter import inner_product


def whiten(data, psd_func, freqs):
    """Whiten frequency-domain data using the detector PSD.

    Divides the data by sqrt(PSD/4) so that the whitened data has
    unit variance under the noise model.

    Parameters
    ----------
    data : array_like
        Complex frequency-domain data.
    psd_func : callable
        Function f(freqs) -> PSD values.
    freqs : array_like
        Frequency array in Hz.

    Returns
    -------
    whitened : ndarray
        Whitened frequency-domain data, same shape as input.
    """
    data = np.asarray(data, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)

    psd = psd_func(freqs)

    # Avoid division by zero
    whitened = np.zeros_like(data)
    valid = np.isfinite(psd) & (psd > 0)
    whitened[valid] = data[valid] / np.sqrt(psd[valid] / 4.0)

    return whitened


def likelihood(data, template, psd_func, freqs):
    """Compute the log-likelihood for a given template.

    .. math::
        \\log L = -\\frac{1}{2} (d - h | d - h) + \\text{const}

    Parameters
    ----------
    data : array_like
        Complex frequency-domain data.
    template : array_like
        Complex frequency-domain template waveform.
    psd_func : callable
        Function f(freqs) -> PSD values.
    freqs : array_like
        Frequency array in Hz.

    Returns
    -------
    log_l : float
        The log-likelihood (up to an additive constant).
    """
    data = np.asarray(data, dtype=np.complex128)
    template = np.asarray(template, dtype=np.complex128)

    residual = data - template
    return -0.5 * inner_product(residual, residual, psd_func, freqs)


def prior_chirp_mass(mc):
    """Uniform prior on chirp mass.

    Returns log(1) = 0 for any finite chirp mass (uniform in Mc).
    Returns -inf for non-positive values.

    Parameters
    ----------
    mc : float
        Chirp mass in solar masses.

    Returns
    -------
    log_prior : float
        Log of the prior probability density.
    """
    if mc <= 0 or not np.isfinite(mc):
        return -np.inf
    return 0.0


def posterior_samples(data, template_func, psd_func, freqs, prior_ranges, n_samples,
                      burn_in=500, seed=None):
    """Simple Metropolis-Hastings MCMC sampler for posterior inference.

    Samples from the posterior p(theta|d) proportional to L(d|theta) * p(theta).

    Parameters
    ----------
    data : array_like
        Complex frequency-domain data.
    template_func : callable
        Function theta -> h(f) that returns a complex frequency-domain template
        given a parameter vector theta.
    psd_func : callable
        Function f(freqs) -> PSD values.
    freqs : array_like
        Frequency array in Hz.
    prior_ranges : list of tuple
        List of (low, high) bounds for each parameter. Uniform priors are assumed
        within these bounds.
    n_samples : int
        Number of posterior samples to collect (after burn-in).
    burn_in : int, optional
        Number of burn-in steps to discard. Default 500.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    samples : ndarray
        Array of shape (n_samples, n_params) with posterior samples.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    data = np.asarray(data, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)

    n_params = len(prior_ranges)

    # Initialize at the center of prior ranges
    theta = np.array([0.5 * (lo + hi) for lo, hi in prior_ranges])

    # Proposal step sizes: 5% of each prior range
    step_sizes = np.array([0.05 * (hi - lo) for lo, hi in prior_ranges])

    def log_prior(t):
        for i, (lo, hi) in enumerate(prior_ranges):
            if t[i] < lo or t[i] > hi:
                return -np.inf
        return 0.0

    # Evaluate initial log-posterior
    template = template_func(theta)
    current_log_l = likelihood(data, template, psd_func, freqs)
    current_log_p = log_prior(theta)
    current_log_post = current_log_l + current_log_p

    total_steps = n_samples + burn_in
    samples = np.zeros((n_samples, n_params))
    accepted = 0
    sample_idx = 0

    for step in range(total_steps):
        # Propose new parameters
        theta_proposed = theta + step_sizes * rng.standard_normal(n_params)

        prop_log_p = log_prior(theta_proposed)
        if np.isfinite(prop_log_p):
            template_proposed = template_func(theta_proposed)
            prop_log_l = likelihood(data, template_proposed, psd_func, freqs)
            prop_log_post = prop_log_l + prop_log_p

            # Metropolis acceptance
            log_alpha = prop_log_post - current_log_post
            if np.log(rng.random()) < log_alpha:
                theta = theta_proposed
                current_log_post = prop_log_post
                accepted += 1

        # Collect sample after burn-in
        if step >= burn_in:
            samples[sample_idx] = theta
            sample_idx += 1

    return samples
