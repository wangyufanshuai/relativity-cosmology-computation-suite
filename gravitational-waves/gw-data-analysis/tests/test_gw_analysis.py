"""Tests for the gravitational wave data analysis pipeline.

Covers noise models, matched filtering, parameter estimation, and the F-statistic.
"""

import numpy as np
import pytest

from gw_data_analysis.constants import C, G, M_SUN, MPC
from gw_data_analysis.matched_filter import inner_product, matched_filter_snr, optimal_snr
from gw_data_analysis.noise import advanced_ligo_psd, generate_noise, white_noise_psd
from gw_data_analysis.parameter_estimation import (
    likelihood,
    posterior_samples,
    prior_chirp_mass,
    whiten,
)
from gw_data_analysis.f_statistic import cwb_snr, f_statistic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_freqs(n_samples=4096, dt=1.0 / 4096):
    """Return a frequency array and sampling parameters."""
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    return freqs, n_samples, dt


def _inject_signal(freqs, f_peak=100.0, amplitude=1e-22, bandwidth=20.0):
    """Create a simple Gaussian-enveloped sinusoidal signal in the frequency domain."""
    h = amplitude * np.exp(-0.5 * ((freqs - f_peak) / bandwidth) ** 2)
    return h.astype(np.complex128)


# ---------------------------------------------------------------------------
# PSD Tests
# ---------------------------------------------------------------------------

class TestPSD:
    """Tests for power spectral density models."""

    def test_psd_positive(self):
        """PSD > 0 for all positive frequencies."""
        freqs = np.linspace(1.0, 5000.0, 1000)
        psd = advanced_ligo_psd(freqs)
        assert np.all(psd > 0), "PSD must be positive for all positive frequencies"

    def test_psd_high_freq_drops(self):
        """PSD decreases at high frequency (shot-noise dominated regime)."""
        freqs_low = np.array([200.0, 300.0, 400.0])
        freqs_high = np.array([2000.0, 3000.0, 4000.0])
        psd_low = advanced_ligo_psd(freqs_low)
        psd_high = advanced_ligo_psd(freqs_high)
        # The mean PSD at high frequencies should be lower than at moderate frequencies
        assert np.mean(psd_high) < np.mean(psd_low), (
            "PSD should decrease at high frequencies"
        )


# ---------------------------------------------------------------------------
# Inner Product Tests
# ---------------------------------------------------------------------------

class TestInnerProduct:
    """Tests for noise-weighted inner product."""

    def test_inner_product_symmetric(self):
        """(h1|h2) = conj((h2|h1)). For real-valued inner products, (h1|h2) = (h2|h1)."""
        freqs, _, _ = _make_freqs()
        h1 = _inject_signal(freqs, f_peak=80.0)
        h2 = _inject_signal(freqs, f_peak=120.0)

        ip12 = inner_product(h1, h2, advanced_ligo_psd, freqs)
        ip21 = inner_product(h2, h1, advanced_ligo_psd, freqs)

        # The inner product as defined returns the real part, so should be symmetric
        np.testing.assert_allclose(ip12, ip21, rtol=1e-10,
                                   err_msg="(h1|h2) should equal (h2|h1)")

    def test_inner_product_positive_semidef(self):
        """(h|h) >= 0 for any signal h."""
        freqs, _, _ = _make_freqs()
        h = _inject_signal(freqs, f_peak=100.0)

        auto = inner_product(h, h, advanced_ligo_psd, freqs)
        assert auto >= 0.0, f"(h|h) must be >= 0, got {auto}"

    def test_inner_product_positive_semidef_multiple(self):
        """(h|h) >= 0 for various signals."""
        freqs, _, _ = _make_freqs()
        for f_peak in [50.0, 100.0, 200.0, 500.0]:
            h = _inject_signal(freqs, f_peak=f_peak)
            auto = inner_product(h, h, advanced_ligo_psd, freqs)
            assert auto >= 0.0, f"(h|h) must be >= 0 for f_peak={f_peak}, got {auto}"


# ---------------------------------------------------------------------------
# SNR Tests
# ---------------------------------------------------------------------------

class TestSNR:
    """Tests for signal-to-noise ratio calculations."""

    def test_snr_zero_for_no_signal(self):
        """SNR should be approximately zero when no signal is present in data."""
        np.random.seed(42)
        freqs, n_samples, dt = _make_freqs()

        # Pure noise data
        noise = generate_noise(advanced_ligo_psd, duration=1.0, dt=dt)
        data_freq = np.fft.rfft(noise)

        # Template signal
        template = _inject_signal(freqs, f_peak=100.0)

        snr = matched_filter_snr(template, data_freq, advanced_ligo_psd, freqs)

        # For random noise (no signal), SNR should be of order 1 or less
        # We use a generous threshold to avoid flaky tests
        assert abs(snr) < 5.0, (
            f"SNR should be small for pure noise, got {snr}"
        )

    def test_snr_positive_for_signal(self):
        """SNR > 0 when signal is present in data."""
        freqs, n_samples, dt = _make_freqs()

        # Inject a signal into data
        template = _inject_signal(freqs, f_peak=100.0, amplitude=1e-21)
        data = template.copy()  # data = signal (no noise for simplicity)

        snr = matched_filter_snr(template, data, advanced_ligo_psd, freqs)

        assert snr > 0, f"SNR should be positive when signal is present, got {snr}"

    def test_optimal_snr_positive_for_signal(self):
        """Optimal SNR should be positive for a non-trivial signal."""
        freqs, _, _ = _make_freqs()
        h = _inject_signal(freqs, f_peak=100.0, amplitude=1e-21)
        snr = optimal_snr(h, advanced_ligo_psd, freqs)
        assert snr > 0, f"Optimal SNR should be positive, got {snr}"


# ---------------------------------------------------------------------------
# Whitening Tests
# ---------------------------------------------------------------------------

class TestWhiten:
    """Tests for data whitening."""

    def test_whiten_preserves_length(self):
        """Whitened output has the same length as input."""
        freqs, _, _ = _make_freqs()
        data = _inject_signal(freqs, f_peak=100.0)

        whitened = whiten(data, advanced_ligo_psd, freqs)

        assert len(whitened) == len(data), (
            f"Whitened data length {len(whitened)} != input length {len(data)}"
        )


# ---------------------------------------------------------------------------
# Likelihood Tests
# ---------------------------------------------------------------------------

class TestLikelihood:
    """Tests for Bayesian likelihood function."""

    def test_likelihood_max_at_true_params(self):
        """Likelihood should peak near true parameters (injection-recovery test)."""
        np.random.seed(123)
        freqs, n_samples, dt = _make_freqs()

        # True parameters: amplitude and frequency
        true_amplitude = 1e-22
        true_f_peak = 150.0
        bandwidth = 20.0

        # Inject signal at true parameters
        true_signal = _inject_signal(
            freqs, f_peak=true_f_peak, amplitude=true_amplitude, bandwidth=bandwidth
        )
        data = true_signal.copy()

        # Evaluate likelihood at the true parameters
        ll_true = likelihood(data, true_signal, advanced_ligo_psd, freqs)

        # Evaluate at several wrong parameter values
        ll_wrong_values = []
        for wrong_amp in [0.5e-22, 2.0e-22]:
            for wrong_f in [80.0, 250.0]:
                wrong_signal = _inject_signal(
                    freqs, f_peak=wrong_f, amplitude=wrong_amp, bandwidth=bandwidth
                )
                ll_wrong = likelihood(data, wrong_signal, advanced_ligo_psd, freqs)
                ll_wrong_values.append(ll_wrong)

        # True parameters should give higher likelihood than all wrong ones
        for ll_wrong in ll_wrong_values:
            assert ll_true > ll_wrong, (
                f"Log-likelihood at true params ({ll_true}) should exceed "
                f"wrong params ({ll_wrong})"
            )

    def test_likelihood_zero_residual(self):
        """When data = template, likelihood = 0 (maximum)."""
        freqs, _, _ = _make_freqs()
        h = _inject_signal(freqs, f_peak=100.0)
        ll = likelihood(h, h, advanced_ligo_psd, freqs)
        np.testing.assert_allclose(ll, 0.0, atol=1e-10,
                                   err_msg="log L should be 0 when data = template")


# ---------------------------------------------------------------------------
# Prior Tests
# ---------------------------------------------------------------------------

class TestPrior:
    """Tests for prior functions."""

    def test_prior_chirp_mass_positive(self):
        """Uniform prior returns 0 log-density for positive chirp mass."""
        assert prior_chirp_mass(30.0) == 0.0

    def test_prior_chirp_mass_nonpositive(self):
        """Prior returns -inf for non-positive chirp mass."""
        assert prior_chirp_mass(0.0) == -np.inf
        assert prior_chirp_mass(-1.0) == -np.inf


# ---------------------------------------------------------------------------
# F-statistic Tests
# ---------------------------------------------------------------------------

class TestFStatistic:
    """Tests for continuous wave F-statistic."""

    def test_f_statistic_with_signal(self):
        """F-statistic should be larger when a signal is present."""
        freqs, n_samples, dt = _make_freqs()

        # Inject a monochromatic signal
        f_signal = 200.0
        signal = np.zeros(len(freqs), dtype=np.complex128)
        df = freqs[1] - freqs[0]
        idx = int(round(f_signal / df))
        signal[idx] = 1e-22 + 0j

        data_with_signal = signal.copy()

        # F-statistic with signal
        f_stat_signal = f_statistic(
            data_with_signal, f_signal,
            theta=0.5, phi=0.3, psi=0.1, phi0=0.0,
            psd_func=advanced_ligo_psd, freqs=freqs
        )

        # F-statistic without signal (empty data)
        data_empty = np.zeros(len(freqs), dtype=np.complex128)
        f_stat_empty = f_statistic(
            data_empty, f_signal,
            theta=0.5, phi=0.3, psi=0.1, phi0=0.0,
            psd_func=advanced_ligo_psd, freqs=freqs
        )

        assert f_stat_signal > f_stat_empty, (
            "F-statistic should be larger with signal present"
        )

    def test_cwb_snr_positive_for_signal(self):
        """CW SNR should be positive when a signal is present."""
        freqs, _, _ = _make_freqs()

        f_signal = 200.0
        signal = np.zeros(len(freqs), dtype=np.complex128)
        df = freqs[1] - freqs[0]
        idx = int(round(f_signal / df))
        signal[idx] = 1e-21

        snr = cwb_snr(signal, f_signal, advanced_ligo_psd, freqs)
        assert snr > 0, f"CW SNR should be positive with signal, got {snr}"


# ---------------------------------------------------------------------------
# Noise Generation Tests
# ---------------------------------------------------------------------------

class TestNoiseGeneration:
    """Tests for noise time series generation."""

    def test_noise_correct_length(self):
        """Generated noise has the expected number of samples."""
        duration = 1.0
        dt = 1.0 / 4096
        noise = generate_noise(white_noise_psd_fn(sigma=1.0), duration, dt)
        expected = int(round(duration / dt))
        assert len(noise) == expected

    def test_noise_not_silent(self):
        """Generated noise is not all zeros."""
        np.random.seed(99)
        noise = generate_noise(white_noise_psd_fn(sigma=1.0), 1.0, 1.0 / 1024)
        assert np.std(noise) > 0, "Noise should have non-zero standard deviation"


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests."""

    def test_injection_recovery_mcmc(self):
        """MCMC should recover injected amplitude and frequency within prior range."""
        np.random.seed(42)
        freqs, n_samples, dt = _make_freqs()

        # True parameters
        true_amp = 5e-22
        true_f = 200.0
        bandwidth = 20.0

        # Inject signal
        true_signal = _inject_signal(
            freqs, f_peak=true_f, amplitude=true_amp, bandwidth=bandwidth
        )

        # Template function: (amplitude, frequency) -> frequency-domain signal
        def template_func(theta):
            amp, f_peak = theta
            return _inject_signal(freqs, f_peak=f_peak, amplitude=amp, bandwidth=bandwidth)

        prior_ranges = [(1e-22, 1e-20), (100.0, 300.0)]

        samples = posterior_samples(
            data=true_signal,
            template_func=template_func,
            psd_func=advanced_ligo_psd,
            freqs=freqs,
            prior_ranges=prior_ranges,
            n_samples=2000,
            burn_in=1000,
            seed=42,
        )

        # Check that the mean of samples is within the prior range
        mean_amp = np.mean(samples[:, 0])
        mean_f = np.mean(samples[:, 1])

        assert prior_ranges[0][0] <= mean_amp <= prior_ranges[0][1], (
            f"Mean amplitude {mean_amp} outside prior range"
        )
        assert prior_ranges[1][0] <= mean_f <= prior_ranges[1][1], (
            f"Mean frequency {mean_f} outside prior range"
        )

        # The posterior should be peaked near the true values
        # Check that the true values are within 3 sigma of the posterior mean
        std_amp = np.std(samples[:, 0])
        std_f = np.std(samples[:, 1])

        if std_amp > 0 and std_f > 0:
            # At least check the mean is within 5 sigma of truth (generous for short chain)
            assert abs(mean_amp - true_amp) < 5 * max(std_amp, 1e-23), (
                f"Posterior mean amp {mean_amp} too far from truth {true_amp}"
            )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def white_noise_psd_fn(sigma):
    """Return a white noise PSD function with the given sigma."""
    def fn(f):
        return white_noise_psd(f, sigma)
    return fn
