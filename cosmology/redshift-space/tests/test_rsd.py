"""Tests for redshift-space distortion models."""

import numpy as np
import pytest

from redshift_space.kaiser import kaiser_factor, rsd_power_spectrum, multipole_Pk
from redshift_space.fog import lorentzian_fog, gaussian_fog, combined_rsd
from redshift_space.ap import alcock_paczynski_alpha, ap_power_spectrum


# ============================================================
# Kaiser effect tests
# ============================================================

class TestKaiserFactor:
    """Tests for the Kaiser RSD factor (1 + beta * mu^2)^2."""

    def test_kaiser_factor_zero_mu(self):
        """At mu=0 (perpendicular to LOS), factor should be 1."""
        result = kaiser_factor(0.1, mu=0.0, beta=0.5)
        np.testing.assert_allclose(result, 1.0)

    def test_kaiser_factor_unity_mu(self):
        """At mu=1 (along LOS), factor should be (1 + beta)^2."""
        beta = 0.4
        result = kaiser_factor(0.1, mu=1.0, beta=beta)
        expected = (1.0 + beta) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_kaiser_factor_boost(self):
        """Kaiser factor should always be >= 1 for positive beta and real mu."""
        mu = np.linspace(0, 1, 50)
        k = np.ones_like(mu) * 0.1
        beta = 0.5
        result = kaiser_factor(k, mu, beta)
        assert np.all(result >= 1.0), "Kaiser boost should be >= 1"

    def test_kaiser_factor_analytic(self):
        """Check against analytic formula for specific values."""
        beta = 0.8
        mu_vals = np.array([0.0, 0.5, 0.707, 1.0])
        k = np.ones_like(mu_vals)
        expected = (1.0 + beta * mu_vals**2) ** 2
        result = kaiser_factor(k, mu_vals, beta)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_kaiser_factor_zero_beta(self):
        """With beta=0 (no growth rate), factor should be 1."""
        mu = np.linspace(0, 1, 30)
        k = np.ones_like(mu)
        result = kaiser_factor(k, mu, beta=0.0)
        np.testing.assert_allclose(result, 1.0)

    def test_kaiser_factor_array(self):
        """Should work with arrays."""
        mu = np.array([0.0, 0.5, 1.0])
        result = kaiser_factor(np.array([0.1, 0.2, 0.3]), mu, beta=1.0)
        assert result.shape == mu.shape


class TestRSDPowerSpectrum:
    """Tests for the full Kaiser RSD power spectrum."""

    def test_rsd_power_equals_kaiser_times_real(self):
        """P_rsd(k, mu) = Kaiser_factor * P_real(k)."""
        k = np.array([0.1, 0.2])
        mu = np.array([0.5, 0.7])
        P_real = np.array([1000.0, 500.0])
        beta = 0.5

        result = rsd_power_spectrum(k, mu, P_real, beta)
        expected = kaiser_factor(k, mu, beta) * P_real
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_rsd_power_scalar(self):
        """Should work with scalar inputs."""
        result = rsd_power_spectrum(k=0.1, mu=0.5, P_real=1000.0, beta=0.5)
        expected = (1.0 + 0.5 * 0.25) ** 2 * 1000.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_rsd_power_positive(self):
        """Redshift-space P(k) should be positive if P_real is positive."""
        k = np.logspace(-2, 0, 20)
        mu = np.random.uniform(0, 1, 20)
        P_real = np.abs(np.random.randn(20)) + 100.0
        result = rsd_power_spectrum(k, mu, P_real, beta=0.5)
        assert np.all(result > 0)


class TestMultipoles:
    """Tests for power spectrum multipole moments."""

    def test_monopole_larger_than_real(self):
        """The monopole (l=0) should be >= P_real for positive beta (Kaiser boost)."""
        P_real = 1000.0
        beta = 0.5
        P0 = multipole_Pk(0.1, P_real, beta, l=0)
        assert P0 >= P_real * 0.99  # small tolerance for numerical integration

    def test_quadrupole_sign(self):
        """The quadrupole (l=2) should be positive for positive beta (prolate distortion)."""
        P_real = 1000.0
        beta = 0.5
        P2 = multipole_Pk(0.1, P_real, beta, l=2)
        assert P2 > 0, "Quadrupole should be positive for Kaiser effect"

    def test_hexadecapole_small(self):
        """The hexadecapole (l=4) should be much smaller than quadrupole."""
        P_real = 1000.0
        beta = 0.5
        P2 = multipole_Pk(0.1, P_real, beta, l=2)
        P4 = multipole_Pk(0.1, P_real, beta, l=4)
        assert abs(P4) < abs(P2), "l=4 should be smaller than l=2"

    def test_invalid_multipole_raises(self):
        """Requesting unsupported multipole order should raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            multipole_Pk(0.1, 1000.0, 0.5, l=1)

    def test_monopole_zero_beta_equals_real(self):
        """With beta=0, the monopole should equal P_real."""
        P_real = 1000.0
        P0 = multipole_Pk(0.1, P_real, beta=0.0, l=0)
        np.testing.assert_allclose(P0, P_real, rtol=1e-3)


# ============================================================
# Fingers-of-God tests
# ============================================================

class TestLorentzianFOG:
    """Tests for Lorentzian FOG damping."""

    def test_lorentzian_unity_at_zero_kmu(self):
        """Damping should be 1 when k*mu = 0."""
        result = lorentzian_fog(k=0.0, mu=0.0, sigma_v=5.0)
        np.testing.assert_allclose(result, 1.0)

    def test_lorentzian_damping_positive(self):
        """Damping factor should always be positive and <= 1."""
        k = np.logspace(-2, 1, 50)
        mu = np.linspace(0, 1, 50)
        K, MU = np.meshgrid(k, mu)
        result = lorentzian_fog(K.ravel(), MU.ravel(), sigma_v=5.0)
        assert np.all(result > 0)
        assert np.all(result <= 1.0 + 1e-12)

    def test_lorentzian_decreases_with_k(self):
        """Damping should decrease as k increases (at fixed mu)."""
        k = np.logspace(-3, 0, 100)
        result = lorentzian_fog(k, mu=1.0, sigma_v=5.0)
        # Should be monotonically decreasing
        assert np.all(np.diff(result) <= 1e-12)


class TestGaussianFOG:
    """Tests for Gaussian FOG damping."""

    def test_gaussian_unity_at_zero_kmu(self):
        """Damping should be 1 when k*mu = 0."""
        result = gaussian_fog(k=0.0, mu=0.0, sigma_v=5.0)
        np.testing.assert_allclose(result, 1.0)

    def test_gaussian_damping_positive(self):
        """Damping should be in [0, 1]."""
        k = np.logspace(-2, 1, 50)
        mu = np.linspace(0, 1, 50)
        K, MU = np.meshgrid(k, mu)
        result = gaussian_fog(K.ravel(), MU.ravel(), sigma_v=5.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1.0 + 1e-12)

    def test_gaussian_stronger_than_lorentzian_at_large_k(self):
        """At large k*mu*sigma_v, Gaussian damping is stronger than Lorentzian."""
        k = 10.0
        mu = 1.0
        sigma_v = 5.0
        g = gaussian_fog(k, mu, sigma_v)
        l = lorentzian_fog(k, mu, sigma_v)
        assert g < l, "Gaussian should damp more at large k"


class TestCombinedRSD:
    """Tests for combined Kaiser + FOG model."""

    def test_combined_equals_kaiser_times_fog_times_real(self):
        """P_combined = Kaiser * FOG_lorentzian * P_real."""
        k = np.array([0.05, 0.1, 0.2])
        mu = np.array([0.3, 0.5, 0.8])
        P_real = np.array([2000.0, 1000.0, 300.0])
        beta = 0.5
        sigma_v = 5.0

        result = combined_rsd(k, mu, P_real, beta, sigma_v)
        expected = kaiser_factor(k, mu, beta) * lorentzian_fog(k, mu, sigma_v) * P_real
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_combined_positive_for_positive_input(self):
        """Combined RSD power should be positive for positive P_real."""
        k = np.logspace(-2, 0, 30)
        mu = np.linspace(0.01, 0.99, 30)
        P_real = np.abs(np.random.randn(30)) * 1000 + 100.0
        result = combined_rsd(k, mu, P_real, beta=0.5, sigma_v=5.0)
        assert np.all(result > 0)

    def test_combined_suppressed_at_large_k(self):
        """At large k, FOG damping should suppress the power spectrum."""
        P_real = 1000.0
        beta = 0.5
        sigma_v = 5.0
        P_small_k = combined_rsd(0.01, 0.5, P_real, beta, sigma_v)
        P_large_k = combined_rsd(1.0, 0.5, P_real, beta, sigma_v)
        assert P_large_k < P_small_k, "FOG should suppress power at large k"


# ============================================================
# Alcock-Paczynski tests
# ============================================================

class TestAlcockPaczynski:
    """Tests for the Alcock-Paczynski effect."""

    def test_unity_scaling_same_cosmology(self):
        """When reference and true cosmology are the same, alpha should be 1."""
        cosmo = {'H0': 70.0, 'Omega_m': 0.3}
        alpha_para, alpha_perp = alcock_paczynski_alpha(
            parallel=50.0, perpendicular=50.0,
            cosmology_ref=cosmo, cosmology_true=cosmo,
        )
        np.testing.assert_allclose(alpha_para, 1.0)
        np.testing.assert_allclose(alpha_perp, 1.0)

    def test_scaling_with_different_H0(self):
        """Different H0 should produce non-unity scaling."""
        cosmo_ref = {'H0': 70.0, 'Omega_m': 0.3}
        cosmo_true = {'H0': 67.0, 'Omega_m': 0.3}
        alpha_para, alpha_perp = alcock_paczynski_alpha(
            parallel=50.0, perpendicular=50.0,
            cosmology_ref=cosmo_ref, cosmology_true=cosmo_true,
        )
        expected_ratio = 70.0 / 67.0
        np.testing.assert_allclose(alpha_para, expected_ratio, rtol=1e-10)
        np.testing.assert_allclose(alpha_perp, expected_ratio, rtol=1e-10)

    def test_ap_power_spectrum_jacobian(self):
        """AP-transformed P(k) should include the Jacobian factor."""
        k = np.array([0.1, 0.2])
        mu = np.array([0.5, 0.5])
        P_true = np.array([1000.0, 500.0])
        alpha_perp = 1.02
        alpha_para = 0.98

        P_obs = ap_power_spectrum(k, mu, P_true, alpha_perp, alpha_para)
        jacobian = (alpha_para * alpha_perp**2) ** (-1)
        np.testing.assert_allclose(P_obs, P_true * jacobian, rtol=1e-10)

    def test_ap_power_spectrum_unity_for_identity(self):
        """With alpha=1, AP power should equal true power."""
        k = np.array([0.1, 0.2])
        mu = np.array([0.3, 0.7])
        P_true = np.array([1000.0, 500.0])

        P_obs = ap_power_spectrum(k, mu, P_true, alpha_perp=1.0, alpha_para=1.0)
        np.testing.assert_allclose(P_obs, P_true, rtol=1e-10)
