"""Tests for sn-cosmology: supernova cosmology distance calculations."""

import numpy as np
import pytest

from sn_cosmology import (
    hubble_distance,
    E,
    comoving_distance,
    luminosity_distance,
    distance_modulus,
    angular_diameter_distance,
    chi_squared,
    salt2_light_curve,
)


class TestHubbleDistance:
    """Test Hubble distance calculation."""

    def test_basic(self):
        """d_H = c / H0."""
        d_H = hubble_distance(67.4)
        expected = 299792.458 / 67.4
        assert d_H == pytest.approx(expected, rel=1e-6)

    def test_positive(self):
        """Hubble distance should be positive."""
        assert hubble_distance(70.0) > 0


class TestEz:
    """Test dimensionless Hubble parameter E(z)."""

    def test_ez_at_zero(self):
        """E(z=0) = 1 for standard cosmology."""
        val = E(0.0)
        assert val == pytest.approx(1.0, rel=1e-6)

    def test_ez_increases_with_z(self):
        """E(z) should increase with redshift in LCDM."""
        assert E(1.0) > E(0.0)
        assert E(2.0) > E(1.0)


class TestDistances:
    """Test cosmological distance calculations."""

    def test_comoving_distance_positive(self):
        """Comoving distance should be positive."""
        d_C = comoving_distance(1.0)
        assert d_C > 0

    def test_luminosity_distance_formula(self):
        """d_L = (1+z) * d_C."""
        z = 0.5
        d_C = comoving_distance(z)
        d_L = luminosity_distance(z)
        assert d_L == pytest.approx((1 + z) * d_C, rel=1e-6)

    def test_angular_diameter_distance_formula(self):
        """d_A = d_C / (1+z)."""
        z = 0.5
        d_C = comoving_distance(z)
        d_A = angular_diameter_distance(z)
        assert d_A == pytest.approx(d_C / (1 + z), rel=1e-6)

    def test_distance_modulus_at_low_z(self):
        """At low z, mu should be approximately 5*log10(c*z/H0) + 25."""
        z = 0.01
        mu = distance_modulus(z)
        assert 30 < mu < 40  # rough range


class TestChiSquared:
    """Test chi-squared computation."""

    def test_chi_squared_zero_for_perfect_data(self):
        """Chi-squared should be ~0 when theory matches observation."""
        z = np.array([0.1, 0.5, 1.0])
        mu = np.array([distance_modulus(zi) for zi in z])
        mu_err = np.ones(3)
        chi2 = chi_squared(z, mu, mu_err)
        assert chi2 == pytest.approx(0.0, abs=0.5)

    def test_chi_squared_positive(self):
        """Chi-squared should be non-negative."""
        z = np.array([0.1, 0.5])
        mu = np.array([35.0, 42.0])
        mu_err = np.array([0.2, 0.3])
        chi2 = chi_squared(z, mu, mu_err)
        assert chi2 >= 0


class TestSALT2:
    """Test SALT2 light curve model."""

    def test_salt2_returns_magnitudes(self):
        """SALT2 should return magnitude values."""
        t = np.linspace(-20, 20, 50)
        m = salt2_light_curve(t)
        assert len(m) == 50
        assert np.all(np.isfinite(m))

    def test_salt2_peak_brightest(self):
        """Peak should be near t=0 (brightest = lowest magnitude)."""
        t = np.linspace(-20, 20, 100)
        m = salt2_light_curve(t, t0=0.0, x0=1.0)
        assert m[50] < m[0]  # peak is brighter
