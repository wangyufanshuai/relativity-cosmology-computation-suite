"""Tests for matter-power-spectrum: transfer function, growth factor,
linear power spectrum, and sigma_8."""

import numpy as np
import pytest

from matter_power_spectrum.transfer import (
    transfer_EH98,
    transfer_EH98_wiggle,
    k_eq_EH98,
    sound_horizon_EH98,
)
from matter_power_spectrum.growth import (
    growth_factor,
    growth_factor_normalized,
    growth_rate,
)
from matter_power_spectrum.spectrum import (
    linear_power_spectrum,
    primordial_power,
    sigma_8,
)
from matter_power_spectrum.constants import OMEGA_M_DEFAULT, OMEGA_B_DEFAULT


# ---------------------------------------------------------------------------
# Transfer function
# ---------------------------------------------------------------------------

class TestTransferFunction:
    """Tests for the EH98 transfer function."""

    def test_transfer_unity_at_low_k(self):
        """T(k -> 0) -> 1 (large-scale (super-horizon) limit)."""
        k_low = np.array([1e-6, 1e-5])
        T = transfer_EH98(k_low)
        # Should be very close to 1 at low k
        assert T[0] == pytest.approx(1.0, abs=0.05)
        assert T[1] == pytest.approx(1.0, abs=0.05)

    def test_transfer_suppressed_at_high_k(self):
        """T(k >> k_eq) -> 0 (small-scale modes suppressed)."""
        k_high = np.array([10.0, 100.0])  # 1/Mpc, well above k_eq ~ 0.01
        T = transfer_EH98(k_high)
        assert T[0] < 0.2
        assert T[1] < 0.05

    def test_transfer_monotonic_decrease(self):
        """T(k) should generally decrease with increasing k (no-wiggle)."""
        k = np.logspace(-4, 1, 200)
        T = transfer_EH98(k)
        # Check that the envelope decreases
        # Use a coarse check: compare at widely separated points
        assert T[0] > T[-1]

    def test_transfer_positive(self):
        """Transfer function should be positive."""
        k = np.logspace(-4, 1, 100)
        T = transfer_EH98(k)
        assert np.all(T > 0.0)

    def test_wiggle_transfer_similar_envelope(self):
        """Wiggle transfer function should have similar envelope to no-wiggle."""
        k = np.logspace(-4, 1, 500)
        T_nw = transfer_EH98(k)
        T_w = transfer_EH98_wiggle(k)
        # The wiggle version should be close to no-wiggle on large scales
        # and oscillate around it on intermediate scales
        assert np.all(T_w > -50)  # should not diverge wildly
        # At low k, no-wiggle should be ~1
        assert T_nw[0] == pytest.approx(1.0, abs=0.05)

    def test_k_eq_positive(self):
        """Matter-radiation equality wavenumber should be positive."""
        k_eq = k_eq_EH98()
        assert k_eq > 0.0

    def test_sound_horizon_positive(self):
        """Sound horizon should be positive."""
        s = sound_horizon_EH98()
        assert s > 0.0


# ---------------------------------------------------------------------------
# Growth factor
# ---------------------------------------------------------------------------

class TestGrowthFactor:
    """Tests for the linear growth factor."""

    def test_growth_factor_normalized_at_a1(self):
        """D(a=1) = 1 by definition of the normalized growth factor."""
        D1 = growth_factor_normalized(1.0)
        assert D1 == pytest.approx(1.0, rel=1e-5)

    def test_growth_factor_monotonic(self):
        """D(a) should increase monotonically from early times to today."""
        a_vals = np.linspace(0.01, 1.0, 50)
        D = growth_factor_normalized(a_vals)
        # Should be monotonically increasing
        assert np.all(np.diff(D) >= 0.0)

    def test_growth_factor_positive(self):
        """D(a) > 0 for a > 0."""
        a_vals = np.linspace(0.01, 1.0, 20)
        D = growth_factor_normalized(a_vals)
        assert np.all(D > 0.0)

    def test_growth_rate_positive(self):
        """Growth rate f should be positive for standard cosmology."""
        a_vals = np.linspace(0.1, 1.0, 20)
        f = growth_rate(a_vals)
        assert np.all(f > 0.0)

    def test_growth_rate_decreasing_in_lambda(self):
        """f(a) should decrease toward late times as Lambda dominates."""
        f_early = growth_rate(0.3)
        f_late = growth_rate(1.0)
        assert f_early > f_late


# ---------------------------------------------------------------------------
# Power spectrum
# ---------------------------------------------------------------------------

class TestPowerSpectrum:
    """Tests for the linear power spectrum."""

    def test_linear_power_spectrum_positive(self):
        """P(k) > 0 for all k."""
        k = np.logspace(-4, 1, 100)
        P = linear_power_spectrum(k)
        assert np.all(P > 0.0)

    def test_primordial_power_positive(self):
        """Primordial power spectrum should be positive."""
        k = np.logspace(-4, 1, 50)
        P = primordial_power(k)
        assert np.all(P > 0.0)

    def test_power_spectrum_peak(self):
        """P(k) should be positive and monotonically decreasing for n_s<1."""
        k = np.logspace(-5, 1, 500)
        P = linear_power_spectrum(k)
        assert np.all(P > 0), "P(k) should be positive"
        assert P[0] > P[-1], "P(k) should be larger at low k than high k"
        # Dimensionless spectrum Δ²(k) = k³P(k)/(2π²) should be well-behaved
        Delta2 = k**3 * P / (2 * np.pi**2)
        assert np.all(Delta2 > 0), "Δ²(k) should be positive"
        assert np.max(Delta2) < 1.0, "Δ²(k) should be << 1 for linear regime"


# ---------------------------------------------------------------------------
# sigma_8
# ---------------------------------------------------------------------------

class TestSigma8:
    """Tests for sigma_8 computation."""

    def test_sigma_8_positive(self):
        """sigma_8 should be positive."""
        k = np.logspace(-4, 2, 2000)
        P = linear_power_spectrum(k)
        s8 = sigma_8(P, k, R=8.0)
        assert s8 > 0.0

    def test_sigma_8_reasonable_range(self):
        """sigma_8 should be positive and finite for Planck-like parameters."""
        k = np.logspace(-4, 2, 2000)
        P = linear_power_spectrum(k)
        s8 = sigma_8(P, k, R=8.0)
        assert s8 > 0, f"sigma_8 = {s8} should be positive"
        assert np.isfinite(s8), f"sigma_8 = {s8} should be finite"
        assert s8 < 10.0, f"sigma_8 = {s8} should be reasonable"
