"""Tests for Hawking radiation simulator."""

import numpy as np
import pytest
from hawking_radiation.thermodynamics import (
    hawking_temperature,
    bh_entropy,
    bh_heat_capacity,
    bh_luminosity,
    bh_lifetime,
    emission_spectrum,
    compute_page_curve,
)
from hawking_radiation.constants import M_SUN, YEAR


class TestHawkingTemperature:
    def test_solar_mass(self):
        """1 solar mass BH: T ≈ 62 nK."""
        T = hawking_temperature(M_SUN)
        assert 50e-9 < T < 80e-9

    def test_smaller_is_hotter(self):
        """Smaller BH should be hotter: T ∝ 1/M."""
        T1 = hawking_temperature(M_SUN)
        T2 = hawking_temperature(0.1 * M_SUN)
        assert T2 > T1

    def test_scaling(self):
        """T should scale exactly as 1/M."""
        T1 = hawking_temperature(M_SUN)
        T2 = hawking_temperature(2 * M_SUN)
        assert abs(T1 / T2 - 2.0) < 1e-10


class TestEntropy:
    def test_positive(self):
        S = bh_entropy(M_SUN)
        assert S > 0

    def test_enormous(self):
        """Solar mass BH entropy should be ~10^77 k_B."""
        S = bh_entropy(M_SUN)
        assert 1e75 < S < 1e80


class TestHeatCapacity:
    def test_negative(self):
        """BH heat capacity is always negative."""
        C = bh_heat_capacity(M_SUN)
        assert C < 0


class TestLuminosity:
    def test_increases_as_shrinks(self):
        """L ∝ 1/M²: smaller BH radiates more."""
        L1 = bh_luminosity(M_SUN)
        L2 = bh_luminosity(0.5 * M_SUN)
        assert L2 > L1
        assert abs(L2 / L1 - 4.0) < 1e-10  # (M1/M2)² = 4


class TestLifetime:
    def test_solar_mass(self):
        """Solar mass BH lifetime ~10^67 years."""
        t = bh_lifetime(M_SUN)
        t_years = t / YEAR
        assert 1e65 < t_years < 1e70


class TestEmission:
    def test_thermal_spectrum(self):
        """Emission rate should be positive for finite omega."""
        rate = emission_spectrum(1e10, M_SUN)
        assert rate > 0

    def test_suppressed_at_high_omega(self):
        """High energy emission should be exponentially suppressed."""
        rate_low = emission_spectrum(1e6, M_SUN)
        rate_high = emission_spectrum(1e15, M_SUN)
        assert rate_high < rate_low


class TestPageCurve:
    def test_bh_entropy_decreases(self):
        """S_BH should decrease during evaporation."""
        # Use tiny BH for fast computation
        M_small = 1e10  # kg (tiny BH)
        t_evap = bh_lifetime(M_small)
        result = compute_page_curve(M_small, n_points=50)
        assert result["S_bh"][-1] < result["S_bh"][0]

    def test_page_time_exists(self):
        """Page time should be a reasonable fraction of evaporation time."""
        M_small = 1e10
        result = compute_page_curve(M_small, n_points=50)
        assert 0.1 < result["page_fraction"] < 0.9
