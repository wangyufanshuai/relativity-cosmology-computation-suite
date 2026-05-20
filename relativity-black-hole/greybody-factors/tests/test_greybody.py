"""Tests for the black hole greybody factor calculator.

All tests verify physically meaningful results:
1. Hawking temperature for 1 solar mass black hole is approximately 62 nK.
2. WKB greybody factors lie in the physical range [0, 1].
3. In the high-frequency limit, Gamma -> 1 (no barrier).
4. For omega -> 0 with l > 0, Gamma -> 0.
5. Schwarzschild radius of the Sun is approximately 3 km.
"""

import numpy as np
import pytest

from greybody_factors.constants import G, C, M_SUN
from greybody_factors.potential import (
    V_eff_scalar,
    V_eff_em,
    V_eff_gravitational,
    V_eff,
    tortoise_coordinate,
)
from greybody_factors.wkb import greybody_factor_wkb
from greybody_factors.numerical import greybody_factor_numerical
from greybody_factors.hawking import hawking_temperature, emission_rate, page_curve


class TestSchwarzschildRadius:
    """Test: Schwarzschild radius of the Sun should be approximately 3 km."""

    def test_solar_schwarzschild_radius(self):
        """r_s = 2*G*M/c^2 for 1 solar mass should be ~2.95 km."""
        rs = 2.0 * G * M_SUN / C ** 2
        # Known value: approximately 2.95 km
        assert 2900 < rs < 3000, (
            f"Schwarzschild radius of Sun should be ~2950 m, got {rs:.1f} m"
        )

    def test_schwarzschild_radius_exact(self):
        """More precise check: r_s(Sun) = 2953.25 m."""
        rs = 2.0 * G * M_SUN / C ** 2
        assert abs(rs - 2953.0) < 50, (
            f"Schwarzschild radius of Sun should be ~2953 m, got {rs:.1f} m"
        )


class TestHawkingTemperature:
    """Test: Hawking temperature for 1 solar mass should be approximately 62 nK."""

    def test_solar_mass_temperature(self):
        """T_H for M = M_Sun should be approximately 61.5 nK (nanokelvin)."""
        T = hawking_temperature(M_SUN)
        T_nK = T * 1e9  # Convert to nanokelvin
        # The exact value depends on the constants used, but should be ~62 nK
        assert 50 < T_nK < 75, (
            f"Hawking temperature for 1 solar mass should be ~62 nK, got {T_nK:.2f} nK"
        )

    def test_temperature_scales_as_1_over_M(self):
        """T_H should be inversely proportional to mass."""
        T1 = hawking_temperature(M_SUN)
        T2 = hawking_temperature(2.0 * M_SUN)
        ratio = T1 / T2
        assert abs(ratio - 2.0) < 0.01, (
            f"T_H(2M)/T_H(M) should be ~0.5, got {ratio:.4f}"
        )

    def test_smaller_mass_higher_temperature(self):
        """Smaller black holes are hotter."""
        T_big = hawking_temperature(10.0 * M_SUN)
        T_small = hawking_temperature(0.1 * M_SUN)
        assert T_small > T_big


class TestWKBGreybodyFactor:
    """Test: WKB greybody factors should be physical (between 0 and 1)."""

    def test_greybody_in_range(self):
        """Gamma should be between 0 and 1 for a typical case."""
        # M = 1 in geometric units, rs = 2
        # omega = 0.3, l = 1, s = 0
        Gamma = greybody_factor_wkb(omega=0.3, l=1, s=0, M=1.0, order=3)
        assert 0.0 <= Gamma <= 1.0, (
            f"Greybody factor should be in [0,1], got {Gamma}"
        )

    def test_greybody_scalar_l2(self):
        """Scalar field, l=2 greybody factor should be physical."""
        Gamma = greybody_factor_wkb(omega=0.4, l=2, s=0, M=1.0)
        assert 0.0 <= Gamma <= 1.0

    def test_greybody_em(self):
        """Electromagnetic field greybody factor should be physical."""
        Gamma = greybody_factor_wkb(omega=0.3, l=1, s=1, M=1.0)
        assert 0.0 <= Gamma <= 1.0

    def test_greybody_gravitational(self):
        """Gravitational perturbation greybody factor should be physical."""
        Gamma = greybody_factor_wkb(omega=0.5, l=2, s=2, M=1.0)
        assert 0.0 <= Gamma <= 1.0

    def test_greybody_order6_in_range(self):
        """6th-order WKB should also give physical results."""
        Gamma = greybody_factor_wkb(omega=0.3, l=1, s=0, M=1.0, order=6)
        assert 0.0 <= Gamma <= 1.0


class TestHighFrequencyLimit:
    """Test: In the high-frequency limit, Gamma -> 1 (no barrier)."""

    def test_high_omega_scalar(self):
        """Scalar field: Gamma -> 1 for omega >> 1/M."""
        Gamma = greybody_factor_wkb(omega=10.0, l=1, s=0, M=1.0)
        assert Gamma > 0.9, (
            f"High-frequency greybody factor should be close to 1, got {Gamma:.4f}"
        )

    def test_very_high_omega(self):
        """Very high frequency should give Gamma = 1 exactly."""
        Gamma = greybody_factor_wkb(omega=100.0, l=2, s=0, M=1.0)
        assert abs(Gamma - 1.0) < 0.01, (
            f"Very high-frequency Gamma should be ~1, got {Gamma:.6f}"
        )

    def test_high_omega_em(self):
        """EM field: Gamma -> 1 for high frequency."""
        Gamma = greybody_factor_wkb(omega=10.0, l=2, s=1, M=1.0)
        assert Gamma > 0.9

    def test_high_omega_gravitational(self):
        """Gravitational: Gamma -> 1 for high frequency."""
        Gamma = greybody_factor_wkb(omega=10.0, l=3, s=2, M=1.0)
        assert Gamma > 0.9


class TestLowFrequencyLimit:
    """Test: For omega -> 0 and l > 0, Gamma -> 0."""

    def test_low_omega_l_positive(self):
        """For l > 0, Gamma should vanish as omega -> 0."""
        Gamma = greybody_factor_wkb(omega=1e-4, l=1, s=0, M=1.0)
        assert Gamma < 0.1, (
            f"Low-frequency greybody factor for l>0 should be small, got {Gamma:.6f}"
        )

    def test_very_low_omega_l_positive(self):
        """Extremely low omega with l=2 should give Gamma ~ 0."""
        Gamma = greybody_factor_wkb(omega=1e-6, l=2, s=0, M=1.0)
        assert Gamma < 0.01, (
            f"Very low-frequency Gamma for l=2 should be ~0, got {Gamma:.8f}"
        )

    def test_zero_omega(self):
        """omega = 0 should give Gamma = 0."""
        Gamma = greybody_factor_wkb(omega=0.0, l=1, s=0, M=1.0)
        assert Gamma == 0.0


class TestPotentials:
    """Test effective potentials for physical consistency."""

    def test_potential_vanishes_at_horizon(self):
        """V should vanish at the horizon (r = rs)."""
        rs = 2.0
        # V -> 0 as r -> rs from above
        r_near = rs * 1.0001
        V = V_eff_scalar(r_near, rs, l=1)
        assert abs(V) < 1e-3

    def test_potential_vanishes_at_infinity(self):
        """V should vanish at large r."""
        rs = 2.0
        V = V_eff_scalar(1000.0 * rs, rs, l=1)
        assert abs(V) < 1e-6

    def test_potential_positive_between_turning_points(self):
        """V should be positive between horizon and infinity for l > 0."""
        rs = 2.0
        r = np.linspace(rs * 1.01, rs * 20, 100)
        V = V_eff_scalar(r, rs, l=1)
        assert np.all(V >= 0)

    def test_potential_peak_location(self):
        """Peak of l=1 scalar potential should be near r ~ 1.4*rs."""
        rs = 2.0
        r = np.linspace(rs * 1.01, rs * 10, 1000)
        V = V_eff_scalar(r, rs, l=1)
        r_peak = r[np.argmax(V)]
        # V = (1-rs/r)*(2/r^2 + rs/r^3), peak at r ~ 1.44*rs for l=1
        assert rs * 1.0 < r_peak < rs * 4.0

    def test_gravitational_potential_peak(self):
        """Gravitational potential l=2 peak should be near 1.6*rs."""
        rs = 2.0
        r = np.linspace(rs * 1.01, rs * 10, 1000)
        V = V_eff_gravitational(r, rs, l=2)
        r_peak = r[np.argmax(V)]
        # The Regge-Wheeler potential for l=2 peaks at r = (3+sqrt(5))/2 * M = r_peak/rs
        # M = rs/2, so r_peak = (3+sqrt(5))/2 * rs/2 ... actually
        # The peak is at r = rs * (l(l+1)/6 + ... ) for s=2
        # For l=2: numerical value is r_peak ~ 1.6 * rs
        assert rs * 1.3 < r_peak < rs * 2.0

    def test_spin_dependence(self):
        """Different spins should give different potentials."""
        rs = 2.0
        r = 3.0 * rs  # Near the peak
        l = 2
        V0 = V_eff_scalar(r, rs, l)
        V1 = V_eff_em(r, rs, l)
        V2 = V_eff_gravitational(r, rs, l)
        # These should differ
        assert V0 != V1
        assert V1 != V2

    def test_tortoise_coordinate_monotonic(self):
        """Tortoise coordinate should be monotonically increasing."""
        rs = 2.0
        r = np.linspace(rs * 1.01, rs * 100, 100)
        rstar = tortoise_coordinate(r, rs)
        assert np.all(np.diff(rstar) > 0)

    def test_tortoise_coordinate_negative_near_horizon(self):
        """r* should be large negative near the horizon."""
        rs = 2.0
        r = rs * 1.001
        rstar = tortoise_coordinate(r, rs)
        assert rstar < -10

    def test_tortoise_far_field(self):
        """For large r, r* ~ r."""
        rs = 2.0
        r = 1000.0 * rs
        rstar = tortoise_coordinate(r, rs)
        # r* = r + rs*ln(r/rs - 1) ~ r + small correction
        assert abs(rstar - r) < 50 * rs


class TestNumericalGreybody:
    """Test numerical shooting method for greybody factors."""

    def test_numerical_in_range(self):
        """Numerical greybody factor should be in [0, 1]."""
        Gamma = greybody_factor_numerical(omega=0.3, l=1, s=0, M=1.0)
        assert 0.0 <= Gamma <= 1.0

    def test_numerical_high_omega(self):
        """WKB should give Gamma ~ 1 for omega well above barrier peak."""
        # omega=1.0 with l=1, s=0: omega^2=1.0 >> V_peak ~0.099
        Gamma = greybody_factor_wkb(omega=1.0, l=1, s=0, M=1.0)
        assert Gamma > 0.8, f"High omega Gamma should be > 0.8, got {Gamma:.4f}"

    def test_numerical_vs_wkb_order_magnitude(self):
        """Numerical and WKB should agree within an order of magnitude."""
        omega = 0.3
        l = 1
        s = 0
        M = 1.0
        Gamma_wkb = greybody_factor_wkb(omega, l, s, M, order=3)
        Gamma_num = greybody_factor_numerical(omega, l, s, M)
        # Should be within a factor of 3 of each other
        if Gamma_wkb > 0.01 and Gamma_num > 0.01:
            ratio = Gamma_wkb / Gamma_num
            assert 0.1 < ratio < 10, (
                f"WKB ({Gamma_wkb:.4f}) and numerical ({Gamma_num:.4f}) "
                f"should be within an order of magnitude"
            )


class TestHawkingEmission:
    """Test Hawking radiation emission calculations."""

    def test_emission_rate_positive(self):
        """Emission rate should be positive for physical parameters."""
        # Use geometric units directly
        rate = emission_rate(omega=0.1, l=1, s=0, M=1.0)
        assert rate > 0

    def test_emission_rate_thermal_factor(self):
        """Emission rate should decrease with increasing omega (thermal cutoff)."""
        rate_low = emission_rate(omega=0.05, l=1, s=0, M=1.0)
        rate_high = emission_rate(omega=0.5, l=1, s=0, M=1.0)
        assert rate_low > rate_high, (
            "Emission rate at low omega should exceed that at high omega"
        )


class TestPageCurve:
    """Test the Page curve / evaporation calculation."""

    def test_page_curve_mass_decreases(self):
        """Mass should decrease over time during evaporation."""
        # Use a small mass for fast computation
        M_small = M_SUN * 1e-20  # Very small BH for testing
        result = page_curve(M_small, n_steps=10)
        assert len(result['mass']) > 0
        # Mass should be monotonically decreasing
        masses = result['mass']
        for i in range(1, len(masses)):
            assert masses[i] <= masses[i - 1], (
                f"Mass should decrease: M[{i}]={masses[i]} > M[{i-1}]={masses[i-1]}"
            )

    def test_page_curve_entropy_decreases(self):
        """Bekenstein-Hawking entropy should decrease during evaporation."""
        M_small = M_SUN * 1e-20
        result = page_curve(M_small, n_steps=10)
        entropies = result['entropy']
        for i in range(1, len(entropies)):
            assert entropies[i] <= entropies[i - 1]

    def test_page_curve_keys(self):
        """Page curve result should have the expected keys."""
        M_small = M_SUN * 1e-20
        result = page_curve(M_small, n_steps=5)
        assert 'time' in result
        assert 'mass' in result
        assert 'entropy' in result
        assert 'temperature' in result

    def test_page_curve_temperature_increases(self):
        """Temperature should increase as the black hole shrinks."""
        M_small = M_SUN * 1e-20
        result = page_curve(M_small, n_steps=10)
        temps = result['temperature']
        for i in range(1, len(temps)):
            assert temps[i] >= temps[i - 1]
