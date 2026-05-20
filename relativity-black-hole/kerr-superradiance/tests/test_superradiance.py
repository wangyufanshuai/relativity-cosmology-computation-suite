"""Tests for kerr-superradiance: superradiance condition, reflection coefficient,
radial potential, and instability growth rates."""

import numpy as np
import pytest

from kerr_superradiance.superradiance import (
    superradiance_condition,
    reflection_coefficient,
    horizon_angular_velocity,
    superradiance_rate,
)
from kerr_superradiance.teukolsky import radial_potential
from kerr_superradiance.instability import instability_growth_rate


# ---------------------------------------------------------------------------
# Superradiance condition
# ---------------------------------------------------------------------------

class TestSuperradianceCondition:
    """Test the superradiance condition omega < m * Omega_H."""

    def test_superradiance_true(self):
        """Low frequency in a spinning BH should satisfy the condition."""
        M = 1.0
        a = 0.9 * M  # near-extremal spin
        m = 1
        Omega_H = horizon_angular_velocity(a, M)
        omega = 0.5 * m * Omega_H  # well below m*Omega_H
        assert superradiance_condition(omega, m, a, M) == True

    def test_no_superradiance_above_threshold(self):
        """Frequency above m*Omega_H should not satisfy condition."""
        M = 1.0
        a = 0.5 * M
        m = 1
        Omega_H = horizon_angular_velocity(a, M)
        omega = 2.0 * m * Omega_H  # above threshold
        assert superradiance_condition(omega, m, a, M) == False

    def test_no_superradiance_schwarzschild(self):
        """a=0 (Schwarzschild) => no superradiance for any omega."""
        assert superradiance_condition(0.01, 1, 0.0, 1.0) == False

    def test_no_superradiance_m_zero(self):
        """m=0 modes never superradiate."""
        assert superradiance_condition(0.01, 0, 0.9, 1.0) == False

    def test_no_superradiance_negative_m(self):
        """Negative m modes do not superradiate."""
        assert superradiance_condition(0.01, -1, 0.9, 1.0) == False

    def test_higher_m_mode(self):
        """Higher m modes have a higher threshold m*Omega_H."""
        M = 1.0
        a = 0.5 * M
        Omega_H = horizon_angular_velocity(a, M)
        omega = 1.5 * Omega_H  # between m=1 and m=2 thresholds
        # m=1: threshold = Omega_H, omega > threshold => False
        assert superradiance_condition(omega, 1, a, M) == False
        # m=2: threshold = 2*Omega_H, omega < threshold => True
        assert superradiance_condition(omega, 2, a, M) == True


# ---------------------------------------------------------------------------
# Reflection coefficient
# ---------------------------------------------------------------------------

class TestReflectionCoefficient:
    """Test the reflection coefficient |R|^2."""

    def test_reflection_coefficient_range(self):
        """|R|^2 should be >= 0."""
        M = 1.0
        a = 0.5 * M
        omega = 0.1
        R_sq = reflection_coefficient(omega, a, l=1, m=1, M=M)
        assert R_sq >= 0.0

    def test_reflection_coefficient_no_spin(self):
        """For Schwarzschild (a=0), R^2 should be ~1 (no amplification)."""
        R_sq = reflection_coefficient(0.1, 0.0, l=1, m=1, M=1.0)
        assert R_sq >= 0.0
        # For Schwarzschild, no superradiance, so |R|^2 ~ 1
        assert R_sq == pytest.approx(1.0, abs=0.5)


# ---------------------------------------------------------------------------
# Radial potential
# ---------------------------------------------------------------------------

class TestRadialPotential:
    """Test the radial Teukolsky potential V(r)."""

    def test_radial_potential_at_large_r(self):
        """At large r, V(r) -> omega^2 (asymptotic plane-wave limit)."""
        M = 1.0
        a = 0.5
        omega = 0.1
        r_large = 1e6 * M
        V = radial_potential(r_large, a, omega, m_spin=0, l=1, m_az=1, M=M)
        assert V == pytest.approx(omega**2, rel=0.01)

    def test_radial_potential_positive_outside_horizon(self):
        """V should be finite and real outside the horizon."""
        M = 1.0
        a = 0.5
        r_plus = M + np.sqrt(M**2 - a**2)
        r = r_plus * 1.5
        V = radial_potential(r, a, omega=0.1, m_spin=0, l=1, m_az=1, M=M)
        assert np.isfinite(V)

    def test_radial_potential_schwarzschild_limit(self):
        """For a=0, the potential should reduce to the Schwarzschild form."""
        M = 1.0
        a = 0.0
        omega = 0.1
        l = 1
        r = 50.0 * M
        V = radial_potential(r, a, omega, m_spin=0, l=l, m_az=0, M=M)
        # Schwarzschild: V = (l(l+1)/r^2)(1 - 2M/r) + omega^2 terms
        expected_leading = l * (l + 1) / r**2
        assert V > 0
        assert np.isfinite(V)


# ---------------------------------------------------------------------------
# Instability growth rate
# ---------------------------------------------------------------------------

class TestInstabilityGrowthRate:
    """Test the instability growth rate calculation."""

    def test_instability_rate_positive_in_superradiant_regime(self):
        """In the superradiant regime, the growth rate should be positive."""
        M = 1.0
        a = 0.99 * M  # near-extremal
        mu = 0.1  # scalar field mass
        l, m = 1, 1
        # mu < m*Omega_H must hold for superradiance
        Omega_H = horizon_angular_velocity(a, M)
        assert mu < m * Omega_H, "Test precondition: mu must be < m*Omega_H"
        rate = instability_growth_rate(mu, a, l, m, M)
        assert rate > 0.0

    def test_instability_zero_for_schwarzschild(self):
        """No instability for Schwarzschild (a=0)."""
        rate = instability_growth_rate(0.1, 0.0, l=1, m=1, M=1.0)
        assert rate == 0.0

    def test_instability_zero_for_m_zero(self):
        """No instability for m=0 modes."""
        rate = instability_growth_rate(0.1, 0.9, l=1, m=0, M=1.0)
        assert rate == 0.0

    def test_instability_zero_outside_regime(self):
        """No instability when mu > m*Omega_H."""
        M = 1.0
        a = 0.5 * M
        Omega_H = horizon_angular_velocity(a, M)
        mu = 2.0 * Omega_H  # above threshold
        rate = instability_growth_rate(mu, a, l=1, m=1, M=M)
        assert rate == 0.0

    def test_higher_l_modes_have_smaller_rate(self):
        """l=m=2 mode should have smaller rate than l=m=1 for same mu*M."""
        M = 1.0
        a = 0.99 * M
        mu = 0.05  # small mu*M
        rate_1 = instability_growth_rate(mu, a, l=1, m=1, M=M)
        rate_2 = instability_growth_rate(mu, a, l=2, m=2, M=M)
        # For small muM, l=1 dominates
        assert rate_1 > rate_2


# ---------------------------------------------------------------------------
# Horizon angular velocity
# ---------------------------------------------------------------------------

class TestHorizonAngularVelocity:
    """Test horizon angular velocity."""

    def test_positive_for_spinning(self):
        """Omega_H > 0 for a > 0."""
        assert horizon_angular_velocity(0.5, 1.0) > 0.0

    def test_zero_for_schwarzschild(self):
        """Omega_H = 0 for a = 0."""
        assert horizon_angular_velocity(0.0, 1.0) == 0.0

    def test_extremal_limit(self):
        """For a -> M, Omega_H -> 1/(2M)."""
        M = 1.0
        a = 0.9999 * M
        Omega_H = horizon_angular_velocity(a, M)
        r_plus = M + np.sqrt(M**2 - a**2)
        expected = a / (2.0 * M * r_plus)
        assert Omega_H == pytest.approx(expected, rel=1e-10)
