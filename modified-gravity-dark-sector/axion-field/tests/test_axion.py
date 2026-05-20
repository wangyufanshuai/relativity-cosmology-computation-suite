"""Tests for axion field dynamics simulator."""

import numpy as np
import pytest

from axion_field.misalignment import (
    axion_mass_temperature,
    oscillation_temperature,
    axion_density,
    axion_mass_from_density,
)


class TestAxionMass:
    """Tests for temperature-dependent axion mass."""

    def test_mass_at_low_T(self):
        """At T << Lambda_QCD, mass equals zero-temperature mass."""
        m_a = 1e-5
        T = 1e6  # Well below Lambda_QCD = 1e9
        m_T = axion_mass_temperature(T, m_a)
        assert abs(m_T - m_a) / m_a < 1e-10

    def test_mass_suppressed_at_high_T(self):
        """At T >> Lambda_QCD, mass is suppressed."""
        m_a = 1e-5
        T = 1e12  # Well above Lambda_QCD
        m_T = axion_mass_temperature(T, m_a)
        assert m_T < m_a

    def test_mass_increases_as_T_decreases(self):
        """Mass should increase as universe cools."""
        m_a = 1e-5
        T_vals = np.array([1e12, 1e10, 1e8])
        m_vals = axion_mass_temperature(T_vals, m_a)
        assert np.all(np.diff(m_vals) > 0), "Mass should increase as T decreases"


class TestOscillationTemperature:
    """Tests for oscillation temperature."""

    def test_tosc_positive(self):
        """Oscillation temperature should be positive."""
        T_osc = oscillation_temperature(1e-5)
        assert T_osc > 0

    def test_tosc_increases_with_mass(self):
        """Heavier axions start oscillating earlier (higher T_osc)."""
        T1 = oscillation_temperature(1e-6)
        T2 = oscillation_temperature(1e-4)
        assert T2 > T1, f"T_osc(m=1e-4)={T2} should be > T_osc(m=1e-6)={T1}"

    def test_tosc_around_qcd_scale(self):
        """For typical axion masses, T_osc should be near QCD scale."""
        T_osc = oscillation_temperature(5e-6)
        assert 1e7 < T_osc < 1e11, f"T_osc={T_osc:.2e} not near QCD scale"


class TestAxionDensity:
    """Tests for axion relic density."""

    def test_density_positive(self):
        """Omega_a h^2 should be positive."""
        Omega_h2 = axion_density(5e-6)
        assert Omega_h2 > 0

    def test_density_around_observed_for_typical_mass(self):
        """For m_a ~ 5e-6 eV and theta_i=1, Omega_a h^2 ~ 0.12."""
        Omega_h2 = axion_density(5e-6, theta_i=1.0)
        assert 0.01 < Omega_h2 < 1.0, f"Omega_a h^2 = {Omega_h2:.4f}"

    def test_density_increases_with_theta(self):
        """Larger misalignment angle gives more axion DM."""
        O1 = axion_density(5e-6, theta_i=0.5)
        O2 = axion_density(5e-6, theta_i=1.0)
        assert O2 > O1

    def test_density_decreases_with_mass(self):
        """Heavier axions contribute less from misalignment (wider decay constant)."""
        O1 = axion_density(1e-6, theta_i=1.0)
        O2 = axion_density(1e-4, theta_i=1.0)
        assert O2 < O1

    def test_mass_roundtrip(self):
        """mass_from_density should invert density calculation."""
        m_a_orig = 5e-6
        Omega_h2 = axion_density(m_a_orig, theta_i=1.0)
        m_a_recovered = axion_mass_from_density(Omega_h2, theta_i=1.0)
        assert abs(m_a_recovered / m_a_orig - 1.0) < 0.01
