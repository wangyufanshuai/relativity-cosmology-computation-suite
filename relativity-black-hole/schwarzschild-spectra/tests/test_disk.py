"""Tests for Schwarzschild accretion disk radiation."""

import numpy as np
import pytest
from schwarzschild_spectra.disk import (
    disk_temperature_profile,
    radiative_efficiency,
    local_blackbody,
    peak_temperature,
    eddington_luminosity,
    eddington_accretion_rate,
    observed_spectrum,
)
from schwarzschild_spectra.constants import G, C, SIGMA_SB, M_SUN, K_B, H


class TestDiskTemperature:
    def test_zero_at_isco(self):
        """Temperature should be zero at r = r_isco (zero-torque BC)."""
        T = disk_temperature_profile(np.array([6.0]), M_SUN, 1e16)
        assert T[0] == 0.0

    def test_positive_beyond_isco(self):
        """Temperature should be positive for r > r_isco."""
        r = np.array([7.0, 10.0, 50.0, 100.0])
        T = disk_temperature_profile(r, M_SUN, 1e16)
        assert np.all(T > 0)

    def test_peak_near_isco(self):
        """Maximum temperature should be near (but outside) r_isco."""
        r = np.linspace(6.01, 100, 1000)
        T = disk_temperature_profile(r, 10 * M_SUN, 1e17)
        r_max = r[np.argmax(T)]
        # Peak should be around r ~ 8-10 M
        assert 6.0 < r_max < 15.0

    def test_decreases_at_large_r(self):
        """Temperature should decrease at large r (approaches zero)."""
        r_far = np.array([500.0, 1000.0, 5000.0])
        T = disk_temperature_profile(r_far, M_SUN, 1e16)
        assert np.all(np.diff(T) < 0)


class TestRadiativeEfficiency:
    def test_schwarzschild_value(self):
        """Schwarzschild ISCO=6M gives η ≈ 5.72%."""
        eta = radiative_efficiency(r_isco=6.0)
        assert abs(eta - 0.0572) < 0.001


class TestBlackbody:
    def test_peak_wien(self):
        """Peak frequency should follow Wien's law: ν_max ≈ 2.82 kT/h."""
        T = 1e7  # K
        nu = np.logspace(10, 20, 10000)
        B = local_blackbody(nu, T)
        nu_peak = nu[np.argmax(B)]
        expected = 2.82 * K_B * T / H
        assert abs(nu_peak / expected - 1.0) < 0.1

    def test_stefan_boltzmann(self):
        """Integral of B_nu should give σT⁴/π."""
        T = 1e6
        nu = np.logspace(5, 22, 100000)
        B = local_blackbody(nu, T)
        integral = np.trapezoid(B, nu)
        expected = SIGMA_SB * T**4 / np.pi
        assert abs(integral / expected - 1.0) < 0.05


class TestEddington:
    def test_eddington_luminosity_solar(self):
        """Solar Eddington luminosity ~1.26 × 10³¹ W."""
        L = eddington_luminosity(M_SUN)
        assert abs(L / 1.26e31 - 1.0) < 0.1

    def test_eddington_accretion_positive(self):
        """Eddington accretion rate should be positive."""
        Mdot = eddington_accretion_rate(10 * M_SUN)
        assert Mdot > 0


class TestObservedSpectrum:
    def test_spectrum_nonzero(self):
        """Observed spectrum should be nonzero for reasonable parameters."""
        nu = np.logspace(10, 19, 100)
        F = observed_spectrum(nu, 10 * M_SUN, 1e17, distance=1e20)
        assert np.any(F > 0)

    def test_face_on_brighter(self):
        """Face-on disk should be brighter than edge-on."""
        nu = np.logspace(12, 18, 50)
        F_face = observed_spectrum(nu, 10 * M_SUN, 1e17, inclination=0.0, distance=1e20)
        F_edge = observed_spectrum(nu, 10 * M_SUN, 1e17, inclination=np.pi / 2 * 0.9, distance=1e20)
        assert np.sum(F_face) > np.sum(F_edge)
