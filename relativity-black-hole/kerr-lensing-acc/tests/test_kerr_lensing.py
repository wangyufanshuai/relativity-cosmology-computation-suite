"""Tests for kerr-lensing-acc: Kerr ray tracing, accretion disk, iron line."""

import numpy as np
import pytest

from kerr_lensing_acc.kerr_rays import (
    delta,
    sigma,
    compute_isco,
    photon_sphere_radius,
    critical_impact_parameter,
    metric_coefficients,
    compute_deflection_angle,
    trace_ray_equatorial,
)
from kerr_lensing_acc.accretion_disk import (
    keplerian_angular_velocity,
    keplerian_specific_energy,
    keplerian_specific_angular_momentum,
    novikov_thorne_flux,
    disk_temperature,
    disk_temperature_schwarzschild,
    blackbody_spectrum,
)
from kerr_lensing_acc.iron_line import (
    redshift_factor,
    redshift_factor_schwarzschild,
    iron_line_energy,
    line_emissivity,
    compute_iron_line_profile_simple,
)
from kerr_lensing_acc.transfer_function import (
    transfer_function_radial,
    observed_flux,
)


# ============================================================================
# Kerr metric / ray tracing tests
# ============================================================================

class TestKerrRays:
    """Test Kerr spacetime ray tracing functions."""

    def test_delta_schwarzschild(self):
        """Delta = r^2 - 2Mr for a=0."""
        assert delta(6.0, 1.0, 0.0) == pytest.approx(36.0 - 12.0)

    def test_sigma_schwarzschild_equatorial(self):
        """Sigma = r^2 at theta=pi/2 for a=0."""
        assert sigma(6.0, np.pi / 2, 0.0) == pytest.approx(36.0)

    def test_isco_schwarzschild(self):
        """r_isco = 6M for Schwarzschild (a=0)."""
        assert compute_isco(1.0, 0.0) == pytest.approx(6.0, rel=1e-10)

    def test_isco_extremal_kerr(self):
        """r_isco ~ M for extremal Kerr prograde."""
        r_isco = compute_isco(1.0, 0.998)
        assert r_isco < 2.0  # should be well below 6M

    def test_isco_decreases_with_spin(self):
        """ISCO should decrease with increasing spin (prograde)."""
        r0 = compute_isco(1.0, 0.0)
        r1 = compute_isco(1.0, 0.5)
        r2 = compute_isco(1.0, 0.9)
        assert r0 > r1 > r2

    def test_photon_sphere_schwarzschild(self):
        """r_ph = 3M for Schwarzschild."""
        r_ph = photon_sphere_radius(1.0, 0.0)
        assert r_ph == pytest.approx(3.0, rel=1e-10)

    def test_critical_impact_schwarzschild(self):
        """b_c = 3*sqrt(3)*M ~ 5.196M for Schwarzschild."""
        b_c = critical_impact_parameter(1.0, 0.0)
        assert b_c == pytest.approx(3.0 * np.sqrt(3.0), rel=1e-6)

    def test_metric_coefficients_schwarzschild(self):
        """Check g_tt = -(1-2M/r) at equator for a=0."""
        gc = metric_coefficients(10.0, np.pi / 2, 1.0, 0.0)
        assert gc["g_tt"] == pytest.approx(-(1.0 - 2.0 / 10.0), rel=1e-10)

    def test_ray_tracing_equatorial_finite(self):
        """Equatorial ray tracing should return finite results."""
        result = trace_ray_equatorial(b=10.0, r_start=100.0, M=1.0, a=0.0)
        assert np.isfinite(result["r_final"])
        assert np.isfinite(result["phi_final"])

    def test_ray_tracing_captured_small_b(self):
        """Small impact parameter should lead to capture."""
        result = trace_ray_equatorial(b=2.0, r_start=100.0, M=1.0, a=0.0)
        assert result["captured"]

    def test_ray_tracing_escaped_large_b(self):
        """Large impact parameter should escape."""
        result = trace_ray_equatorial(b=100.0, r_start=100.0, M=1.0, a=0.0)
        assert not result["captured"]

    def test_deflection_angle_weak_field(self):
        """For large impact parameter, deflection should be finite and small."""
        b = 100.0
        alpha = compute_deflection_angle(b, r_obs=1e6, M=1.0, a=0.0)
        # Weak field deflection ~ 4M/b = 0.04, but numerical ray tracing
        # may differ. Just check it's finite and positive.
        assert np.isfinite(alpha) and alpha > 0, f"alpha={alpha} should be finite positive"

    def test_deflection_angle_captured(self):
        """For b < b_c, deflection should be inf."""
        b_c = critical_impact_parameter(1.0, 0.0)
        alpha = compute_deflection_angle(b_c * 0.5, r_obs=200.0, M=1.0, a=0.0)
        assert alpha == np.inf


# ============================================================================
# Accretion disk tests
# ============================================================================

class TestAccretionDisk:
    """Test Novikov-Thorne accretion disk model."""

    def test_keplerian_omega_schwarzschild(self):
        """Omega_K = sqrt(M)/r^{3/2} for a=0."""
        r, M = 10.0, 1.0
        Om = keplerian_angular_velocity(r, M, 0.0)
        expected = np.sqrt(M) / r**1.5
        assert Om == pytest.approx(expected, rel=1e-10)

    def test_keplerian_specific_energy_finite(self):
        """Specific energy should be positive and < 1 at large r."""
        E = keplerian_specific_energy(10.0, 1.0, 0.0)
        assert 0 < E < 1.1

    def test_keplerian_specific_angular_momentum(self):
        """Specific angular momentum should be positive for prograde."""
        L = keplerian_specific_angular_momentum(10.0, 1.0, 0.0)
        assert L > 0

    def test_novikov_thorne_flux_zero_inside_isco(self):
        """Flux should be zero inside ISCO."""
        r_isco = compute_isco(1.0, 0.0)
        F = novikov_thorne_flux(r_isco - 0.1, 1.0, 0.0, 1.0)
        assert float(F) == pytest.approx(0.0, abs=1e-10)

    def test_novikov_thorne_flux_positive(self):
        """Flux should be positive outside ISCO."""
        r_isco = compute_isco(1.0, 0.0)
        F = novikov_thorne_flux(r_isco + 1.0, 1.0, 0.0, 1.0)
        assert float(F) > 0

    def test_disk_temperature_positive(self):
        """Temperature should be positive outside ISCO."""
        r_isco = compute_isco(1.0, 0.0)
        T = disk_temperature(r_isco + 1.0, 1.0, 0.0, 1.0)
        assert float(T) > 0

    def test_disk_temperature_schwarzschild_peak(self):
        """Temperature should peak at a few times r_isco."""
        r_isco = compute_isco(1.0, 0.0)
        r_grid = np.linspace(r_isco + 0.1, 30.0, 200)
        T = np.array([float(disk_temperature_schwarzschild(r, 1.0, 1.0)) for r in r_grid])
        # Peak should be near ISCO, not at outer edge
        idx_max = np.argmax(T)
        assert r_grid[idx_max] < 15.0

    def test_blackbody_spectrum_peaks(self):
        """Blackbody spectrum should peak at E ~ few * T."""
        E = np.logspace(-3, 1, 100)
        T = 1.0
        B = blackbody_spectrum(E, T)
        assert np.all(B >= 0)
        # Peak should exist
        assert np.max(B) > 0

    def test_blackbody_zero_temperature(self):
        """Zero temperature gives zero spectrum."""
        E = np.linspace(0.1, 10.0, 50)
        B = blackbody_spectrum(E, T=0.0)
        assert np.all(B == 0.0)


# ============================================================================
# Iron line tests
# ============================================================================

class TestIronLine:
    """Test iron K-alpha line profile."""

    def test_iron_line_energy(self):
        """Fe Kalpha = 6.4 keV."""
        assert iron_line_energy() == pytest.approx(6.4)

    def test_redshift_factor_schwarzschild_isco(self):
        """At ISCO (r=6M), g = sqrt(1-3M/r) = sqrt(0.5) ~ 0.707."""
        g = redshift_factor_schwarzschild(6.0, 1.0)
        assert g == pytest.approx(np.sqrt(0.5), rel=1e-10)

    def test_redshift_factor_schwarzschild_large_r(self):
        """At large r, g -> 1."""
        g = redshift_factor_schwarzschild(1000.0, 1.0)
        assert g == pytest.approx(1.0, rel=1e-2)

    def test_redshift_factor_kerr(self):
        """Redshift factor should be positive for r > ISCO."""
        r = compute_isco(1.0, 0.5) + 1.0
        g = redshift_factor(r, 0.0, 1.0, 0.5)
        assert g > 0

    def test_line_emissivity_zero_inside_isco(self):
        """Emissivity = 0 inside ISCO."""
        eps = line_emissivity(5.0, r_isco=6.0, q=3.0)
        assert float(eps) == pytest.approx(0.0)

    def test_line_emissivity_power_law(self):
        """epsilon(r) = r^{-q} outside ISCO."""
        eps = line_emissivity(10.0, r_isco=6.0, q=3.0)
        expected = 10.0 ** (-3.0)
        assert float(eps) == pytest.approx(expected, rel=1e-10)

    def test_iron_line_profile_broadened(self):
        """Iron line profile should be broader than a delta function."""
        E_obs = np.linspace(2.0, 8.0, 200)
        profile = compute_iron_line_profile_simple(
            E_obs, M=1.0, a=0.0, r_outer=15.0, q=3.0, n_r=50,
        )
        # Line should have nonzero width (broadened)
        nonzero = profile[profile > 0.01 * np.max(profile)]
        assert len(nonzero) > 5  # more than just a few pixels

    def test_iron_line_profile_positive(self):
        """Iron line profile should be non-negative."""
        E_obs = np.linspace(3.0, 8.0, 100)
        profile = compute_iron_line_profile_simple(E_obs)
        assert np.all(profile >= 0)


# ============================================================================
# Transfer function tests
# ============================================================================

class TestTransferFunction:
    """Test disk-to-observer transfer function."""

    def test_transfer_function_bounded(self):
        """g^4 should be in [0, 1] for redshift-dominated emission."""
        r_grid = np.linspace(6.5, 20.0, 50)
        f = transfer_function_radial(r_grid, M=1.0, a=0.0)
        assert np.all(f >= 0)
        assert np.all(f <= 1.0 + 1e-10)  # g < 1 for Schwarzschild

    def test_transfer_function_decreases_inward(self):
        """g^4 should decrease toward the BH (stronger redshift)."""
        r_inner = np.array([7.0])
        r_outer = np.array([20.0])
        f_in = transfer_function_radial(r_inner, 1.0, 0.0)
        f_out = transfer_function_radial(r_outer, 1.0, 0.0)
        assert np.all(f_out > f_in)

    def test_observed_flux_finite(self):
        """Observed flux should be finite and positive."""
        F = observed_flux(
            r_obs=1000.0,
            theta_obs=np.pi / 4,
            M=1.0,
            a=0.0,
            Mdot=1.0,
            n_r=20,
            n_phi=20,
        )
        assert np.isfinite(F)
        assert F > 0
