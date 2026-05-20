"""Tests for the cosmic recombination history calculator.

Uses standard Planck-like cosmological parameters:
    H0 = 67.4 km/s/Mpc, h = 0.674
    Omega_m = 0.315, Omega_b = 0.049
    Omega_r = 9.1e-5 (photons + neutrinos)
    Omega_lambda = 1 - Omega_m - Omega_r
    Y_p = 0.245
"""

import numpy as np
import pytest

from recombination_history.constants import T_CMB0
from recombination_history.background import hubble, temperature, baryon_density
from recombination_history.saha import saha_xe
from recombination_history.peebles import solve_recombination
from recombination_history.observables import (
    thomson_opacity_full,
    visibility_function,
    last_scattering_z,
    sound_horizon,
)


# ---- Standard cosmological parameters ----
H0 = 67.4       # km/s/Mpc
h = 0.674
Omega_m = 0.315
Omega_b = 0.049
Omega_r = 9.1e-5
Omega_lambda = 1.0 - Omega_m - Omega_r
Y_p = 0.245


class TestHubble:
    """Tests for the Hubble parameter."""

    def test_hubble_present(self):
        """H(0) should equal H0."""
        H_z0 = hubble(0.0, H0, Omega_m, Omega_r, Omega_lambda)
        H0_si = H0 * 1.0e3 / 3.0856775814913673e22
        # At z=0: H(0) = H0 * sqrt(Omega_lambda) in a flat universe
        # But H(0) = H0 * sqrt(Omega_m + Omega_r + Omega_lambda) = H0 * sqrt(1) = H0
        expected = H0_si * np.sqrt(Omega_m + Omega_r + Omega_lambda)
        assert abs(H_z0 - H0_si) / H0_si < 1e-10, \
            f"H(0) = {H_z0:.6e}, expected {H0_si:.6e}"

    def test_hubble_increases_with_z(self):
        """H(z) should increase with redshift in a matter+radiation dominated era."""
        z_vals = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 3000.0])
        H_vals = hubble(z_vals, H0, Omega_m, Omega_r, Omega_lambda)
        for i in range(1, len(H_vals)):
            assert H_vals[i] > H_vals[i - 1], \
                f"H(z={z_vals[i]}) = {H_vals[i]} not > H(z={z_vals[i-1]}) = {H_vals[i-1]}"


class TestTemperature:
    """Tests for CMB temperature."""

    def test_temperature_cmb(self):
        """T(z=0) should equal 2.7255 K."""
        T0 = temperature(0.0)
        assert abs(T0 - T_CMB0) < 1e-10, \
            f"T(0) = {T0}, expected {T_CMB0}"

    def test_temperature_scales_linearly(self):
        """T(z) = T0 * (1+z)."""
        T_at_1100 = temperature(1100.0)
        expected = T_CMB0 * 1101.0
        assert abs(T_at_1100 - expected) / expected < 1e-10


class TestSaha:
    """Tests for the Saha equation."""

    def test_saha_full_ionization_early(self):
        """At z >> 1000, the Saha equation should give x_e close to 1."""
        z_early = 2000.0
        T = temperature(z_early)
        n_b = baryon_density(z_early, Omega_b, h)
        x_e = saha_xe(T, n_b, Y_p)
        assert x_e > 0.99, \
            f"At z={z_early}, Saha x_e = {x_e:.4f}, expected > 0.99"

    def test_saha_neutral_late(self):
        """At z << 800, the Saha equation should give x_e close to 0."""
        z_late = 500.0
        T = temperature(z_late)
        n_b = baryon_density(z_late, Omega_b, h)
        x_e = saha_xe(T, n_b, Y_p)
        assert x_e < 0.01, \
            f"At z={z_late}, Saha x_e = {x_e:.4f}, expected < 0.01"


class TestPeebles:
    """Tests for the Peebles TLA recombination solver."""

    def test_peebles_recombination_occurs(self):
        """The TLA should produce x_e dropping from ~1 to ~0 around z ~ 800-1200."""
        z_array = np.linspace(400.0, 2000.0, 200)
        x_e = solve_recombination(z_array, H0, Omega_m, Omega_r, Omega_lambda,
                                  Omega_b, h, Y_p)

        # At high z (z > 1500), x_e should be close to 1
        mask_high = z_array > 1500
        assert np.all(x_e[mask_high] > 0.8), \
            f"At z > 1500, min x_e = {np.min(x_e[mask_high]):.4f}, expected > 0.8"

        # At low z (z < 500), x_e should be small
        mask_low = z_array < 500
        assert np.all(x_e[mask_low] < 0.1), \
            f"At z < 500, max x_e = {np.max(x_e[mask_low]):.4f}, expected < 0.1"

        # At intermediate z (~800-1200), x_e should be transitioning
        mask_mid = (z_array > 800) & (z_array < 1200)
        x_e_mid = x_e[mask_mid]
        # There should be significant variation in this range
        assert (np.max(x_e_mid) - np.min(x_e_mid)) > 0.3, \
            "x_e should change significantly between z=800 and z=1200"


class TestObservables:
    """Tests for derived observables."""

    @classmethod
    def setup_class(cls):
        """Compute recombination history once for all observable tests."""
        cls.z_array = np.linspace(10.0, 2500.0, 3000)
        cls.x_e = solve_recombination(
            cls.z_array, H0, Omega_m, Omega_r, Omega_lambda,
            Omega_b, h, Y_p
        )
        cls.g, cls.tau = visibility_function(
            cls.z_array, cls.x_e, Omega_b, h,
            H0, Omega_m, Omega_r, Omega_lambda
        )

    def test_thomson_opacity_peaks(self):
        """dtau/dz should be positive and finite."""
        dtaudz = thomson_opacity_full(
            self.z_array, self.x_e, Omega_b, h,
            H0, Omega_m, Omega_r, Omega_lambda
        )
        assert np.all(dtaudz > 0), "Thomson opacity should be positive"
        assert np.all(np.isfinite(dtaudz)), "Thomson opacity should be finite"

    def test_visibility_function_normalized(self):
        """Integral of g(z) dz should be approximately 1."""
        integral = np.trapezoid(self.g, self.z_array)
        assert abs(integral - 1.0) < 0.5, \
            f"Integral of g(z) = {integral:.4f}, expected ~ 1.0"

    def test_last_scattering_around_1100(self):
        """z_* (peak of visibility function) should be around 1050-1200."""
        z_star = last_scattering_z(self.g, self.z_array)
        assert 800 <= z_star <= 1400, \
            f"z_* = {z_star:.1f}, expected in [1000, 1300]"

    def test_sound_horizon_positive(self):
        """Sound horizon should be positive."""
        z_drag = 1060.0
        rs = sound_horizon(z_drag, H0, Omega_m, Omega_r, Omega_b, h)
        assert rs > 0, f"Sound horizon r_s = {rs:.4f} Mpc, expected > 0"
        # Expected value is roughly ~150 Mpc for standard cosmology
        assert 50 < rs < 300, \
            f"Sound horizon r_s = {rs:.4f} Mpc, expected in [50, 300] Mpc"
