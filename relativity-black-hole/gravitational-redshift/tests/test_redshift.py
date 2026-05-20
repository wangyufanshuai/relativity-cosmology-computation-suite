"""Tests for gravitational redshift and time dilation calculations."""

import math

import numpy as np
import pytest

from gravitational_redshift.constants import (
    EARTH_MASS,
    EARTH_RADIUS,
    G,
    GPS_ORBIT_ALT,
    c,
)
from gravitational_redshift.pulsar import hulse_taylor_parameters, pulsar_binary_orbit
from gravitational_redshift.redshift import (
    doppler_shift,
    gps_clock_correction,
    kerr_redshift,
    sagnac_effect,
    schwarzschild_redshift,
)


# ===========================================================================
# Schwarzschild redshift
# ===========================================================================

class TestSchwarzschildRedshift:
    """Tests for schwarzschild_redshift."""

    def test_earth_surface_redshift(self):
        """Redshift at Earth's surface should be ~ 7e-10."""
        z = schwarzschild_redshift(EARTH_RADIUS, EARTH_MASS)
        assert 6.5e-10 < z < 8.0e-10, f"Earth surface z = {z:.3e}, expected ~7e-10"

    def test_schwarzschild_radius_raises(self):
        """Asking for redshift at or inside the Schwarzschild radius should raise."""
        r_s = 2 * G * EARTH_MASS / c**2
        with pytest.raises(ValueError):
            schwarzschild_redshift(r_s * 0.5, EARTH_MASS)

    def test_zero_mass_no_shift(self):
        """Zero mass should produce zero redshift."""
        z = schwarzschild_redshift(1.0, 0.0)
        assert z == pytest.approx(0.0)

    def test_array_input(self):
        """Function should accept numpy arrays."""
        r = np.array([1e8, 1e9, 1e10])
        z = schwarzschild_redshift(r, EARTH_MASS)
        assert z.shape == (3,)
        assert np.all(z > 0)


# ===========================================================================
# Kerr redshift
# ===========================================================================

class TestKerrRedshift:
    """Tests for kerr_redshift."""

    def test_reduces_to_schwarzschild_when_a_zero(self):
        """Kerr redshift with a=0 should match the Schwarzschild circular-orbit
        expression (which includes orbital velocity): z = (1-3M/r)^{-1/2} - 1."""
        M_geo = G * EARTH_MASS / c**2  # geometrised mass
        r = np.array([100.0 * M_geo, 1000.0 * M_geo])

        # The Kerr formula at a=0 reduces to (1 - 3M/r)^{-1/2} - 1,
        # which is the total redshift for circular equatorial orbits in
        # Schwarzschild (gravitational + transverse Doppler).
        z_schwarzschild_circular = 1.0 / np.sqrt(1.0 - 3.0 * M_geo / r) - 1.0

        z_kerr = kerr_redshift(r, np.pi / 2, M_geo, 0.0)
        np.testing.assert_allclose(z_kerr, z_schwarzschild_circular, rtol=1e-10)

    def test_positive_spin_shifts_redshift(self):
        """A positive spin (prograde) should reduce the redshift relative to a=0."""
        M_geo = G * EARTH_MASS / c**2
        r = 500.0 * M_geo
        z_a0 = kerr_redshift(r, np.pi / 2, M_geo, 0.0)
        z_a_pos = kerr_redshift(r, np.pi / 2, M_geo, 0.5)
        assert z_a_pos < z_a0, "Prograde spin should reduce redshift"


# ===========================================================================
# GPS clock correction
# ===========================================================================

class TestGPSClockCorrection:
    """Tests for gps_clock_correction."""

    def test_total_correction_38_microseconds(self):
        """GPS total clock correction should be ~38 us/day."""
        result = gps_clock_correction(GPS_ORBIT_ALT)
        total = result["total_us_day"]
        assert 36.0 < total < 40.0, f"GPS total correction = {total:.2f} us/day, expected ~38"

    def test_gravitational_positive(self):
        """Gravitational component should be positive (clock runs faster)."""
        result = gps_clock_correction(GPS_ORBIT_ALT)
        assert result["gravitational_us_day"] > 0

    def test_special_relativistic_negative(self):
        """Special-relativistic component should be negative (clock runs slower)."""
        result = gps_clock_correction(GPS_ORBIT_ALT)
        assert result["special_us_day"] < 0

    def test_gravitational_magnitude_45(self):
        """Gravitational component should be ~45 us/day."""
        result = gps_clock_correction(GPS_ORBIT_ALT)
        grav = result["gravitational_us_day"]
        assert 43.0 < grav < 47.0, f"Gravitational correction = {grav:.2f} us/day, expected ~45"

    def test_special_magnitude_7(self):
        """Special-relativistic component should be ~ -7 us/day."""
        result = gps_clock_correction(GPS_ORBIT_ALT)
        spec = result["special_us_day"]
        assert -9.0 < spec < -5.0, f"SR correction = {spec:.2f} us/day, expected ~-7"


# ===========================================================================
# Doppler shift
# ===========================================================================

class TestDopplerShift:
    """Tests for doppler_shift."""

    def test_zero_velocity_no_shift(self):
        """Zero velocity should produce no shift."""
        z = doppler_shift(0.0, 0.0)
        assert z == pytest.approx(0.0)

    def test_receding_redshift(self):
        """Source moving directly away (angle=0) should produce redshift."""
        z = doppler_shift(0.1 * c, 0.0)
        assert z > 0

    def test_approaching_blueshift(self):
        """Source moving toward observer (angle=pi) should produce blueshift."""
        z = doppler_shift(0.1 * c, np.pi)
        assert z < 0

    def test_transverse_dilation(self):
        """Transverse Doppler (angle=pi/2) should give redshift from time dilation."""
        z = doppler_shift(0.5 * c, np.pi / 2, include_transverse=True)
        assert z > 0  # purely transverse → time dilation → redshift

    def test_no_transverse_zero(self):
        """Without transverse term, pi/2 should give zero shift."""
        z = doppler_shift(0.5 * c, np.pi / 2, include_transverse=False)
        assert z == pytest.approx(0.0)

    def test_relativistic_formula_symmetry(self):
        """At v << c, the result should approach the classical formula."""
        v = 1e3  # 1 km/s
        angle = 0.0
        z = doppler_shift(v, angle, include_transverse=False)
        z_classical = v / c
        assert z == pytest.approx(z_classical, rel=1e-6)


# ===========================================================================
# Sagnac effect
# ===========================================================================

class TestSagnacEffect:
    """Tests for sagnac_effect."""

    def test_positive_direction(self):
        """Co-rotating beam should give positive delay."""
        dt = sagnac_effect(1.0, 1.0, np.array([0.0, 0.0, 1.0]), +1)
        assert dt > 0

    def test_counter_rotating_negative(self):
        """Counter-rotating beam should give negative delay."""
        dt = sagnac_effect(1.0, 1.0, np.array([0.0, 0.0, 1.0]), -1)
        assert dt < 0

    def test_magnitude(self):
        """Check approximate magnitude against known formula."""
        omega = 7.2921e-5  # Earth rotation rate [rad/s]
        R = 0.5            # 50 cm ring
        dt = sagnac_effect(omega, R, np.array([0.0, 0.0, 1.0]), +1)
        expected = omega * np.pi * R**2 / c**2
        assert dt == pytest.approx(expected, rel=1e-10)


# ===========================================================================
# Hulse-Taylor PSR B1913+16
# ===========================================================================

class TestHulseTaylor:
    """Tests for PSR B1913+16 parameters."""

    def test_periastron_advance_observed(self):
        """Observed periastron advance should be ~4.226 deg/yr."""
        params = hulse_taylor_parameters()
        assert 4.2 < params["omega_dot_obs"] < 4.3

    def test_periastron_advance_gr_matches_observed(self):
        """GR prediction should match the observed periastron advance."""
        params = hulse_taylor_parameters()
        assert params["omega_dot_gr"] == pytest.approx(
            params["omega_dot_obs"], rel=0.01
        )

    def test_orbital_period(self):
        """Orbital period should be ~7.75 hours."""
        params = hulse_taylor_parameters()
        Pb_hours = params["Pb"] / 3600.0
        assert 7.7 < Pb_hours < 7.8

    def test_binary_orbit_consistency(self):
        """pulsar_binary_orbit should reproduce periastron advance."""
        params = hulse_taylor_parameters()
        orbit = pulsar_binary_orbit(
            params["Pb"], params["ecc"], params["M1"], params["M2"]
        )
        omega_dot_gr_deg_yr = orbit["omega_dot_gr"] * (180.0 / np.pi) * (
            365.25 * 86400.0
        )
        assert omega_dot_gr_deg_yr == pytest.approx(
            params["omega_dot_obs"], rel=0.01
        )


# ===========================================================================
# Lense-Thirring
# ===========================================================================

class TestLenseThirring:
    """Tests for lense_thirring_rate."""

    def test_positive_angular_momentum(self):
        """Positive J should give positive precession rate."""
        from gravitational_redshift.redshift import lense_thirring_rate

        # Earth angular momentum: J = I omega ~ 8.04e34 kg m^2/s
        J_earth = 8.04e34
        Omega = lense_thirring_rate(J_earth, EARTH_MASS, EARTH_RADIUS, np.pi / 2)
        assert Omega > 0

    def test_equatorial_vs_polar(self):
        """Polar precession should differ from equatorial by known factor."""
        from gravitational_redshift.redshift import lense_thirring_rate

        J_earth = 8.04e34
        r = 2.0 * EARTH_RADIUS  # LAGEOS altitude ~ 12,270 km
        Omega_eq = lense_thirring_rate(J_earth, EARTH_MASS, r, np.pi / 2)
        Omega_pole = lense_thirring_rate(J_earth, EARTH_MASS, r, 0.0)
        # At theta=0: factor = 1 + 1.5*cos^2(0) = 2.5
        # At theta=pi/2: factor = 1 + 1.5*cos^2(pi/2) = 1.0
        assert Omega_pole / Omega_eq == pytest.approx(2.5, rel=1e-10)
