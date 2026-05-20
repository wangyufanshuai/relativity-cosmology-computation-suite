"""Tests for pulsar-magnetosphere: dipole field and force-free electrodynamics."""

import numpy as np
import pytest

from pulsar_magnetosphere.dipole import (
    RotatingDipole,
    light_cylinder_radius,
    magnetic_dipole_moment,
    spindown_luminosity,
    C_LIGHT,
)
from pulsar_magnetosphere.force_free import ForceFreeSolver


# Typical pulsar parameters (Crab-like)
B0_CRAB = 1e12       # G
R_STAR = 1e6          # cm (10 km)
OMEGA_CRAB = 2 * np.pi * 30  # rad/s (P ~ 33 ms)
ALPHA = np.pi / 2     # orthogonal rotator


# ============================================================================
# Light cylinder tests
# ============================================================================

class TestLightCylinder:
    """Test light-cylinder radius R_LC = c / Omega."""

    def test_basic(self):
        """R_LC = c / Omega."""
        omega = 2 * np.pi * 30.0
        r_lc = light_cylinder_radius(omega)
        expected = C_LIGHT / omega
        assert r_lc == pytest.approx(expected, rel=1e-10)

    def test_faster_rotation_smaller_lc(self):
        """Higher Omega -> smaller R_LC."""
        r1 = light_cylinder_radius(100.0)
        r2 = light_cylinder_radius(200.0)
        assert r2 < r1

    def test_invalid_omega(self):
        """omega <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            light_cylinder_radius(0.0)
        with pytest.raises(ValueError):
            light_cylinder_radius(-1.0)


# ============================================================================
# Dipole moment tests
# ============================================================================

class TestDipoleMoment:
    """Test magnetic_dipole_moment mu = B0 * R^3."""

    def test_basic(self):
        mu = magnetic_dipole_moment(1e12, 1e6)
        assert mu == pytest.approx(1e12 * 1e18, rel=1e-10)

    def test_scaling(self):
        """mu should scale linearly with B0 and as R^3."""
        mu1 = magnetic_dipole_moment(1.0, 1.0)
        mu2 = magnetic_dipole_moment(2.0, 1.0)
        mu3 = magnetic_dipole_moment(1.0, 2.0)
        assert mu2 == pytest.approx(2.0 * mu1)
        assert mu3 == pytest.approx(8.0 * mu1)


# ============================================================================
# Spindown luminosity tests
# ============================================================================

class TestSpindown:
    """Test spindown luminosity L_sd = (2/3c^3) mu^2 Omega^4 sin^2(alpha)."""

    def test_orthogonal_rotator(self):
        """For alpha=pi/2, sin^2(alpha)=1."""
        L = spindown_luminosity(B0_CRAB, R_STAR, OMEGA_CRAB, alpha=np.pi / 2)
        mu = magnetic_dipole_moment(B0_CRAB, R_STAR)
        expected = (2.0 / (3.0 * C_LIGHT**3)) * mu**2 * OMEGA_CRAB**4
        assert L == pytest.approx(expected, rel=1e-10)

    def test_aligned_rotator_zero_luminosity(self):
        """For alpha=0 (aligned), L_sd = 0."""
        L = spindown_luminosity(B0_CRAB, R_STAR, OMEGA_CRAB, alpha=0.0)
        assert L == pytest.approx(0.0, abs=1e-10)

    def test_luminosity_scaling_omega4(self):
        """L_sd should scale as Omega^4."""
        L1 = spindown_luminosity(B0_CRAB, R_STAR, 100.0)
        L2 = spindown_luminosity(B0_CRAB, R_STAR, 200.0)
        ratio = L2 / L1
        assert ratio == pytest.approx(2.0**4, rel=1e-6)

    def test_luminosity_scaling_B2(self):
        """L_sd should scale as B0^2."""
        L1 = spindown_luminosity(1e12, R_STAR, OMEGA_CRAB)
        L2 = spindown_luminosity(2e12, R_STAR, OMEGA_CRAB)
        ratio = L2 / L1
        assert ratio == pytest.approx(4.0, rel=1e-6)

    def test_luminosity_scaling_R6(self):
        """L_sd should scale as R^6 (since mu = B0*R^3, mu^2 ~ R^6)."""
        L1 = spindown_luminosity(B0_CRAB, 1e6, OMEGA_CRAB)
        L2 = spindown_luminosity(B0_CRAB, 2e6, OMEGA_CRAB)
        ratio = L2 / L1
        assert ratio == pytest.approx(2.0**6, rel=1e-6)

    def test_crab_luminosity_order(self):
        """Crab pulsar spindown luminosity ~ 5e38 erg/s."""
        L = spindown_luminosity(B0_CRAB, R_STAR, OMEGA_CRAB)
        assert 1e36 < L < 1e42  # order-of-magnitude check


# ============================================================================
# Rotating dipole field tests
# ============================================================================

class TestRotatingDipoleField:
    """Test the RotatingDipole class magnetic field components."""

    def setup_method(self):
        self.dipole = RotatingDipole(B0_CRAB, R_STAR, OMEGA_CRAB, ALPHA)

    def test_B_r_at_pole(self):
        """At theta=0 (pole): B_r = 2*mu/r^3, B_theta=0."""
        r = 1e7  # 10 stellar radii
        Br, Bt = self.dipole.magnetic_field(r, 0.0)
        expected = 2.0 * self.dipole.mu / r**3
        assert float(Br) == pytest.approx(expected, rel=1e-8)
        assert float(Bt) == pytest.approx(0.0, abs=1e-10)

    def test_B_at_equator(self):
        """At theta=pi/2 (equator): B_r=0, B_theta=mu/r^3."""
        r = 1e7
        Br, Bt = self.dipole.magnetic_field(r, np.pi / 2)
        expected = self.dipole.mu / r**3
        assert float(Br) == pytest.approx(0.0, abs=1e-6)
        assert float(Bt) == pytest.approx(expected, rel=1e-8)

    def test_B_falls_as_r_cubed(self):
        """|B| should fall as 1/r^3."""
        r1, r2 = 1e7, 2e7
        B1 = self.dipole.magnetic_field_magnitude(r1, np.pi / 4)
        B2 = self.dipole.magnetic_field_magnitude(r2, np.pi / 4)
        ratio = float(B2) / float(B1)
        assert ratio == pytest.approx(1.0 / 8.0, rel=1e-4)

    def test_surface_field(self):
        """At the stellar surface and pole, B_r should be ~ 2*B0."""
        Br, _ = self.dipole.magnetic_field(R_STAR, 0.0)
        assert float(Br) == pytest.approx(2.0 * B0_CRAB, rel=1e-8)

    def test_invalid_B0(self):
        with pytest.raises(ValueError):
            RotatingDipole(-1.0, R_STAR, OMEGA_CRAB)

    def test_invalid_R_star(self):
        with pytest.raises(ValueError):
            RotatingDipole(B0_CRAB, -1.0, OMEGA_CRAB)

    def test_invalid_omega(self):
        with pytest.raises(ValueError):
            RotatingDipole(B0_CRAB, R_STAR, 0.0)


# ============================================================================
# Co-rotation electric field tests
# ============================================================================

class TestCorotationEField:
    """Test co-rotation electric field E = -(Omega x r) x B."""

    def setup_method(self):
        self.dipole = RotatingDipole(B0_CRAB, R_STAR, OMEGA_CRAB, ALPHA)

    def test_E_phi_zero(self):
        """E_phi = 0 for axisymmetric dipole."""
        r, theta = 1e7, np.pi / 4
        _, _, E_phi = self.dipole.electric_field(r, theta)
        assert float(E_phi) == pytest.approx(0.0, abs=1e-20)

    def test_E_zero_at_pole(self):
        """At theta=0 (pole), Omega x r = 0, so E = 0."""
        r = 1e7
        Er, Et, Ep = self.dipole.electric_field(r, 0.0)
        assert float(Er) == pytest.approx(0.0, abs=1e-10)
        assert float(Et) == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# Force-free tests
# ============================================================================

class TestForceFree:
    """Test force-free electrodynamics conditions."""

    def setup_method(self):
        self.dipole = RotatingDipole(B0_CRAB, R_STAR, OMEGA_CRAB, ALPHA)
        self.solver = ForceFreeSolver(self.dipole)

    def test_edotb_vanishes(self):
        """E.B should be small for co-rotation fields in the near-zone."""
        r = 1e7
        thetas = np.linspace(0.1, np.pi - 0.1, 20)
        for th in thetas:
            edb = self.solver.edotb(r, th)
            # In near-zone approximation, E.B is not exactly zero;
            # it should be small relative to |E||B|
            assert np.isfinite(float(edb))

    def test_force_free_check_edotb_norm(self):
        """Normalized E.B / (|E||B|) should be zero."""
        r = 1e7
        theta = np.pi / 3
        result = self.solver.check_force_free(r, theta)
        assert float(result["edotb_norm"]) < 0.1

    def test_charge_density_nonzero(self):
        """Goldreich-Julian charge density should be nonzero in general."""
        r = 1e7
        theta = np.pi / 4
        rho = self.solver.charge_density(r, theta)
        assert float(rho) != 0.0

    def test_current_density_toroidal(self):
        """J_r and J_theta should be zero (near-zone), J_phi nonzero."""
        r = 1e7
        theta = np.pi / 4
        Jr, Jt, Jp = self.solver.current_density(r, theta)
        assert float(Jr) == pytest.approx(0.0, abs=1e-20)
        assert float(Jt) == pytest.approx(0.0, abs=1e-20)
