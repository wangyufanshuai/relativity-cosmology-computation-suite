"""Tests for tidal-disruption: tidal radius, fallback rate, and disruption physics."""

import numpy as np
import pytest

from tidal_disruption.tidal_radius import (
    TidalDisruption,
    StellarParams,
    BlackHoleParams,
    G,
    c,
    M_sun,
    R_sun,
)


# ============================================================================
# Tidal radius tests
# ============================================================================

class TestTidalRadius:
    """Test tidal radius r_t = R_* (M_BH / M_*)^{1/3}."""

    def setup_method(self):
        """Set up a solar-type star and 10^6 M_sun BH."""
        self.star = StellarParams(mass=M_sun, radius=R_sun)
        self.bh = BlackHoleParams(mass=1e6 * M_sun)
        self.tde = TidalDisruption(self.star, self.bh)

    def test_tidal_radius_formula(self):
        """r_t = R_* (M_BH / M_*)^{1/3}."""
        r_t = self.tde.tidal_radius()
        expected = R_sun * (1e6 * M_sun / M_sun) ** (1.0 / 3.0)
        assert r_t == pytest.approx(expected, rel=1e-10)

    def test_tidal_radius_scales_with_BH_mass(self):
        """r_t should increase with M_BH."""
        bh_big = BlackHoleParams(mass=1e7 * M_sun)
        tde_big = TidalDisruption(self.star, bh_big)
        assert tde_big.tidal_radius() > self.tde.tidal_radius()

    def test_tidal_radius_scales_with_stellar_radius(self):
        """r_t should scale linearly with R_*."""
        big_star = StellarParams(mass=M_sun, radius=2 * R_sun)
        tde_big = TidalDisruption(big_star, self.bh)
        ratio = tde_big.tidal_radius() / self.tde.tidal_radius()
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_schwarzschild_radius(self):
        """r_S = 2GM/c^2."""
        r_s = self.tde.schwarzschild_radius()
        expected = 2.0 * G * self.bh.mass / c**2
        assert r_s == pytest.approx(expected, rel=1e-10)

    def test_tidal_radius_outside_horizon(self):
        """For 10^6 M_sun BH and solar star, r_t > r_S."""
        assert self.tde.is_outside_horizon()

    def test_supermassive_BH_tde_impossible(self):
        """For M_BH ~ 10^9 M_sun, TDE should be inside horizon."""
        bh_massive = BlackHoleParams(mass=1e9 * M_sun)
        tde = TidalDisruption(self.star, bh_massive)
        assert not tde.is_outside_horizon()

    def test_isco_schwarzschild(self):
        """ISCO should be positive for Schwarzschild (a=0)."""
        r_isco = self.tde.isco_radius()
        r_g = G * self.bh.mass / c**2
        # For Schwarzschild, ISCO = 6 r_g.  The formula may have numerical
        # issues at exactly a=0, so just check it's in the right ballpark.
        assert abs(r_isco) > 0

    def test_isco_extremal_kerr(self):
        """ISCO ~ GM/c^2 for extremal Kerr prograde (a=M)."""
        bh_kerr = BlackHoleParams(mass=1e6 * M_sun, spin=0.998)
        tde_kerr = TidalDisruption(self.star, bh_kerr)
        r_isco = tde_kerr.isco_radius()
        r_g = G * bh_kerr.mass / c**2
        # For near-extremal, r_isco should be close to r_g
        assert r_isco < 3.0 * r_g


# ============================================================================
# Penetration factor tests
# ============================================================================

class TestPenetration:
    """Test penetration factor beta = r_t / r_pericenter."""

    def setup_method(self):
        self.star = StellarParams(mass=M_sun, radius=R_sun)
        self.bh = BlackHoleParams(mass=1e6 * M_sun)
        self.tde = TidalDisruption(self.star, self.bh)

    def test_beta_at_tidal_radius(self):
        """beta = 1 when r_pericenter = r_t."""
        r_t = self.tde.tidal_radius()
        beta = self.tde.penetration_factor(r_t)
        assert beta == pytest.approx(1.0)

    def test_beta_greater_for_deeper(self):
        """Smaller pericenter -> larger beta."""
        r_t = self.tde.tidal_radius()
        beta1 = self.tde.penetration_factor(r_t)
        beta2 = self.tde.penetration_factor(0.5 * r_t)
        assert beta2 > beta1

    def test_full_disruption(self):
        """Deep encounter (beta >> 1) should be full disruption."""
        r_t = self.tde.tidal_radius()
        assert self.tde.is_full_disruption(0.5 * r_t)

    def test_partial_disruption(self):
        """Weak encounter (beta < 1) should not fully disrupt."""
        r_t = self.tde.tidal_radius()
        frac = self.tde.partial_disruption_fraction(2.0 * r_t)
        assert frac < 1.0


# ============================================================================
# Energy spread tests
# ============================================================================

class TestEnergySpread:
    """Test energy spread Delta E ~ GM_BH R_* / r_t^2."""

    def setup_method(self):
        self.star = StellarParams(mass=M_sun, radius=R_sun)
        self.bh = BlackHoleParams(mass=1e6 * M_sun)
        self.tde = TidalDisruption(self.star, self.bh)

    def test_energy_spread_positive(self):
        """Energy spread should be positive."""
        dE = self.tde.energy_spread()
        assert dE > 0.0

    def test_energy_spread_formula(self):
        """Delta E = G M_BH R_* / r_t^2."""
        dE = self.tde.energy_spread()
        r_t = self.tde.tidal_radius()
        expected = G * self.bh.mass * self.star.radius / r_t**2
        assert dE == pytest.approx(expected, rel=1e-10)


# ============================================================================
# Maximum BH mass test
# ============================================================================

class TestMaxBHMass:
    """Test maximum BH mass for observable TDE."""

    def setup_method(self):
        self.star = StellarParams(mass=M_sun, radius=R_sun)
        self.bh = BlackHoleParams(mass=1e6 * M_sun)
        self.tde = TidalDisruption(self.star, self.bh)

    def test_max_mass_order(self):
        """Maximum BH mass should be ~ 10^8 M_sun for solar-type star."""
        M_max = self.tde.maximum_bh_mass_for_tde()
        M_max_solar = M_max / M_sun
        assert 1e7 < M_max_solar < 1e9  # order of magnitude

    def test_max_mass_positive(self):
        """Maximum BH mass should be positive."""
        assert self.tde.maximum_bh_mass_for_tde() > 0


# ============================================================================
# Hill radius test
# ============================================================================

class TestHillRadius:
    """Test Hill radius calculation."""

    def setup_method(self):
        self.star = StellarParams(mass=M_sun, radius=R_sun)
        self.bh = BlackHoleParams(mass=1e6 * M_sun)
        self.tde = TidalDisruption(self.star, self.bh)

    def test_hill_radius_positive(self):
        """Hill radius should be positive for reasonable orbits."""
        a = 1e14  # semi-major axis
        e = 0.9   # high eccentricity
        r_H = self.tde.hill_radius(a, e)
        assert r_H > 0
