"""Tests for Kerr black hole tidal heating and energy extraction."""

import numpy as np
import pytest

from kerr_tidal_heating.kerr import (
    frame_dragging_omega,
    kerr_ergosphere,
    kerr_horizon,
    kerr_isco,
)
from kerr_tidal_heating.penrose import (
    max_penrose_efficiency,
    negative_energy_orbit,
    penrose_efficiency,
)
from kerr_tidal_heating.blandford_znajek import bz_power, bz_power_numerical
from kerr_tidal_heating.tidal import tidal_heating_rate, tidal_phase_shift


# ---------- Kerr metric ----------

class TestKerrHorizon:
    """Tests for kerr_horizon()."""

    def test_kerr_horizon_extreme(self):
        """Extreme Kerr a=M => r+ = M."""
        M = 1.0
        a = M
        assert kerr_horizon(M, a) == pytest.approx(M, rel=1e-12)

    def test_kerr_horizon_schwarzschild(self):
        """Schwarzschild a=0 => r+ = 2M."""
        M = 1.0
        a = 0.0
        assert kerr_horizon(M, a) == pytest.approx(2.0 * M, rel=1e-12)

    def test_kerr_horizon_intermediate(self):
        """Intermediate spin a=0.5M."""
        M = 1.0
        a = 0.5
        expected = M + np.sqrt(M**2 - a**2)
        assert kerr_horizon(M, a) == pytest.approx(expected, rel=1e-12)


class TestErgosphere:
    """Tests for kerr_ergosphere()."""

    def test_ergosphere_equatorial(self):
        """theta=pi/2 => r_ergo = 2M (independent of a)."""
        M = 1.0
        a = 0.8
        r_ergo = kerr_ergosphere(M, a, theta=np.pi / 2)
        assert r_ergo == pytest.approx(2.0 * M, rel=1e-12)

    def test_ergosphere_pole(self):
        """theta=0 => r_ergo = r+ (ergosphere touches horizon at poles)."""
        M = 1.0
        a = 0.6
        r_ergo = kerr_ergosphere(M, a, theta=0.0)
        r_plus = kerr_horizon(M, a)
        assert r_ergo == pytest.approx(r_plus, rel=1e-12)

    def test_ergosphere_extreme_equatorial(self):
        """Extreme Kerr at equator: r_ergo = 2M."""
        M = 1.0
        a = 1.0
        r_ergo = kerr_ergosphere(M, a, theta=np.pi / 2)
        assert r_ergo == pytest.approx(2.0, rel=1e-12)

    def test_ergosphere_extreme_pole(self):
        """Extreme Kerr at pole: r_ergo = r+ = M."""
        M = 1.0
        a = 1.0
        r_ergo = kerr_ergosphere(M, a, theta=0.0)
        assert r_ergo == pytest.approx(M, rel=1e-12)


class TestISCO:
    """Tests for kerr_isco()."""

    def test_isco_schwarzschild(self):
        """a=0 => r_isco = 6M (Schwarzschild)."""
        M = 1.0
        a = 0.0
        assert kerr_isco(M, a, prograde=True) == pytest.approx(6.0 * M, rel=1e-10)

    def test_isco_extreme_prograde(self):
        """a -> M => r_isco -> M (extreme Kerr prograde)."""
        M = 1.0
        # Near-extreme spin; convergence is ~(1-a/M)^(2/3)
        a = 0.9999 * M
        r_isco = kerr_isco(M, a, prograde=True)
        assert r_isco == pytest.approx(M, rel=0.1)

    def test_isco_extreme_retrograde(self):
        """a -> M => r_isco -> 9M (extreme Kerr retrograde)."""
        M = 1.0
        a = 0.9999 * M
        r_isco = kerr_isco(M, a, prograde=False)
        assert r_isco == pytest.approx(9.0 * M, rel=0.15)

    def test_isco_retrograde_schwarzschild(self):
        """a=0 retrograde => same as prograde = 6M."""
        M = 1.0
        a = 0.0
        assert kerr_isco(M, a, prograde=False) == pytest.approx(6.0 * M, rel=1e-10)


class TestFrameDragging:
    """Tests for frame_dragging_omega()."""

    def test_frame_dragging_positive(self):
        """omega > 0 for a > 0."""
        M = 1.0
        a = 0.5
        r = 6.0
        omega = frame_dragging_omega(M, a, r)
        assert omega > 0.0

    def test_frame_dragging_zero_spin(self):
        """omega = 0 for a = 0 (no frame dragging without spin)."""
        M = 1.0
        a = 0.0
        r = 6.0
        omega = frame_dragging_omega(M, a, r)
        assert omega == pytest.approx(0.0, abs=1e-15)

    def test_frame_dragging_decreases_with_r(self):
        """Frame dragging decreases with distance."""
        M = 1.0
        a = 0.8
        omega_near = frame_dragging_omega(M, a, 4.0)
        omega_far = frame_dragging_omega(M, a, 10.0)
        assert omega_near > omega_far


# ---------- Penrose process ----------

class TestPenrose:
    """Tests for Penrose process functions."""

    def test_penrose_efficiency_positive(self):
        """Efficiency should be positive within the ergosphere."""
        M = 1.0
        a = 0.9
        # Break up halfway between horizon and ergosphere edge
        r_plus = kerr_horizon(M, a)
        r_ergo = kerr_ergosphere(M, a, np.pi / 2)
        r_breakup = (r_plus + r_ergo) / 2.0
        E_in = 1.0
        eta = penrose_efficiency(M, a, r_breakup, E_in)
        assert eta > 0.0

    def test_penrose_efficiency_outside_ergosphere(self):
        """Efficiency should be zero outside the ergosphere."""
        M = 1.0
        a = 0.5
        r_breakup = 10.0  # well outside ergosphere
        E_in = 1.0
        eta = penrose_efficiency(M, a, r_breakup, E_in)
        assert eta == pytest.approx(0.0, abs=1e-15)

    def test_max_penrose_extreme(self):
        """Maximum Penrose efficiency ~ 0.207 for extreme Kerr."""
        M = 1.0
        a = M
        eta_max = max_penrose_efficiency(M, a)
        expected = (np.sqrt(2.0) - 1.0) / 2.0
        assert eta_max == pytest.approx(expected, rel=1e-12)
        assert eta_max == pytest.approx(0.2071, rel=1e-3)

    def test_max_penrose_zero_spin(self):
        """Maximum Penrose efficiency = 0 for Schwarzschild."""
        M = 1.0
        a = 0.0
        assert max_penrose_efficiency(M, a) == pytest.approx(0.0, abs=1e-15)

    def test_negative_energy_orbit(self):
        """Orbits with E < 0 should return True."""
        assert negative_energy_orbit(1.0, 0.9, 1.5, -1.0, -0.1) is True
        assert negative_energy_orbit(1.0, 0.9, 1.5, 1.0, 0.5) is False


# ---------- Blandford-Znajek ----------

class TestBlandfordZnajek:
    """Tests for BZ mechanism functions."""

    def test_bz_power_increases_with_spin(self):
        """P_BZ(a1) < P_BZ(a2) for a1 < a2."""
        M = 1.0
        B = 1.0
        P1 = bz_power(M, 0.3, B)
        P2 = bz_power(M, 0.7, B)
        assert P1 < P2

    def test_bz_power_zero_spin(self):
        """a=0 => P_BZ = 0 (no spin, no extraction)."""
        M = 1.0
        a = 0.0
        B = 1.0
        assert bz_power(M, a, B) == pytest.approx(0.0, abs=1e-30)

    def test_bz_power_positive(self):
        """Power should be positive for a > 0, B > 0."""
        M = 1.0
        a = 0.5
        B = 1.0
        assert bz_power(M, a, B) > 0.0

    def test_bz_power_numerical(self):
        """Angle-dependent power should vanish at poles."""
        M = 1.0
        a = 0.5
        B = 1.0
        P_pole = bz_power_numerical(M, a, B, theta=0.0)
        P_equator = bz_power_numerical(M, a, B, theta=np.pi / 2)
        assert P_pole == pytest.approx(0.0, abs=1e-30)
        assert P_equator > 0.0
        # At equator, sin^2(pi/2) = 1, so should equal angle-averaged power
        P0 = bz_power(M, a, B)
        assert P_equator == pytest.approx(P0, rel=1e-12)


# ---------- Tidal heating ----------

class TestTidalHeating:
    """Tests for tidal heating functions."""

    def test_tidal_heating_positive(self):
        """Tidal heating rate should be positive."""
        M = 1.0
        a = 0.5
        r = 10.0
        m = 0.01
        rate = tidal_heating_rate(M, a, r, m)
        assert rate > 0.0

    def test_tidal_heating_zero_companion(self):
        """Tidal heating should vanish for m=0."""
        rate = tidal_heating_rate(1.0, 0.5, 10.0, 0.0)
        assert rate == pytest.approx(0.0, abs=1e-30)

    def test_tidal_heating_decreases_with_r(self):
        """Heating rate should decrease with separation."""
        M = 1.0
        a = 0.5
        m = 0.01
        rate_near = tidal_heating_rate(M, a, 5.0, m)
        rate_far = tidal_heating_rate(M, a, 20.0, m)
        assert rate_near > rate_far

    def test_tidal_phase_shift_positive_for_positive_spin(self):
        """Phase shift should be positive for prograde orbit and a > 0."""
        M = 1.0
        a = 0.5
        r = 10.0
        m = 0.01
        omega_orb = np.sqrt(M / r**3)
        dpsi = tidal_phase_shift(M, a, r, m, omega_orb)
        assert dpsi > 0.0

    def test_tidal_phase_shift_zero_spin(self):
        """Phase shift should be zero for a=0."""
        M = 1.0
        a = 0.0
        r = 10.0
        m = 0.01
        omega_orb = np.sqrt(M / r**3)
        dpsi = tidal_phase_shift(M, a, r, m, omega_orb)
        assert dpsi == pytest.approx(0.0, abs=1e-30)
