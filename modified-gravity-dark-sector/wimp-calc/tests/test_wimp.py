"""Tests for wimp-calc: thermal relic density, thermal cross-section,
and freeze-out behavior."""

import numpy as np
import pytest

from wimp_calc.constants import M_PL, GEV_TO_KG, GEV_TO_JOULE
from wimp_calc.thermal import (
    relic_density,
    equilibrium_abundance,
    thermal_average_sv,
    solve_freezeout,
)


# ---------------------------------------------------------------------------
# Constant cross-section (canonical WIMP)
# ---------------------------------------------------------------------------

def _const_sv(E_cm):
    """Constant cross-section: sigma*v = 3e-26 cm^3/s (thermal relic benchmark)."""
    return 3.0e-26


# ---------------------------------------------------------------------------
# Relic density
# ---------------------------------------------------------------------------

class TestRelicDensity:
    """Test relic density calculation."""

    def test_relic_density_positive(self):
        """Omega h^2 should be positive for a physical WIMP."""
        m_chi = 100.0  # GeV
        result = solve_freezeout(m_chi, _const_sv)
        Y_inf = result['Y_inf']
        oh2 = relic_density(Y_inf, m_chi)
        assert oh2 > 0.0

    def test_relic_density_reasonable(self):
        """For canonical cross-section, Omega h^2 ~ 0.1."""
        m_chi = 100.0  # GeV
        result = solve_freezeout(m_chi, _const_sv)
        oh2 = relic_density(result['Y_inf'], m_chi)
        # The canonical 3e-26 cm^3/s gives ~0.1 (within an order of magnitude)
        assert 0.001 < oh2 < 10.0, f"Omega h^2 = {oh2} outside reasonable range"

    def test_relic_density_increases_with_mass(self):
        """For fixed <sigma v>, Omega h^2 ~ m_chi (approximately)."""
        m1 = 50.0
        m2 = 200.0
        r1 = solve_freezeout(m1, _const_sv)
        r2 = solve_freezeout(m2, _const_sv)
        oh2_1 = relic_density(r1['Y_inf'], m1)
        oh2_2 = relic_density(r2['Y_inf'], m2)
        # Higher mass should give higher relic density (for same sigma*v)
        assert oh2_2 > oh2_1


# ---------------------------------------------------------------------------
# Thermal cross-section
# ---------------------------------------------------------------------------

class TestThermalCrossSection:
    """Test thermal average cross-section."""

    def test_thermal_cross_section_positive(self):
        """<sigma v> should be positive for a positive cross-section."""
        m_chi = 100.0
        T = 10.0
        sv = thermal_average_sv(m_chi, T, _const_sv)
        assert sv > 0.0

    def test_thermal_cross_section_const(self):
        """For constant sigma*v, the thermal average should equal that constant."""
        m_chi = 100.0
        T = 50.0  # high T relative to m
        sv = thermal_average_sv(m_chi, T, _const_sv)
        assert sv == pytest.approx(3.0e-26, rel=0.5)


# ---------------------------------------------------------------------------
# Freeze-out behavior
# ---------------------------------------------------------------------------

class TestFreezeOut:
    """Test freeze-out dynamics."""

    def test_freeze_out_occurs(self):
        """Y(x) should drop significantly around x ~ 20-30."""
        m_chi = 100.0
        result = solve_freezeout(m_chi, _const_sv)
        x = result['x']
        Y = result['Y']

        # Find Y at x ~ 10 (before freeze-out) and x ~ 100 (after)
        idx_before = np.argmin(np.abs(x - 10.0))
        idx_after = np.argmin(np.abs(x - 100.0))

        Y_before = Y[idx_before]
        Y_after = Y[idx_after]

        # After freeze-out, Y should be much smaller than before
        assert Y_after < Y_before * 0.1

    def test_equilibrium_abundance_decreasing(self):
        """Y_eq should decrease exponentially for x >> 1 (Boltzmann suppression)."""
        m_chi = 100.0
        Y_x10 = equilibrium_abundance(m_chi, m_chi / 10.0)
        Y_x30 = equilibrium_abundance(m_chi, m_chi / 30.0)
        assert Y_x30 < Y_x10

    def test_equilibrium_abundance_positive(self):
        """Y_eq should always be positive."""
        m_chi = 100.0
        T_vals = [200.0, 100.0, 50.0, 10.0, 1.0]
        for T in T_vals:
            Y = equilibrium_abundance(m_chi, T)
            assert Y > 0.0

    def test_solve_freezeout_returns_dict(self):
        """solve_freezeout should return a dict with expected keys."""
        m_chi = 100.0
        result = solve_freezeout(m_chi, _const_sv)
        assert 'x' in result
        assert 'Y' in result
        assert 'Y_inf' in result
        assert len(result['x']) > 10
        assert len(result['Y']) == len(result['x'])
