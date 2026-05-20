"""Tests for mercury-precession analytical formulae."""

import numpy as np
import pytest

from mercury_precession.analytical import (
    schwarzschild_precession,
    schwarzschild_2pn,
    quadrupole_precession,
    total_analytical,
    precession_vs_eccentricity,
)
from mercury_precession.constants import A_MERCURY, C, E_MERCURY, G, M_SUN
from mercury_precession.ppn import precession_ppn


class TestSchwarzschildPrecession:
    def test_well_known_value(self):
        """Mercury's GR precession should be ~42.98 arcsec/century."""
        result = total_analytical(include_2pn=False, include_J2=False)
        # Well-known value: ~42.98 arcsec/century
        assert abs(result["total_arcsec_per_century"] - 42.98) < 0.05

    def test_analytical_formula(self):
        """Verify Δφ = 6πGM / [a(1-e²)c²]."""
        expected = 6.0 * np.pi * G * M_SUN / (A_MERCURY * (1.0 - E_MERCURY**2) * C**2)
        actual = schwarzschild_precession()
        assert abs(actual - expected) / expected < 1e-10

    def test_circular_orbit(self):
        """For e=0, precession reduces to 6πGM/(ac²)."""
        delta_phi = schwarzschild_precession(e=0.0)
        expected = 6.0 * np.pi * G * M_SUN / (A_MERCURY * C**2)
        assert abs(delta_phi - expected) / expected < 1e-10

    def test_higher_eccentricity_more_precession(self):
        """Higher eccentricity → more precession per orbit."""
        phi_low = schwarzschild_precession(e=0.1)
        phi_high = schwarzschild_precession(e=0.5)
        assert phi_high > phi_low


class Test2PN:
    def test_2pn_is_small(self):
        """2PN correction should be much smaller than 1PN."""
        phi_1pn = schwarzschild_precession()
        phi_2pn = schwarzschild_2pn()
        # 2PN is O(v⁴/c⁴) smaller
        assert abs(phi_2pn / phi_1pn) < 1e-6


class TestQuadrupole:
    def test_j2_is_tiny(self):
        """Solar J2 contribution should be tiny (~0.02 arcsec/century)."""
        phi_J2 = quadrupole_precession()
        # Should be orders of magnitude smaller than GR
        phi_GR = schwarzschild_precession()
        assert phi_J2 / phi_GR < 0.01


class TestPPN:
    def test_gr_recovered(self):
        """PPN with γ=β=1 should reproduce GR."""
        phi_ppn = precession_ppn(gamma=1.0, beta=1.0)
        phi_gr = schwarzschild_precession()
        assert abs(phi_ppn - phi_gr) / phi_gr < 1e-10

    def test_gamma_effect(self):
        """Larger γ → more precession."""
        phi_low = precession_ppn(gamma=0.9)
        phi_high = precession_ppn(gamma=1.1)
        assert phi_high > phi_low

    def test_brans_dicke(self):
        """Brans-Dicke with ω→∞ should approach GR."""
        omega = 1e10
        gamma_bd = (omega + 1.0) / (omega + 2.0)
        phi_bd = precession_ppn(gamma=gamma_bd, beta=1.0)
        phi_gr = precession_ppn(gamma=1.0, beta=1.0)
        assert abs(phi_bd - phi_gr) / phi_gr < 1e-8


class TestArrayFunctions:
    def test_precession_vs_eccentricity(self):
        """Should return array of same size as input."""
        e_range = np.linspace(0.01, 0.99, 50)
        result = precession_vs_eccentricity(e_range)
        assert result.shape == e_range.shape
        # Should be monotonically increasing
        assert np.all(np.diff(result) > 0)
