"""Tests for PPN framework."""

import numpy as np
import pytest
from ppn_framework.metric import PPNParameters
from ppn_framework.observables import (
    light_deflection,
    perihelion_precession,
    shapiro_delay,
)
from ppn_framework.constants import G, C, M_SUN, R_SUN, MERCURY, AU


class TestPPNParameters:
    def test_gr_is_default(self):
        ppn = PPNParameters()
        assert ppn.is_gr()

    def test_brans_dicke_limit(self):
        """Brans-Dicke with omega→∞ should approach GR."""
        ppn = PPNParameters.brans_dicke(omega=1e10)
        assert abs(ppn.gamma - 1.0) < 1e-8
        assert ppn.beta == 1.0

    def test_brans_dicke_gamma(self):
        """Check γ_BD = (ω+1)/(ω+2)."""
        ppn = PPNParameters.brans_dicke(omega=1.0)
        assert abs(ppn.gamma - 2.0 / 3.0) < 1e-10


class TestLightDeflection:
    def test_gr_solar_deflection(self):
        """GR predicts 1.75 arcsec for light grazing the Sun."""
        delta = light_deflection(R_SUN, M_SUN)
        delta_arcsec = delta * 180.0 * 3600.0 / np.pi
        assert abs(delta_arcsec - 1.75) < 0.01

    def test_gamma_scaling(self):
        """Deflection should scale as (1+γ)/2."""
        ppn_half = PPNParameters(gamma=0.5)
        ppn_gr = PPNParameters(gamma=1.0)
        d_half = light_deflection(R_SUN, M_SUN, ppn=ppn_half)
        d_gr = light_deflection(R_SUN, M_SUN, ppn=ppn_gr)
        expected_ratio = 0.75 / 1.0  # (1+0.5)/2 / (1+1)/2
        assert abs(d_half / d_gr - expected_ratio) < 1e-10


class TestPerihelionPrecession:
    def test_gr_mercury(self):
        """GR prediction: ~42.98 arcsec/century."""
        ppn = PPNParameters()
        omega_dot = perihelion_precession(MERCURY["a"], MERCURY["e"], M_SUN, ppn)
        T = 2.0 * np.pi * np.sqrt(MERCURY["a"]**3 / (G * M_SUN))
        dphi_per_orbit = omega_dot * T
        orbits_per_century = 100.0 * 365.25 * 86400.0 / T
        arcsec = dphi_per_orbit * orbits_per_century * 180.0 * 3600.0 / np.pi
        assert abs(arcsec - 42.98) < 0.1

    def test_brans_dicke_less_precession(self):
        """Brans-Dicke with ω=1 should give less precession than GR."""
        ppn_bd = PPNParameters.brans_dicke(omega=1.0)
        ppn_gr = PPNParameters()
        prec_bd = perihelion_precession(MERCURY["a"], MERCURY["e"], M_SUN, ppn_bd)
        prec_gr = perihelion_precession(MERCURY["a"], MERCURY["e"], M_SUN, ppn_gr)
        assert prec_bd < prec_gr


class TestShapiroDelay:
    def test_positive_delay(self):
        """Shapiro delay should always be positive."""
        delay = shapiro_delay(AU, AU, R_SUN, M_SUN)
        assert delay > 0

    def test_gr_order_of_magnitude(self):
        """Shapiro delay for Cassini experiment ~100 microseconds."""
        # Approximate: Earth-Saturn configuration
        delay = shapiro_delay(1.0 * AU, 10.0 * AU, R_SUN * 5, M_SUN)
        # Should be order 10-1000 microseconds
        assert 1e-6 < delay < 1e-2
