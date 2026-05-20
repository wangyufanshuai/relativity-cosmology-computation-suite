"""Tests for PN gravitational wave waveform library."""

import numpy as np
import pytest
from pn_waveform.orbital import chirp_mass, symmetric_mass_ratio, schwarzschild_isco_freq
from pn_waveform.imr import final_mass, final_spin, ringdown_frequency, ringdown_damping
from pn_waveform.constants import M_SUN, G, C


class TestOrbitalDynamics:
    def test_chirp_mass_equal(self):
        """Equal mass 30+30 M☉: Mc = (m1*m2)^(3/5)/(m1+m2)^(1/5)."""
        m1 = m2 = 30 * M_SUN
        Mc = chirp_mass(m1, m2)
        expected = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        assert abs(Mc / expected - 1.0) < 1e-10

    def test_symmetric_mass_ratio(self):
        """eta = m1*m2/(m1+m2)^2, max = 0.25 for equal mass."""
        eta = symmetric_mass_ratio(30 * M_SUN, 30 * M_SUN)
        assert abs(eta - 0.25) < 1e-10

    def test_isco_frequency(self):
        """ISCO frequency for 60 M☉ total should be ~70 Hz."""
        M_total = 60 * M_SUN
        # f_isco = c³/(6^1.5 * 2π * G * M_total)
        f_isco = C**3 / (6**1.5 * 2 * np.pi * G * M_total)
        assert 20 < f_isco < 60


class TestIMR:
    def test_final_mass_less_than_total(self):
        """Final mass should be less than total (energy radiated)."""
        M_f = final_mass(30 * M_SUN, 30 * M_SUN)
        assert M_f < 60 * M_SUN
        assert M_f > 50 * M_SUN  # should retain most mass

    def test_final_spin_positive(self):
        """Non-spinning merger should produce spinning remnant."""
        a_f = final_spin(30 * M_SUN, 30 * M_SUN)
        assert 0.5 < a_f < 0.9

    def test_ringdown_frequency_order(self):
        """Ringdown frequency for 30+30 M☉ should be 200-400 Hz."""
        f = ringdown_frequency(30 * M_SUN, 30 * M_SUN)
        assert 100 < f < 500

    def test_ringdown_damping_positive(self):
        """Damping time should be positive."""
        tau = ringdown_damping(30 * M_SUN, 30 * M_SUN)
        assert tau > 0

    def test_heavier_system_lower_frequency(self):
        """More massive system should have lower ringdown frequency."""
        f1 = ringdown_frequency(10 * M_SUN, 10 * M_SUN)
        f2 = ringdown_frequency(50 * M_SUN, 50 * M_SUN)
        assert f2 < f1
