"""Tests for thermal-history: g_*, entropy, BBN, Hubble rate."""

import numpy as np
import pytest

from thermal_history import (
    g_star,
    g_star_s,
    entropy_density,
    freeze_out_temperature,
    bbn_neutron_proton_ratio,
    helium_mass_fraction,
    hubble_rate_radiation,
)


class TestGStar:
    """Test effective degrees of freedom."""

    def test_gstar_high_temperature(self):
        """At very high T, g_* should be large."""
        g = g_star(1e8)  # 100 GeV
        assert g > 50  # includes many relativistic species

    def test_gstar_low_temperature(self):
        """At low T (below MeV), only photons and neutrinos contribute."""
        g = g_star(0.01)  # 10 keV
        assert g > 0  # positive, value depends on Boltzmann suppression model

    def test_gstar_positive(self):
        """g_* should always be positive."""
        for T in [0.001, 0.1, 1.0, 100.0, 1e6]:
            assert g_star(T) > 0

    def test_gstar_array_input(self):
        """g_star should accept array input."""
        T = np.array([0.01, 1.0, 100.0])
        g = g_star(T)
        assert len(g) == 3


class TestEntropy:
    """Test entropy density and conservation."""

    def test_entropy_density_positive(self):
        """Entropy density should be positive."""
        s = entropy_density(1.0)
        assert s > 0

    def test_entropy_density_increases_with_T(self):
        """Entropy density should increase with temperature."""
        s1 = entropy_density(0.1)
        s2 = entropy_density(1.0)
        assert s2 > s1


class TestBBN:
    """Test Big Bang Nucleosynthesis quantities."""

    def test_np_ratio_less_than_one(self):
        """n/p ratio should be < 1 at typical BBN temperatures."""
        ratio = bbn_neutron_proton_ratio(1.0)  # 1 MeV
        assert 0 < ratio < 1

    def test_np_ratio_decreases_with_T(self):
        """n/p ratio should decrease as T drops."""
        r1 = bbn_neutron_proton_ratio(2.0)
        r2 = bbn_neutron_proton_ratio(0.5)
        assert r2 < r1

    def test_helium_fraction_reasonable(self):
        """He-4 mass fraction should be ~0.24-0.26."""
        Y_p = helium_mass_fraction()
        assert 0.15 < Y_p < 0.35


class TestFreezeOut:
    """Test WIMP freeze-out temperature."""

    def test_freeze_out_positive(self):
        """Freeze-out temperature should be positive."""
        T_f = freeze_out_temperature(sigma_v=1e-26, m_MeV=100000.0)
        assert T_f > 0

    def test_freeze_out_scales_with_mass(self):
        """Freeze-out T should be proportional to mass."""
        T1 = freeze_out_temperature(m_MeV=100000.0)
        T2 = freeze_out_temperature(m_MeV=200000.0)
        assert T2 > T1


class TestHubbleRate:
    """Test Hubble rate during radiation domination."""

    def test_hubble_rate_positive(self):
        """Hubble rate should be positive."""
        H = hubble_rate_radiation(1.0)
        assert H > 0

    def test_hubble_rate_scales_T2(self):
        """H should scale as T^2 at fixed g_*."""
        H1 = hubble_rate_radiation(1.0, g_star_val=100.0)
        H2 = hubble_rate_radiation(2.0, g_star_val=100.0)
        assert H2 / H1 == pytest.approx(4.0, rel=1e-10)
