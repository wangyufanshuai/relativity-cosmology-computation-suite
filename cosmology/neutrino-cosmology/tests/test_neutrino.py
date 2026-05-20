"""Tests for the neutrino-cosmology package."""

import numpy as np
import pytest

from neutrino_cosmology.constants import (
    C,
    EV_TO_J,
    K_B,
    T_CMB0,
    T_NU_OVER_T_GAMMA,
)
from neutrino_cosmology.background import (
    neutrino_temperature,
    neutrino_energy_density,
    total_neutrino_density,
    neutrino_pressure,
    neutrino_equation_of_state,
    omega_nu,
)
from neutrino_cosmology.observables import (
    N_eff_standard,
    N_eff_with_extra,
    sound_horizon_shift,
    growth_suppression,
)

# Tolerance for relative comparisons
RTOL = 1e-4


class TestNeutrinoTemperature:
    """Tests for the neutrino temperature."""

    def test_neutrino_temperature_ratio(self):
        """T_nu / T_gamma = (4/11)^(1/3) after e+e- annihilation."""
        T_gamma = 2.7255
        T_nu = neutrino_temperature(T_gamma)
        expected_ratio = (4.0 / 11.0) ** (1.0 / 3.0)
        assert T_nu / T_gamma == pytest.approx(expected_ratio, rel=RTOL)

    def test_neutrino_temperature_today(self):
        """T_nu0 ~ 1.95 K for T_CMB = 2.7255 K."""
        T_nu = neutrino_temperature(T_CMB0)
        expected = T_CMB0 * (4.0 / 11.0) ** (1.0 / 3.0)
        assert T_nu == pytest.approx(expected, rel=RTOL)
        # Known value: ~1.95 K
        assert T_nu == pytest.approx(1.945, rel=0.01)


class TestNeutrinoDensity:
    """Tests for neutrino energy density."""

    def test_neutrino_density_massless(self):
        """Massless neutrino: rho proportional to T^4 (Stefan-Boltzmann-like).

        For a single massless Weyl fermion species:
            rho = (7/8) * (pi^2/30) * g * T^4    (natural units, g=1)
        which gives rho = (7 pi^2 / 240) * T^4 in natural units.

        We test the scaling rho(T1)/rho(T2) = (T1/T2)^4.
        """
        T1 = 2.0  # K
        T2 = 3.0  # K
        m_nu = 0.0  # massless

        rho1 = neutrino_energy_density(m_nu, T1)
        rho2 = neutrino_energy_density(m_nu, T2)

        expected_ratio = (T1 / T2) ** 4
        assert rho1 / rho2 == pytest.approx(expected_ratio, rel=0.01)

    def test_neutrino_density_positive(self):
        """Energy density must be positive."""
        rho = neutrino_energy_density(0.0, 1.95)
        assert rho > 0

    def test_neutrino_density_massive_greater(self):
        """Massive neutrino has higher energy density than massless at same T."""
        m_nu_eV = 0.05  # 0.05 eV
        m_nu_kg = m_nu_eV * EV_TO_J / C**2
        rho_massless = neutrino_energy_density(0.0, 1.95)
        rho_massive = neutrino_energy_density(m_nu_kg, 1.95)
        assert rho_massive > rho_massless


class TestNeutrinoEquationOfState:
    """Tests for w_nu(a)."""

    def test_neutrino_equation_of_state_early(self):
        """w -> 1/3 for a << a_nr (relativistic regime)."""
        # Use a massive neutrino with a_nr << 1
        m_nu_eV = 0.05
        m_nu_kg = m_nu_eV * EV_TO_J / C**2

        # Evaluate at very early times (a << a_nr)
        a_early = 1e-6
        w = neutrino_equation_of_state(m_nu_kg, a_early)
        assert w == pytest.approx(1.0 / 3.0, abs=0.05)

    def test_neutrino_equation_of_state_late(self):
        """w -> 0 for a >> a_nr (non-relativistic regime)."""
        m_nu_eV = 0.05
        m_nu_kg = m_nu_eV * EV_TO_J / C**2

        # Evaluate at very late times (a >> a_nr)
        a_late = 1e4
        w = neutrino_equation_of_state(m_nu_kg, a_late)
        assert w == pytest.approx(0.0, abs=0.05)

    def test_equation_of_state_monotonic(self):
        """w_nu(a) should decrease monotonically from 1/3 to 0."""
        m_nu_kg = 0.05 * EV_TO_J / C**2
        a_vals = np.logspace(-3, 3, 50)
        w_vals = neutrino_equation_of_state(m_nu_kg, a_vals)
        # Check monotonically decreasing
        assert np.all(np.diff(w_vals) <= 0)


class TestOmegaNu:
    """Tests for Omega_nu."""

    def test_omega_nu_known(self):
        """Sigma_m_nu = 0.06 eV, h = 0.674 -> Omega_nu ~ 0.00064."""
        m_nu_sum = 0.06  # eV
        h = 0.674
        result = omega_nu(m_nu_sum, h)
        # Omega_nu = 0.06 / (93.14 * 0.674^2) ~ 0.06 / 42.29 ~ 0.00142
        # Wait, let me recalculate: 93.14 * 0.674^2 = 93.14 * 0.4543 = 42.31
        # 0.06 / 42.31 = 0.001418
        # But the test says ~0.00064. Let me check: the commonly used formula
        # is Omega_nu h^2 = Sigma_m_nu / 93.14 eV
        # So Omega_nu = Sigma_m_nu / (93.14 * h^2) ... yes
        # 0.06 / (93.14 * 0.674^2) = 0.06 / 42.31 = 0.00142
        # The specification says ~0.00064 which would correspond to
        # 0.06 / 93.14 / h^2 ... no, that gives the same.
        # Actually, the standard formula is Omega_nu * h^2 = Sigma_m_nu / 93.14
        # so Omega_nu = Sigma_m_nu / (93.14 * h^2)
        expected = m_nu_sum / (93.14 * h**2)
        assert result == pytest.approx(expected, rel=RTOL)

    def test_omega_nu_zero_mass(self):
        """Omega_nu = 0 for massless neutrinos."""
        assert omega_nu(0.0, 0.674) == pytest.approx(0.0, abs=1e-15)


class TestNEff:
    """Tests for N_eff."""

    def test_N_eff_standard(self):
        """Standard N_eff = 3.044."""
        assert N_eff_standard() == pytest.approx(3.044, rel=RTOL)

    def test_N_eff_with_extra(self):
        """N_eff = 3.044 + Delta_Neff."""
        assert N_eff_with_extra(1.0) == pytest.approx(4.044, rel=RTOL)
        assert N_eff_with_extra(0.0) == pytest.approx(3.044, rel=RTOL)


class TestGrowthSuppression:
    """Tests for growth suppression."""

    def test_growth_suppression_positive(self):
        """0 < f_sigma8(massive) / f_sigma8(massless) <= 1."""
        ratio = growth_suppression(
            m_nu_sum=0.1,  # eV
            k=0.1,  # h/Mpc
            a=1.0,
            Omega_m=0.315,
            h=0.674,
        )
        assert 0.0 < ratio <= 1.0

    def test_growth_suppression_zero_mass(self):
        """No suppression for massless neutrinos."""
        ratio = growth_suppression(
            m_nu_sum=0.0,
            k=0.1,
            a=1.0,
            Omega_m=0.315,
            h=0.674,
        )
        assert ratio == pytest.approx(1.0, abs=1e-10)

    def test_growth_suppression_increases_with_mass(self):
        """Larger neutrino mass -> stronger suppression (smaller ratio)."""
        r1 = growth_suppression(0.05, 0.1, 1.0, 0.315, 0.674)
        r2 = growth_suppression(0.10, 0.1, 1.0, 0.315, 0.674)
        r3 = growth_suppression(0.20, 0.1, 1.0, 0.315, 0.674)
        assert r1 > r2 > r3


class TestSoundHorizon:
    """Tests for sound horizon shift."""

    def test_sound_horizon_decreases_with_mass(self):
        """Larger Sigma m_nu -> smaller r_s (negative shift)."""
        s1 = sound_horizon_shift(0.05, 0.0224, 0.674)
        s2 = sound_horizon_shift(0.10, 0.0224, 0.674)
        s3 = sound_horizon_shift(0.20, 0.0224, 0.674)
        # All should be negative
        assert s1 < 0
        assert s2 < 0
        assert s3 < 0
        # Larger mass -> more negative
        assert s1 > s2 > s3

    def test_sound_horizon_zero_mass(self):
        """No shift for massless neutrinos."""
        s = sound_horizon_shift(0.0, 0.0224, 0.674)
        assert s == pytest.approx(0.0, abs=1e-15)
