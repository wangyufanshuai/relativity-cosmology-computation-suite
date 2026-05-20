"""Tests for fR-gravity: Hu-Sawicki, Starobinsky, and chameleon models."""

import numpy as np
import pytest

from fR_gravity.hu_sawicki import HuSawickiModel
from fR_gravity.starobinsky import StarobinskyModel
from fR_gravity.chameleon import ChameleonScreening


# ============================================================================
# Hu-Sawicki model tests
# ============================================================================

class TestHuSawicki:
    """Test Hu-Sawicki f(R) gravity model."""

    def setup_method(self):
        """Use typical cosmological parameters."""
        self.model = HuSawickiModel(m2=1.0, c1=1.0, c2=1.0, n=1)

    def test_f_negative(self):
        """f(R) should be negative for positive R."""
        f = self.model.f(1.0)
        assert f < 0

    def test_f_zero_at_zero(self):
        """f(0) = 0."""
        assert self.model.f(0.0) == pytest.approx(0.0, abs=1e-14)

    def test_f_R_negative(self):
        """f_R should be negative for positive R (standard Hu-Sawicki)."""
        fR = self.model.f_R(1.0)
        assert fR < 0

    def test_f_R_approaches_zero_high_R(self):
        """f_R -> 0 as R -> infinity (reduces to GR)."""
        fR_small = self.model.f_R(1.0)
        fR_large = self.model.f_R(1000.0)
        assert abs(fR_large) < abs(fR_small)

    def test_lcdm_limit(self):
        """For very small coupling, f(R) ~ 0 (LCDM limit)."""
        model_weak = HuSawickiModel(m2=1.0, c1=1e-12, c2=1.0, n=1)
        assert model_weak.lcdm_limit_check(R=1.0, atol=1e-8)

    def test_effective_cosmological_constant(self):
        """Lambda_eff = m^2 * c1 / (2*c2)."""
        Lambda = self.model.effective_cosmological_constant(100.0)
        expected = 1.0 * 1.0 / (2.0 * 1.0)
        assert Lambda == pytest.approx(expected)

    def test_scalaron_mass_positive(self):
        """Scalaron mass should be positive (tachyon-free)."""
        m_phi = self.model.scalaron_mass(1.0)
        assert m_phi > 0

    def test_scalaron_mass_squared_positive(self):
        """Scalaron mass squared should be positive."""
        m2 = self.model.scalaron_mass_squared(1.0)
        assert m2 > 0

    def test_f_RR_sensible(self):
        """f_RR should be nonzero for the scalaron to exist."""
        frr = self.model.f_RR(1.0)
        assert frr != 0.0

    def test_invalid_n_raises(self):
        """n < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            HuSawickiModel(m2=1.0, c1=1.0, c2=1.0, n=0)

    def test_array_input(self):
        """Functions should handle array inputs."""
        R = np.array([0.1, 1.0, 10.0])
        f = self.model.f(R)
        fR = self.model.f_R(R)
        assert f.shape == R.shape
        assert fR.shape == R.shape

    def test_n_equals_2(self):
        """Model with n=2 should run without errors."""
        model2 = HuSawickiModel(m2=1.0, c1=1.0, c2=1.0, n=2)
        f = model2.f(1.0)
        fR = model2.f_R(1.0)
        assert np.isfinite(f)
        assert np.isfinite(fR)


# ============================================================================
# Starobinsky model tests
# ============================================================================

class TestStarobinsky:
    """Test Starobinsky R^2 inflation model."""

    def setup_method(self):
        self.alpha = 1.0  # simplified
        self.model = StarobinskyModel(self.alpha)

    def test_f_equals_R_plus_alpha_R2(self):
        """f(R) = R + alpha*R^2."""
        R = 2.0
        expected = R + self.alpha * R**2
        assert self.model.f(R) == pytest.approx(expected)

    def test_f_R_equals_1_plus_2alphaR(self):
        """f_R = 1 + 2*alpha*R."""
        R = 3.0
        expected = 1.0 + 2.0 * self.alpha * R
        assert self.model.f_R(R) == pytest.approx(expected)

    def test_f_R_equals_GR_for_small_R(self):
        """f_R -> 1 as R -> 0 (recovers GR)."""
        fR = self.model.f_R(1e-10)
        assert fR == pytest.approx(1.0, rel=1e-6)

    def test_f_RR_constant(self):
        """f_RR = 2*alpha (constant)."""
        R = np.array([1.0, 5.0, 100.0])
        frr = self.model.f_RR(R)
        expected = 2.0 * self.alpha
        np.testing.assert_allclose(frr, expected)

    def test_scalaron_mass_squared_positive(self):
        """m_phi^2 = 1/(6*alpha) > 0."""
        m2 = self.model.scalaron_mass_squared(1.0)
        expected = 1.0 / (6.0 * self.alpha)
        assert m2 == pytest.approx(expected)
        assert m2 > 0

    def test_scalaron_mass_positive(self):
        """m_phi = 1/sqrt(6*alpha) > 0."""
        m = self.model.scalaron_mass(1.0)
        assert m > 0

    def test_spectral_index(self):
        """n_s ~ 1 - 2/N for N=55."""
        ns = StarobinskyModel.spectral_index(55.0)
        expected = 1.0 - 2.0 / 55.0
        assert ns == pytest.approx(expected, rel=1e-10)
        # Should be slightly less than 1
        assert 0.96 < ns < 1.0

    def test_tensor_to_scalar_ratio(self):
        """r ~ 12/N^2 for N=55."""
        r = StarobinskyModel.tensor_to_scalar_ratio(55.0)
        expected = 12.0 / 55.0**2
        assert r == pytest.approx(expected, rel=1e-10)
        # Should be very small
        assert r < 0.01

    def test_consistency_relation(self):
        """r = 8(1-n_s)/3 * (1-n_s) for Starobinsky approximately."""
        N = 55.0
        ns = StarobinskyModel.spectral_index(N)
        r = StarobinskyModel.tensor_to_scalar_ratio(N)
        # Check approximate consistency: r ~ 3/2 * (1 - ns)^2
        # Actually r = 12/N^2 and 1-ns = 2/N, so r = 3*(1-ns)^2
        assert r == pytest.approx(3.0 * (1.0 - ns)**2, rel=1e-10)

    def test_slow_roll_epsilon(self):
        """epsilon should be small during inflation (large R)."""
        eps = self.model.slow_roll_epsilon(1000.0)
        assert eps > 0
        assert eps < 1.0

    def test_slow_roll_eta(self):
        """eta should be small during inflation (large R)."""
        eta = self.model.slow_roll_eta(1000.0)
        assert eta > 0

    def test_e_folds_positive(self):
        """e-folds should be positive."""
        N = self.model.e_folds(R_end=1.0, R_start=100.0)
        assert N > 0

    def test_invalid_alpha(self):
        """alpha <= 0 should raise ValueError."""
        with pytest.raises(ValueError):
            StarobinskyModel(0.0)
        with pytest.raises(ValueError):
            StarobinskyModel(-1.0)


# ============================================================================
# Chameleon screening tests
# ============================================================================

class TestChameleon:
    """Test chameleon screening mechanism."""

    def setup_method(self):
        self.M_Pl = 1.0  # simplified units
        self.chameleon = ChameleonScreening(self.M_Pl)

    def test_beta_value(self):
        """beta = 1/sqrt(6) for metric f(R) gravity."""
        assert self.chameleon.beta == pytest.approx(1.0 / np.sqrt(6.0))

    def test_scalaron_from_fR_zero(self):
        """phi(f_R=0) = 0."""
        phi = self.chameleon.scalaron_from_fR(0.0)
        assert phi == pytest.approx(0.0, abs=1e-14)

    def test_scalaron_from_fR_positive(self):
        """For f_R > 0, phi should be positive."""
        phi = self.chameleon.scalaron_from_fR(0.1)
        assert phi > 0

    def test_V_eff_positive(self):
        """Effective potential should be positive for positive inputs."""
        V_eff = self.chameleon.V_eff(phi=0.1, V=0.01, rho=1.0)
        assert V_eff > 0

    def test_V_eff_increases_with_density(self):
        """V_eff should increase with ambient density."""
        V1 = self.chameleon.V_eff(phi=0.1, V=0.01, rho=1.0)
        V2 = self.chameleon.V_eff(phi=0.1, V=0.01, rho=10.0)
        assert V2 > V1

    def test_thin_shell_parameter_positive(self):
        """DeltaR/R should be positive."""
        dR_over_R = self.chameleon.thin_shell_parameter(
            rho=1.0, R_obj=1.0, phi_inf=0.1
        )
        assert dR_over_R > 0

    def test_screening_for_massive_object(self):
        """A massive enough object should be screened."""
        # Large Newtonian potential -> screened
        screened = self.chameleon.screening_condition(
            Phi_obj=10.0, phi_inf=0.01
        )
        assert screened

    def test_no_screening_for_light_object(self):
        """A light object should NOT be screened."""
        screened = self.chameleon.screening_condition(
            Phi_obj=1e-10, phi_inf=10.0
        )
        assert not screened

    def test_is_screened_alias(self):
        """is_screened should match screening_condition."""
        assert self.chameleon.is_screened(10.0, 0.01) == \
            self.chameleon.screening_condition(10.0, 0.01)

    def test_fifth_force_decays(self):
        """Fifth force should decay exponentially with distance."""
        F1 = self.chameleon.fifth_force_magnitude(
            m_phi=1.0, rho=1.0, grad_rho=1.0, r=1.0
        )
        F2 = self.chameleon.fifth_force_magnitude(
            m_phi=1.0, rho=1.0, grad_rho=1.0, r=10.0
        )
        assert F2 < F1

    def test_fifth_force_yukawa_suppression(self):
        """Yukawa factor exp(-m*r) should be in (0, 1]."""
        y = self.chameleon.fifth_force_yukawa_suppression(m_phi=1.0, r=1.0)
        assert 0 < y <= 1.0
        y2 = self.chameleon.fifth_force_yukawa_suppression(m_phi=1.0, r=0.0)
        assert y2 == pytest.approx(1.0)

    def test_effective_coupling_screened(self):
        """Effective coupling should be smaller when screened."""
        beta_eff_screened = self.chameleon.effective_coupling(
            Phi_obj=100.0, phi_inf=0.01
        )
        beta_eff_unscreened = self.chameleon.effective_coupling(
            Phi_obj=1e-10, phi_inf=0.01
        )
        assert beta_eff_screened < beta_eff_unscreened

    def test_effective_coupling_unscreened_equals_beta(self):
        """For unscreened objects, beta_eff = beta."""
        beta_eff = self.chameleon.effective_coupling(
            Phi_obj=1e-10, phi_inf=10.0
        )
        assert beta_eff == pytest.approx(self.chameleon.beta)

    def test_custom_beta(self):
        """Should accept custom beta."""
        cham = ChameleonScreening(M_Pl=1.0, beta=0.5)
        assert cham.beta == pytest.approx(0.5)
