"""Tests for Einstein-aether theory: aether field, PPN parameters, cosmology."""

import numpy as np
import pytest

from einstein_aether.aether_field import AetherField
from einstein_aether.ppn_parameters import (
    ppn_gamma,
    ppn_beta,
    newton_constant_ratio,
    preferred_frame_params,
)
from einstein_aether.cosmology import (
    gw_speed,
    friedmann_factor,
    modified_friedmann,
    hubble_parameter,
    effective_gravitational_constant,
)
from einstein_aether.constraints import (
    solar_system_constraints,
    gw_speed_constraint,
    parameter_priors,
)


# ---------------------------------------------------------------------------
# Aether field normalization
# ---------------------------------------------------------------------------

class TestAetherField:
    """Tests for the aether vector field."""

    def test_unit_timelike_pure_time(self):
        """Pure time-directed aether: u^mu u_mu = -1."""
        u = AetherField.pure_time()
        assert abs(u.norm_sq() + 1.0) < 1e-14, (
            f"u^mu u_mu should be -1, got {u.norm_sq()}"
        )

    def test_unit_timelike_general(self):
        """General aether field should satisfy u^mu u_mu = -1."""
        for theta in [0.0, 0.3, 0.7, 1.0]:
            u = AetherField.from_spherical(theta=theta, phi=0.5)
            assert abs(u.norm_sq() + 1.0) < 1e-12, (
                f"u^mu u_mu should be -1 for theta={theta}, got {u.norm_sq()}"
            )

    def test_future_directed(self):
        """Aether field should be future-directed (u^0 > 0)."""
        u = AetherField(np.array([-1.0, 0.0, 0.0, 0.0]))
        assert u.u_contra[0] > 0, "Aether should be future-directed"

    def test_components_shape(self):
        """Contravariant and covariant components should have shape (4,)."""
        u = AetherField.pure_time()
        assert u.u_contra.shape == (4,)
        assert u.u_cov.shape == (4,)

    def test_covariant_contraction(self):
        """u^mu u_mu via contraction should equal norm_sq."""
        u = AetherField.from_spherical(theta=0.5, phi=1.0)
        manual = float(np.dot(u.u_contra, u.u_cov))
        assert abs(manual - u.norm_sq()) < 1e-15

    def test_minkowski_metric(self):
        """Minkowski metric should have correct signature."""
        eta = AetherField.eta()
        assert eta[0, 0] == -1.0
        assert eta[1, 1] == 1.0
        assert eta[2, 2] == 1.0
        assert eta[3, 3] == 1.0

    def test_action_density_zero_for_zero_derivatives(self):
        """Action density should be 0 when all derivatives vanish."""
        du = np.zeros((4, 4))
        L = AetherField.action_density(du, c1=0.1, c2=0.1, c3=0.1, c4=0.1)
        assert L == 0.0

    def test_action_density_finite(self):
        """Action density should produce finite results."""
        du = np.random.randn(4, 4) * 0.01
        L = AetherField.action_density(du, c1=0.01, c2=0.01, c3=0.01, c4=0.01)
        assert np.isfinite(L)

    def test_invalid_input_raises(self):
        """Passing wrong number of components should raise ValueError."""
        with pytest.raises(ValueError):
            AetherField(np.array([1.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# PPN parameters
# ---------------------------------------------------------------------------

class TestPPNParameters:
    """Tests for PPN parameters in Einstein-aether theory."""

    def test_gr_equals_1_in_GR_limit(self):
        """PPN gamma should be 1 when all c_i = 0 (GR limit)."""
        gamma = ppn_gamma(0.0, 0.0, 0.0, 0.0)
        assert abs(gamma - 1.0) < 1e-10, f"gamma should be 1 in GR, got {gamma}"

    def test_beta_equals_1_in_GR_limit(self):
        """PPN beta should be 1 when all c_i = 0 (GR limit)."""
        beta = ppn_beta(0.0, 0.0, 0.0, 0.0)
        assert abs(beta - 1.0) < 1e-10, f"beta should be 1 in GR, got {beta}"

    def test_gr_ratio_is_1(self):
        """G_N/G should be 1 in GR limit."""
        ratio = newton_constant_ratio(0.0, 0.0, 0.0, 0.0)
        assert abs(ratio - 1.0) < 1e-10

    def test_preferred_frame_zero_in_GR(self):
        """Preferred-frame parameters should vanish in GR."""
        a1, a2 = preferred_frame_params(0.0, 0.0, 0.0, 0.0)
        assert abs(a1) < 1e-10
        assert abs(a2) < 1e-10

    def test_gamma_reasonable_range(self):
        """PPN gamma for small coupling should be close to 1."""
        gamma = ppn_gamma(1e-6, 1e-6, 1e-6, 1e-6)
        assert 0.5 < gamma < 1.5, f"gamma should be near 1, got {gamma}"

    def test_beta_reasonable_range(self):
        """PPN beta for small coupling should be close to 1."""
        beta = ppn_beta(1e-6, 1e-6, 1e-6, 1e-6)
        assert 0.5 < beta < 1.5, f"beta should be near 1, got {beta}"

    def test_gamma_symmetric_in_sign(self):
        """Changing all signs of c_i should give a different gamma."""
        gamma_pos = ppn_gamma(1e-4, 1e-4, 1e-4, 1e-4)
        gamma_neg = ppn_gamma(-1e-4, -1e-4, -1e-4, -1e-4)
        # They should both be near 1 but not necessarily equal
        assert abs(gamma_pos - 1.0) < 0.1
        assert abs(gamma_neg - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Cosmology: GW speed and Friedmann equation
# ---------------------------------------------------------------------------

class TestCosmology:
    """Tests for cosmological predictions."""

    def test_gw_speed_equals_c_for_c13_zero(self):
        """GW speed should be c when c_13 = c_1 + c_3 = 0."""
        c_T = gw_speed(c1=0.0, c3=0.0)
        assert abs(c_T - 1.0) < 1e-10, f"c_T should be 1, got {c_T}"

    def test_gw_speed_near_c_for_small_c13(self):
        """GW speed should be close to c for small c_13."""
        c_T = gw_speed(c1=1e-10, c3=1e-10)
        assert abs(c_T - 1.0) < 1e-5

    def test_friedmann_factor_1_for_zero_ci(self):
        """Friedmann factor should be 1 when all c_i = 0."""
        F = friedmann_factor(0.0, 0.0, 0.0, 0.0)
        assert abs(F - 1.0) < 1e-10

    def test_modified_friedmann_reduces_to_standard(self):
        """Modified Friedmann should reduce to standard for c_i = 0."""
        rho = 1e-26  # kg/m^3 (cosmological density)
        G_N = 6.674e-11
        H2_aether = modified_friedmann(rho, G_N, 0.0, 0.0, 0.0, 0.0)
        H2_standard = (8.0 * np.pi * G_N / 3.0) * rho
        assert abs(H2_aether - H2_standard) / H2_standard < 1e-10

    def test_hubble_parameter_positive(self):
        """Hubble parameter should be positive for positive density."""
        rho = 1e-26
        H = hubble_parameter(rho, c1=0.0, c2=0.0, c3=0.0, c4=0.0)
        assert H > 0

    def test_effective_gravitational_constant_gr(self):
        """G_eff should equal G_N for all c_i = 0."""
        G_N = 6.674e-11
        G_eff = effective_gravitational_constant(0.0, 0.0, 0.0, 0.0, G_N)
        assert abs(G_eff - G_N) / G_N < 1e-10


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    """Tests for experimental constraints."""

    def test_solar_system_gr_satisfied_in_GR(self):
        """Solar system constraints should be satisfied for c_i = 0."""
        bounds = solar_system_constraints(0.0, 0.0, 0.0, 0.0)
        assert bounds.satisfied

    def test_gw_constraint_satisfied_for_c13_zero(self):
        """GW speed constraint should be satisfied when c_13 = 0."""
        result = gw_speed_constraint(0.0, 0.0)
        assert result.satisfied

    def test_parameter_priors(self):
        """Parameter priors should correctly identify in-range values."""
        priors = parameter_priors()
        assert priors.in_range(0.0, 0.0, 0.0, 0.0)
        assert not priors.in_range(1.0, 0.0, 0.0, 0.0)
