"""Tests for geodesic-deviation: Jacobi equation, tidal forces, Raychaudhuri."""

import numpy as np
import pytest

from geodesic_deviation import (
    jacobi_equation_flat,
    solve_flat_deviation,
    schwarzschild_tidal_radial,
    schwarzschild_tidal_transverse,
    raychaudhuri_expansion,
    raychaudhuri_geodesic_congruence,
    riemann_schwarzschild,
    tidal_tensor_trace,
)


class TestFlatDeviation:
    """Test geodesic deviation in flat space."""

    def test_jacobi_flat_rhs(self):
        """In flat space, d^2 xi / dlambda^2 = 0."""
        result = jacobi_equation_flat(0.0, [1.0, 0.1])
        assert result == [0.1, 0.0]

    def test_flat_deviation_linear_growth(self):
        """Separation should grow linearly: xi = xi0 + dxi0 * lambda."""
        xi0, dxi0 = 1.0, 0.5
        sol = solve_flat_deviation(xi0, dxi0, lambda_span=(0, 10))
        xi_end = sol.y[0][-1]
        expected = xi0 + dxi0 * 10.0
        assert xi_end == pytest.approx(expected, rel=0.01)

    def test_flat_deviation_rate_constant(self):
        """Rate of change should be constant in flat space."""
        sol = solve_flat_deviation(1.0, 0.3, lambda_span=(0, 5))
        rates = sol.y[1]
        np.testing.assert_allclose(rates, 0.3, atol=0.01)


class TestTidalForces:
    """Test Schwarzschild tidal forces."""

    def test_radial_tidal_negative(self):
        """Radial tidal force coefficient should be negative (stretching)."""
        val = schwarzschild_tidal_radial(10.0, M=1.0)
        assert val < 0

    def test_transverse_tidal_positive(self):
        """Transverse tidal force coefficient should be positive (compression)."""
        val = schwarzschild_tidal_transverse(10.0, M=1.0)
        assert val > 0

    def test_tidal_decay_r3(self):
        """Tidal forces should decay as 1/r^3."""
        r1, r2 = 10.0, 20.0
        f1 = abs(schwarzschild_tidal_radial(r1))
        f2 = abs(schwarzschild_tidal_radial(r2))
        ratio = f2 / f1
        assert ratio == pytest.approx(1.0 / 8.0, rel=0.01)


class TestRaychaudhuri:
    """Test Raychaudhuri equation."""

    def test_focusing_with_shear(self):
        """Expansion should decrease when shear is nonzero."""
        dtheta = raychaudhuri_expansion(0, 1.0, sigma_sq=1.0, omega_sq=0.0, R_munu_un_un=0.0)
        assert dtheta < 0

    def test_anti_focusing_with_vorticity(self):
        """Vorticity should counteract focusing."""
        d1 = raychaudhuri_expansion(0, 1.0, sigma_sq=0.0, omega_sq=0.0)
        d2 = raychaudhuri_expansion(0, 1.0, sigma_sq=0.0, omega_sq=1.0)
        assert d2 > d1

    def test_congruence_expansion_decreases(self):
        """Expansion should decrease over time for irrotational congruence."""
        sol = raychaudhuri_geodesic_congruence(theta0=1.0, sigma_sq=0.1)
        theta_start = sol.y[0][0]
        theta_end = sol.y[0][-1]
        assert theta_end < theta_start


class TestRiemann:
    """Test Riemann tensor components."""

    def test_riemann_components_exist(self):
        """All key Riemann components should be returned."""
        R = riemann_schwarzschild(10.0, np.pi / 4, M=1.0)
        assert "R_trtr" in R
        assert "R_tptp" in R

    def test_tidal_tensor_trace_free(self):
        """In vacuum, the tidal tensor should be trace-free."""
        # E_rr + E_theta_theta + E_phi_phi = 0
        r = 10.0
        E_rr = schwarzschild_tidal_radial(r, M=1.0)
        E_tt = schwarzschild_tidal_transverse(r, M=1.0)
        trace = tidal_tensor_trace([E_rr, E_tt, E_tt])
        assert trace == pytest.approx(0.0, abs=1e-10)
