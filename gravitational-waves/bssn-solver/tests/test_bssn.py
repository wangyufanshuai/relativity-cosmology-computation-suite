"""Tests for BSSN formulation: variable decomposition, constraints, initial data."""

import numpy as np
import pytest

from bssn_solver.bssn_variables import (
    BSSNState,
    physical_to_bssn,
    bssn_to_physical,
    compute_det_gamma_tilde,
    compute_trace_A_tilde,
    _determinant_3x3,
    _inverse_3x3,
)
from bssn_solver.constraints import (
    hamiltonian_constraint,
    momentum_constraint,
    hamiltonian_constraint_simplified,
    compute_constraint_norm,
)
from bssn_solver.gauge import (
    gradient,
    divergence,
    partial_i,
    one_plus_log_rhs,
    gamma_driver_rhs,
)
from bssn_solver.initial_data import (
    flat_data,
    schwarzschild_puncture,
    schwarzschild_puncture_bssn_direct,
    brill_wave,
    binary_puncture,
    make_grid,
)


def _make_flat_state(N=5):
    """Create a small flat-space BSSNState for testing."""
    shape = (N, N, N)
    phi = np.zeros(shape)
    gamma_tilde = np.zeros((3, 3, N, N, N))
    for i in range(3):
        gamma_tilde[i, i] = 1.0
    K = np.zeros(shape)
    A_tilde = np.zeros((3, 3, N, N, N))
    Lambda_tilde = np.zeros((3, N, N, N))
    alpha = np.ones(shape)
    beta = np.zeros((3, N, N, N))
    return BSSNState(phi, gamma_tilde, K, A_tilde, Lambda_tilde, alpha, beta)


# ---------------------------------------------------------------------------
# BSSN variable decomposition roundtrip
# ---------------------------------------------------------------------------

class TestBSSNDecomposition:
    """Tests for physical <-> BSSN conversion."""

    def test_roundtrip_flat(self):
        """Flat space: physical -> BSSN -> physical should recover original."""
        N = 5
        shape = (N, N, N)
        gamma_ij = np.zeros((3, 3, N, N, N))
        for i in range(3):
            gamma_ij[i, i] = 1.0
        K_ij = np.zeros((3, 3, N, N, N))
        alpha = np.ones(shape)
        beta = np.zeros((3, N, N, N))

        state = physical_to_bssn(gamma_ij, K_ij, alpha, beta)
        g_out, K_out, a_out, b_out = bssn_to_physical(state)

        np.testing.assert_allclose(g_out, gamma_ij, atol=1e-12)
        np.testing.assert_allclose(K_out, K_ij, atol=1e-12)
        np.testing.assert_allclose(a_out, alpha, atol=1e-12)
        np.testing.assert_allclose(b_out, beta, atol=1e-12)

    def test_roundtrip_perturbed(self):
        """Perturbed metric roundtrip."""
        N = 5
        shape = (N, N, N)
        # Conformally flat with small perturbation
        phi_pert = 0.1 * np.ones(shape)
        e4phi = np.exp(4.0 * phi_pert)
        gamma_ij = np.zeros((3, 3, N, N, N))
        for i in range(3):
            gamma_ij[i, i] = e4phi
        K_ij = np.zeros((3, 3, N, N, N))
        alpha = np.ones(shape)
        beta = np.zeros((3, N, N, N))

        state = physical_to_bssn(gamma_ij, K_ij, alpha, beta)
        g_out, K_out, a_out, b_out = bssn_to_physical(state)

        np.testing.assert_allclose(g_out, gamma_ij, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(K_out, K_ij, rtol=1e-10, atol=1e-12)

    def test_conformal_metric_unit_determinant(self):
        """After decomposition, det(gamma_tilde) should be ~1."""
        N = 5
        shape = (N, N, N)
        phi_pert = 0.05 * np.ones(shape)
        e4phi = np.exp(4.0 * phi_pert)
        gamma_ij = np.zeros((3, 3, N, N, N))
        for i in range(3):
            gamma_ij[i, i] = e4phi
        K_ij = np.zeros((3, 3, N, N, N))
        alpha = np.ones(shape)
        beta = np.zeros((3, N, N, N))

        state = physical_to_bssn(gamma_ij, K_ij, alpha, beta)
        det_gt = compute_det_gamma_tilde(state.gamma_tilde)
        np.testing.assert_allclose(det_gt, 1.0, atol=1e-10)

    def test_trace_A_tilde_near_zero(self):
        """Trace of A_tilde w.r.t. gamma_tilde should be ~0."""
        state = _make_flat_state(5)
        trace = compute_trace_A_tilde(state.A_tilde, state.gamma_tilde)
        np.testing.assert_allclose(trace, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for 3x3 matrix helpers."""

    def test_determinant_identity(self):
        """Determinant of identity should be 1."""
        N = 3
        a = np.zeros((3, 3, N, N, N))
        for i in range(3):
            a[i, i] = 1.0
        det = _determinant_3x3(a)
        np.testing.assert_allclose(det, 1.0, atol=1e-12)

    def test_inverse_identity(self):
        """Inverse of identity should be identity."""
        N = 3
        a = np.zeros((3, 3, N, N, N))
        for i in range(3):
            a[i, i] = 1.0
        inv = _inverse_3x3(a)
        for i in range(3):
            for j in range(3):
                expected = 1.0 if i == j else 0.0
                np.testing.assert_allclose(inv[i, j], expected, atol=1e-12)

    def test_determinant_inverse_consistency(self):
        """A * inv(A) should be identity."""
        N = 4
        np.random.seed(42)
        a = np.zeros((3, 3, N, N, N))
        # Create a positive-definite symmetric matrix
        for i in range(3):
            a[i, i] = 2.0 + np.random.rand(N, N, N)
        for i in range(3):
            for j in range(i + 1, 3):
                a[i, j] = 0.1 * np.random.rand(N, N, N)
                a[j, i] = a[i, j]

        inv_a = _inverse_3x3(a)
        # a @ inv_a should be identity
        for i in range(3):
            for j in range(3):
                result = np.zeros((N, N, N))
                for k in range(3):
                    result += a[i, k] * inv_a[k, j]
                expected = 1.0 if i == j else 0.0
                np.testing.assert_allclose(result, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Initial data
# ---------------------------------------------------------------------------

class TestInitialData:
    """Tests for initial data generators."""

    def test_flat_data(self):
        """Flat data should have phi=0, K=0, alpha=1, gamma_tilde=identity."""
        state, X, Y, Z, dx = flat_data(N=5)
        assert state.phi.shape == (5, 5, 5)
        np.testing.assert_allclose(state.phi, 0.0, atol=1e-12)
        np.testing.assert_allclose(state.K, 0.0, atol=1e-12)
        np.testing.assert_allclose(state.alpha, 1.0, atol=1e-12)
        for i in range(3):
            assert state.gamma_tilde[i, i].mean() == pytest.approx(1.0)

    def test_schwarzschild_puncture_positive_psi(self):
        """Schwarzschild puncture conformal factor psi should be > 1."""
        state, X, Y, Z, dx = schwarzschild_puncture(N=10, M=1.0, x_range=(-5.0, 5.0))
        # phi = ln(psi), psi = 1 + M/(2r) > 1 for r > 0
        assert np.all(state.phi >= 0), "phi should be non-negative for Schwarzschild"

    def test_schwarzschild_puncture_direct(self):
        """Direct BSSN Schwarzschild should have flat conformal metric."""
        state, X, Y, Z, dx = schwarzschild_puncture_bssn_direct(N=10, M=1.0)
        for i in range(3):
            np.testing.assert_allclose(state.gamma_tilde[i, i], 1.0, atol=1e-12)
        np.testing.assert_allclose(state.K, 0.0, atol=1e-12)

    def test_schwarzschild_puncture_K_zero(self):
        """Time-symmetric Schwarzschild should have K=0."""
        state, X, Y, Z, dx = schwarzschild_puncture(N=10, M=1.0)
        np.testing.assert_allclose(state.K, 0.0, atol=1e-10)

    def test_brill_wave_finite(self):
        """Brill wave data should produce finite BSSN variables."""
        state, X, Y, Z, dx = brill_wave(N=10, amplitude=0.1)
        assert np.all(np.isfinite(state.phi))
        assert np.all(np.isfinite(state.K))

    def test_binary_puncture_finite(self):
        """Binary puncture data should be finite."""
        state, X, Y, Z, dx = binary_puncture(N=10, M1=1.0, M2=1.0, separation=4.0)
        assert np.all(np.isfinite(state.phi))

    def test_make_grid(self):
        """Grid creation should produce correct shapes."""
        X, Y, Z, dx = make_grid(10, (-1.0, 1.0))
        assert X.shape == (10, 10, 10)
        assert dx > 0


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    """Tests for Hamiltonian and momentum constraints."""

    def test_hamiltonian_flat(self):
        """Hamiltonian constraint should be ~0 for flat data."""
        state, X, Y, Z, dx = flat_data(N=5)
        H = hamiltonian_constraint(state, dx)
        # For flat data with K=0, A=0, phi=0, constraint should vanish
        assert np.allclose(H, 0.0, atol=1e-10), (
            f"Hamiltonian constraint should vanish for flat data, got max |H| = {np.max(np.abs(H))}"
        )

    def test_momentum_flat(self):
        """Momentum constraint should be ~0 for flat data."""
        state, X, Y, Z, dx = flat_data(N=5)
        M = momentum_constraint(state, dx)
        assert np.allclose(M, 0.0, atol=1e-10), (
            f"Momentum constraint should vanish for flat data, got max |M| = {np.max(np.abs(M))}"
        )

    def test_schwarzschild_satisfies_hamiltonian(self):
        """Schwarzschild puncture should approximately satisfy Hamiltonian constraint."""
        state, X, Y, Z, dx = schwarzschild_puncture_bssn_direct(N=15, M=1.0, x_range=(-5.0, 5.0))
        H = hamiltonian_constraint_simplified(state, dx)
        # Exclude the puncture region (small r) where numerical errors dominate
        r = np.sqrt(X**2 + Y**2 + Z**2)
        mask = r > 1.0
        H_outside = H[mask]
        H_norm = np.sqrt(np.mean(H_outside**2))
        assert H_norm < 1.0, (
            f"Hamiltonian constraint should be small outside puncture, got ||H||={H_norm}"
        )

    def test_constraint_norm_positive(self):
        """Constraint norm should be non-negative."""
        state, X, Y, Z, dx = flat_data(N=5)
        H = hamiltonian_constraint(state, dx)
        M = momentum_constraint(state, dx)
        H_norm, M_norm = compute_constraint_norm(H, M, dx)
        assert H_norm >= 0
        assert M_norm >= 0


# ---------------------------------------------------------------------------
# Gauge conditions
# ---------------------------------------------------------------------------

class TestGauge:
    """Tests for gauge conditions and finite differences."""

    def test_gradient_constant(self):
        """Gradient of a constant should be zero."""
        N = 10
        f = np.ones((N, N, N)) * 5.0
        dx = 0.1
        grad = gradient(f, dx, order=2)
        assert grad.shape == (3, N, N, N)
        np.testing.assert_allclose(grad, 0.0, atol=1e-12)

    def test_gradient_linear(self):
        """Gradient of f = x should be (1, 0, 0)."""
        N = 20
        dx = 0.1
        x = np.arange(N) * dx
        f = np.zeros((N, N, N))
        for i in range(N):
            f[i, :, :] = x[i]
        grad = gradient(f, dx, order=2)
        # Interior should be approximately 1 in x-direction
        np.testing.assert_allclose(grad[0, 2:-2, 2:-2, 2:-2], 1.0, atol=0.05)
        np.testing.assert_allclose(grad[1, 2:-2, 2:-2, 2:-2], 0.0, atol=0.05)
        np.testing.assert_allclose(grad[2, 2:-2, 2:-2, 2:-2], 0.0, atol=0.05)

    def test_divergence_zero_for_constant(self):
        """Divergence of a constant vector should be zero."""
        N = 10
        vec = np.ones((3, N, N, N))
        dx = 0.1
        div = divergence(vec, dx, order=2)
        np.testing.assert_allclose(div, 0.0, atol=1e-12)

    def test_one_plus_log_rhs_zero_for_flat(self):
        """1+log RHS should give -2K for flat data with zero shift."""
        N = 5
        alpha = np.ones((N, N, N))
        K = np.zeros((N, N, N))
        beta = np.zeros((3, N, N, N))
        d_alpha = np.zeros((3, N, N, N))
        dx = 0.1
        rhs = one_plus_log_rhs(alpha, K, beta, d_alpha, dx)
        np.testing.assert_allclose(rhs, 0.0, atol=1e-12)

    def test_gamma_driver_rhs_structure(self):
        """Gamma-driver RHS should return correct shapes."""
        N = 5
        beta = np.zeros((3, N, N, N))
        B = np.zeros((3, N, N, N))
        Lambda_dot = np.ones((3, N, N, N)) * 0.1
        rhs_beta, rhs_B = gamma_driver_rhs(beta, B, Lambda_dot, eta=0.75)
        assert rhs_beta.shape == (3, N, N, N)
        assert rhs_B.shape == (3, N, N, N)

    def test_flat_evolution_stable(self):
        """Flat space should remain stable for a single 1+log step."""
        state, X, Y, Z, dx = flat_data(N=5)
        dt = 0.1 * dx
        # One step of lapse evolution
        rhs_alpha = one_plus_log_rhs(
            state.alpha, state.K, state.beta,
            gradient(state.alpha, dx), dx
        )
        alpha_new = state.alpha + dt * rhs_alpha
        # Should remain positive
        assert np.all(alpha_new > 0), "Lapse should remain positive"
