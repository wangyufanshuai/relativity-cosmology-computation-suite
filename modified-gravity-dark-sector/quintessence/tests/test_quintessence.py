"""Tests for quintessence: potentials, evolution, equation of state, and
CPL parameterization."""

import numpy as np
import pytest

from quintessence.potentials import (
    QuadraticPotential,
    InversePowerPotential,
    SUGRAPotential,
    ExponentialPotential,
)
from quintessence.evolution import (
    evolve_quintessence,
    equation_of_state,
)
from quintessence.analysis import w0_wa_fit, classify_model


# ---------------------------------------------------------------------------
# Potential tests
# ---------------------------------------------------------------------------

class TestQuadraticPotential:
    """Tests for the quadratic (mass-like) potential."""

    def test_potential_positive(self):
        """V(phi) = 0.5 * m^2 * phi^2 >= 0 for all phi."""
        pot = QuadraticPotential(m2=1.0)
        phi_vals = np.linspace(-5.0, 5.0, 100)
        for phi in phi_vals:
            assert pot.V(phi) >= 0.0

    def test_potential_minimum_at_zero(self):
        """V(0) = 0 for the quadratic potential."""
        pot = QuadraticPotential(m2=1.0)
        assert pot.V(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_derivative(self):
        """dV/dphi = m^2 * phi."""
        pot = QuadraticPotential(m2=2.0)
        assert pot.dV(1.0) == pytest.approx(2.0, rel=1e-10)
        assert pot.dV(0.0) == pytest.approx(0.0, abs=1e-15)


class TestInversePowerPotential:
    """Tests for the Ratra-Peebles potential."""

    def test_potential_positive(self):
        """V(phi) = M^(4+n) * phi^(-n) > 0 for phi > 0."""
        pot = InversePowerPotential(M=1.0, n=2.0)
        phi_vals = np.linspace(0.1, 10.0, 50)
        for phi in phi_vals:
            assert pot.V(phi) > 0.0

    def test_potential_decreasing(self):
        """V should decrease as phi increases (negative slope)."""
        pot = InversePowerPotential(M=1.0, n=2.0)
        assert pot.V(1.0) > pot.V(2.0)

    def test_n_must_be_positive(self):
        """Constructor should reject n <= 0."""
        with pytest.raises(ValueError):
            InversePowerPotential(M=1.0, n=0.0)
        with pytest.raises(ValueError):
            InversePowerPotential(M=1.0, n=-1.0)


class TestSUGRAPotential:
    """Tests for the SUGRA potential."""

    def test_potential_positive(self):
        """V(phi) = V0 * (1 + cosh(phi/M)) > 0 for all phi."""
        pot = SUGRAPotential(V0=1.0, M=1.0)
        phi_vals = np.linspace(-10.0, 10.0, 100)
        for phi in phi_vals:
            assert pot.V(phi) > 0.0


class TestExponentialPotential:
    """Tests for the exponential potential."""

    def test_potential_positive(self):
        """V(phi) = V0 * exp(-lambda*phi/M_Pl) > 0 for all phi."""
        pot = ExponentialPotential(V0=1.0, lam=1.0)
        phi_vals = np.linspace(-10.0, 10.0, 50)
        for phi in phi_vals:
            assert pot.V(phi) > 0.0


# ---------------------------------------------------------------------------
# Equation of state
# ---------------------------------------------------------------------------

class TestEquationOfState:
    """Test the equation of state parameter w."""

    def test_equation_of_state_range(self):
        """w should be between -1 and 1 for reasonable quintessence models."""
        pot = QuadraticPotential(m2=1.0)
        phi = 1.0
        phi_dot = 0.5
        w = equation_of_state(phi, phi_dot, pot)
        K = 0.5 * phi_dot**2
        V = pot.V(phi)
        w_expected = (K - V) / (K + V)
        assert w == pytest.approx(w_expected, rel=1e-10)
        assert -1.0 <= w <= 1.0

    def test_cosmological_constant_limit(self):
        """For a constant potential with phi_dot -> 0, w -> -1."""
        pot = SUGRAPotential(V0=1.0, M=1.0)
        phi_dot_tiny = 1e-15
        w = equation_of_state(0.0, phi_dot_tiny, pot)
        assert w == pytest.approx(-1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

class TestEvolution:
    """Test full quintessence evolution."""

    def test_evolution_runs(self):
        """Full evolution should complete without error."""
        pot = QuadraticPotential(m2=1e-3)
        result = evolve_quintessence(
            pot,
            phi0=1.0,
            phi_dot0=0.0,
            Omega_m0=0.3,
            Omega_r0=1e-4,
            a_range=(1e-2, 1.0),
            n_points=100,
        )
        assert 'a' in result
        assert 'phi' in result
        assert 'w' in result
        assert 'H' in result
        assert len(result['a']) == 100

    def test_evolution_w_range(self):
        """w should be in a physically reasonable range during evolution."""
        pot = QuadraticPotential(m2=1e-3)
        result = evolve_quintessence(
            pot,
            phi0=1.0,
            phi_dot0=0.0,
            Omega_m0=0.3,
            Omega_r0=1e-4,
            a_range=(1e-2, 1.0),
            n_points=200,
        )
        assert np.all(result['w'] >= -1.0 - 0.01)
        assert np.all(result['w'] <= 1.0 + 0.01)


# ---------------------------------------------------------------------------
# CPL fit (w0, wa)
# ---------------------------------------------------------------------------

class TestCPLFit:
    """Test the CPL (w0, wa) parameterization fit."""

    def test_w0_wa_fit_sensible(self):
        """CPL fit should return w0 near -1 and modest |wa|."""
        pot = QuadraticPotential(m2=1e-3)
        result = evolve_quintessence(
            pot,
            phi0=1.0,
            phi_dot0=0.0,
            Omega_m0=0.3,
            Omega_r0=1e-4,
            a_range=(0.1, 1.0),
            n_points=200,
        )
        w0, wa = w0_wa_fit(result['w'], result['a'])
        assert -1.5 < w0 < 0.5, f"w0 = {w0} outside expected range"
        assert -2.0 < wa < 2.0, f"wa = {wa} outside expected range"

    def test_w0_wa_exact_cpl(self):
        """Fit should recover exact CPL parameters from synthetic data."""
        a = np.linspace(0.1, 1.0, 200)
        w0_true, wa_true = -0.9, 0.2
        w = w0_true + wa_true * (1.0 - a)
        w0_fit, wa_fit = w0_wa_fit(w, a)
        assert w0_fit == pytest.approx(w0_true, abs=0.05)
        assert wa_fit == pytest.approx(wa_true, abs=0.05)
