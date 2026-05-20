"""Tests for the inflation-solver package."""

from __future__ import annotations

import numpy as np
import pytest

from inflation_solver.potentials import (
    AlphaAttractorPotential,
    HilltopPotential,
    NaturalPotential,
    QuadraticPotential,
    QuarticPotential,
    StarobinskyPotential,
)
from inflation_solver.slow_roll import (
    N_efolds,
    epsilon_V,
    eta_V,
    n_s,
    phi_end,
    phi_start,
    planck_constraints,
    r_tensor,
)
from inflation_solver.background import InflationBackground


# ---------------------------------------------------------------------------
# Potential positivity
# ---------------------------------------------------------------------------

class TestPotentialPositivity:
    """All potentials should give V > 0 for physical phi range."""

    @pytest.mark.parametrize(
        "pot",
        [
            QuadraticPotential(m=1e-5),
            QuarticPotential(lambda_=1e-10),
            StarobinskyPotential(M=1e-5),
            AlphaAttractorPotential(V0=1e-10, alpha=1.0),
        ],
    )
    def test_positive_potential(self, pot):
        phi = np.linspace(0.1, 20.0, 200)
        assert np.all(pot.V(phi) > 0)

    def test_natural_potential_positive(self):
        pot = NaturalPotential(V0=1e-10, f=5.0)
        phi = np.linspace(-15.0, 15.0, 200)
        # V = V0(1 + cos(phi/f)) >= 0 always since cos >= -1
        assert np.all(pot.V(phi) >= -1e-15)

    def test_hilltop_potential_positive_in_range(self):
        pot = HilltopPotential(V0=1e-10, n=4, phi_c=10.0)
        # Near the hilltop (phi << phi_c), V ~ V0 > 0
        phi = np.linspace(0.01, 5.0, 100)
        assert np.all(pot.V(phi) > 0)


# ---------------------------------------------------------------------------
# Starobinsky model: n_s ~ 0.965, r ~ 0.0035 at N=60
# ---------------------------------------------------------------------------

class TestStarobinsky:
    """Starobinsky model should reproduce well-known analytical predictions."""

    @pytest.fixture
    def staro(self):
        return StarobinskyPotential(M=1e-5)

    def test_ns_approximately_0965(self, staro):
        phi_s = phi_start(staro, N_target=60)
        ns_val = n_s(staro, phi_s)
        # Analytic: n_s ~ 1 - 2/N = 1 - 1/30 ~ 0.9667 at N=60
        # Numerically around 0.964-0.968
        assert abs(ns_val - 0.965) < 0.01, f"n_s = {ns_val}"

    def test_r_approximately_00035(self, staro):
        phi_s = phi_start(staro, N_target=60)
        r_val = r_tensor(staro, phi_s)
        # Analytic: r ~ 12/N^2 = 12/3600 ~ 0.00333
        assert abs(r_val - 0.0035) < 0.002, f"r = {r_val}"


# ---------------------------------------------------------------------------
# Slow-roll parameters
# ---------------------------------------------------------------------------

class TestSlowRoll:
    """Tests for slow-roll parameter computation."""

    def test_epsilon_at_end_equals_one(self):
        """epsilon_V = 1 should define the end of inflation."""
        pot = QuadraticPotential(m=1e-5)
        phi_e = phi_end(pot)
        eps = epsilon_V(pot, phi_e)
        assert abs(eps - 1.0) < 1e-3, f"epsilon(phi_end) = {eps}"

    def test_epsilon_small_at_horizon_exit(self):
        """epsilon_V should be << 1 at horizon exit."""
        pot = QuadraticPotential(m=1e-5)
        phi_s = phi_start(pot, N_target=60)
        eps = epsilon_V(pot, phi_s)
        assert eps < 0.01, f"epsilon(phi_start) = {eps}"

    def test_N_efolds_positive(self):
        """N_efolds should be positive for phi_start > phi_end."""
        pot = QuadraticPotential(m=1e-5)
        phi_e = phi_end(pot)
        phi_s = phi_start(pot, N_target=60)
        N = N_efolds(pot, phi_s, phi_e)
        assert N > 0, f"N_efolds = {N}"

    def test_N_efolds_matches_target(self):
        """N_efolds from phi_start should be close to N_target."""
        pot = QuadraticPotential(m=1e-5)
        phi_e = phi_end(pot)
        phi_s = phi_start(pot, N_target=60)
        N = N_efolds(pot, phi_s, phi_e)
        assert abs(N - 60.0) < 1.0, f"N_efolds = {N}, expected ~60"


# ---------------------------------------------------------------------------
# Quadratic model: r = 8/N at leading order
# ---------------------------------------------------------------------------

class TestQuadratic:
    """Quadratic (chaotic) inflation: r ~ 8/N."""

    @pytest.fixture
    def quad(self):
        return QuadraticPotential(m=1e-5)

    def test_r_approximately_8_over_N(self, quad):
        """At N=60, r ~ 8/60 ~ 0.133."""
        phi_s = phi_start(quad, N_target=60)
        r_val = r_tensor(quad, phi_s)
        expected = 8.0 / 60.0
        assert abs(r_val - expected) < 0.01, f"r = {r_val}, expected ~{expected}"


# ---------------------------------------------------------------------------
# Planck constraints
# ---------------------------------------------------------------------------

class TestPlanckConstraints:
    """Check that the Planck constraint dict is sensible."""

    def test_ns_central_value(self):
        pc = planck_constraints()
        assert abs(pc["n_s"][0] - 0.9649) < 1e-10

    def test_r_upper_bound(self):
        pc = planck_constraints()
        assert pc["r"][2] == 0.036


# ---------------------------------------------------------------------------
# Background evolution
# ---------------------------------------------------------------------------

class TestBackground:
    """Tests for exact background evolution."""

    def test_evolve_runs(self):
        pot = QuadraticPotential(m=1e-5)
        bg = InflationBackground(pot, phi0=15.0)
        bg.evolve()
        assert bg.N is not None
        assert len(bg.N) > 10

    def test_epsilon_increases(self):
        """epsilon should generally increase during inflation."""
        pot = QuadraticPotential(m=1e-5)
        bg = InflationBackground(pot, phi0=15.0)
        bg.evolve()
        eps = bg.epsilon()
        # epsilon should be increasing (from small to ~1)
        assert eps[-1] > eps[0]

    def test_background_properties_shape(self):
        """All property arrays should have the same length."""
        pot = QuadraticPotential(m=1e-5)
        bg = InflationBackground(pot, phi0=15.0)
        bg.evolve()
        n = len(bg.N)
        assert len(bg.phi) == n
        assert len(bg.phi_dot) == n
        assert len(bg.H) == n
        assert len(bg.epsilon()) == n
        assert len(bg.n_s()) == n
        assert len(bg.r()) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
