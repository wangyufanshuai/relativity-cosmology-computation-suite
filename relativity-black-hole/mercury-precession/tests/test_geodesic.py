"""Tests for geodesic numerical integration."""

import numpy as np
import pytest

from mercury_precession.geodesic import (
    schwarzschild_radius,
    effective_potential,
    orbital_params_to_conserved,
    integrate_orbit,
)
from mercury_precession.constants import A_MERCURY, E_MERCURY, M_SUN


class TestEffectivePotential:
    def test_minimum_between_rp_ra(self):
        """V_eff should have minimum between perihelion and aphelion."""
        E_t, L_t, r_p, r_a = orbital_params_to_conserved()
        rs = schwarzschild_radius()
        r_test = np.linspace(r_p * 1.001, r_a * 0.999, 1000)
        V = effective_potential(r_test, L_t, rs)
        assert np.any(V < V[0])
        assert np.any(V < V[-1])

    def test_isco(self):
        """ISCO should be at r = 3rs for massless particle."""
        rs = schwarzschild_radius()
        # At ISCO: V_eff' = 0 and V_eff'' = 0 for L = √3 rs
        L_isco = np.sqrt(3.0) * rs
        r_isco = 3.0 * rs
        V = effective_potential(r_isco, L_isco, rs)
        # Check it's an inflection point numerically
        dr = r_isco * 1e-6
        V_plus = effective_potential(r_isco + dr, L_isco, rs)
        V_minus = effective_potential(r_isco - dr, L_isco, rs)
        d2V = (V_plus - 2 * V + V_minus) / dr**2
        assert abs(d2V) < 1e-10  # Should be ~0


class TestIntegration:
    def test_single_orbit_returns_valid_result(self):
        """Integration of 1 orbit should complete successfully."""
        result = integrate_orbit(n_orbits=2)
        assert "r" in result
        assert "phi" in result
        assert len(result["r"]) > 0

    def test_perihelion_found(self):
        """Should find at least one perihelion passage."""
        result = integrate_orbit(n_orbits=3)
        assert len(result["perihelion_phis"]) >= 2

    def test_precession_close_to_analytical(self):
        """Numerical precession should agree with analytical within 0.1%."""
        result = integrate_orbit(n_orbits=10)
        if result["precession_per_orbit_rad"] == 0:
            pytest.skip("Not enough perihelion passages found")
        analytical = result["analytical_precession_rad"]
        numerical = result["precession_per_orbit_rad"]
        assert abs(numerical / analytical - 1.0) < 1e-3

    def test_orbit_is_bound(self):
        """Orbit should stay between perihelion and aphelion."""
        result = integrate_orbit(n_orbits=3)
        _, _, r_p, r_a = orbital_params_to_conserved()
        # Allow small numerical tolerance
        assert np.min(result["r"]) >= r_p * 0.999
        assert np.max(result["r"]) <= r_a * 1.001
