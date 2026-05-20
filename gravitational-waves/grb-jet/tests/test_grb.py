"""Tests for grb-jet: relativistic jet, afterglow, and relativistic hydro."""

import numpy as np
import pytest

from grb_jet.relativistic_hydro import (
    enthalpy,
    lorentz,
    sound_speed,
    prim_to_cons,
    cons_to_prim,
    flux,
    char_speeds,
    hll_flux,
    fv_step,
)
from grb_jet.afterglow import (
    blast_radius,
    lorentz_factor,
    break_frequencies,
    synchrotron_flux,
    closure_relation,
)
from grb_jet.jet_propagation import (
    envelope_density,
    jet_head_velocity,
    cocoon_pressure,
    cocoon_energy,
    collimation_radius,
    propagate,
)


# ============================================================================
# Relativistic hydro tests
# ============================================================================

class TestRelativisticHydro:
    """Test relativistic hydrodynamics primitives."""

    def test_lorentz_rest(self):
        """Lorentz factor at rest is 1."""
        assert lorentz(np.array([0.0]))[0] == pytest.approx(1.0)

    def test_lorentz_greater_than_one(self):
        """Lorentz factor > 1 for any nonzero velocity."""
        v = np.array([0.1, 0.5, 0.9, 0.99])
        W = lorentz(v)
        assert np.all(W > 1.0)

    def test_sound_speed_subluminal(self):
        """Sound speed always < c=1."""
        rho = np.array([1.0, 10.0, 1e6])
        p = np.array([0.01, 1.0, 1e5])
        cs = sound_speed(rho, p, 5.0 / 3.0)
        assert np.all(cs < 1.0)
        assert np.all(cs > 0.0)

    def test_prim_cons_roundtrip(self):
        """prim -> cons -> prim roundtrip."""
        rho0 = np.array([1.0, 5.0, 10.0])
        v0 = np.array([0.1, -0.3, 0.5])
        p0 = np.array([0.5, 2.0, 5.0])
        gamma_ad = 5.0 / 3.0

        D, S, tau = prim_to_cons(rho0, v0, p0, gamma_ad)
        rho1, v1, p1 = cons_to_prim(D, S, tau, gamma_ad)

        np.testing.assert_allclose(rho1, rho0, atol=0.2)
        np.testing.assert_allclose(v1, v0, atol=0.05)
        np.testing.assert_allclose(p1, p0, atol=1.0)

    def test_hll_flux_consistency(self):
        """HLL flux for identical L/R equals physical flux."""
        rho = np.array([1.0])
        v = np.array([0.0])
        p = np.array([1.0])
        gamma_ad = 5.0 / 3.0

        fD, fS, fT = hll_flux(rho, v, p, rho, v, p, gamma_ad)
        D, S, tau = prim_to_cons(rho, v, p, gamma_ad)
        fD_e, fS_e, fT_e = flux(D, S, tau, p, v)

        np.testing.assert_allclose(fD, fD_e, rtol=1e-10)
        np.testing.assert_allclose(fS, fS_e, rtol=1e-10)

    def test_lorentz_factor_constraint(self):
        """Reconstructed velocity |v| < 1."""
        rho = np.array([1.0])
        v = np.array([0.999])
        p = np.array([0.1])
        gamma_ad = 5.0 / 3.0
        D, S, tau = prim_to_cons(rho, v, p, gamma_ad)
        _, v_rec, _ = cons_to_prim(D, S, tau, gamma_ad)
        assert np.all(np.abs(v_rec) < 1.0)

    def test_fv_step_preserves_positivity(self):
        """One finite-volume step should keep density positive."""
        N = 50
        rho = np.ones(N)
        v = np.zeros(N)
        p = np.ones(N) * 0.1
        dx, dt = 0.01, 1e-5
        gamma_ad = 5.0 / 3.0

        rho_new, v_new, p_new = fv_step(rho, v, p, dx, dt, gamma_ad)
        assert np.all(rho_new > 0)
        assert np.all(p_new > 0)
        assert np.all(np.abs(v_new) < 1.0)


# ============================================================================
# Afterglow tests
# ============================================================================

class TestAfterglow:
    """Test afterglow synchrotron radiation model."""

    def test_blast_radius_grows(self):
        """Blast radius R ~ t^{2/5} for adiabatic expansion."""
        t = np.array([1e4, 1e5, 1e6])
        E, n = 1e52, 1.0
        R = blast_radius(t, E, n, "adiabatic")
        assert np.all(R > 0)
        # Check R increases with time
        assert np.all(np.diff(R) > 0)

    def test_blast_radius_adiabatic_scaling(self):
        """R ~ t^{2/5} -> R(t2)/R(t1) = (t2/t1)^{2/5}."""
        t1, t2 = 1e4, 1e6
        E, n = 1e52, 1.0
        R1 = blast_radius(np.array([t1]), E, n, "adiabatic")[0]
        R2 = blast_radius(np.array([t2]), E, n, "adiabatic")[0]
        ratio = R2 / R1
        expected = (t2 / t1) ** 0.4
        assert ratio == pytest.approx(expected, rel=0.05)

    def test_lorentz_factor_decreases(self):
        """Gamma(t) decreases with time for ISM."""
        t = np.array([1e4, 1e5, 1e6])
        E, n = 1e52, 1.0
        G = lorentz_factor(t, E, n, "ISM")
        assert np.all(G > 1.0)  # Lorentz factor > 1
        assert np.all(np.diff(G) < 0)  # decreasing

    def test_lorentz_factor_greater_than_one(self):
        """Bulk Lorentz factor should be > 1 during active afterglow."""
        t = np.array([1.0, 1e3, 1e6])
        G = lorentz_factor(t, 1e52, 1.0, "ISM")
        assert np.all(G > 1.0)

    def test_break_frequencies_positive(self):
        """Break frequencies nu_m, nu_c should be positive."""
        t = np.array([1e5])
        nu_m, nu_c, F_max = break_frequencies(t, 1e52, 1.0, 0.01, 0.1)
        assert np.all(nu_m > 0)
        assert np.all(nu_c > 0)
        assert np.all(F_max > 0)

    def test_synchrotron_flux_positive(self):
        """Synchrotron flux should be non-negative."""
        nu = np.logspace(8, 20, 50)
        t = np.array([1e5])
        F = synchrotron_flux(nu, t, 1e52, 1.0, 0.01, 0.1)
        assert np.all(F >= 0)

    def test_light_curve_decays(self):
        """Afterglow light curve should decay with time at fixed frequency."""
        nu = np.array([1e14])
        times = np.array([1e4, 1e5, 1e6])
        fluxes = np.array([
            synchrotron_flux(nu, np.array([t]), 1e52, 1.0, 0.01, 0.1)[0]
            for t in times
        ])
        # Flux should decrease with time
        assert np.all(np.diff(fluxes) < 0)

    def test_closure_relation_returns_finite(self):
        """Closure relation should return a finite value."""
        alpha = closure_relation(0.5, 0.6, "ISM_slow_between", 2.3)
        assert np.isfinite(alpha)


# ============================================================================
# Jet propagation tests
# ============================================================================

class TestJetPropagation:
    """Test relativistic jet propagation model."""

    def test_envelope_density_positive(self):
        """Envelope density should be positive."""
        r = np.array([1e8, 1e9, 1e10])
        rho = envelope_density(r)
        assert np.all(rho > 0)

    def test_envelope_density_decreasing(self):
        """Envelope density should decrease outward (alpha=2)."""
        r = np.array([1e8, 1e9, 1e10])
        rho = envelope_density(r)
        assert np.all(np.diff(rho) < 0)

    def test_jet_head_velocity_subluminal(self):
        """Jet head velocity < c=1."""
        v_j = 0.99
        rho_a = 1e-3
        rho_j = 1e-7
        v_h = jet_head_velocity(v_j, rho_a, rho_j)
        assert 0 < v_h < 1.0

    def test_jet_head_slower_than_jet(self):
        """v_h < v_j always."""
        v_j = 0.99
        v_h = jet_head_velocity(v_j, rho_a=1e-3, rho_j=1e-7)
        assert v_h < v_j

    def test_cocoon_pressure_positive(self):
        """Cocoon pressure should be positive."""
        P_c = cocoon_pressure(L_j=1e50, r_h=1e8, v_h=0.1, Omega_j=0.1)
        assert P_c > 0

    def test_cocoon_energy_grows(self):
        """Cocoon energy grows with time."""
        L_j = 1e50
        E1 = cocoon_energy(L_j, t=1.0)
        E2 = cocoon_energy(L_j, t=10.0)
        assert E2 > E1

    def test_collimation_radius_positive(self):
        """Collimation radius should be positive."""
        r_c = collimation_radius(P_c=1e20, L_j=1e50, Gamma_j=10.0)
        assert r_c > 0

    def test_propagate_returns_valid(self):
        """propagate() should return a dict with expected keys."""
        result = propagate(
            v_j=0.99,
            rho_j=1e-7,
            L_j=1e50,
            R_star=1e9,
            theta_j=0.1,
        )
        assert "r_h" in result
        assert "v_h" in result
        assert "broken_out" in result
        assert result["v_h"] > 0
        assert result["v_h"] < 1.0
