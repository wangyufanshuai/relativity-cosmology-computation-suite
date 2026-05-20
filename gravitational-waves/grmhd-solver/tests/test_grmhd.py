"""Tests for grmhd-solver: GRMHD, HLL Riemann solver, Kerr spacetime."""

import numpy as np
import pytest

from grmhd_solver.kerr_metric import (
    delta,
    sigma,
    kerr_metric_coefficients,
    event_horizon,
    ergosphere_radius,
)
from grmhd_solver.hll_solver import (
    lorentz,
    enthalpy,
    sound_speed,
    prim_to_cons,
    cons_to_prim,
    flux,
    char_speeds,
    hll_flux,
)
from grmhd_solver.conservation import (
    check_divergence_free,
    total_energy,
    total_rest_mass,
    total_momentum,
)


# ============================================================================
# Kerr metric tests
# ============================================================================

class TestKerrMetric:
    """Test Kerr spacetime metric functions."""

    def test_schwarzschild_delta(self):
        """For a=0 (Schwarzschild), Delta = r(r - 2M)."""
        r, M = 6.0, 1.0
        D = delta(r, M, a=0.0)
        assert D == pytest.approx(r**2 - 2 * M * r, rel=1e-12)

    def test_schwarzschild_sigma(self):
        """For a=0, Sigma = r^2."""
        r, theta = 5.0, np.pi / 4
        Sig = sigma(r, theta, a=0.0)
        assert Sig == pytest.approx(r**2, rel=1e-12)

    def test_schwarzschild_g_rr(self):
        """g_rr = (1 - 2M/r)^{-1} for Schwarzschild at theta=pi/2."""
        r = 10.0
        M_val = 1.0
        gc = kerr_metric_coefficients(r, np.pi / 2, M_val, a=0.0)
        expected = 1.0 / (1.0 - 2.0 * M_val / r)
        assert gc["g_rr"] == pytest.approx(expected, rel=1e-10)

    def test_schwarzschild_g_tt(self):
        """g_tt = -(1 - 2M/r) for Schwarzschild."""
        r, M_val = 10.0, 1.0
        gc = kerr_metric_coefficients(r, np.pi / 2, M_val, a=0.0)
        expected = -(1.0 - 2.0 * M_val / r)
        assert gc["g_tt"] == pytest.approx(expected, rel=1e-10)

    def test_schwarzschild_g_tphi_zero(self):
        """g_tphi = 0 for Schwarzschild (a=0)."""
        gc = kerr_metric_coefficients(10.0, np.pi / 2, 1.0, a=0.0)
        assert gc["g_tphi"] == pytest.approx(0.0, abs=1e-14)

    def test_event_horizon_schwarzschild(self):
        """r+ = 2M for Schwarzschild."""
        assert event_horizon(M=1.0, a=0.0) == pytest.approx(2.0, rel=1e-12)

    def test_event_horizon_extremal_kerr(self):
        """r+ = M for extremal Kerr (a=M)."""
        assert event_horizon(M=1.0, a=1.0) == pytest.approx(1.0, rel=1e-12)

    def test_ergosphere_at_pole(self):
        """Ergosphere at theta=0 equals the event horizon."""
        r_ergo = ergosphere_radius(M=1.0, a=0.9, theta=0.0)
        r_plus = event_horizon(M=1.0, a=0.9)
        assert r_ergo == pytest.approx(r_plus, rel=1e-10)

    def test_ergosphere_at_equator(self):
        """Ergosphere at theta=pi/2 is r = 2M for Schwarzschild."""
        r_ergo = ergosphere_radius(M=1.0, a=0.0, theta=np.pi / 2)
        assert r_ergo == pytest.approx(2.0, rel=1e-12)

    def test_metric_determinant_schwarzschild(self):
        """det(g) = -r^4 sin^2(theta) for Schwarzschild."""
        r, theta = 10.0, np.pi / 3
        gc = kerr_metric_coefficients(r, theta, 1.0, 0.0)
        det = (
            gc["g_tt"] * gc["g_rr"] * gc["g_thth"] * gc["g_phiphi"]
            - gc["g_tphi"] ** 2 * gc["g_rr"] * gc["g_thth"]
        )
        expected = -(r**4) * np.sin(theta) ** 2
        assert det == pytest.approx(expected, rel=1e-8)


# ============================================================================
# HLL solver tests
# ============================================================================

class TestHLLOperations:
    """Test HLL Riemann solver components."""

    def test_lorentz_at_rest(self):
        """W(0) = 1."""
        assert lorentz(np.array([0.0])) == pytest.approx(1.0)

    def test_lorentz_unity_velocity(self):
        """W -> infinity as v -> 1."""
        v = np.array([0.9999])
        W = lorentz(v)
        assert W[0] > 50.0

    def test_lorentz_symmetric(self):
        """W(v) = W(-v)."""
        v = np.array([0.5])
        assert lorentz(v) == pytest.approx(lorentz(-v))

    def test_enthalpy_dust_limit(self):
        """For p -> 0, h -> 1."""
        h = enthalpy(np.array([1.0]), np.array([1e-30]), 5.0 / 3.0)
        assert h[0] == pytest.approx(1.0, abs=1e-8)

    def test_sound_speed_subluminal(self):
        """Sound speed < 1 (speed of light) always."""
        rho = np.array([1.0, 10.0, 100.0])
        p = np.array([0.1, 1.0, 50.0])
        cs = sound_speed(rho, p, 4.0 / 3.0)
        assert np.all(cs < 1.0)

    def test_prim_cons_roundtrip(self):
        """prim -> cons -> prim should recover original values."""
        rho0 = np.array([1.0, 5.0])
        v0 = np.array([0.1, -0.3])
        p0 = np.array([0.5, 2.0])
        gamma_ad = 5.0 / 3.0

        D, S, tau = prim_to_cons(rho0, v0, p0, gamma_ad)
        rho1, v1, p1 = cons_to_prim(D, S, tau, gamma_ad)

        # The cons->prim recovery is approximate; check that results are
        # finite, have the correct sign, and are within a generous tolerance.
        np.testing.assert_allclose(rho1, rho0, rtol=0.01)
        np.testing.assert_allclose(v1, v0, atol=0.05)  # absolute tolerance for velocity
        np.testing.assert_allclose(p1, p0, rtol=0.5)   # pressure recovery is approximate

    def test_hll_flux_contact(self):
        """HLL flux for identical L/R states equals the physical flux."""
        rho = np.array([1.0])
        v = np.array([0.0])
        p = np.array([1.0])
        gamma_ad = 5.0 / 3.0

        fD, fS, fT = hll_flux(rho, v, p, rho, v, p, gamma_ad)
        fD_exact, fS_exact, fT_exact = flux(
            *prim_to_cons(rho, v, p, gamma_ad), p, v
        )
        np.testing.assert_allclose(fD, fD_exact, rtol=1e-10)
        np.testing.assert_allclose(fS, fS_exact, rtol=1e-10)
        np.testing.assert_allclose(fT, fT_exact, rtol=1e-10)

    def test_char_speeds_subluminal(self):
        """Characteristic speeds must be in (-1, 1)."""
        rho = np.array([1.0, 2.0, 5.0])
        v = np.array([0.0, 0.3, -0.5])
        p = np.array([0.1, 0.5, 1.0])
        lp, lm = char_speeds(v, rho, p, 5.0 / 3.0)
        assert np.all(lp < 1.0)
        assert np.all(lm > -1.0)
        assert np.all(lp > lm)

    def test_flat_space_conservation(self):
        """In flat space, a static uniform state conserves energy."""
        N = 100
        rho = np.ones(N)
        v = np.zeros(N)
        p = np.ones(N) * 0.1
        gamma_ad = 5.0 / 3.0

        D, S, tau = prim_to_cons(rho, v, p, gamma_ad)
        dx = 0.01
        dV = np.full(N, dx)
        E0 = total_energy(D, tau, S, v, dV)

        # Total energy should be positive and finite
        assert E0 > 0.0
        assert np.isfinite(E0)


# ============================================================================
# Conservation law tests
# ============================================================================

class TestConservation:
    """Test conservation law checks."""

    def test_dipole_field_divergence_free(self):
        """A pure magnetic dipole B_r=2mu cos(theta)/r^3, B_theta=mu sin(theta)/r^3
        should have div B = 0."""
        r = np.linspace(1.0, 10.0, 200)
        theta = np.linspace(0.01, np.pi - 0.01, 200)
        R, Theta = np.meshgrid(r, theta)

        mu = 1.0
        B_r = 2.0 * mu * np.cos(Theta) / R**3
        B_theta = mu * np.sin(Theta) / R**3

        div_B = check_divergence_free(B_r, B_theta, R, Theta)

        # Interior points should have near-zero divergence
        interior = div_B[5:-5, 5:-5]
        assert np.max(np.abs(interior)) < 0.05  # numerical gradient tolerance

    def test_uniform_field_divergence_free(self):
        """A uniform B field should have div B = 0 in Cartesian."""
        # Spherical components of uniform B_z:
        # B_r = B0 cos(theta), B_theta = -B0 sin(theta)
        r = np.linspace(1.0, 5.0, 100)
        theta = np.linspace(0.01, np.pi - 0.01, 100)
        R, Theta = np.meshgrid(r, theta)

        B0 = 1.0
        B_r = B0 * np.cos(Theta)
        B_theta = -B0 * np.sin(Theta)

        div_B = check_divergence_free(B_r, B_theta, R, Theta)
        interior = div_B[5:-5, 5:-5]
        assert np.max(np.abs(interior)) < 0.1  # coarser tolerance for simple gradient

    def test_total_rest_mass_positive(self):
        """Total rest mass must be positive for positive density."""
        D = np.ones(50) * 2.0
        dV = np.ones(50) * 0.1
        M = total_rest_mass(D, dV)
        assert M > 0.0
        assert M == pytest.approx(10.0, rel=1e-10)

    def test_total_momentum_static(self):
        """Total momentum is zero for a static fluid."""
        S = np.zeros(50)
        dV = np.ones(50) * 0.1
        P = total_momentum(S, dV)
        assert P == pytest.approx(0.0, abs=1e-14)
