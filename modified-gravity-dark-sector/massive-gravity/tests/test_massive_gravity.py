"""Tests for massive gravity: dRGT potential, Vainshtein screening, cosmology."""

import numpy as np
import pytest

from massive_gravity.dRGT import (
    interaction_matrix,
    compute_Un,
    compute_all_Un,
    dRGT_potential,
    flat_space_potential_vanishes,
    _identity_4d,
)
from massive_gravity.vainshtein import (
    gravitational_radius,
    vainshtein_radius,
    screening_factor,
    fifth_force,
    newton_force,
    total_force,
    force_suppressed_inside,
    graviton_mass_eV_to_inv_m,
    graviton_mass_eV_to_m,
)
from massive_gravity.cosmology import (
    hubble_parameter,
    hubble_parameter_lcdm,
    gw_speed_dRGT,
    gw_speed_constraint_gw170817,
    comoving_distance,
    luminosity_distance,
    reduces_to_lcdm,
    check_graviton_mass_bound,
)
from massive_gravity.bimetric import (
    frw_metric,
    desitter_metric,
    compute_y_ratio,
    y_ratio_finite,
)


# ---------------------------------------------------------------------------
# dRGT potential
# ---------------------------------------------------------------------------

class TestDRGT:
    """Tests for dRGT massive gravity potential."""

    def test_identity_metric(self):
        """Identity (Minkowski) metric should have correct signature."""
        eta = _identity_4d()
        assert eta[0, 0] == -1.0
        assert np.all(np.diag(eta)[1:] == 1.0)

    def test_interaction_matrix_identity(self):
        """Interaction matrix of eta with eta should be identity-like."""
        eta = _identity_4d()
        gamma = interaction_matrix(eta, eta)
        # eigenvalues should all be 1
        eigvals = np.linalg.eigvalsh(gamma)
        assert np.allclose(eigvals, 1.0, atol=1e-8)

    def test_Un_correct_values_for_identity(self):
        """U_n for Minkowski-Minkowski: U_0=1, U_1=4, U_4 should be reasonable."""
        eta = _identity_4d()
        U0 = compute_Un(eta, eta, 0)
        assert abs(U0 - 1.0) < 1e-10, f"U_0 should be 1, got {U0}"

        U1 = compute_Un(eta, eta, 1)
        # For identity eigenvalues (1,1,1,1), U_1 = sum = 4
        assert abs(U1 - 4.0) < 1e-8, f"U_1 should be 4, got {U1}"

    def test_all_Un_returns_five_values(self):
        """compute_all_Un should return 5 values."""
        eta = _identity_4d()
        Un = compute_all_Un(eta, eta)
        assert len(Un) == 5

    def test_dRGT_potential_has_correct_dimensions(self):
        """V should be proportional to m^2, i.e., V ~ m^2 * dimensionless."""
        eta = _identity_4d()
        betas = [1.0, 1.0, 1.0, 1.0, 1.0]
        m1 = 1.0
        m2 = 2.0
        V1 = dRGT_potential(eta, eta, m1, betas)
        V2 = dRGT_potential(eta, eta, m2, betas)
        # V proportional to m^2
        if abs(V1) > 1e-10:
            ratio = V2 / V1
            expected = (m2 / m1) ** 2
            assert abs(ratio - expected) < 0.01, (
                f"V should scale as m^2: ratio={ratio}, expected={expected}"
            )

    def test_potential_is_scalar(self):
        """dRGT potential should return a single float."""
        eta = _identity_4d()
        V = dRGT_potential(eta, eta, 1.0, [1.0, 1.0, 1.0, 1.0, 1.0])
        assert isinstance(V, float)
        assert np.isfinite(V)


# ---------------------------------------------------------------------------
# Vainshtein screening
# ---------------------------------------------------------------------------

class TestVainshtein:
    """Tests for the Vainshtein screening mechanism."""

    def test_gravitational_radius_positive(self):
        """Schwarzschild radius should be positive for positive mass."""
        M_sun = 1.989e30  # kg
        r_g = gravitational_radius(M_sun)
        assert r_g > 0
        assert abs(r_g - 2953.0) < 100, (  # ~3 km for Sun
            f"r_g for Sun should be ~2953 m, got {r_g}"
        )

    def test_vainshtein_radius_positive(self):
        """Vainshtein radius should be positive and large."""
        M_sun = 1.989e30
        m_g = 1e-22  # eV (near LIGO bound)
        r_V = vainshtein_radius(M_sun, m_g)
        assert r_V > 0, f"r_V should be positive, got {r_V}"
        assert r_V > 1e6, f"r_V should be > 1000 km for solar mass, got {r_V}"

    def test_screening_factor_inside(self):
        """Screening factor should be < 1 inside r_V."""
        r_V = 1e10
        r_inside = 0.01 * r_V
        xi = screening_factor(r_inside, r_V)
        assert 0 < xi < 1, f"Screening should be < 1 inside r_V, got {xi}"

    def test_screening_factor_outside(self):
        """Screening factor should approach 1 outside r_V."""
        r_V = 1e10
        r_outside = 100.0 * r_V
        xi = screening_factor(r_outside, r_V)
        assert abs(xi - 1.0) < 0.01, f"Screening should be ~1 outside r_V, got {xi}"

    def test_force_suppressed_inside(self):
        """Fifth force should be suppressed inside Vainshtein radius."""
        M_sun = 1.989e30
        m_g = 1e-22
        assert force_suppressed_inside(M_sun, m_g, r_fraction=0.01)

    def test_fifth_force_positive(self):
        """Fifth force magnitude should be positive."""
        M = 1.989e30
        m_g = 1e-22
        r = 1e8  # 100 km
        F5 = fifth_force(r, M, m_g)
        assert F5 > 0
        assert np.isfinite(F5)

    def test_newton_force_inverse_square(self):
        """Newton force should follow 1/r^2."""
        M = 1.0
        r1, r2 = 1.0, 2.0
        F1 = newton_force(r1, M)
        F2 = newton_force(r2, M)
        assert abs(F1 / F2 - 4.0) < 1e-10

    def test_total_force_greater_than_newton(self):
        """Total force should be >= Newton force."""
        M = 1e30
        m_g = 1e-25
        r = 1e15
        F_total = total_force(r, M, m_g, alpha=1.0)
        F_newton = newton_force(r, M)
        assert F_total >= F_newton

    def test_mass_conversion(self):
        """Graviton mass conversion should be self-consistent."""
        m_eV = 1e-22
        m_inv_m = graviton_mass_eV_to_inv_m(m_eV)
        m_m = graviton_mass_eV_to_m(m_eV)
        assert abs(m_m * m_inv_m - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Cosmology
# ---------------------------------------------------------------------------

class TestCosmology:
    """Tests for massive gravity cosmology."""

    def test_hubble_parameter_positive(self):
        """H(a) should be positive for valid inputs."""
        a = np.array([0.5, 0.8, 1.0])
        H0 = 2.184e-18  # s^{-1}
        H = hubble_parameter(a, H0, Omega_m=0.3)
        assert np.all(H > 0)

    def test_reduces_to_lcdm(self):
        """H(a) should reduce to LCDM for m_g -> 0."""
        assert reduces_to_lcdm(m_g_eV_small=1e-30, beta_eff=1.0)

    def test_lcdm_hubble_matches(self):
        """LCDM Hubble should match massive gravity with m_g=0."""
        a = 0.5
        H0 = 2.184e-18
        H_mg = hubble_parameter(a, H0, 0.3, m_g_eV=0.0)
        H_lcdm = hubble_parameter_lcdm(a, H0, 0.3)
        assert abs(H_mg - H_lcdm) / H_lcdm < 1e-10

    def test_gw_speed_equals_c(self):
        """GW speed should be exactly c in pure dRGT."""
        v_g = gw_speed_dRGT()
        assert abs(v_g - 2.99792458e8) < 1e-5

    def test_gw170817_satisfied(self):
        """Pure dRGT GW speed should satisfy GW170817 bound."""
        v_g = gw_speed_dRGT()
        assert gw_speed_constraint_gw170817(v_g)

    def test_comoving_distance_positive(self):
        """Comoving distance should be positive."""
        chi = comoving_distance(0.5, 2.184e-18, 0.3)
        assert chi > 0
        assert np.isfinite(chi)

    def test_luminosity_distance_larger_than_comoving(self):
        """d_L = (1+z) * chi > chi for z > 0."""
        z = 0.5
        H0 = 2.184e-18
        d_L = luminosity_distance(z, H0, 0.3)
        chi = comoving_distance(z, H0, 0.3)
        assert d_L > chi

    def test_graviton_mass_bound(self):
        """Mass at LIGO bound should satisfy the bound."""
        assert check_graviton_mass_bound(1e-23)
        assert not check_graviton_mass_bound(1e-20)


# ---------------------------------------------------------------------------
# Bimetric theory
# ---------------------------------------------------------------------------

class TestBimetric:
    """Tests for bimetric cosmology."""

    def test_frw_metric_signature(self):
        """FRW metric should have correct signature."""
        g = frw_metric(a=1.0)
        assert g[0, 0] == -1.0
        assert g[1, 1] == 1.0
        assert g[2, 2] == 1.0
        assert g[3, 3] == 1.0

    def test_frw_metric_scale_factor(self):
        """FRW metric spatial part should scale as a^2."""
        a = 2.0
        g = frw_metric(a)
        assert abs(g[1, 1] - a**2) < 1e-10

    def test_desitter_metric_positive_spatial(self):
        """de Sitter spatial components should be positive."""
        g = desitter_metric(H_f=1.0, t=0.5)
        assert g[1, 1] > 0
        assert g[2, 2] > 0
        assert g[3, 3] > 0

    def test_y_ratio_finite(self):
        """Scale factor ratio should be finite."""
        a_g = np.linspace(0.01, 1.0, 50)
        t = np.linspace(0.01, 1.0, 50)
        assert y_ratio_finite(a_g, t, rho_f=1e-3)

    def test_y_ratio_positive(self):
        """Scale factor ratio should be positive."""
        a_g = np.linspace(0.01, 1.0, 50)
        t = np.linspace(0.01, 1.0, 50)
        y = compute_y_ratio(a_g, t, rho_f=1e-3)
        assert np.all(y > 0)
