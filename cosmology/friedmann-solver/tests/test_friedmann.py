"""Tests for the Friedmann equation solver.

Validates against known cosmological constraints:
- Planck18 age ~ 13.8 Gyr
- H0 = 67.4 km/s/Mpc
- Matter-radiation equality at z ~ 3400
- Flat universe constraint
- Low-z Hubble law
- Lambda-CDM consistency
"""

import numpy as np
import pytest

from friedmann_solver import Cosmology
from friedmann_solver.constants import C, MPC_IN_M
from friedmann_solver.planck18 import planck18_params, planck18_derived, fisher_matrix
from friedmann_solver.background import solve_background, conformal_time, horizon_scale


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------


@pytest.fixture
def planck18():
    """Planck18 fiducial cosmology."""
    return planck18_params()


@pytest.fixture
def planck18_custom():
    """Planck-like cosmology with explicit Omega_r for reproducibility."""
    return Cosmology(
        H0=67.4,
        Omega_m=0.315,
        Omega_r=9.15e-5,  # approximate value for Planck18
        Omega_lambda=0.685,
        Omega_k=None,
        w0=-1.0,
        wa=0.0,
    )


# ---------------------------------------------------------------
# Test: Age of universe
# ---------------------------------------------------------------


class TestAge:
    """Age of the universe should be ~13.8 Gyr for Planck18."""

    def test_age_planck18(self, planck18):
        age = planck18.age()
        assert 13.5 < age < 14.0, f"Age = {age} Gyr, expected ~13.8 Gyr"

    def test_age_precise(self, planck18_custom):
        age = planck18_custom.age()
        # Planck 2018 reports 13.797 Gyr; allow generous tolerance
        assert abs(age - 13.8) < 0.3, f"Age = {age} Gyr, expected ~13.8 Gyr"

    def test_age_more_de_older(self):
        """More dark energy => faster recent expansion => older apparent age for same H0.

        A matter-only universe is younger than a Lambda-dominated one at fixed H0,
        because the matter-dominated expansion was decelerating.
        """
        cosmo_lcdm = Cosmology(H0=70.0, Omega_m=0.3, Omega_lambda=0.7)
        cosmo_matter = Cosmology(H0=70.0, Omega_m=1.0, Omega_lambda=0.0, Omega_r=0.0, Omega_k=0.0)
        assert cosmo_lcdm.age() > cosmo_matter.age()


# ---------------------------------------------------------------
# Test: H0
# ---------------------------------------------------------------


class TestH0:
    """Hubble constant consistency checks."""

    def test_h0_value(self, planck18):
        assert planck18.H0 == 67.4

    def test_h_at_a1_equals_h0(self, planck18):
        """H(a=1) must equal H0."""
        H_today = planck18.H(1.0)
        assert abs(H_today - 67.4) < 0.01, f"H(a=1) = {H_today}, expected 67.4"

    def test_h_decreases_at_low_z(self, planck18):
        """In LCDM, H should be larger in the past (higher z)."""
        a_vals = np.array([0.5, 0.8, 1.0])
        H_vals = planck18.H(a_vals)
        assert np.all(np.diff(H_vals) < 0), "H should decrease toward a=1 in LCDM"


# ---------------------------------------------------------------
# Test: Matter-radiation equality
# ---------------------------------------------------------------


class TestMatterRadiationEquality:
    """Matter-radiation equality at z ~ 3400."""

    def test_equality_redshift(self, planck18):
        """z_eq = Omega_r / Omega_m - 1 should be ~ 3400."""
        a_eq = planck18.Omega_r / planck18.Omega_m
        z_eq = 1.0 / a_eq - 1.0
        assert 3000 < z_eq < 4000, f"z_eq = {z_eq}, expected ~3400"

    def test_equality_redshift_custom(self, planck18_custom):
        a_eq = planck18_custom.Omega_r / planck18_custom.Omega_m
        z_eq = 1.0 / a_eq - 1.0
        assert 3000 < z_eq < 4000, f"z_eq = {z_eq}, expected ~3400"


# ---------------------------------------------------------------
# Test: Flat universe
# ---------------------------------------------------------------


class TestFlatUniverse:
    """Omega_m + Omega_Lambda = 1 => Omega_k = 0."""

    def test_flat_sum(self, planck18):
        total = planck18.Omega_m + planck18.Omega_lambda + planck18.Omega_r + planck18.Omega_k
        assert abs(total - 1.0) < 1e-10, f"Sum = {total}, expected 1.0"

    def test_omega_k_zero(self, planck18):
        assert abs(planck18.Omega_k) < 1e-10, f"Omega_k = {planck18.Omega_k}, expected ~0"

    def test_explicit_flat(self):
        cosmo = Cosmology(H0=70.0, Omega_m=0.3, Omega_lambda=0.7, Omega_k=0.0)
        assert abs(cosmo.Omega_k) < 1e-10


# ---------------------------------------------------------------
# Test: Hubble law at low z
# ---------------------------------------------------------------


class TestHubbleLaw:
    """At low z, d_L should approximate cz/H0."""

    def test_luminosity_distance_hubble_law(self, planck18):
        """d_L ~ cz/H0 at low z."""
        z = 0.01
        dL = planck18.luminosity_distance(z)
        cz_over_H0 = C * z / (planck18.H0 * 1e3 / MPC_IN_M) / MPC_IN_M
        # cz/H0 in Mpc
        cz_H0_mpc = C * z / planck18.H0_si / MPC_IN_M
        # Allow ~2% relative error since z is small but nonzero
        rel_err = abs(dL - cz_H0_mpc) / cz_H0_mpc
        assert rel_err < 0.02, (
            f"d_L = {dL} Mpc, cz/H0 = {cz_H0_mpc} Mpc, rel_err = {rel_err}"
        )

    def test_hubble_law_array(self, planck18):
        """Test Hubble law for an array of low redshifts."""
        z_vals = np.array([0.001, 0.005, 0.01])
        dL = planck18.luminosity_distance(z_vals)
        cz_H0 = C * z_vals / planck18.H0_si / MPC_IN_M
        rel_err = np.abs(dL - cz_H0) / cz_H0
        assert np.all(rel_err < 0.02), f"Relative errors: {rel_err}"


# ---------------------------------------------------------------
# Test: Cosmological constant (w0=-1, wa=0)
# ---------------------------------------------------------------


class TestCosmologicalConstant:
    """For w0=-1, wa=0, dark energy is a cosmological constant."""

    def test_w_de_constant(self, planck18):
        """w(a) should be -1 for all a when w0=-1, wa=0."""
        a_vals = np.array([0.1, 0.5, 1.0, 2.0])
        w_vals = planck18.w_de(a_vals)
        np.testing.assert_allclose(w_vals, -1.0, atol=1e-15)

    def test_rho_de_constant(self, planck18):
        """Omega_DE(a) should equal Omega_lambda for all a when w=-1."""
        a_vals = np.array([0.1, 0.5, 1.0])
        rho_de = planck18.rho_de(a_vals)
        np.testing.assert_allclose(rho_de, planck18.Omega_lambda, rtol=1e-10)

    def test_rho_de_evolution(self):
        """For w != -1, dark energy density should evolve."""
        cosmo = Cosmology(
            H0=70.0,
            Omega_m=0.3,
            Omega_lambda=0.7,
            w0=-0.9,
            wa=0.1,
        )
        a_vals = np.array([0.5, 1.0])
        rho = cosmo.rho_de(a_vals)
        assert rho[0] != rho[1], "rho_de should evolve for w != -1"


# ---------------------------------------------------------------
# Test: Distance relations
# ---------------------------------------------------------------


class TestDistanceRelations:
    """Distance reciprocity: d_L = (1+z)^2 * d_A."""

    def test_dl_da_reciprocity(self, planck18):
        z_vals = np.array([0.1, 0.5, 1.0, 2.0])
        dL = planck18.luminosity_distance(z_vals)
        dA = planck18.angular_diameter_distance(z_vals)
        ratio = dL / (dA * (1.0 + z_vals) ** 2)
        np.testing.assert_allclose(ratio, 1.0, rtol=1e-10)

    def test_distance_modulus(self, planck18):
        """mu = 5*log10(dL/Mpc) + 25."""
        z = 0.5
        dL = planck18.luminosity_distance(z)
        mu = planck18.distance_modulus(z)
        mu_expected = 5.0 * np.log10(dL) + 25.0
        assert abs(mu - mu_expected) < 1e-10


# ---------------------------------------------------------------
# Test: Deceleration parameter
# ---------------------------------------------------------------


class TestDecelerationParameter:
    """Check q(z) behavior for LCDM."""

    def test_q_today_negative(self, planck18):
        """In LCDM, q0 should be negative (accelerating expansion)."""
        q0 = planck18.deceleration_parameter(0.0)
        assert q0 < 0, f"q0 = {q0}, expected < 0 (accelerating)"

    def test_q_high_z_positive(self, planck18):
        """At high z, q should be positive (matter dominated, decelerating)."""
        q_high = planck18.deceleration_parameter(5.0)
        assert q_high > 0, f"q(z=5) = {q_high}, expected > 0 (decelerating)"


# ---------------------------------------------------------------
# Test: Background evolution
# ---------------------------------------------------------------


class TestBackgroundEvolution:
    """Test background integration."""

    def test_solve_background_shape(self, planck18):
        bg = solve_background(planck18, z_max=1e4, n_points=100)
        assert bg.t.shape == (100,)
        assert bg.a.shape == (100,)
        assert bg.z.shape == (100,)
        assert bg.H.shape == (100,)

    def test_solve_background_monotonic(self, planck18):
        """a(t) should be monotonically increasing."""
        bg = solve_background(planck18, z_max=1e4, n_points=100)
        assert np.all(np.diff(bg.a) > 0), "a(t) should be monotonically increasing"
        assert np.all(np.diff(bg.t) > 0), "t should be monotonically increasing"

    def test_background_h_at_a1(self, planck18):
        """H at a~1 should be close to H0."""
        bg = solve_background(planck18, z_max=1e2, n_points=50)
        idx = np.argmin(np.abs(bg.a - 1.0))
        H_near_1 = bg.H[idx]
        assert abs(H_near_1 - 67.4) < 1.0, f"H(a~1) = {H_near_1}, expected ~67.4"


# ---------------------------------------------------------------
# Test: Conformal time
# ---------------------------------------------------------------


class TestConformalTime:
    """Conformal time tests."""

    def test_conformal_time_positive(self, planck18):
        eta = conformal_time(planck18, 1.0)
        assert eta > 0, f"eta_0 = {eta} Mpc, should be positive"
        # For Planck18, eta_0 ~ 14000 Mpc
        assert 10000 < eta < 16000, f"eta_0 = {eta} Mpc, expected ~14000 Mpc"

    def test_conformal_time_increasing(self, planck18):
        """Conformal time should increase with a."""
        a_vals = np.array([0.1, 0.5, 1.0])
        eta = conformal_time(planck18, a_vals)
        assert np.all(np.diff(eta) > 0), "eta should increase with a"


# ---------------------------------------------------------------
# Test: Horizon scales
# ---------------------------------------------------------------


class TestHorizonScale:
    """Horizon scale tests."""

    def test_horizon_scale_values(self, planck18):
        scales = horizon_scale(planck18)
        assert scales.eta_0 > 0
        assert scales.eta_rec > 0
        assert scales.eta_eq > 0
        assert scales.z_rec == 1090.0
        assert 3000 < scales.z_eq < 4000

    def test_horizon_ordering(self, planck18):
        """eta_0 > eta_rec > eta_eq (later epochs have larger horizons)."""
        scales = horizon_scale(planck18)
        assert scales.eta_0 > scales.eta_rec
        assert scales.eta_rec > scales.eta_eq


# ---------------------------------------------------------------
# Test: Planck18 derived parameters
# ---------------------------------------------------------------


class TestPlanck18Derived:
    """Planck18 derived parameter checks."""

    def test_planck18_derived(self):
        derived = planck18_derived()
        assert "age" in derived
        assert "sigma8" in derived
        assert "r_drag" in derived
        assert 13.5 < derived["age"] < 14.0
        assert abs(derived["sigma8"] - 0.8111) < 0.01
        assert 140 < derived["r_drag"] < 155, f"r_drag = {derived['r_drag']} Mpc"


# ---------------------------------------------------------------
# Test: Fisher matrix
# ---------------------------------------------------------------


class TestFisherMatrix:
    """Fisher matrix forecast tests."""

    def test_fisher_positive_definite(self, planck18):
        result = fisher_matrix(
            planck18,
            param_names=["H0", "Omega_m"],
            sigma_H0=0.5,
            sigma_omega_m=0.007,
        )
        F = result["fisher"]
        eigenvalues = np.linalg.eigvalsh(F)
        assert np.all(eigenvalues > 0), "Fisher matrix should be positive definite"

    def test_fisher_errors_exist(self, planck18):
        result = fisher_matrix(
            planck18,
            param_names=["H0", "Omega_m"],
            sigma_H0=0.5,
            sigma_omega_m=0.007,
        )
        for name in ["H0", "Omega_m"]:
            assert name in result["errors"]
            assert result["errors"][name] > 0


# ---------------------------------------------------------------
# Test: Sound horizon
# ---------------------------------------------------------------


class TestSoundHorizon:
    """Sound horizon tests."""

    def test_sound_horizon_planck18(self, planck18):
        r_s = planck18.sound_horizon(z_drag=1060.0)
        # Planck18 reports r_drag ~ 147.09 Mpc
        assert 140 < r_s < 155, f"r_s = {r_s} Mpc, expected ~147 Mpc"


# ---------------------------------------------------------------
# Test: Curvature distances
# ---------------------------------------------------------------


class TestCurvatureDistances:
    """Test distance calculations in non-flat cosmologies."""

    def test_open_universe_distances(self):
        """Open universe: f_k(chi) > chi."""
        cosmo = Cosmology(
            H0=70.0,
            Omega_m=0.3,
            Omega_lambda=0.5,
            Omega_k=0.2,
        )
        z = 1.0
        dL = cosmo.luminosity_distance(z)
        assert dL > 0

    def test_closed_universe_distances(self):
        """Closed universe: f_k(chi) < chi."""
        cosmo = Cosmology(
            H0=70.0,
            Omega_m=0.3,
            Omega_lambda=0.9,
            Omega_k=-0.2,
        )
        z = 1.0
        dL = cosmo.luminosity_distance(z)
        assert dL > 0
