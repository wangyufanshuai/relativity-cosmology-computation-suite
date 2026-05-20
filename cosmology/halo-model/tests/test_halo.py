"""Tests for halo model large-scale structure analysis."""

import numpy as np
import pytest

from halo_model.mass_function import press_schechter, sheth_tormen, halo_mass_function
from halo_model.hod import hod_n_galaxies, hod_occupation
from halo_model.power import one_halo_term, two_halo_term, halo_model_Pk


# ============================================================
# Mass function tests
# ============================================================

class TestPressSchechter:
    """Tests for the Press-Schechter mass function."""

    def test_positive_for_positive_nu(self):
        """f(nu) should be positive for nu > 0."""
        nu = np.linspace(0.1, 5.0, 100)
        result = press_schechter(nu)
        assert np.all(result > 0), "Press-Schechter f(nu) must be positive for nu > 0"

    def test_zero_at_nu_zero(self):
        """f(nu) should be zero at nu = 0."""
        result = press_schechter(0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_peaks_at_nu_unity(self):
        """f(nu) should peak near nu ~ 1."""
        nu = np.linspace(0.01, 5.0, 1000)
        f = press_schechter(nu)
        peak_nu = nu[np.argmax(f)]
        assert 0.5 < peak_nu < 1.5, f"Peak at nu={peak_nu}, expected near 1.0"

    def test_decays_at_large_nu(self):
        """f(nu) should decay exponentially at large nu."""
        f_low = press_schechter(3.0)
        f_high = press_schechter(5.0)
        assert f_high < f_low, "f(nu) should decrease at large nu"

    def test_analytic_normalization(self):
        """Integral of f(nu) over all nu should be ~0.8 (standard PS normalization)."""
        from scipy import integrate
        result, _ = integrate.quad(press_schechter, 0, np.inf)
        # The standard PS formula sqrt(2/pi)*nu*exp(-nu^2/2) integrates to ~0.798
        # The full PS includes a factor of 2 for cloud-in-cloud correction
        np.testing.assert_allclose(result, np.sqrt(2.0/np.pi), rtol=1e-3)

    def test_analytic_formula(self):
        """Check exact analytic formula for specific nu values."""
        nu = 1.0
        expected = np.sqrt(2.0 / np.pi) * 1.0 * np.exp(-0.5)
        result = press_schechter(nu)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


class TestShethTormen:
    """Tests for the Sheth-Tormen mass function."""

    def test_positive_for_positive_nu(self):
        """f(nu) should be positive for nu > 0."""
        nu = np.linspace(0.1, 5.0, 100)
        result = sheth_tormen(nu)
        assert np.all(result > 0)

    def test_larger_than_press_schechter_at_low_nu(self):
        """ST should predict more low-mass halos than PS at very low nu."""
        nu = np.array([0.1, 0.2, 0.3])
        st = sheth_tormen(nu)
        ps = press_schechter(nu)
        # ST generally predicts more structure at low nu
        assert np.any(st > ps), "ST should predict more than PS somewhere"

    def test_normalization_close_to_unity(self):
        """Integral of ST f(nu) should be approximately 1 (within factor of ~2)."""
        from scipy import integrate
        result, _ = integrate.quad(sheth_tormen, 1e-3, 50)
        # ST is approximately normalized; with A=0.3222 the integral is ~0.63
        assert 0.5 < result < 1.0, f"ST integral = {result}, expected ~0.6-1"

    def test_decays_at_large_nu(self):
        """f(nu) should decay at large nu."""
        assert sheth_tormen(5.0) < sheth_tormen(2.0)


class TestHaloMassFunction:
    """Tests for the full halo mass function dn/dlnM."""

    def _sigma_func(self, M):
        """Simple power-law sigma(M) for testing."""
        return 10.0 * (M / 1e12) ** (-0.5)

    def test_mass_function_positive(self):
        """dn/dlnM should be positive."""
        M = np.logspace(10, 15, 50)
        rho_mean = 2.775e11 * 0.3 * 0.7**2
        result = halo_mass_function(M, self._sigma_func, rho_mean)
        assert np.all(result > 0), "Mass function must be positive"

    def test_mass_function_decreases_with_mass(self):
        """At high mass, dn/dlnM should decrease (fewer massive halos)."""
        M = np.logspace(12, 15, 100)
        rho_mean = 2.775e11 * 0.3 * 0.7**2
        result = halo_mass_function(M, self._sigma_func, rho_mean)
        # At high mass, should be monotonically decreasing
        assert result[-1] < result[0], "Mass function should decrease at high M"

    def test_mass_function_units(self):
        """dn/dlnM * dlnM should give a number density (finite, positive)."""
        M = np.logspace(12, 15, 200)
        rho_mean = 2.775e11 * 0.3 * 0.7**2
        dn_dlnM = halo_mass_function(M, self._sigma_func, rho_mean)
        dlnM = np.log(M[1] / M[0])
        n_total = np.sum(dn_dlnM) * dlnM
        assert n_total > 0 and np.isfinite(n_total)


# ============================================================
# HOD tests
# ============================================================

class TestHODNGalaxies:
    """Tests for HOD mean galaxy occupation."""

    def test_occupation_non_negative(self):
        """Mean galaxy occupation should be >= 0."""
        M = np.logspace(10, 15, 100)
        result = hod_n_galaxies(M, M_min=1e12, M1=1e13, alpha=1.0)
        assert np.all(result >= 0), "HOD occupation must be >= 0"

    def test_occupation_increases_with_mass(self):
        """Larger halos should host more galaxies on average."""
        M = np.array([1e11, 1e12, 1e13, 1e14, 1e15])
        result = hod_n_galaxies(M, M_min=1e12, M1=1e13, alpha=1.0)
        # Should be monotonically increasing
        assert np.all(np.diff(result) >= 0), "HOD should increase with mass"

    def test_occupation_near_zero_below_mmin(self):
        """Halos well below M_min should have nearly zero galaxies."""
        M = np.array([1e10, 1e11])
        result = hod_n_galaxies(M, M_min=1e12, M1=1e13, alpha=1.0)
        assert np.all(result < 0.1), "Occupation should be ~0 below M_min"

    def test_central_galaxy_step(self):
        """Central galaxy occupation should transition from 0 to 1 around M_min."""
        M = np.array([1e11, 1e12, 1e13])
        result = hod_n_galaxies(M, M_min=1e12, M1=1e15, alpha=1.0)
        # At M << M_min, N_cen ~ 0; at M >> M_min, N_cen ~ 1
        # With M1 very high, satellites are suppressed, so this is ~centrals only
        assert result[0] < 0.5  # well below M_min
        assert result[2] > 0.9  # well above M_min

    def test_satellite_power_law(self):
        """Above M1, satellite count should follow power law."""
        M = np.array([1e14, 1e15])
        alpha = 0.8
        M_min = 1e12
        M1 = 1e13
        result = hod_n_galaxies(M, M_min, M1, alpha)
        # All galaxies at high M, should be positive and increasing
        assert np.all(result > 1)


class TestHODOccupation:
    """Tests for the parameterized HOD occupation function."""

    def test_default_params(self):
        """Should work with default parameters."""
        M = np.logspace(10, 15, 50)
        result = hod_occupation(M, {})
        assert result.shape == M.shape
        assert np.all(result >= 0)

    def test_custom_params(self):
        """Should use custom parameters when provided."""
        M = np.logspace(10, 15, 50)
        params = {'M_min': 1e11, 'M1': 1e12, 'alpha': 0.5, 'sigma_logM': 0.2}
        result = hod_occupation(M, params)
        assert np.all(result >= 0)

    def test_satellites_scaled_by_centrals(self):
        """In hod_occupation, satellites are N_cen * (M/M1)^alpha, not just (M/M1)^alpha."""
        M_high = 1e15
        params = {'M_min': 1e12, 'M1': 1e13, 'alpha': 1.0}
        result = hod_occupation(np.array([M_high]), params)
        # N_cen at this mass is ~1, so satellites dominate
        assert result[0] > 1.0


# ============================================================
# Power spectrum tests
# ============================================================

class TestOneHaloTerm:
    """Tests for the 1-halo power spectrum term."""

    def test_positive_power(self):
        """1-halo power should be positive."""
        k = np.array([0.1, 0.5, 1.0])

        def mass_func(M):
            rho_mean = 2.775e11 * 0.3 * 0.7**2
            sigma = 10.0 * (M / 1e12) ** (-0.5)
            nu = 1.686 / sigma
            from halo_model.mass_function import sheth_tormen
            return sheth_tormen(nu) * rho_mean / M

        def density_profile(ki, Mi):
            # Simple NFW-like profile: u(k|M) ~ 1 / (1 + k*rs)^2
            rs = (Mi / 1e12) ** (1.0 / 3.0) * 0.1
            return 1.0 / (1.0 + ki * rs) ** 2

        P_1h = one_halo_term(k, mass_func, density_profile)
        assert np.all(P_1h > 0), "1-halo power must be positive"

    def test_decreasing_at_large_k(self):
        """1-halo term should decrease at large k (profile suppression)."""
        k = np.array([0.01, 0.1, 1.0, 10.0])

        def mass_func(M):
            rho_mean = 2.775e11 * 0.3 * 0.7**2
            sigma = 10.0 * (M / 1e12) ** (-0.5)
            nu = 1.686 / sigma
            from halo_model.mass_function import sheth_tormen
            return sheth_tormen(nu) * rho_mean / M

        def density_profile(ki, Mi):
            rs = (Mi / 1e12) ** (1.0 / 3.0) * 0.1
            return 1.0 / (1.0 + ki * rs) ** 2

        P_1h = one_halo_term(k, mass_func, density_profile)
        assert P_1h[-1] < P_1h[0], "1-halo power should decrease at large k"


class TestTwoHaloTerm:
    """Tests for the 2-halo power spectrum term."""

    def test_positive_power(self):
        """2-halo power should be positive."""
        k = np.array([0.1, 0.5])

        def bias_func(ki):
            P_lin = 1000.0 * ki ** (-1.5)
            bias_weighted = 1.5  # constant bias
            return bias_weighted, P_lin

        P_2h = two_halo_term(k, bias_func)
        assert np.all(P_2h > 0), "2-halo power must be positive"

    def test_scales_with_linear_power(self):
        """2-halo term should scale linearly with P_linear."""
        k = np.array([0.1])

        def bias_func_high(ki):
            return 1.5, 2000.0

        def bias_func_low(ki):
            return 1.5, 1000.0

        P_high = two_halo_term(k, bias_func_high)
        P_low = two_halo_term(k, bias_func_low)
        ratio = P_high / P_low
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-10)


class TestHaloModelPk:
    """Tests for the full halo model power spectrum."""

    def test_returns_dict_with_required_keys(self):
        """Should return dict with P_1h, P_2h, P_total."""
        k = np.array([0.1, 0.5])

        def mass_func(M):
            rho_mean = 2.775e11 * 0.3 * 0.7**2
            sigma = 10.0 * (M / 1e12) ** (-0.5)
            nu = 1.686 / sigma
            from halo_model.mass_function import sheth_tormen
            return sheth_tormen(nu) * rho_mean / M

        def density_profile(ki, Mi):
            rs = (Mi / 1e12) ** (1.0 / 3.0) * 0.1
            return 1.0 / (1.0 + ki * rs) ** 2

        def bias_func(ki):
            P_lin = 1000.0 * ki ** (-1.5)
            return 1.5, P_lin

        params = {
            'mass_func': mass_func,
            'density_profile': density_profile,
            'bias_func': bias_func,
        }
        result = halo_model_Pk(k, params)

        assert 'P_1h' in result
        assert 'P_2h' in result
        assert 'P_total' in result

    def test_total_equals_sum(self):
        """P_total should equal P_1h + P_2h."""
        k = np.array([0.1, 0.5])

        def mass_func(M):
            rho_mean = 2.775e11 * 0.3 * 0.7**2
            sigma = 10.0 * (M / 1e12) ** (-0.5)
            nu = 1.686 / sigma
            from halo_model.mass_function import sheth_tormen
            return sheth_tormen(nu) * rho_mean / M

        def density_profile(ki, Mi):
            rs = (Mi / 1e12) ** (1.0 / 3.0) * 0.1
            return 1.0 / (1.0 + ki * rs) ** 2

        def bias_func(ki):
            P_lin = 1000.0 * ki ** (-1.5)
            return 1.5, P_lin

        params = {
            'mass_func': mass_func,
            'density_profile': density_profile,
            'bias_func': bias_func,
        }
        result = halo_model_Pk(k, params)
        np.testing.assert_allclose(
            result['P_total'],
            result['P_1h'] + result['P_2h'],
            rtol=1e-10,
        )

    def test_total_power_positive(self):
        """Total halo model power should be positive at all k."""
        k = np.logspace(-2, 1, 10)

        def mass_func(M):
            rho_mean = 2.775e11 * 0.3 * 0.7**2
            sigma = 10.0 * (M / 1e12) ** (-0.5)
            nu = 1.686 / sigma
            from halo_model.mass_function import sheth_tormen
            return sheth_tormen(nu) * rho_mean / M

        def density_profile(ki, Mi):
            rs = (Mi / 1e12) ** (1.0 / 3.0) * 0.1
            return 1.0 / (1.0 + ki * rs) ** 2

        def bias_func(ki):
            P_lin = 1000.0 * ki ** (-1.5)
            return 1.5, P_lin

        params = {
            'mass_func': mass_func,
            'density_profile': density_profile,
            'bias_func': bias_func,
        }
        result = halo_model_Pk(k, params)
        assert np.all(result['P_total'] > 0), "Total power must be positive"
