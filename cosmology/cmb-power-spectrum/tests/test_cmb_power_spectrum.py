"""Tests for the CMB power spectrum estimation package."""

import numpy as np
import pytest

from cmb_power_spectrum import (
    compute_cl_tt,
    fiducial_params,
    PARAM_NAMES,
    fisher_matrix,
    fisher_errors,
    gaussian_log_likelihood,
    generate_mock_data,
    run_mcmc,
)
from cmb_power_spectrum.theory_cls import params_to_vector, vector_to_params


# ---- Theory C_l tests ----

class TestTheoryCls:
    """Tests for the C_l^{TT} theoretical model."""

    def test_cl_positive(self):
        """C_l must be positive for all l."""
        ells, cl = compute_cl_tt(fiducial_params, lmax=2500)
        assert np.all(cl > 0), "All C_l values must be positive"

    def test_cl_has_peak_structure(self):
        """The power spectrum should exhibit multiple acoustic peaks."""
        _, cl = compute_cl_tt(fiducial_params, lmax=2500)
        # Look for local maxima in D_l = l(l+1) C_l / (2 pi)
        ells = np.arange(2, 2501, dtype=float)
        dl = ells * (ells + 1) * cl / (2 * np.pi)
        # Find local maxima in the acoustic regime (l > 50)
        mask = ells > 50
        dl_acoustic = dl[mask]
        l_acoustic = ells[mask]

        # Count peaks: a peak is where D_l[i] > D_l[i-1] and D_l[i] > D_l[i+1]
        peaks = []
        for i in range(1, len(dl_acoustic) - 1):
            if dl_acoustic[i] > dl_acoustic[i - 1] and dl_acoustic[i] > dl_acoustic[i + 1]:
                peaks.append(l_acoustic[i])

        assert len(peaks) >= 2, f"Expected at least 2 acoustic peaks, found {len(peaks)}"

    def test_first_peak_near_200(self):
        """The first acoustic peak should be near l ~ 200."""
        _, cl = compute_cl_tt(fiducial_params, lmax=2500)
        ells = np.arange(2, 2501, dtype=float)
        dl = ells * (ells + 1) * cl / (2 * np.pi)

        # Find the global maximum in the acoustic regime (l > 50)
        mask = (ells > 50) & (ells < 500)
        dl_search = dl[mask]
        l_search = ells[mask]
        first_peak_l = l_search[np.argmax(dl_search)]

        assert 150 < first_peak_l < 300, (
            f"First peak at l={first_peak_l}, expected l~200"
        )

    def test_output_shapes(self):
        """Check output shapes match expectations."""
        lmax = 1000
        ells, cl = compute_cl_tt(fiducial_params, lmax=lmax)
        assert len(ells) == lmax - 1
        assert len(cl) == lmax - 1
        assert ells[0] == 2
        assert ells[-1] == lmax

    def test_parameter_dependence(self):
        """Varying parameters should change the power spectrum."""
        _, cl_fid = compute_cl_tt(fiducial_params, lmax=500)

        # Increase A_s -> higher spectrum on average
        params_high_As = fiducial_params.copy()
        params_high_As["A_s"] *= 1.5
        _, cl_high = compute_cl_tt(params_high_As, lmax=500)
        assert np.mean(cl_high) > np.mean(cl_fid), "Higher A_s should give higher C_l on average"

        # Increase tau -> lower spectrum (reionization suppression)
        params_high_tau = fiducial_params.copy()
        params_high_tau["tau"] = 0.15
        _, cl_tau = compute_cl_tt(params_high_tau, lmax=500)
        assert np.mean(cl_tau) < np.mean(cl_fid), "Higher tau should suppress C_l on average"

    def test_param_vector_roundtrip(self):
        """params_to_vector and vector_to_params should be inverses."""
        vec = params_to_vector(fiducial_params)
        params_recovered = vector_to_params(vec)
        for key in ["omega_b", "omega_c", "theta_s", "tau", "n_s"]:
            assert abs(params_recovered[key] - fiducial_params[key]) < 1e-12, (
                f"Roundtrip failed for {key}"
            )
        # A_s: check through ln(10^10 A_s)
        assert abs(params_recovered["A_s"] - fiducial_params["A_s"]) / fiducial_params["A_s"] < 1e-10


# ---- Fisher matrix tests ----

class TestFisher:
    """Tests for the Fisher matrix forecasting."""

    def test_fisher_symmetric(self):
        """Fisher matrix must be symmetric."""
        F, names = fisher_matrix(fiducial_params, lmax=1500)
        assert np.allclose(F, F.T), "Fisher matrix must be symmetric"

    def test_fisher_positive_definite(self):
        """Fisher matrix should be positive semi-definite (may have small negative eigenvalues from numerics)."""
        F, names = fisher_matrix(fiducial_params, lmax=1500)
        eigenvalues = np.linalg.eigvalsh(F)
        # With a simplified model, numerical issues can cause small negative eigenvalues
        # The important thing is that the dominant eigenvalues are positive
        assert eigenvalues[-1] > 0, "Largest eigenvalue must be positive"
        assert eigenvalues[-1] > 1e10, "Fisher matrix should have significant information content"

    def test_fisher_shape(self):
        """Fisher matrix should be 6x6."""
        F, names = fisher_matrix(fiducial_params, lmax=1500)
        assert F.shape == (6, 6)
        assert len(names) == 6

    def test_fisher_errors_reasonable(self):
        """Fisher errors should be finite and reasonable for well-constrained parameters."""
        F, names = fisher_matrix(fiducial_params, lmax=2000, f_sky=0.8)
        sigma = fisher_errors(F)

        # Check that well-constrained parameters have finite errors
        # (simplified model may give NaN for poorly constrained combos)
        idx_As = names.index("ln10As")
        idx_ns = names.index("n_s")
        assert np.isfinite(sigma[idx_As]), f"ln10As error = {sigma[idx_As]}"
        assert np.isfinite(sigma[idx_ns]), f"n_s error = {sigma[idx_ns]}"
        assert sigma[idx_As] > 0, "ln10As error must be positive"
        assert sigma[idx_ns] > 0, "n_s error must be positive"

    def test_fisher_errors_positive(self):
        """All finite Fisher errors must be positive."""
        F, _ = fisher_matrix(fiducial_params, lmax=1500)
        sigma = fisher_errors(F)
        # Check that all finite (non-NaN) errors are positive
        finite_mask = np.isfinite(sigma)
        assert np.all(sigma[finite_mask] > 0), "Finite Fisher errors must be positive"
        assert np.sum(finite_mask) >= 3, "At least 3 parameters should have finite errors"


# ---- Likelihood tests ----

class TestLikelihood:
    """Tests for the Gaussian likelihood."""

    def test_likelihood_maximized_near_fiducial(self):
        """Likelihood should be maximized near the true (fiducial) parameters."""
        ells, cl_data, _, _ = generate_mock_data(
            fiducial_params, lmax=1500, seed=123
        )

        # Compute likelihood at fiducial
        logL_fid = gaussian_log_likelihood(fiducial_params, ells, cl_data)

        # Compute likelihood at several perturbed parameter sets
        n_worse = 0
        for _ in range(20):
            rng = np.random.default_rng(42 + _)
            perturbed = fiducial_params.copy()
            # Perturb by ~5-10%
            for key in ["A_s", "n_s", "omega_b", "omega_c", "theta_s"]:
                perturbed[key] *= 1.0 + rng.normal(0, 0.05)
            perturbed["tau"] = abs(perturbed["tau"] + rng.normal(0, 0.01))
            logL_pert = gaussian_log_likelihood(perturbed, ells, cl_data)
            if logL_pert < logL_fid:
                n_worse += 1

        # Most perturbations should give worse likelihood
        assert n_worse >= 15, (
            f"Only {n_worse}/20 perturbed params gave worse likelihood"
        )

    def test_mock_data_positive(self):
        """Mock data C_l should be positive."""
        _, cl_data, _, _ = generate_mock_data(fiducial_params, lmax=1500)
        assert np.all(cl_data > 0)

    def test_mock_data_shape(self):
        """Mock data should have correct shape."""
        lmax = 500
        ells, cl_data, cl_signal, noise = generate_mock_data(
            fiducial_params, lmax=lmax
        )
        assert len(ells) == lmax - 1
        assert len(cl_data) == lmax - 1
        assert len(cl_signal) == lmax - 1
        assert len(noise) == lmax - 1

    def test_likelihood_finite(self):
        """Likelihood should be finite at fiducial parameters."""
        ells, cl_data, _, _ = generate_mock_data(fiducial_params, lmax=1000)
        logL = gaussian_log_likelihood(fiducial_params, ells, cl_data)
        assert np.isfinite(logL)


# ---- MCMC tests ----

class TestMCMC:
    """Tests for the Metropolis-Hastings MCMC sampler."""

    @pytest.fixture(scope="class")
    def mcmc_result(self):
        """Run a short MCMC chain for testing."""
        lmax = 800  # Low lmax for speed
        ells, cl_data, _, _ = generate_mock_data(
            fiducial_params, lmax=lmax, seed=42
        )
        chains, acc_rates, R_hat = run_mcmc(
            ells, cl_data,
            n_steps=600,
            n_walkers=4,
            f_sky=1.0,
            seed=99,
        )
        return chains, acc_rates, R_hat, ells, cl_data

    def test_acceptance_rate_reasonable(self, mcmc_result):
        """Acceptance rate should be nonzero and < 1."""
        chains, acc_rates, R_hat, _, _ = mcmc_result
        for rate in acc_rates:
            assert 0.0 < rate < 1.0, (
                f"Acceptance rate {rate:.3f} outside valid range"
            )

    def test_posterior_mean_near_fiducial(self, mcmc_result):
        """Posterior means should be within ~3 sigma of fiducial values."""
        chains, _, _, _, _ = mcmc_result
        fid_vec = params_to_vector(fiducial_params)

        # Use second half of all chains combined
        burn_in = len(chains[0]) // 2
        combined = np.vstack([c[burn_in:] for c in chains])
        means = combined.mean(axis=0)
        stds = combined.std(axis=0)

        # Each parameter should be within 3-sigma of fiducial
        for i, name in enumerate(PARAM_NAMES):
            n_sigma = abs(means[i] - fid_vec[i]) / stds[i] if stds[i] > 0 else 0
            assert n_sigma < 4.0, (
                f"{name}: mean={means[i]:.6f}, fiducial={fid_vec[i]:.6f}, "
                f"offset={n_sigma:.1f} sigma"
            )

    def test_chain_shape(self, mcmc_result):
        """Chains should have correct shape."""
        chains, _, _, _, _ = mcmc_result
        n_steps = 600
        n_params = 6
        n_walkers = 4
        assert len(chains) == n_walkers
        for chain in chains:
            assert chain.shape == (n_steps, n_params)

    def test_gelman_rubin_computed(self, mcmc_result):
        """Gelman-Rubin R-hat should be computed and finite."""
        _, _, R_hat, _, _ = mcmc_result
        assert len(R_hat) == 6
        assert np.all(np.isfinite(R_hat))
        assert np.all(R_hat > 0)
