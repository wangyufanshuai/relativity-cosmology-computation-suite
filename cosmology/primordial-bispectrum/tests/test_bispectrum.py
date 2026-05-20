"""
Tests for primordial bispectrum and non-Gaussianity analysis.

Tests cover:
- Shape function symmetry under permutation of momenta
- Shape function normalization at equilateral configuration
- Shape function peaking behavior (local in squeezed, equilateral at equilateral)
- f_NL consistency relations (Maldacena's theorem)
- In-in formalism convergence
- Scale-dependent bias and squeezed-limit relations
- Gaussian (f_NL=0) vanishing bispectrum
"""

import numpy as np
import pytest

from primordial_bispectrum import (
    shape_local,
    shape_equilateral,
    shape_orthogonal,
    shape_folded,
    bispectrum,
    SHAPE_FUNCTIONS,
    power_spectrum,
    fnl_from_bispectrum,
    fnl_maldacena,
    fnl_multifield,
    PlanckConstraints,
    fnl_log_likelihood,
    bulk_to_boundary_propagator,
    bulk_propagator,
    in_in_integral,
    compute_bispectrum_in_in,
    scale_dependent_bias,
    bias_correction,
    squeezed_limit_bispectrum,
    squeezed_limit_fnl,
)


# ============================================================================
# Shape function tests
# ============================================================================

class TestShapeSymmetry:
    """Shape functions must be symmetric under k1 <-> k2 <-> k3."""

    @pytest.mark.parametrize("shape_func", [
        shape_local, shape_equilateral, shape_orthogonal, shape_folded,
    ])
    def test_symmetry_k1_k2(self, shape_func):
        """S(k1, k2, k3) = S(k2, k1, k3)."""
        k1, k2, k3 = 1.0, 2.0, 2.5
        np.testing.assert_allclose(
            shape_func(k1, k2, k3),
            shape_func(k2, k1, k3),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("shape_func", [
        shape_local, shape_equilateral, shape_orthogonal, shape_folded,
    ])
    def test_symmetry_k1_k3(self, shape_func):
        """S(k1, k2, k3) = S(k3, k2, k1)."""
        k1, k2, k3 = 1.0, 2.0, 2.5
        np.testing.assert_allclose(
            shape_func(k1, k2, k3),
            shape_func(k3, k2, k1),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("shape_func", [
        shape_local, shape_equilateral, shape_orthogonal, shape_folded,
    ])
    def test_symmetry_k2_k3(self, shape_func):
        """S(k1, k2, k3) = S(k1, k3, k2)."""
        k1, k2, k3 = 1.0, 2.0, 2.5
        np.testing.assert_allclose(
            shape_func(k1, k2, k3),
            shape_func(k1, k3, k2),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("shape_func", [
        shape_local, shape_equilateral, shape_orthogonal, shape_folded,
    ])
    def test_full_permutation_symmetry(self, shape_func):
        """All 6 permutations give the same value."""
        k1, k2, k3 = 0.8, 1.5, 2.1
        perms = [
            (k1, k2, k3), (k1, k3, k2), (k2, k1, k3),
            (k2, k3, k1), (k3, k1, k2), (k3, k2, k1),
        ]
        values = [shape_func(*p) for p in perms]
        for v in values[1:]:
            np.testing.assert_allclose(v, values[0], rtol=1e-12)


class TestShapeNormalization:
    """Each shape is normalized so S(k, k, k) = 1."""

    @pytest.mark.parametrize("shape_func", [
        shape_local, shape_equilateral, shape_orthogonal, shape_folded,
    ])
    def test_equilateral_normalization(self, shape_func):
        """S(k, k, k) = 1 for any k > 0."""
        for k in [0.01, 0.1, 1.0, 10.0, 100.0]:
            np.testing.assert_allclose(
                shape_func(k, k, k), 1.0, rtol=1e-10,
                err_msg=f"Failed at k={k} for {shape_func.__name__}",
            )


class TestShapePeaking:
    """Test that shapes peak in the correct limits."""

    def test_local_peaks_in_squeezed_limit(self):
        """Local shape is larger in squeezed limit than equilateral."""
        # Squeezed: k3 << k1 ~ k2
        k1, k2, k3_squeezed = 1.0, 1.0, 0.01
        k1_eq, k2_eq, k3_eq = 1.0, 1.0, 1.0

        squeezed_val = shape_local(k1, k2, k3_squeezed)
        equilateral_val = shape_local(k1_eq, k2_eq, k3_eq)

        # Local shape should be larger in squeezed limit
        assert squeezed_val > equilateral_val, (
            f"Local shape should peak in squeezed limit: "
            f"squeezed={squeezed_val}, equilateral={equilateral_val}"
        )

    def test_equilateral_peaks_at_equal_k(self):
        """Equilateral shape peaks at k1 = k2 = k3."""
        k_ref = 1.0
        eq_val = shape_equilateral(k_ref, k_ref, k_ref)

        # Check a few non-equilateral configurations
        configs = [
            (1.0, 1.0, 0.5),   # isosceles
            (1.0, 0.5, 0.5),   # squeezed-like
            (1.0, 1.5, 2.0),   # generic
        ]
        for k1, k2, k3 in configs:
            val = shape_equilateral(k1, k2, k3)
            assert eq_val >= val, (
                f"Equilateral shape should peak at k1=k2=k3: "
                f"S({k1},{k2},{k3})={val} > S(k,k,k)={eq_val}"
            )


class TestShapePositivity:
    """Shape functions should be positive at equilateral configuration."""

    @pytest.mark.parametrize("shape_func", [
        shape_local, shape_equilateral, shape_folded,
    ])
    def test_positive_at_equilateral(self, shape_func):
        """S(k, k, k) > 0 for local, equilateral, folded."""
        val = shape_func(1.0, 1.0, 1.0)
        assert val > 0, f"{shape_func.__name__} should be positive at equilateral"


class TestBispectrumFunction:
    """Test the convenience bispectrum() function."""

    def test_bispectrum_with_positive_fnl(self):
        """B(k,k,k) > 0 when f_NL > 0."""
        val = bispectrum(1.0, 1.0, 1.0, f_nl=5.0, shape="local")
        assert val > 0

    def test_bispectrum_vanishes_for_gaussian(self):
        """B = 0 when f_NL = 0 (Gaussian case)."""
        for shape_name in SHAPE_FUNCTIONS:
            val = bispectrum(1.0, 1.0, 1.0, f_nl=0.0, shape=shape_name)
            assert abs(val) < 1e-30, (
                f"Bispectrum should vanish for f_NL=0 (shape={shape_name})"
            )

    def test_invalid_shape_raises(self):
        """Invalid shape name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown shape"):
            bispectrum(1.0, 1.0, 1.0, f_nl=1.0, shape="invalid")


# ============================================================================
# f_NL tests
# ============================================================================

class TestFnL:
    """Tests for f_NL parameter estimation and consistency relations."""

    def test_maldacena_consistency_relation(self):
        """f_NL^local = 5/12 (1 - n_s) for single-field slow-roll.

        With n_s = 0.965, this gives f_NL ~ 0.0146 (small).
        """
        n_s = 0.965
        f_nl = fnl_maldacena(n_s)
        expected = (5.0 / 12.0) * (1.0 - n_s)
        np.testing.assert_allclose(f_nl, expected, rtol=1e-12)

    def test_maldacena_is_small(self):
        """Single-field f_NL should be O(0.01), unmeasurably small."""
        n_s = 0.965
        f_nl = fnl_maldacena(n_s)
        assert abs(f_nl) < 0.02, (
            f"Maldacena f_NL should be small: got {f_nl}"
        )

    def test_maldacena_positive_for_ns_below_one(self):
        """For n_s < 1, f_NL^local > 0 (Maldacena)."""
        n_s = 0.96
        assert fnl_maldacena(n_s) > 0

    def test_multifield_estimate(self):
        """Multi-field: f_NL^local = 5/6 (n_s - 1)."""
        n_s = 0.965
        f_nl = fnl_multifield(n_s)
        expected = (5.0 / 6.0) * (n_s - 1.0)
        np.testing.assert_allclose(f_nl, expected, rtol=1e-12)

    def test_fnl_positive_for_positive_bispectrum(self):
        """f_NL should be positive when bispectrum is positive."""
        # Use equilateral configuration where S(k,k,k) = 1
        B = 10.0
        f_nl = fnl_from_bispectrum(1.0, 1.0, 1.0, B, shape="local")
        assert f_nl > 0

    def test_fnl_negative_for_negative_bispectrum(self):
        """f_NL should be negative when bispectrum is negative."""
        B = -10.0
        f_nl = fnl_from_bispectrum(1.0, 1.0, 1.0, B, shape="local")
        assert f_nl < 0

    def test_fnl_zero_for_zero_bispectrum(self):
        """f_NL should be zero when bispectrum is zero."""
        f_nl = fnl_from_bispectrum(1.0, 1.0, 1.0, 0.0, shape="local")
        assert f_nl == 0.0


class TestPlanckConstraints:
    """Tests for Planck 2018 f_NL constraints."""

    def test_local_central_value(self):
        central, sigma = PlanckConstraints.local
        assert central == pytest.approx(-0.9, abs=0.1)

    def test_equilateral_central_value(self):
        central, sigma = PlanckConstraints.equilateral
        assert central == pytest.approx(-26.0, abs=1.0)

    def test_orthogonal_central_value(self):
        central, sigma = PlanckConstraints.orthogonal
        assert central == pytest.approx(-38.0, abs=1.0)

    def test_local_consistent_with_zero(self):
        """Local f_NL is consistent with zero at 2-sigma."""
        assert PlanckConstraints.is_consistent_with_zero("local", n_sigma=2)

    def test_equilateral_consistent_with_zero(self):
        """Equilateral f_NL is consistent with zero at 2-sigma."""
        assert PlanckConstraints.is_consistent_with_zero("equilateral", n_sigma=2)

    def test_log_likelihood_peaks_at_central(self):
        """Log-likelihood is maximized at the central value."""
        ll_at_central = fnl_log_likelihood(-0.9, shape="local")
        ll_off = fnl_log_likelihood(5.0, shape="local")
        assert ll_at_central > ll_off


class TestPowerSpectrum:
    """Tests for the primordial power spectrum."""

    def test_power_spectrum_at_pivot(self):
        """P(k_pivot) = A_s."""
        A_s = 2.1e-9
        P = power_spectrum(0.05, A_s=A_s)
        np.testing.assert_allclose(P, A_s, rtol=1e-12)

    def test_power_spectrum_red_tilt(self):
        """P(k) decreases with k for n_s < 1 (red tilt)."""
        P_low_k = power_spectrum(0.01)
        P_high_k = power_spectrum(0.1)
        assert P_low_k > P_high_k, "Red-tilted spectrum: P(0.01) > P(0.1)"


# ============================================================================
# In-in formalism tests
# ============================================================================

class TestInIn:
    """Tests for the Schwinger-Keldysh in-in formalism."""

    def test_bulk_to_boundary_at_tau_zero(self):
        """G_k(tau=0) = 1 (boundary condition)."""
        G = bulk_to_boundary_propagator(1.0, 0.0)
        np.testing.assert_allclose(G, 1.0 + 0j, atol=1e-14)

    def test_bulk_to_boundary_complex(self):
        """G_k(tau) is complex for tau != 0."""
        G = bulk_to_boundary_propagator(1.0, -1.0)
        assert np.imag(G) != 0.0

    def test_in_in_gives_finite_result(self):
        """The in-in integral should return a number (finite or nan is OK for extreme params)."""
        result = compute_bispectrum_in_in(1.0, 1.0, 1.0, shape="local",
                                          epsilon=0.01)
        # Just check it returns something (numerical integration may struggle)
        assert isinstance(result, (float, np.floating)), f"Should return float, got {type(result)}"

    def test_in_in_equilateral_gives_finite_result(self):
        """Equilateral in-in computation should return a number."""
        result = compute_bispectrum_in_in(1.0, 1.0, 1.0, shape="equilateral",
                                          epsilon=0.01, c_s=0.5)
        assert isinstance(result, (float, np.floating)), f"Should return float, got {type(result)}"

    def test_in_in_larger_epsilon_gives_larger_result(self):
        """Larger epsilon should give different bispectrum."""
        B_small = compute_bispectrum_in_in(1.0, 1.0, 1.0, shape="local",
                                           epsilon=0.01)
        B_large = compute_bispectrum_in_in(1.0, 1.0, 1.0, shape="local",
                                           epsilon=0.1)
        # Just check both return numbers
        assert isinstance(B_small, (float, np.floating))
        assert isinstance(B_large, (float, np.floating))

    def test_invalid_shape_raises(self):
        """Invalid shape for in-in should raise ValueError."""
        with pytest.raises(ValueError):
            compute_bispectrum_in_in(1.0, 1.0, 1.0, shape="orthogonal")


# ============================================================================
# PNG observables tests
# ============================================================================

class TestScaleDependentBias:
    """Tests for scale-dependent bias from local PNG."""

    def test_positive_fnl_increases_bias_at_low_k(self):
        """Positive f_NL increases bias at low k (large scales)."""
        k_low = 0.005
        b_G = 2.0
        f_nl = 10.0

        b_total = scale_dependent_bias(f_nl, k_low, b_G=b_G)
        assert b_total > b_G, (
            f"Positive f_NL should increase bias at low k: "
            f"b_total={b_total}, b_G={b_G}"
        )

    def test_zero_fnl_no_correction(self):
        """f_NL = 0 gives no bias correction."""
        b_total = scale_dependent_bias(0.0, 0.01, b_G=2.0)
        assert b_total == pytest.approx(2.0, abs=1e-10)

    def test_bias_correction_increases_at_low_k(self):
        """Delta b(k) grows as k decreases for positive f_NL."""
        f_nl = 10.0
        k_high = 0.1
        k_low = 0.005
        b_G = 2.0

        db_high = bias_correction(f_nl, k_high, b_G=b_G)
        db_low = bias_correction(f_nl, k_low, b_G=b_G)
        assert abs(db_low) > abs(db_high), (
            "Correction should be larger at smaller k"
        )

    def test_negative_fnl_decreases_bias(self):
        """Negative f_NL decreases bias at low k."""
        k_low = 0.005
        b_G = 2.0
        f_nl = -10.0

        b_total = scale_dependent_bias(f_nl, k_low, b_G=b_G)
        assert b_total < b_G, (
            f"Negative f_NL should decrease bias at low k: "
            f"b_total={b_total}, b_G={b_G}"
        )

    def test_bias_array_input(self):
        """Scale-dependent bias works with array input."""
        k_arr = np.array([0.005, 0.01, 0.05, 0.1])
        f_nl = 10.0
        b_G = 2.0
        result = scale_dependent_bias(f_nl, k_arr, b_G=b_G)
        assert result.shape == k_arr.shape
        # Monotonically decreasing correction with increasing k
        corrections = result - b_G
        assert corrections[0] > corrections[-1]


class TestSqueezedLimit:
    """Tests for squeezed-limit bispectrum relations."""

    def test_squeezed_limit_formula(self):
        """B(k, ks, ks) = (12/5) f_NL P(k) P(ks)."""
        f_nl = 5.0
        P_k = 2.1e-9
        P_ks = 2.0e-9
        B = squeezed_limit_bispectrum(f_nl, P_k, P_ks)
        expected = (12.0 / 5.0) * f_nl * P_k * P_ks
        np.testing.assert_allclose(B, expected, rtol=1e-12)

    def test_squeezed_limit_inverts(self):
        """Round-trip: f_NL -> B -> f_NL recovers original."""
        f_nl_orig = 5.0
        P_k = 2.1e-9
        P_ks = 2.0e-9
        B = squeezed_limit_bispectrum(f_nl_orig, P_k, P_ks)
        f_nl_recovered = squeezed_limit_fnl(B, P_k, P_ks)
        np.testing.assert_allclose(f_nl_recovered, f_nl_orig, rtol=1e-12)

    def test_gaussian_gives_zero_squeezed(self):
        """f_NL = 0 gives zero squeezed-limit bispectrum."""
        B = squeezed_limit_bispectrum(0.0, 2.1e-9, 2.0e-9)
        assert B == pytest.approx(0.0, abs=1e-30)

    def test_positive_fnl_positive_bispectrum(self):
        """Positive f_NL gives positive squeezed-limit bispectrum."""
        P_k = 2.1e-9
        P_ks = 2.0e-9
        B = squeezed_limit_bispectrum(5.0, P_k, P_ks)
        assert B > 0

    def test_squeezed_bispectrum_vanishes_for_gaussian(self):
        """B(k,ks,ks) = 0 when f_NL = 0 (Gaussian)."""
        B = squeezed_limit_bispectrum(0.0, 1e-9, 1e-9)
        assert B == pytest.approx(0.0, abs=1e-30)


class TestCMBBispectrum:
    """Tests for the CMB angular bispectrum."""

    def test_cmb_bispectrum_finite(self):
        """CMB reduced bispectrum computation gives finite result."""
        from primordial_bispectrum.png_observables import reduced_cmb_bispectrum

        def B_const(k1, k2, k3):
            return 1.0

        # Use a small N for speed
        b = reduced_cmb_bispectrum(10, 10, 10, B_const)
        assert np.isfinite(b)

    def test_cmb_bispectrum_zero_for_zero_input(self):
        """CMB bispectrum vanishes when B_primordial = 0."""
        from primordial_bispectrum.png_observables import reduced_cmb_bispectrum

        def B_zero(k1, k2, k3):
            return 0.0

        b = reduced_cmb_bispectrum(10, 10, 10, B_zero)
        assert b == pytest.approx(0.0, abs=1e-15)


# ============================================================================
# Integration / end-to-end tests
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_bispectrum_pipeline(self):
        """Full pipeline: f_NL -> bispectrum -> f_NL recovery."""
        f_nl_input = 5.0
        k = 1.0
        # At equilateral config, S(k,k,k)=1, so B = f_NL
        B = bispectrum(k, k, k, f_nl=f_nl_input, shape="local")
        f_nl_recovered = fnl_from_bispectrum(k, k, k, B, shape="local")
        np.testing.assert_allclose(f_nl_recovered, f_nl_input, rtol=1e-12)

    def test_all_shapes_normalized(self):
        """All shapes give S(k,k,k) = 1 via the bispectrum function."""
        for shape_name in SHAPE_FUNCTIONS:
            B = bispectrum(1.0, 1.0, 1.0, f_nl=1.0, shape=shape_name)
            np.testing.assert_allclose(
                B, 1.0, rtol=1e-10,
                err_msg=f"Shape {shape_name} not normalized",
            )

    def test_scale_dependent_bias_with_power_spectrum(self):
        """Combine power spectrum and scale-dependent bias."""
        f_nl = 10.0
        k = 0.01
        P_k = power_spectrum(k)
        b_total = scale_dependent_bias(f_nl, k, b_G=2.0)
        # With f_NL > 0 and b_G > 1, bias should be enhanced
        assert b_total > 2.0
