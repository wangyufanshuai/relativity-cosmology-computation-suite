"""Tests for the Unruh Effect Detector package."""

import numpy as np
import pytest

from unruh_effect.constants import C, G_EARTH, HBAR, K_B
from unruh_effect.rindler import (
    minkowski_to_rindler,
    rindler_christoffel,
    rindler_metric,
    rindler_to_minkowski,
)
from unruh_effect.temperature import (
    inverse_unruh_temperature,
    thermal_spectrum,
    unruh_temperature,
)
from unruh_effect.detector import (
    detector_response_function,
    detector_response_rate,
    wightman_function_minkowski,
    wightman_function_rindler,
)


# -----------------------------------------------------------------------
# Temperature tests
# -----------------------------------------------------------------------

class TestUnruhTemperature:
    """Tests for the Unruh temperature formula."""

    def test_unruh_temperature_zero_accel(self):
        """Zero acceleration must give zero temperature."""
        T = float(unruh_temperature(0.0))
        assert T == 0.0

    def test_unruh_temperature_1g(self):
        """At 1 g the Unruh temperature is extremely small, ~4e-20 K."""
        T = float(unruh_temperature(G_EARTH))
        assert 3e-20 < T < 5e-20, f"T = {T}"

    def test_inverse_roundtrip(self):
        """inverse_unruh_temperature(unruh_temperature(a)) == a."""
        a = 1e18  # large acceleration for numerical stability
        T = float(unruh_temperature(a))
        a2 = float(inverse_unruh_temperature(T))
        assert np.isclose(a, a2, rtol=1e-12)


# -----------------------------------------------------------------------
# Rindler coordinate tests
# -----------------------------------------------------------------------

class TestRindlerCoordinates:
    """Tests for Rindler coordinate transformations and metric."""

    def test_rindler_roundtrip(self):
        """Minkowski -> Rindler -> Minkowski roundtrip recovers original."""
        a = 1.0
        t0, x0 = 0.5, 2.0  # in the right wedge: x > |t|
        eta, xi = minkowski_to_rindler(t0, x0, a)
        t1, x1 = rindler_to_minkowski(eta, xi, a)
        assert np.isclose(t1, t0, rtol=1e-12)
        assert np.isclose(x1, x0, rtol=1e-12)

    def test_rindler_metric_at_xi_equals_one(self):
        """g_00 at xi=1 with a=1 should be -1 (i.e. -(a*1)^2 = -1)."""
        a = 1.0
        g = rindler_metric(1.0, a)
        # g is (2,2) array for scalar xi input
        assert np.isclose(g[0, 0], -(a * 1.0) ** 2)
        assert np.isclose(g[1, 1], 1.0)
        assert np.isclose(g[0, 1], 0.0)

    def test_rindler_metric_general(self):
        """g_00 = -(a*xi)^2, g_11 = 1 for general xi."""
        a = 2.5
        xi = 3.0
        g = rindler_metric(xi, a)
        assert np.isclose(g[0, 0], -(a * xi) ** 2)
        assert np.isclose(g[1, 1], 1.0)

    def test_rindler_christoffel(self):
        """Non-zero Christoffel symbols at known xi."""
        a = 1.0
        xi = 2.0
        G = rindler_christoffel(a)(xi)
        # Gamma^0_{01} = Gamma^0_{10} = 1/xi
        assert np.isclose(G[0, 0, 1], 1.0 / xi)
        assert np.isclose(G[0, 1, 0], 1.0 / xi)
        # Gamma^1_{00} = a^2 * xi
        assert np.isclose(G[1, 0, 0], a**2 * xi)
        # All others zero
        assert np.isclose(G[1, 1, 1], 0.0)
        assert np.isclose(G[0, 0, 0], 0.0)

    def test_rindler_transformation_known_point(self):
        """At eta=0: t=0, x=xi (observer at closest approach)."""
        a = 1.5
        eta = 0.0
        xi = 3.0
        t, x = rindler_to_minkowski(eta, xi, a)
        assert np.isclose(t, 0.0)
        assert np.isclose(x, xi)


# -----------------------------------------------------------------------
# Thermal spectrum tests
# -----------------------------------------------------------------------

class TestThermalSpectrum:
    """Tests for the Planck thermal distribution."""

    def test_thermal_spectrum_planck(self):
        """Known value: n(omega) at specific omega, T."""
        # Choose omega and T so that hbar*omega/(k_B*T) = 1
        # => n = 1/(e - 1) = 1/(e-1)
        T = 1.0  # arbitrary
        omega = K_B * T / HBAR  # makes hbar*omega/(k_B*T) = 1
        n = float(thermal_spectrum(omega, T))
        expected = 1.0 / (np.e - 1.0)
        assert np.isclose(n, expected, rtol=1e-10)

    def test_thermal_spectrum_zero_T(self):
        """At T=0 the occupation number must vanish."""
        n = float(thermal_spectrum(1e15, 0.0))
        assert n == 0.0

    def test_thermal_spectrum_high_T(self):
        """At high T, n -> k_B*T / (hbar*omega) (classical limit)."""
        omega = 1e10
        T = 1e10
        n = float(thermal_spectrum(omega, T))
        classical = K_B * T / (HBAR * omega)
        assert np.isclose(n, classical, rtol=0.01)


# -----------------------------------------------------------------------
# Detector tests
# -----------------------------------------------------------------------

class TestDetector:
    """Tests for the Unruh-DeWitt detector."""

    def test_detector_response_positive(self):
        """Transition rate must be non-negative."""
        a = 1e20  # large acceleration for measurable rates
        omega = 1e15
        R = detector_response_rate(omega, a, method="analytical")
        assert R >= 0.0

    def test_detector_response_zero_accel(self):
        """Zero acceleration gives zero response."""
        R = detector_response_rate(1e15, 0.0)
        assert R == 0.0

    def test_detector_response_thermal(self):
        """Analytical response must be proportional to Planck distribution.

        R(omega) = omega / (2 pi) * 1 / (exp(hbar omega / k_B T_U) - 1)
                 = omega / (2 pi) * n(omega, T_U)
        """
        a = 1e20
        omega = 1e15
        T_U = float(unruh_temperature(a))
        R = detector_response_rate(omega, a, method="analytical")
        n_th = float(thermal_spectrum(omega, T_U))
        expected = omega / (2.0 * np.pi) * n_th
        assert np.isclose(R, expected, rtol=1e-8)

    def test_detector_response_numerical_matches_analytical(self):
        """Numerical integration should agree with analytical result."""
        a = 1e20
        omega = 1e15
        R_ana = detector_response_rate(omega, a, method="analytical")
        R_num = detector_response_rate(omega, a, method="numerical")
        # Numerical integration has finite precision; allow 20% tolerance
        assert np.isclose(R_ana, R_num, rtol=0.2), (
            f"analytical={R_ana:.6e}, numerical={R_num:.6e}"
        )

    def test_detector_response_function(self):
        """Response function over an array should be consistent."""
        # Need large enough acceleration so kT/hbar ~ omega range
        a = 1e22  # T ~ 40 K, kT/hbar ~ 5e12 rad/s
        omegas = np.logspace(6, 12, 10)
        R = detector_response_function(omegas, a)
        assert R.shape == omegas.shape
        assert np.all(R >= 0.0)
        # There should be nonzero response in the thermal range
        assert np.any(R > 0)


# -----------------------------------------------------------------------
# Wightman function tests
# -----------------------------------------------------------------------

class TestWightman:
    """Tests for Wightman function properties."""

    def test_wightman_positive(self):
        """The Wightman function must satisfy positivity:
        sum_ij G^+(x_i, x_j) c_i c_j* >= 0 for any set of points and
        coefficients.  For a single point this reduces to positivity of
        the autocorrelation, i.e. Im G^+(x,x) >= 0 (or at least
        finite).  We test the simpler property that G+(x,x') evaluated
        at coincident points is well-defined (finite after i*eps).
        """
        # Coincident-point limit: G+(t,x; t,x) = -1/(4 pi^2) / (-eps^2 - 0)
        G = wightman_function_minkowski(1.0, 1.0, 1.0, 1.0)
        assert np.isfinite(G)

    def test_wightman_hermiticity(self):
        """G^+(x, x') = [G^+(x', x)]* (Hermiticity of Wightman function)."""
        G1 = wightman_function_minkowski(1.0, 2.0, 0.5, 1.5)
        G2 = wightman_function_minkowski(0.5, 1.5, 1.0, 2.0)
        assert np.isclose(G1, np.conj(G2))

    def test_wightman_rindler_consistency(self):
        """Rindler Wightman function should agree with Minkowski after
        transforming coordinates."""
        a = 1.0
        eta1, xi1 = 0.5, 2.0
        eta2, xi2 = 1.0, 2.0
        G_rindler = wightman_function_rindler(eta1, xi1, eta2, xi2, a)
        t1, x1 = rindler_to_minkowski(eta1, xi1, a)
        t2, x2 = rindler_to_minkowski(eta2, xi2, a)
        G_mink = wightman_function_minkowski(t1, x1, t2, x2)
        assert np.isclose(G_rindler, G_mink, rtol=1e-12)
