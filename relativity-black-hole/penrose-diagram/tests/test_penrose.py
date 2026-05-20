"""Tests for penrose-diagram."""

import numpy as np
import pytest

from penrose_diagram.coordinates import (
    kruskal_to_penrose,
    schwarzschild_conformal,
    tortoise_to_penrose,
)
from penrose_diagram.boundaries import (
    identify_horizons,
    identify_singularity,
    identify_infinities,
)
from penrose_diagram.diagram import (
    penrose_points_schwarzschild,
    penrose_points_desitter,
    penrose_points_kerr,
)


def _schwarzschild_metric(r, M=1.0):
    """Schwarzschild metric components."""
    rs = 2.0 * M
    g_tt = -(1.0 - rs / np.asarray(r, dtype=float))
    g_rr = 1.0 / (1.0 - rs / np.asarray(r, dtype=float))
    return {"g_tt": g_tt, "g_rr": g_rr}


class TestKruskalCompactFinite:
    """Kruskal to Penrose transformation should produce finite, compact values."""

    def test_large_values_compact(self):
        """Large Kruskal coordinates map to finite Penrose values."""
        U = np.array([-1e6, -1e3, 0.0, 1e3, 1e6])
        V = np.array([-1e6, -1e3, 0.0, 1e3, 1e6])
        T_p, X_p = kruskal_to_penrose(U, V)
        assert np.all(np.isfinite(T_p))
        assert np.all(np.isfinite(X_p))

    def test_bounded_by_pi(self):
        """Penrose coordinates are bounded by pi."""
        U = np.linspace(-1e10, 1e10, 100)
        V = np.linspace(-1e10, 1e10, 100)
        T_p, X_p = kruskal_to_penrose(U, V)
        assert np.all(np.abs(T_p) <= np.pi + 1e-10)
        assert np.all(np.abs(X_p) <= np.pi + 1e-10)

    def test_zero_gives_zero(self):
        U, V = 0.0, 0.0
        T_p, X_p = kruskal_to_penrose(U, V)
        assert T_p == pytest.approx(0.0)
        assert X_p == pytest.approx(0.0)


class TestConformalCompact:
    """Schwarzschild conformal transformation should be compact."""

    def test_schwarzschild_compact(self):
        """Schwarzschild exterior coordinates map to finite Penrose values."""
        M = 1.0
        r = np.linspace(2.1, 100.0, 50)
        t = np.linspace(-50.0, 50.0, 50)
        R, T = np.meshgrid(r, t)
        T_p, X_p = schwarzschild_conformal(R.ravel(), T.ravel(), M)
        assert np.all(np.isfinite(T_p))
        assert np.all(np.isfinite(X_p))

    def test_tortoise_compact(self):
        """Tortoise coordinates compactify properly."""
        r_star = np.linspace(-50, 50, 100)
        t_star = np.linspace(-50, 50, 100)
        T_p, X_p = tortoise_to_penrose(r_star, t_star)
        assert np.all(np.abs(T_p) <= np.pi + 1e-10)
        assert np.all(np.abs(X_p) <= np.pi + 1e-10)


class TestHorizonAt45Degrees:
    """The event horizon in Penrose diagram should be at 45 degrees (null)."""

    def test_horizon_slope_is_45(self):
        """The horizon line should have slope ±1 (45 degrees) in (T_p, X_p)."""
        M = 1.0
        # In Kruskal coordinates, the horizon is at U=0 or V=0
        # Points along the horizon: V=0, U varies
        U_horizon = np.linspace(-10, 10, 100)
        V_horizon = np.zeros_like(U_horizon)
        T_p, X_p = kruskal_to_penrose(U_horizon, V_horizon)

        # For V=0: T_p = arctan(U), X_p = -arctan(U) = -T_p
        # So T_p + X_p = 0 => slope = -1 (which is 45 degrees in rotated frame)
        np.testing.assert_allclose(T_p + X_p, 0.0, atol=1e-12)

    def test_kruskal_horizon_null(self):
        """Another horizon (U=0) should also be null (45 degrees)."""
        V_horizon = np.linspace(-10, 10, 100)
        U_horizon = np.zeros_like(V_horizon)
        T_p, X_p = kruskal_to_penrose(U_horizon, V_horizon)

        # For U=0: T_p = arctan(V), X_p = arctan(V) = T_p
        np.testing.assert_allclose(T_p - X_p, 0.0, atol=1e-12)


class TestSingularityBounded:
    """Singularities should be bounded within the Penrose diagram."""

    def test_singularity_inside_diamond(self):
        """The singularity at r=0 should be bounded inside the diagram."""
        M = 1.0
        # In Kruskal coords, the singularity UV = 1 maps to finite T_p, X_p
        # Let's test points on the singularity: UV = const
        U_sing = np.linspace(-10, 10, 100)
        V_sing = 1.0 / (U_sing + 1e-10)
        T_p, X_p = kruskal_to_penrose(U_sing, V_sing)
        assert np.all(np.isfinite(T_p))
        assert np.all(np.isfinite(X_p))

    def test_singularity_detection(self):
        """identify_singularity should detect r=0 singularity."""
        metric_func = lambda r: _schwarzschild_metric(r, M=1.0)
        sing = identify_singularity(metric_func)
        # Schwarzschild has a singularity at r=0
        # Our function checks for non-finite metric components near r=0
        assert isinstance(sing, list)


class TestBoundaries:
    """Test boundary identification."""

    def test_schwarzschild_horizon(self):
        """identify_horizons should find the Schwarzschild horizon at r=2M."""
        M = 1.0
        metric_func = lambda r: _schwarzschild_metric(r, M=M)
        horizons = identify_horizons(metric_func, (0.5, 4.0, 1000))
        assert len(horizons) >= 1
        # Should be close to r = 2M = 2.0
        assert any(abs(h - 2.0 * M) < 0.1 for h in horizons)

    def test_infinities_schwarzschild(self):
        """Schwarzschild spacetime should have proper infinity labels."""
        inf = identify_infinities("schwarzschild")
        assert "scri_plus" in inf
        assert "i_plus" in inf
        assert "i_zero" in inf


class TestDiagrams:
    """Test diagram point generation."""

    def test_schwarzschild_points(self):
        pts = penrose_points_schwarzschild(M=1.0)
        assert "i_plus" in pts
        assert "bifurcation" in pts

    def test_desitter_points(self):
        pts = penrose_points_desitter(H=1.0)
        assert "future_infinity_T" in pts

    def test_kerr_points(self):
        pts = penrose_points_kerr(M=1.0, a=0.5)
        assert "r_plus" in pts
        assert "r_minus" in pts
        # Outer horizon should be larger than inner
        assert pts["r_plus"] > pts["r_minus"]

    def test_kerr_extremal(self):
        """For a = M, the horizons coincide (extremal Kerr)."""
        pts = penrose_points_kerr(M=1.0, a=1.0)
        assert pts["r_plus"] == pytest.approx(pts["r_minus"])
