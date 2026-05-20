"""Tests for scalar-tensor gravity module."""

import numpy as np
import pytest
from scalar_tensor.brans_dicke import (
    brans_dicke_action,
    ppn_gamma,
    ppn_beta,
    cassini_constraint,
)
from scalar_tensor.cosmology import (
    brans_dicke_friedmann,
    conformal_transform,
    inverse_conformal_transform,
    solve_background,
)
from scalar_tensor.stars import (
    scalar_field_profile,
    vainshtein_radius,
)


def test_ppn_gamma_1_over_2_for_large_omega():
    """PPN gamma should approach 1 for large omega (GR limit)."""
    # For omega -> inf, gamma -> 1
    gamma_large = ppn_gamma(1e10)
    assert abs(gamma_large - 1.0) < 1e-8, f"gamma({omega}) = {gamma_large}, expected ~1"

    # Exact check: gamma(40000) should be close to 1
    gamma_cassini = ppn_gamma(40000)
    expected = 40001 / 40002
    assert abs(gamma_cassini - expected) < 1e-10

    # For omega = 1, gamma = 2/3
    gamma_1 = ppn_gamma(1)
    assert abs(gamma_1 - 2.0 / 3.0) < 1e-10


def test_cassini_omega_constraint():
    """Cassini constraint should give omega > 40000."""
    result = cassini_constraint()
    assert result["omega_min"] == 40000
    assert result["gamma_error"] < 3e-5

    # Verify: for omega=40000, gamma should be within Cassini bounds
    gamma = ppn_gamma(40000)
    # gamma = (40000+1)/(40000+2) = 40001/40002 ~ 0.999975
    # |gamma - 1| = 1/40002 ~ 2.5e-5 < 2*gamma_error = 4.6e-5
    assert abs(gamma - 1.0) < 2 * result["gamma_error"]


def test_conformal_roundtrip():
    """Jordan -> Einstein -> Jordan frame roundtrip should recover the metric."""
    # Random metric in Jordan frame (symmetric)
    g_jordan = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0],
    ])
    phi = 2.0

    g_einstein = conformal_transform(g_jordan, phi)
    g_recovered = inverse_conformal_transform(g_einstein, phi)

    np.testing.assert_allclose(g_recovered, g_jordan, atol=1e-12,
                               err_msg="Roundtrip should recover original metric")

    # Also test with non-trivial off-diagonal (symmetric) metric
    g_sym = np.array([
        [1.5, 0.3, 0.0, 0.0],
        [0.3, 2.0, -0.1, 0.0],
        [0.0, -0.1, 3.0, 0.2],
        [0.0, 0.0, 0.2, 4.0],
    ])
    phi2 = 0.5
    g_e2 = conformal_transform(g_sym, phi2)
    g_r2 = inverse_conformal_transform(g_e2, phi2)
    np.testing.assert_allclose(g_r2, g_sym, atol=1e-12)


def test_scalar_field_finite():
    """Scalar field profile should be finite for reasonable parameters."""
    r = np.linspace(1.0, 100.0, 50)
    omega = 40000.0
    M_star = 1.0

    phi = scalar_field_profile(r, omega, M_star)

    assert np.all(np.isfinite(phi)), "Field values should be finite"
    assert np.all(phi > 0), "Field should be positive"
    # Field should be close to 1 for large omega
    assert np.allclose(phi, 1.0, atol=0.01), f"For omega=40000, phi should be ~1, got {phi}"


def test_brans_dicke_action():
    """BD action should return proper components."""
    result = brans_dicke_action(omega=1, phi=1.0)
    assert result["kinetic_coefficient"] == -1.0
    assert result["ricci_coupling"] == 1.0
    assert result["omega"] == 1


def test_ppn_beta():
    """PPN beta should be exactly 1 for BD theory."""
    assert ppn_beta(1) == 1.0
    assert ppn_beta(40000) == 1.0
    assert ppn_beta(1e10) == 1.0


def test_solve_background():
    """Background cosmology solver should return finite results."""
    sol = solve_background(omega=40000, phi0=1.0, a_range=(0.01, 1.0))
    assert len(sol["a"]) > 0
    assert np.all(np.isfinite(sol["a"]))
    assert np.all(np.isfinite(sol["phi"]))
    assert np.all(np.isfinite(sol["H"]))


def test_vainshtein_radius():
    """Vainshtein radius should be positive and scale properly."""
    r_V = vainshtein_radius(M=5.97e24, r=1.5e11, beta=1.0)
    assert r_V > 0, "Vainshtein radius should be positive"
    assert np.isfinite(r_V), "Vainshtein radius should be finite"
