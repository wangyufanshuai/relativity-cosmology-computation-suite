"""Tests for pic-simulation: particle-mesh cosmological simulation."""

import numpy as np
import pytest

from pic_simulation.particles import generate_uniform_grid, initialize_particles
from pic_simulation.mesh import cic_assign, compute_density_contrast
from pic_simulation.gravity import solve_poisson_fft
from pic_simulation.integrator import hubble_factor


class TestParticles:
    """Test particle initialization."""

    def test_uniform_grid_count(self):
        """Uniform grid should have N^3 particles."""
        pos = generate_uniform_grid(4, 100.0)
        assert pos.shape == (64, 3)

    def test_uniform_grid_range(self):
        """Particles should be in [0, L)."""
        N, L = 4, 100.0
        pos = generate_uniform_grid(N, L)
        assert np.all(pos >= 0)
        assert np.all(pos < L)

    def test_initialize_particles(self):
        """initialize_particles should return positions, velocities, mass."""
        pos, vel, mass = initialize_particles(N=4, L=100.0, a_start=0.02, seed=42)
        assert pos.shape == (64, 3)
        assert vel.shape == (64, 3)
        assert mass > 0


class TestMesh:
    """Test CIC mass assignment."""

    def test_cic_mass_conservation(self):
        """Total mass on grid should equal total particle mass."""
        N, L = 4, 100.0
        pos = generate_uniform_grid(N, L)
        masses = np.ones(N**3)
        grid = cic_assign(pos, masses, N, L)
        assert np.sum(grid) == pytest.approx(N**3, rel=1e-6)

    def test_density_contrast_zero_mean(self):
        """Density contrast should have zero mean."""
        grid = np.random.default_rng(42).uniform(0.5, 1.5, (4, 4, 4))
        delta = compute_density_contrast(grid)
        assert np.mean(delta) == pytest.approx(0.0, abs=1e-10)


class TestGravity:
    """Test FFT Poisson solver."""

    def test_poisson_potential_finite(self):
        """Potential should be finite for uniform density."""
        delta = np.zeros((4, 4, 4))
        phi = solve_poisson_fft(delta, N=4, L=100.0, a=0.1)
        assert np.all(np.isfinite(phi))

    def test_poisson_zero_density_zero_potential(self):
        """Zero density contrast should give zero potential."""
        delta = np.zeros((4, 4, 4))
        phi = solve_poisson_fft(delta, N=4, L=100.0, a=0.1)
        np.testing.assert_allclose(phi, 0.0, atol=1e-10)


class TestIntegrator:
    """Test Hubble factor and integrator utilities."""

    def test_hubble_factor_positive(self):
        """Hubble factor should be positive."""
        H = hubble_factor(0.5)
        assert H > 0

    def test_hubble_factor_at_a1(self):
        """H(a=1) should equal H0 for matter + Lambda = 1."""
        H = hubble_factor(1.0, omega_m=0.3, h0=0.7)
        # H0 = 70, Omega_m=0.3, Omega_L=0.7 => H(1) = 70*sqrt(1) = 70
        assert H == pytest.approx(70.0, rel=1e-10)
