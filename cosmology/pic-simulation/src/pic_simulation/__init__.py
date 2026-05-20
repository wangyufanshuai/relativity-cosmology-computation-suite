"""
pic_simulation: Particle-Mesh (PM) cosmological N-body simulation.

Modules
-------
particles : Particle initialization with 2LPT displacements
mesh : Cloud-in-Cell mass assignment and force interpolation
gravity : FFT-based Poisson solver
integrator : Kick-Drift-Kick leapfrog time integrator
"""

from .particles import initialize_particles, generate_uniform_grid
from .mesh import cic_assign, compute_density_contrast, cic_interpolate, cic_interpolate_vector
from .gravity import solve_poisson_fft, compute_forces_from_potential, compute_gravitational_forces
from .integrator import kdk_step, run_simulation

__version__ = "0.1.0"

__all__ = [
    "initialize_particles",
    "generate_uniform_grid",
    "cic_assign",
    "compute_density_contrast",
    "cic_interpolate",
    "cic_interpolate_vector",
    "solve_poisson_fft",
    "compute_forces_from_potential",
    "compute_gravitational_forces",
    "kdk_step",
    "run_simulation",
]
