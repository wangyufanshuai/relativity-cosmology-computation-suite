"""
Kick-Drift-Kick (KDK) leapfrog integrator for the PM simulation.

The KDK scheme is a symplectic, second-order integrator:
    v_{n+1/2} = v_n + F(x_n) * dt/2            (kick)
    x_{n+1}   = x_n + v_{n+1/2} * dt            (drift)
    v_{n+1}   = v_{n+1/2} + F(x_{n+1}) * dt/2   (kick)

Time stepping is in scale factor 'a'.

In comoving coordinates the equations of motion are:
    dx/dt = v / a           (drift)
    dv/dt = -grad(Phi) / a  (kick)

Converting to scale-factor time using da = a * H(a) * dt:
    dx/da = v / (a^2 * H(a))
    dv/da = -grad(Phi) / (a^2 * H(a))
"""

import numpy as np
from numpy.typing import NDArray

from .mesh import cic_assign, compute_density_contrast, cic_interpolate_vector
from .gravity import compute_gravitational_forces


def hubble_factor(a: float, omega_m: float = 0.3, h0: float = 0.7) -> float:
    """
    Compute H(a) = H0 * sqrt(Omega_m / a^3 + Omega_Lambda).

    Flat LCDM cosmology: Omega_m + Omega_Lambda = 1.

    Parameters
    ----------
    a : float
        Scale factor.
    omega_m : float
        Matter density parameter.
    h0 : float
        Hubble parameter / 100 km/s/Mpc.

    Returns
    -------
    H : float
        Hubble parameter at scale factor a, in km/s/Mpc.
    """
    H0 = h0 * 100.0
    omega_lambda = 1.0 - omega_m
    return H0 * np.sqrt(omega_m / a**3 + omega_lambda)


def compute_acceleration(
    positions: NDArray[np.floating],
    particle_mass: float,
    N: int,
    L: float,
    a: float,
    omega_m: float = 0.3,
    h0: float = 0.7,
) -> NDArray[np.floating]:
    """
    Compute the gravitational acceleration at each particle position.

    Pipeline:
    1. CIC mass assignment -> density grid
    2. Density contrast delta
    3. FFT Poisson solve -> force field on grid
    4. CIC interpolation -> force at particle positions
    5. Convert to peculiar acceleration: a_pec = F / a

    Parameters
    ----------
    positions : ndarray, shape (n_particles, 3)
        Particle positions.
    particle_mass : float
        Mass per particle.
    N : int
        Grid resolution.
    L : float
        Box side length.
    a : float
        Scale factor.
    omega_m : float
        Matter density parameter.
    h0 : float
        Hubble parameter.

    Returns
    -------
    acceleration : ndarray, shape (n_particles, 3)
        Peculiar acceleration at each particle position.
    """
    n_particles = positions.shape[0]
    masses = np.full(n_particles, particle_mass)

    # 1. CIC mass assignment
    density = cic_assign(positions, masses, N, L)

    # 2. Density contrast
    delta = compute_density_contrast(density)

    # 3. FFT Poisson solve -> force field on grid
    force_x, force_y, force_z = compute_gravitational_forces(
        delta, N, L, a, omega_m, h0
    )

    # 4. CIC interpolation of force to particle positions
    forces = cic_interpolate_vector(force_x, force_y, force_z, positions, N, L)

    # 5. Peculiar acceleration = F / a
    acceleration = forces / a

    return acceleration


def kdk_step(
    positions: NDArray[np.floating],
    velocities: NDArray[np.floating],
    particle_mass: float,
    N: int,
    L: float,
    a_current: float,
    da: float,
    omega_m: float = 0.3,
    h0: float = 0.7,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """
    Perform one Kick-Drift-Kick leapfrog step.

    Parameters
    ----------
    positions : ndarray, shape (n_particles, 3)
        Current particle positions.
    velocities : ndarray, shape (n_particles, 3)
        Current particle velocities.
    particle_mass : float
        Mass per particle.
    N : int
        Grid resolution.
    L : float
        Box side length.
    a_current : float
        Current scale factor.
    da : float
        Scale factor step size.
    omega_m : float
        Matter density parameter.
    h0 : float
        Hubble parameter.

    Returns
    -------
    new_positions : ndarray, shape (n_particles, 3)
    new_velocities : ndarray, shape (n_particles, 3)
    a_new : float
    """
    a_new = a_current + da

    # --- First Kick (half step) ---
    H_current = hubble_factor(a_current, omega_m, h0)
    kick_factor_1 = da / (2.0 * a_current**2 * H_current)

    acc_current = compute_acceleration(
        positions, particle_mass, N, L, a_current, omega_m, h0
    )
    velocities_half = velocities + acc_current * kick_factor_1

    # --- Drift (full step) ---
    a_mid = 0.5 * (a_current + a_new)
    H_mid = hubble_factor(a_mid, omega_m, h0)
    drift_factor = da / (a_mid**2 * H_mid)

    new_positions = positions + velocities_half * drift_factor

    # Periodic boundary conditions
    new_positions = new_positions % L

    # --- Second Kick (half step) ---
    H_new = hubble_factor(a_new, omega_m, h0)
    kick_factor_2 = da / (2.0 * a_new**2 * H_new)

    acc_new = compute_acceleration(
        new_positions, particle_mass, N, L, a_new, omega_m, h0
    )
    new_velocities = velocities_half + acc_new * kick_factor_2

    return new_positions, new_velocities, a_new


def run_simulation(
    N: int = 16,
    L: float = 100.0,
    a_start: float = 0.02,
    a_end: float = 0.1,
    n_steps: int = 5,
    omega_m: float = 0.3,
    h0: float = 0.7,
    seed: int = 42,
) -> dict:
    """
    Run a complete PM cosmological simulation.

    Parameters
    ----------
    N : int
        Grid resolution per dimension (N^3 particles).
    L : float
        Box side length in Mpc/h.
    a_start : float
        Initial scale factor.
    a_end : float
        Final scale factor.
    n_steps : int
        Number of time steps.
    omega_m : float
        Matter density parameter.
    h0 : float
        Hubble parameter / 100 km/s/Mpc.
    seed : int
        Random seed.

    Returns
    -------
    results : dict with 'positions', 'velocities', 'a_final', 'n_particles', 'history'.
    """
    from .particles import initialize_particles

    positions, velocities, particle_mass = initialize_particles(
        N, L, a_start, omega_m, h0, seed=seed
    )

    a_values = np.linspace(a_start, a_end, n_steps + 1)
    da = a_values[1] - a_values[0]

    history = []
    a_current = a_start

    for i in range(n_steps):
        positions, velocities, a_current = kdk_step(
            positions, velocities, particle_mass, N, L, a_current, da, omega_m, h0
        )
        history.append({
            "step": i + 1,
            "a": a_current,
            "positions": positions.copy(),
            "velocities": velocities.copy(),
        })

    return {
        "positions": positions,
        "velocities": velocities,
        "a_final": a_current,
        "n_particles": N**3,
        "history": history,
    }
