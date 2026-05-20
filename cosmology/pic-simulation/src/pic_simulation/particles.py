"""
Particle initialization using 2LPT (second-order Lagrangian perturbation theory).

Generates N^3 particles on a uniform Lagrangian grid in a cubic box of
side length L, then displaces them using both first-order (Zel'dovich / 1LPT)
and second-order (2LPT) displacements:

    Psi_1(q) = -D_1(a) * grad(Phi(q))                    [1LPT / Zel'dovich]
    Psi_2(q) ~ D_2(a) * grad(Phi_2(q))                   [2LPT correction]

    x = q + Psi_1 + Psi_2
    v = a * H(a) * f_1 * Psi_1 + a * H(a) * f_2 * Psi_2

where Phi is a Gaussian random potential drawn from an input power spectrum P(k),
D_1 and D_2 are first and second-order growth factors, and f_i = d ln D_i / d ln a.
"""

import numpy as np
from numpy.typing import NDArray


def generate_uniform_grid(N: int, L: float) -> NDArray[np.floating]:
    """
    Generate particle positions on a uniform Lagrangian grid.

    Places N^3 particles at the cell centers of an N x N x N grid in a
    periodic box of side length L.

    Parameters
    ----------
    N : int
        Number of grid cells per dimension (total particles = N^3).
    L : float
        Side length of the cubic simulation box.

    Returns
    -------
    positions : ndarray, shape (N^3, 3)
        Lagrangian (grid) positions of all particles.
    """
    cell_size = L / N
    coords = (np.arange(N, dtype=np.float64) + 0.5) * cell_size
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    positions = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return positions


def generate_potential_from_power_spectrum(
    N: int,
    L: float,
    Pk=None,
    seed: int = 42,
) -> NDArray[np.complexfloating]:
    """
    Generate a Gaussian random potential Phi(k) from an input power spectrum.

    Parameters
    ----------
    N : int
        Grid resolution per dimension.
    L : float
        Box side length.
    Pk : callable or None
        Function Pk(k) returning the power spectrum amplitude at wavenumber k.
        If None, uses a Harrison-Zel'dovich-like P(k) ~ k with a smooth
        transfer function cutoff at the Nyquist frequency.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    phi_k : ndarray, shape (N, N, N), complex
        Fourier-space displacement potential.
    """
    rng = np.random.default_rng(seed)

    cell_size = L / N

    # Wave numbers for each dimension (cycles per unit length)
    kx = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    kz = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")

    k_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)
    k_mag_safe = k_mag.copy()
    k_mag_safe[0, 0, 0] = 1.0  # avoid division by zero

    if Pk is not None:
        power_spectrum = Pk(k_mag)
    else:
        # Default: Harrison-Zel'dovich P(k) ~ k with smooth cutoff
        k_nyquist = np.pi * N / L
        transfer = np.exp(-(k_mag_safe / k_nyquist) ** 2)
        power_spectrum = k_mag_safe * transfer**2

    power_spectrum = np.asarray(power_spectrum, dtype=np.float64)
    power_spectrum[0, 0, 0] = 0.0

    # Gaussian random field: amplitude ~ sqrt(P(k)), random phase
    amplitude = np.sqrt(power_spectrum)
    phases = rng.uniform(0, 2 * np.pi, size=(N, N, N))
    phi_k = amplitude * np.exp(1j * phases)
    phi_k[0, 0, 0] = 0.0

    return phi_k


def compute_2lpt_displacement(
    N: int,
    L: float,
    a_start: float,
    omega_m: float = 0.3,
    h0: float = 0.7,
    Pk=None,
    seed: int = 42,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute 1LPT + 2LPT particle displacements and velocities.

    First-order (Zel'dovich / 1LPT):
        Psi_1(q) = -D_1(a) * grad(Phi(q))

    Second-order (2LPT) correction:
        Psi_2(q) proportional to D_2(a) * grad(Phi_2(q))
        where Phi_2 is sourced by the tidal tensor of Psi_1.

    Parameters
    ----------
    N : int
        Number of grid cells per dimension.
    L : float
        Box side length.
    a_start : float
        Initial scale factor (e.g. 0.02 for z=49).
    omega_m : float
        Matter density parameter Omega_m.
    h0 : float
        Hubble constant in units of 100 km/s/Mpc.
    Pk : callable or None
        Input power spectrum function Pk(k). If None, uses default.
    seed : int
        Random seed.

    Returns
    -------
    displacement : ndarray, shape (N^3, 3)
        Total displacement Psi_1 + Psi_2 to add to Lagrangian positions.
    velocity : ndarray, shape (N^3, 3)
        Peculiar velocity in simulation units.
    """
    H0 = h0 * 100.0  # km/s/Mpc

    # Growth factors in matter-dominated approximation
    D1 = a_start  # D_1(a) ~ a
    D2 = -3.0 / 7.0 * D1**2 * omega_m ** (-1.0 / 7.0)  # 2LPT growth factor

    # Growth rates f = d ln D / d ln a
    f1 = 1.0  # In matter domination, f_1 ~ 1 for D_1 ~ a
    f2 = 2.0  # For D_2 ~ a^2, f_2 = 2

    cell_size = L / N

    # Generate random potential in Fourier space
    phi_k = generate_potential_from_power_spectrum(N, L, Pk, seed)

    # Wave number grids
    kx = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    kz = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")

    k_mag_sq = kx_grid**2 + ky_grid**2 + kz_grid**2
    k_mag_sq[0, 0, 0] = 1.0  # avoid division by zero

    # --- 1LPT displacement: Psi_1_k = -D1 * i*k * phi_k / |k|^2 ---
    psi1_kx = -D1 * (1j * kx_grid) * phi_k / k_mag_sq
    psi1_ky = -D1 * (1j * ky_grid) * phi_k / k_mag_sq
    psi1_kz = -D1 * (1j * kz_grid) * phi_k / k_mag_sq

    # Transform to real space: shape (N, N, N) each
    psi1_x = np.real(np.fft.ifftn(psi1_kx))
    psi1_y = np.real(np.fft.ifftn(psi1_ky))
    psi1_z = np.real(np.fft.ifftn(psi1_kz))

    # --- 2LPT correction ---
    # Compute derivatives of 1LPT displacement in Fourier space
    # d(psi1_x)/dq_x_k = i*kx * psi1_kx, etc.
    dpsi1x_dx = np.real(np.fft.ifftn(1j * kx_grid * psi1_kx))
    dpsi1x_dy = np.real(np.fft.ifftn(1j * ky_grid * psi1_kx))
    dpsi1x_dz = np.real(np.fft.ifftn(1j * kz_grid * psi1_kx))

    dpsi1y_dx = np.real(np.fft.ifftn(1j * kx_grid * psi1_ky))
    dpsi1y_dy = np.real(np.fft.ifftn(1j * ky_grid * psi1_ky))
    dpsi1y_dz = np.real(np.fft.ifftn(1j * kz_grid * psi1_ky))

    dpsi1z_dx = np.real(np.fft.ifftn(1j * kx_grid * psi1_kz))
    dpsi1z_dy = np.real(np.fft.ifftn(1j * ky_grid * psi1_kz))
    dpsi1z_dz = np.real(np.fft.ifftn(1j * kz_grid * psi1_kz))

    # Source term for 2LPT potential:
    # S = sum_{i,j} (dPsi1_i/dq_j)(dPsi1_j/dq_i)
    source = (
        dpsi1x_dx * dpsi1x_dx
        + dpsi1x_dy * dpsi1y_dx
        + dpsi1x_dz * dpsi1z_dx
        + dpsi1y_dx * dpsi1x_dy
        + dpsi1y_dy * dpsi1y_dy
        + dpsi1y_dz * dpsi1z_dy
        + dpsi1z_dx * dpsi1x_dz
        + dpsi1z_dy * dpsi1y_dz
        + dpsi1z_dz * dpsi1z_dz
    )

    # Transform source to Fourier space and solve for 2LPT potential
    source_k = np.fft.fftn(source)
    phi2_k = D2 * source_k / k_mag_sq
    phi2_k[0, 0, 0] = 0.0

    # 2LPT displacement: Psi_2 = -grad(phi_2)
    psi2_kx = -(1j * kx_grid) * phi2_k
    psi2_ky = -(1j * ky_grid) * phi2_k
    psi2_kz = -(1j * kz_grid) * phi2_k

    psi2_x = np.real(np.fft.ifftn(psi2_kx))
    psi2_y = np.real(np.fft.ifftn(psi2_ky))
    psi2_z = np.real(np.fft.ifftn(psi2_kz))

    # Total displacement on the grid
    disp_x = psi1_x + psi2_x
    disp_y = psi1_y + psi2_y
    disp_z = psi1_z + psi2_z

    displacement = np.column_stack(
        [disp_x.ravel(), disp_y.ravel(), disp_z.ravel()]
    )

    # Velocities: v = a * H(a) * f * Psi
    omega_lambda = 1.0 - omega_m
    H_a = H0 * np.sqrt(omega_m / a_start**3 + omega_lambda)

    psi1_flat = np.column_stack([psi1_x.ravel(), psi1_y.ravel(), psi1_z.ravel()])
    psi2_flat = np.column_stack([psi2_x.ravel(), psi2_y.ravel(), psi2_z.ravel()])

    velocity = (
        a_start * H_a * f1 * psi1_flat
        + a_start * H_a * f2 * psi2_flat
    )

    return displacement, velocity


def initialize_particles(
    N: int,
    L: float,
    a_start: float,
    omega_m: float = 0.3,
    h0: float = 0.7,
    Pk=None,
    seed: int = 42,
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """
    Initialize particle positions and velocities using 2LPT.

    Parameters
    ----------
    N : int
        Grid resolution per dimension (N^3 particles total).
    L : float
        Box side length in Mpc/h.
    a_start : float
        Initial scale factor.
    omega_m : float
        Matter density parameter.
    h0 : float
        Hubble parameter / 100 km/s/Mpc.
    Pk : callable or None
        Input power spectrum Pk(k). If None, uses default.
    seed : int
        Random seed.

    Returns
    -------
    positions : ndarray, shape (N^3, 3)
        Displaced particle positions x = q + Psi_1 + Psi_2.
    velocities : ndarray, shape (N^3, 3)
        Particle peculiar velocities.
    particle_mass : float
        Mass per particle (total mass normalized to 1.0).
    """
    q = generate_uniform_grid(N, L)
    displacement, velocity = compute_2lpt_displacement(
        N, L, a_start, omega_m, h0, Pk, seed
    )
    positions = q + displacement
    positions = positions % L  # periodic boundary conditions

    n_particles = N**3
    particle_mass = 1.0 / n_particles  # total mass = 1.0

    return positions, velocity, particle_mass
