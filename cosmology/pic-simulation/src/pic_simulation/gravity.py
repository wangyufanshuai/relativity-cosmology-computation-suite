"""
FFT-based Poisson solver for gravitational potential and forces.

Solves the Poisson equation:
    nabla^2 Phi = (3/2) * Omega_m * H0^2 * a * delta

In Fourier space:
    Phi_k = -(3/2) * Omega_m * H0^2 * a * delta_k / k^2

Forces:
    F_k = -i * k * Phi_k
"""

import numpy as np
from numpy.typing import NDArray


def solve_poisson_fft(
    delta: NDArray[np.floating],
    N: int,
    L: float,
    a: float,
    omega_m: float = 0.3,
    h0: float = 0.7,
) -> NDArray[np.floating]:
    """
    Solve the Poisson equation using FFT and return the gravitational potential.

    Parameters
    ----------
    delta : ndarray, shape (N, N, N)
        Density contrast field delta = rho/rho_bar - 1.
    N : int
        Grid resolution per dimension.
    L : float
        Box side length in Mpc/h.
    a : float
        Scale factor.
    omega_m : float
        Matter density parameter.
    h0 : float
        Hubble parameter / 100 km/s/Mpc.

    Returns
    -------
    phi : ndarray, shape (N, N, N)
        Gravitational potential in real space.
    """
    H0 = h0 * 100.0  # km/s/Mpc
    cell_size = L / N

    prefactor = 1.5 * omega_m * H0**2 * a

    # FFT of density contrast
    delta_k = np.fft.fftn(delta)

    # Wavenumber grids (physical wavenumbers in rad / [L units])
    kx = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    kz = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")

    k_sq = kx_grid**2 + ky_grid**2 + kz_grid**2
    k_sq[0, 0, 0] = 1.0  # avoid division by zero for k=0 mode

    # Solve: Phi_k = -prefactor * delta_k / k^2
    phi_k = -prefactor * delta_k / k_sq
    phi_k[0, 0, 0] = 0.0  # zero mean potential

    phi = np.real(np.fft.ifftn(phi_k))
    return phi


def compute_forces_from_potential(
    phi: NDArray[np.floating],
    N: int,
    L: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute the gravitational force field F = -grad(Phi) from the potential.

    Parameters
    ----------
    phi : ndarray, shape (N, N, N)
        Gravitational potential in real space.
    N : int
        Grid resolution per dimension.
    L : float
        Box side length.

    Returns
    -------
    force_x, force_y, force_z : ndarray, shape (N, N, N)
        Components of the gravitational force field on the grid.
    """
    cell_size = L / N

    phi_k = np.fft.fftn(phi)

    kx = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    ky = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    kz = np.fft.fftfreq(N, d=cell_size) * 2.0 * np.pi
    kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing="ij")

    force_x = np.real(np.fft.ifftn(-1j * kx_grid * phi_k))
    force_y = np.real(np.fft.ifftn(-1j * ky_grid * phi_k))
    force_z = np.real(np.fft.ifftn(-1j * kz_grid * phi_k))

    return force_x, force_y, force_z


def compute_gravitational_forces(
    delta: NDArray[np.floating],
    N: int,
    L: float,
    a: float,
    omega_m: float = 0.3,
    h0: float = 0.7,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Full gravity pipeline: density contrast -> Poisson solve -> force field.

    Parameters
    ----------
    delta : ndarray, shape (N, N, N)
        Density contrast field.
    N : int
        Grid resolution per dimension.
    L : float
        Box side length.
    a : float
        Scale factor.
    omega_m : float
        Matter density parameter.
    h0 : float
        Hubble parameter / 100 km/s/Mpc.

    Returns
    -------
    force_x, force_y, force_z : ndarray, shape (N, N, N)
        Gravitational force field components on the grid.
    """
    phi = solve_poisson_fft(delta, N, L, a, omega_m, h0)
    return compute_forces_from_potential(phi, N, L)
