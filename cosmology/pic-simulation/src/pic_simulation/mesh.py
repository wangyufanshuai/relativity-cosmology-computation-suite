"""
Cloud-in-Cell (CIC) mass assignment and force interpolation.

CIC distributes each particle's mass to the 2^d nearest grid cells
(d = number of dimensions, here 3) using linear (trilinear) weights.
The same kernel is used in reverse for interpolating grid-based forces
to particle positions.
"""

import numpy as np
from numpy.typing import NDArray


def cic_assign(
    positions: NDArray[np.floating],
    masses: NDArray[np.floating],
    N: int,
    L: float,
) -> NDArray[np.floating]:
    """
    Cloud-in-Cell mass assignment: deposit particle masses onto a 3D grid.

    Each particle contributes to the 8 (=2^3) nearest grid cells with
    trilinear (CIC) weights. The weight for each cell is the product of
    the 1D linear weights along each axis.

    Parameters
    ----------
    positions : ndarray, shape (n_particles, 3)
        Particle positions in the periodic box [0, L).
    masses : ndarray, shape (n_particles,)
        Mass of each particle. If scalar, same mass for all particles.
    N : int
        Number of grid cells per dimension.
    L : float
        Side length of the simulation box.

    Returns
    -------
    density_grid : ndarray, shape (N, N, N)
        Mass density deposited on the grid. The total mass on the grid
        equals the total mass of all particles (mass conservation).
    """
    cell_size = L / N
    n_particles = positions.shape[0]

    # Handle scalar mass
    if np.ndim(masses) == 0:
        masses = np.full(n_particles, float(masses))

    density_grid = np.zeros((N, N, N), dtype=np.float64)

    # Normalized positions in grid units
    x_grid = positions[:, 0] / cell_size
    y_grid = positions[:, 1] / cell_size
    z_grid = positions[:, 2] / cell_size

    for p in range(n_particles):
        # Index of the nearest lower grid point
        ix = int(np.floor(x_grid[p]))
        iy = int(np.floor(y_grid[p]))
        iz = int(np.floor(z_grid[p]))

        # Fractional distance from the lower grid point
        dx = x_grid[p] - ix
        dy = y_grid[p] - iy
        dz = z_grid[p] - iz

        # CIC weights: 8 neighbors
        # (ix, iy, iz) gets weight (1-dx)(1-dy)(1-dz)
        # (ix+1, iy, iz) gets weight dx*(1-dy)*(1-dz), etc.
        weights = np.array(
            [
                (1 - dx) * (1 - dy) * (1 - dz),
                dx * (1 - dy) * (1 - dz),
                (1 - dx) * dy * (1 - dz),
                dx * dy * (1 - dz),
                (1 - dx) * (1 - dy) * dz,
                dx * (1 - dy) * dz,
                (1 - dx) * dy * dz,
                dx * dy * dz,
            ]
        )

        # Cell indices with periodic wrapping
        ix1 = (ix + 1) % N
        iy1 = (iy + 1) % N
        iz1 = (iz + 1) % N
        ix0 = ix % N
        iy0 = iy % N
        iz0 = iz % N

        # Deposit mass with CIC weights
        m = masses[p]
        density_grid[ix0, iy0, iz0] += m * weights[0]
        density_grid[ix1, iy0, iz0] += m * weights[1]
        density_grid[ix0, iy1, iz0] += m * weights[2]
        density_grid[ix1, iy1, iz0] += m * weights[3]
        density_grid[ix0, iy0, iz1] += m * weights[4]
        density_grid[ix1, iy0, iz1] += m * weights[5]
        density_grid[ix0, iy1, iz1] += m * weights[6]
        density_grid[ix1, iy1, iz1] += m * weights[7]

    return density_grid


def compute_density_contrast(
    density_grid: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute the density contrast delta = rho / rho_bar - 1.

    Parameters
    ----------
    density_grid : ndarray, shape (N, N, N)
        Mass density on the grid.

    Returns
    -------
    delta : ndarray, shape (N, N, N)
        Density contrast field. Has zero mean by construction:
        mean(delta) = 0.
    """
    mean_density = np.mean(density_grid)
    if mean_density == 0:
        return np.zeros_like(density_grid)
    delta = density_grid / mean_density - 1.0
    return delta


def cic_interpolate(
    field: NDArray[np.floating],
    positions: NDArray[np.floating],
    N: int,
    L: float,
) -> NDArray[np.floating]:
    """
    Interpolate a 3D grid field to particle positions using CIC weights.

    Uses the same trilinear kernel as the mass assignment (Cloud-in-Cell),
    ensuring that the assignment-interpolation pair is self-consistent
    (satisfies the "convolution" property needed for PM force accuracy).

    Parameters
    ----------
    field : ndarray, shape (N, N, N)
        Grid field to interpolate (e.g., force components).
    positions : ndarray, shape (n_particles, 3)
        Particle positions in the periodic box [0, L).
    N : int
        Grid resolution per dimension.
    L : float
        Box side length.

    Returns
    -------
    interpolated : ndarray, shape (n_particles,)
        Field values interpolated to particle positions.
    """
    cell_size = L / N
    n_particles = positions.shape[0]

    # Normalized positions in grid units
    x_grid = positions[:, 0] / cell_size
    y_grid = positions[:, 1] / cell_size
    z_grid = positions[:, 2] / cell_size

    result = np.zeros(n_particles, dtype=np.float64)

    for p in range(n_particles):
        ix = int(np.floor(x_grid[p]))
        iy = int(np.floor(y_grid[p]))
        iz = int(np.floor(z_grid[p]))

        dx = x_grid[p] - ix
        dy = y_grid[p] - iy
        dz = z_grid[p] - iz

        weights = np.array(
            [
                (1 - dx) * (1 - dy) * (1 - dz),
                dx * (1 - dy) * (1 - dz),
                (1 - dx) * dy * (1 - dz),
                dx * dy * (1 - dz),
                (1 - dx) * (1 - dy) * dz,
                dx * (1 - dy) * dz,
                (1 - dx) * dy * dz,
                dx * dy * dz,
            ]
        )

        ix0 = ix % N
        iy0 = iy % N
        iz0 = iz % N
        ix1 = (ix + 1) % N
        iy1 = (iy + 1) % N
        iz1 = (iz + 1) % N

        result[p] = (
            field[ix0, iy0, iz0] * weights[0]
            + field[ix1, iy0, iz0] * weights[1]
            + field[ix0, iy1, iz0] * weights[2]
            + field[ix1, iy1, iz0] * weights[3]
            + field[ix0, iy0, iz1] * weights[4]
            + field[ix1, iy0, iz1] * weights[5]
            + field[ix0, iy1, iz1] * weights[6]
            + field[ix1, iy1, iz1] * weights[7]
        )

    return result


def cic_interpolate_vector(
    field_x: NDArray[np.floating],
    field_y: NDArray[np.floating],
    field_z: NDArray[np.floating],
    positions: NDArray[np.floating],
    N: int,
    L: float,
) -> NDArray[np.floating]:
    """
    Interpolate a vector field (3 components) from the grid to particle positions.

    Convenience wrapper that calls cic_interpolate for each component.

    Parameters
    ----------
    field_x, field_y, field_z : ndarray, shape (N, N, N)
        Grid components of the vector field (e.g., force).
    positions : ndarray, shape (n_particles, 3)
        Particle positions.
    N : int
        Grid resolution per dimension.
    L : float
        Box side length.

    Returns
    -------
    result : ndarray, shape (n_particles, 3)
        Interpolated vector field at particle positions.
    """
    fx = cic_interpolate(field_x, positions, N, L)
    fy = cic_interpolate(field_y, positions, N, L)
    fz = cic_interpolate(field_z, positions, N, L)
    return np.column_stack([fx, fy, fz])
