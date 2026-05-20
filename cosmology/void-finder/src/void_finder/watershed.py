"""Watershed void finding algorithm."""

import numpy as np
from scipy import ndimage


def watershed_voids(density_field, positions):
    """Identify cosmic voids using a watershed algorithm on the density field.

    Parameters
    ----------
    density_field : array_like
        Density contrast field (delta = rho/rho_mean - 1) on a grid.
    positions : array_like
        Particle/galaxy positions, shape (N, 3).

    Returns
    -------
    dict
        Void catalog with keys:
        - 'labels': integer label array for each voxel (0 = not a void)
        - 'centers': (M, 3) array of void centers
        - 'radii': (M,) array of effective void radii
        - 'n_voids': number of voids found
    """
    density_field = np.asarray(density_field, dtype=float)

    # Identify underdense regions (density below mean)
    mask = density_field < 0.0

    if not np.any(mask):
        return {
            'labels': np.zeros_like(density_field, dtype=int),
            'centers': np.empty((0, 3)),
            'radii': np.empty(0),
            'n_voids': 0,
        }

    # Label connected underdense regions
    labels, n_features = ndimage.label(mask)

    # Compute centers and radii for each void
    centers = []
    radii = []

    for i in range(1, n_features + 1):
        void_mask = labels == i
        # Center of mass in grid coordinates
        coords = np.argwhere(void_mask)
        center = coords.mean(axis=0)
        centers.append(center)

        # Effective radius from volume: V = 4/3 * pi * R^3
        n_cells = void_mask.sum()
        r_eff = (3.0 * n_cells / (4.0 * np.pi)) ** (1.0 / 3.0)
        radii.append(r_eff)

    # Filter out very small voids (less than a few cells)
    valid = [i for i, r in enumerate(radii) if r >= 1.0]
    if not valid:
        return {
            'labels': np.zeros_like(density_field.shape, dtype=int),
            'centers': np.empty((0, 3)),
            'radii': np.empty(0),
            'n_voids': 0,
        }

    centers = np.array(centers)[valid]
    radii = np.array(radii)[valid]

    return {
        'labels': labels,
        'centers': centers,
        'radii': radii,
        'n_voids': len(radii),
    }


def void_radii(void_catalog):
    """Extract void radii from a void catalog.

    Parameters
    ----------
    void_catalog : dict
        Void catalog from watershed_voids.

    Returns
    -------
    array_like
        Array of void effective radii.
    """
    return void_catalog['radii']


def void_centers(void_catalog):
    """Extract void centers from a void catalog.

    Parameters
    ----------
    void_catalog : dict
        Void catalog from watershed_voids.

    Returns
    -------
    array_like
        (M, 3) array of void center positions.
    """
    return void_catalog['centers']
