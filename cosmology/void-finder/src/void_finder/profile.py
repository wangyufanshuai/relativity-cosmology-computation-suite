"""Void density profile analysis."""

import numpy as np


def radial_density_profile(r_array, void_center, positions, densities):
    """Compute the radial density profile around a void center.

    Parameters
    ----------
    r_array : array_like
        Radial bins (edges or centers) at which to compute the profile.
    void_center : array_like
        3D position of the void center.
    positions : array_like
        (N, 3) array of particle positions.
    densities : array_like
        (N,) array of density values at each particle position.

    Returns
    -------
    array_like
        Mean density in each radial bin, normalized by mean density.
    """
    r_array = np.asarray(r_array, dtype=float)
    void_center = np.asarray(void_center, dtype=float)
    positions = np.asarray(positions, dtype=float)
    densities = np.asarray(densities, dtype=float)

    # Compute distances from void center
    dr = positions - void_center
    dist = np.sqrt(np.sum(dr**2, axis=1))

    mean_density = densities.mean()
    if mean_density == 0:
        mean_density = 1.0

    profile = np.zeros(len(r_array))
    for i, r in enumerate(r_array):
        # Use shell of width dr around r
        mask = (dist >= r * 0.8) & (dist < r * 1.2)
        if mask.sum() > 0:
            profile[i] = densities[mask].mean() / mean_density
        else:
            profile[i] = 1.0  # default to mean density

    return profile


def stacked_profile(void_catalog, positions, densities, n_bins):
    """Compute the stacked (averaged) density profile over all voids.

    Parameters
    ----------
    void_catalog : dict
        Void catalog from watershed_voids.
    positions : array_like
        (N, 3) array of particle positions.
    densities : array_like
        (N,) array of density values.
    n_bins : int
        Number of radial bins (in units of void radius).

    Returns
    -------
    tuple
        (r_normalized, profile) where r_normalized is in units of mean void radius
        and profile is the stacked normalized density profile.
    """
    centers = void_catalog['centers']
    radii = void_catalog['radii']

    if len(radii) == 0:
        r_norm = np.linspace(0, 2, n_bins)
        return r_norm, np.ones(n_bins)

    r_norm = np.linspace(0, 2, n_bins)
    all_profiles = []

    for i, (center, R) in enumerate(zip(centers, radii)):
        if R <= 0:
            continue
        r_physical = r_norm * R
        prof = radial_density_profile(r_physical, center, positions, densities)
        all_profiles.append(prof)

    if not all_profiles:
        return r_norm, np.ones(n_bins)

    stacked = np.mean(all_profiles, axis=0)
    return r_norm, stacked
