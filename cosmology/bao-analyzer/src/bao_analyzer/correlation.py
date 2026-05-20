"""
Two-point correlation function estimation.

Implements the Landy-Szalay estimator and Fourier-space correlation
via Hankel transform of the power spectrum.
"""

import numpy as np
from scipy.spatial import cKDTree

from .utils import hankel_transform, logspace_k, linspace_s
from .power_spectrum import linear_power_spectrum, no_wiggle_power_spectrum


# ---------------------------------------------------------------------------
# Landy-Szalay estimator from galaxy positions
# ---------------------------------------------------------------------------

def landy_szalay_estimator(positions, random_positions, s_bins,
                           box_size=None):
    """Compute 2-point correlation function using the Landy-Szalay estimator.

    xi(s) = (DD - 2*DR + RR) / RR

    Parameters
    ----------
    positions : ndarray
        Galaxy positions, shape (N, 3) in Mpc/h.
    random_positions : ndarray
        Random catalog positions, shape (M, 3) in Mpc/h.
    s_bins : ndarray
        Separation bin edges in Mpc/h.
    box_size : float, optional
        Periodic box side length. If None, no periodic wrapping is applied.

    Returns
    -------
    s_mid : ndarray
        Bin centers in Mpc/h.
    xi : ndarray
        Correlation function values.
    """
    positions = np.asarray(positions, dtype=float)
    random_positions = np.asarray(random_positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if random_positions.ndim != 2 or random_positions.shape[1] != 3:
        raise ValueError("random_positions must have shape (M, 3)")

    s_mid = 0.5 * (s_bins[:-1] + s_bins[1:])
    n_bins = len(s_mid)
    N = len(positions)
    M = len(random_positions)

    # DD pair counts
    dd_counts = _pair_counts(positions, positions, s_bins, box_size, exclude_zero=True)
    # RR pair counts
    rr_counts = _pair_counts(random_positions, random_positions, s_bins, box_size, exclude_zero=True)
    # DR cross-pair counts
    dr_counts = _pair_counts(positions, random_positions, s_bins, box_size, exclude_zero=False)

    # Normalize
    dd_norm = dd_counts / (0.5 * N * (N - 1)) if N > 1 else dd_counts
    rr_norm = rr_counts / (0.5 * M * (M - 1)) if M > 1 else rr_counts
    dr_norm = dr_counts / (N * M)

    # Landy-Szalay estimator
    rr_safe = np.where(rr_norm > 0, rr_norm, 1.0)
    xi = (dd_norm - 2 * dr_norm + rr_norm) / rr_safe

    return s_mid, xi


def _pair_counts(pos1, pos2, bins, box_size=None, exclude_zero=True):
    """Count pairs in separation bins using KDTree.

    Parameters
    ----------
    pos1, pos2 : ndarray
        Position arrays of shape (N, 3).
    bins : ndarray
        Separation bin edges.
    box_size : float, optional
        Periodic box size.
    exclude_zero : bool
        If True, exclude self-pairs (used for DD, RR).

    Returns
    -------
    ndarray
        Pair counts per bin.
    """
    if box_size is not None:
        # Wrap positions into [0, box_size)
        pos1 = pos1 % box_size
        pos2 = pos2 % box_size

    tree1 = cKDTree(pos1, boxsize=box_size)
    tree2 = cKDTree(pos2, boxsize=box_size)

    # Count pairs
    counts = np.zeros(len(bins) - 1, dtype=float)
    max_r = bins[-1]

    if exclude_zero and np.array_equal(pos1, pos2):
        # Self-pairs: count pairs and subtract self-count
        for i in range(len(bins) - 1):
            r_min, r_max = bins[i], bins[i + 1]
            # Query ball for all points
            pairs = tree1.count_neighbors(tree2, r_max)
            pairs_inner = tree1.count_neighbors(tree2, r_min) if r_min > 0 else len(pos1)
            counts[i] = pairs - pairs_inner
        # Subtract self-pairs (distance = 0)
        # The first bin might include self-pairs; the count_neighbors
        # for same tree already handles this properly
    else:
        for i in range(len(bins) - 1):
            r_min, r_max = bins[i], bins[i + 1]
            pairs = tree1.count_neighbors(tree2, r_max)
            pairs_inner = tree1.count_neighbor(tree2, r_min) if r_min > 0 else 0
            counts[i] = pairs - pairs_inner

    return counts


# ---------------------------------------------------------------------------
# Fourier-space correlation function
# ---------------------------------------------------------------------------

def power_to_correlation(k, pk, s=None):
    """Compute correlation function xi(s) via Hankel transform of P(k).

    xi(s) = 1/(2 pi^2) int_0^inf k^2 dk sin(ks)/(ks) P(k)

    Parameters
    ----------
    k : ndarray
        Wavenumber array in h/Mpc.
    pk : ndarray
        Power spectrum in (Mpc/h)^3.
    s : ndarray, optional
        Separation array in Mpc/h. Defaults to linspace_s().

    Returns
    -------
    s : ndarray
        Separation array.
    xi : ndarray
        Correlation function.
    """
    if s is None:
        s = linspace_s()

    xi = hankel_transform(k, pk, s, ell=0)
    return s, xi


def correlation_function(s=None, k=None, cosmo=None, use_wiggles=True):
    """Compute the full correlation function with BAO peak.

    Parameters
    ----------
    s : ndarray, optional
        Separation array in Mpc/h.
    k : ndarray, optional
        Wavenumber array in h/Mpc.
    cosmo : dict, optional
        Cosmological parameters.
    use_wiggles : bool
        If True, use full P(k) with BAO. If False, use no-wiggle P_nw(k).

    Returns
    -------
    s : ndarray
        Separation array.
    xi : ndarray
        Correlation function.
    """
    if k is None:
        k = logspace_k(nk=2000, k_min=1e-4, k_max=50.0)
    if s is None:
        s = linspace_s(ns=300, s_min=1.0, s_max=300.0)
    if cosmo is None:
        from .utils import DEFAULT_COSMO
        cosmo = DEFAULT_COSMO

    if use_wiggles:
        pk = linear_power_spectrum(k, cosmo)
    else:
        pk = no_wiggle_power_spectrum(k, cosmo)

    xi = hankel_transform(k, pk, s, ell=0)
    return s, xi


def no_wiggle_correlation(s=None, k=None, cosmo=None):
    """Compute the no-wiggle (smooth) correlation function.

    Parameters
    ----------
    s : ndarray, optional
        Separation array.
    k : ndarray, optional
        Wavenumber array.
    cosmo : dict, optional
        Cosmological parameters.

    Returns
    -------
    s : ndarray
        Separation array.
    xi_nw : ndarray
        Smooth correlation function.
    """
    return correlation_function(s, k, cosmo, use_wiggles=False)
