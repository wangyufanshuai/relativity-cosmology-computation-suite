"""
Bispectrum shape functions for primordial non-Gaussianity.

Implements the standard template shapes used to parametrize the primordial
bispectrum B(k1, k2, k3) = <zeta(k1) zeta(k2) zeta(k3)>.

Each shape function S(k1, k2, k3) is normalized so that S(k, k, k) = 1
at the equilateral configuration.  The shapes depend only on ratios of the
wavenumbers, making them scale-free.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(k, eps=1e-30):
    return max(k, eps)


def _ratios(k1, k2, k3):
    """Normalize k's so k1=1 and return (1, p, q)."""
    k1 = _clamp(k1)
    return 1.0, k2 / k1, k3 / k1


# ---------------------------------------------------------------------------
# Shape kernels defined in terms of dimensionless ratios
# ---------------------------------------------------------------------------

def _local_ratio_shape(k1, k2, k3):
    """Local shape: symmetric under permutation.

    S_local ~ 1/(k1^3 k2^3) + 1/(k2^3 k3^3) + 1/(k1^3 k3^3)
    At equilateral k1=k2=k3: S = 3/k^6.
    """
    k1, k2, k3 = _clamp(k1), _clamp(k2), _clamp(k3)
    return 1.0 / (k1**3 * k2**3) + 1.0 / (k2**3 * k3**3) + 1.0 / (k1**3 * k3**3)


def _equilateral_ratio_shape(k1, k2, k3):
    """Equilateral shape: peaks at k1=k2=k3.

    Uses a symmetric Gaussian in the ratios k_i/k_j that is maximal
    when all three momenta are equal.

    Template:
      S_eq ~ exp( -((k1-k2)^2 + (k2-k3)^2 + (k1-k3)^2) / sigma^2 / k_avg^2 )
    """
    k1, k2, k3 = _clamp(k1), _clamp(k2), _clamp(k3)
    k_avg = (k1 + k2 + k3) / 3.0
    sigma = 1.0
    deviation = ((k1 - k2)**2 + (k2 - k3)**2 + (k1 - k3)**2) / (k_avg**2 * sigma**2)
    return np.exp(-deviation)


def _orthogonal_ratio_shape(p, q):
    """Orthogonal shape in terms of ratios.

    Constructed as a linear combination that has a distinctive sign pattern
    with opposite signs in squeezed vs flattened limits.

    S_orth = (8/3) * S_local - S_equilateral
    """
    # Convert ratios (p,q) back to (k1,k2,k3) = (1, p, q) for 3-arg functions
    loc = _local_ratio_shape(1.0, p, q)
    eq = _equilateral_ratio_shape(1.0, p, q)
    return (8.0 / 3.0) * loc - eq


def _folded_ratio_shape(p, q):
    """Folded (flattened) shape in terms of ratios.

    Peaks when one momentum equals the sum of the other two,
    e.g. q ~ 1 + p (flattened triangle).
    """
    p, q = _clamp(p), _clamp(q)
    # Triangle inequality factors
    a = 1.0 + p - q   # large when q << 1+p
    b = 1.0 + q - p   # large when p << 1+q
    c = p + q - 1.0    # large when p+q >> 1

    # Folded: peaks when c is maximized, i.e., p+q >> 1
    folded_piece = a * b * c / (p * q)**2

    # Background
    background = 1.0 / p**3 + 1.0 / q**3 + 1.0 / (p**3 * q**3)

    return 3.0 * folded_piece - 2.0 * background


# ---------------------------------------------------------------------------
# Normalization constants (value of each kernel at p=q=1)
# ---------------------------------------------------------------------------

_LOCAL_NORM = _local_ratio_shape(1.0, 1.0, 1.0)
_EQUILATERAL_NORM = _equilateral_ratio_shape(1.0, 1.0, 1.0)
_ORTHOGONAL_NORM = _orthogonal_ratio_shape(1.0, 1.0)
_FOLDED_NORM = _folded_ratio_shape(1.0, 1.0)


# ---------------------------------------------------------------------------
# Normalized public shape functions
# ---------------------------------------------------------------------------

def shape_local(k1, k2, k3):
    """Normalized local shape function.

    Peaks in the squeezed limit (k3 << k1 ~ k2).
    Normalized so that S(k, k, k) = 1.

    Parameters
    ----------
    k1, k2, k3 : float
        Wavenumbers forming a triangle.

    Returns
    -------
    float
        Shape function value.
    """
    k1, k2, k3 = _clamp(k1), _clamp(k2), _clamp(k3)
    # Normalize to make scale-invariant: divide by k_avg^6
    k_avg = (k1 + k2 + k3) / 3.0
    return _local_ratio_shape(k1, k2, k3) * k_avg**6 / _LOCAL_NORM


def shape_equilateral(k1, k2, k3):
    """Normalized equilateral shape function.

    Peaks at k1 = k2 = k3.
    Normalized so that S(k, k, k) = 1.
    """
    k1, k2, k3 = _clamp(k1), _clamp(k2), _clamp(k3)
    return _equilateral_ratio_shape(k1, k2, k3) / _EQUILATERAL_NORM


def shape_orthogonal(k1, k2, k3):
    """Normalized orthogonal shape function.

    Has opposite-sign peaks in squeezed and flattened limits.
    Normalized so that S(k, k, k) = 1. Fully symmetric under permutations.
    """
    k1, k2, k3 = _clamp(k1), _clamp(k2), _clamp(k3)
    # Average over all permutations for exact symmetry
    from itertools import permutations
    vals = []
    for ki, kj, kk in permutations([k1, k2, k3]):
        _, p, q = _ratios(ki, kj, kk)
        vals.append(_orthogonal_ratio_shape(p, q))
    return np.mean(vals) / _ORTHOGONAL_NORM


def shape_folded(k1, k2, k3):
    """Normalized folded (flattened) shape function.

    Peaks in the flattened limit where one k equals the sum of the other two.
    Normalized so that S(k, k, k) = 1. Fully symmetric under permutations.
    """
    k1, k2, k3 = _clamp(k1), _clamp(k2), _clamp(k3)
    from itertools import permutations
    vals = []
    for ki, kj, kk in permutations([k1, k2, k3]):
        _, p, q = _ratios(ki, kj, kk)
        vals.append(_folded_ratio_shape(p, q))
    return np.mean(vals) / _FOLDED_NORM


# ---------------------------------------------------------------------------
# Convenience: build the full bispectrum B(k1, k2, k3) = f_NL * S(k1,k2,k3)
# ---------------------------------------------------------------------------

SHAPE_FUNCTIONS = {
    "local": shape_local,
    "equilateral": shape_equilateral,
    "orthogonal": shape_orthogonal,
    "folded": shape_folded,
}


def bispectrum(k1, k2, k3, f_nl=1.0, shape="local"):
    """Compute the bispectrum B(k1, k2, k3) = f_NL * S(k1, k2, k3).

    Parameters
    ----------
    k1, k2, k3 : float
        Wavenumbers.
    f_nl : float
        Non-linearity parameter f_NL.
    shape : str
        Shape name: 'local', 'equilateral', 'orthogonal', 'folded'.

    Returns
    -------
    float
        Bispectrum amplitude.
    """
    if shape not in SHAPE_FUNCTIONS:
        raise ValueError(f"Unknown shape '{shape}'. Choose from {list(SHAPE_FUNCTIONS)}")
    return f_nl * SHAPE_FUNCTIONS[shape](k1, k2, k3)
