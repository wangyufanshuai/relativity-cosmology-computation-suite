"""
Non-Gaussianity parameter f_NL estimation and constraints.

Implements f_NL definitions, theorems, and observational bounds:

- Maldacena's consistency relation for single-field slow-roll inflation
- Multi-field f_NL estimates
- Planck 2018 observational constraints
- f_NL estimation from a measured bispectrum
"""

import numpy as np
from .bispectrum_shapes import shape_local, shape_equilateral, shape_orthogonal, shape_folded


# ---------------------------------------------------------------------------
# Power spectrum (dimensionless, log-space amplitude)
# ---------------------------------------------------------------------------

def power_spectrum(k, A_s=2.1e-9, n_s=0.965, k_pivot=0.05):
    """Primordial scalar power spectrum P(zeta, k).

    P(k) = A_s * (k / k_pivot)^{n_s - 1}

    Parameters
    ----------
    k : float or ndarray
        Wavenumber in Mpc^{-1}.
    A_s : float
        Scalar amplitude at the pivot scale.
    n_s : float
        Scalar spectral index.
    k_pivot : float
        Pivot scale in Mpc^{-1}.

    Returns
    -------
    float or ndarray
    """
    return A_s * (k / k_pivot) ** (n_s - 1.0)


# ---------------------------------------------------------------------------
# f_NL from bispectrum amplitude
# ---------------------------------------------------------------------------

def fnl_from_bispectrum(k1, k2, k3, B_measured, shape="local"):
    """Estimate f_NL from a measured bispectrum amplitude.

    f_NL = B(k1, k2, k3) / S(k1, k2, k3)

    where S is the normalized shape function with S(k,k,k) = 1.

    Parameters
    ----------
    k1, k2, k3 : float
        Triangle wavenumbers.
    B_measured : float
        Measured bispectrum amplitude.
    shape : str
        Shape template name.

    Returns
    -------
    float
        Estimated f_NL value.
    """
    shape_func = {
        "local": shape_local,
        "equilateral": shape_equilateral,
        "orthogonal": shape_orthogonal,
        "folded": shape_folded,
    }
    if shape not in shape_func:
        raise ValueError(f"Unknown shape '{shape}'")
    S = shape_func[shape](k1, k2, k3)
    if abs(S) < 1e-30:
        return 0.0
    return B_measured / S


# ---------------------------------------------------------------------------
# Theoretical f_NL predictions
# ---------------------------------------------------------------------------

def fnl_maldacena(n_s=0.965):
    """Maldacena's consistency relation for single-field slow-roll inflation.

    f_NL^local = 5/12 * (1 - n_s)

    This is unmeasurably small (O(0.01)) and serves as a null test:
    a detection of f_NL >> 0.01 rules out all single-field models.

    Parameters
    ----------
    n_s : float
        Scalar spectral index.

    Returns
    -------
    float
        Predicted local f_NL.
    """
    return (5.0 / 12.0) * (1.0 - n_s)


def fnl_multifield(n_s=0.965):
    """Simplified multi-field inflation estimate for local f_NL.

    f_NL^local = 5/6 * (n_s - 1)

    This is a rough estimate; actual multi-field models can produce
    a wide range of f_NL values depending on field space geometry,
    potential, and initial conditions.

    Parameters
    ----------
    n_s : float
        Scalar spectral index.

    Returns
    -------
    float
        Estimated local f_NL.
    """
    return (5.0 / 6.0) * (n_s - 1.0)


# ---------------------------------------------------------------------------
# Planck 2018 constraints
# ---------------------------------------------------------------------------

class PlanckConstraints:
    """Planck 2018 f_NL constraints (68% CL).

    References
    ----------
    Planck Collaboration, Astron. Astrophys. 641, A10 (2020), Table 4.
    """

    # (central_value, +-error)
    local = (-0.9, 5.1)
    equilateral = (-26.0, 47.0)
    orthogonal = (-38.0, 24.0)

    @classmethod
    def get(cls, shape):
        """Return (central, sigma) for a given shape."""
        return getattr(cls, shape)

    @classmethod
    def is_consistent_with_zero(cls, shape, n_sigma=2):
        """Check whether the Planck measurement is consistent with zero
        at the given number of sigma."""
        central, sigma = cls.get(shape)
        return abs(central) < n_sigma * sigma


# ---------------------------------------------------------------------------
# f_NL likelihood helper
# ---------------------------------------------------------------------------

def fnl_log_likelihood(f_nl, shape="local"):
    """Gaussian log-likelihood for f_NL given Planck 2018 constraints.

    Parameters
    ----------
    f_nl : float
        Test value of f_NL.
    shape : str
        Shape type.

    Returns
    -------
    float
        Log-likelihood (up to a constant).
    """
    central, sigma = PlanckConstraints.get(shape)
    return -0.5 * ((f_nl - central) / sigma) ** 2


# ---------------------------------------------------------------------------
# Convenience: bispectrum from f_NL and shape
# ---------------------------------------------------------------------------

def bispectrum_from_fnl(k1, k2, k3, f_nl, shape="local"):
    """Compute the full bispectrum B(k1, k2, k3) given f_NL and a shape.

    B(k1,k2,k3) = f_NL * S(k1,k2,k3)

    where S is the normalized shape template.

    Parameters
    ----------
    k1, k2, k3 : float
        Wavenumbers.
    f_nl : float
        Non-linearity parameter.
    shape : str
        Shape template name.

    Returns
    -------
    float
        Bispectrum amplitude.
    """
    shape_func = {
        "local": shape_local,
        "equilateral": shape_equilateral,
        "orthogonal": shape_orthogonal,
        "folded": shape_folded,
    }
    if shape not in shape_func:
        raise ValueError(f"Unknown shape '{shape}'")
    return f_nl * shape_func[k1, k2, k3](k1, k2, k3) if isinstance(k1, np.ndarray) else f_nl * shape_func[shape](k1, k2, k3)
