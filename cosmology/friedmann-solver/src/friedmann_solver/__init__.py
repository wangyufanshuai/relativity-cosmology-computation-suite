"""Friedmann equation solver for cosmological background evolution."""

__version__ = "0.1.0"

from .cosmology import Cosmology
from .background import solve_background, conformal_time, horizon_scale
from .planck18 import planck18_params, planck18_derived, fisher_matrix

__all__ = [
    "Cosmology",
    "solve_background",
    "conformal_time",
    "horizon_scale",
    "planck18_params",
    "planck18_derived",
    "fisher_matrix",
]
