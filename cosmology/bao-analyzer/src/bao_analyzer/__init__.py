"""
BAO Analyzer: Baryon Acoustic Oscillation analysis toolbox.

Provides tools for computing correlation functions, power spectra,
and performing BAO peak fitting with template matching and anisotropic analysis.
"""

from .power_spectrum import (
    eisenstein_hu_transfer_nw,
    linear_power_spectrum,
    no_wiggle_power_spectrum,
    wiggle_power_spectrum,
)
from .correlation import (
    landy_szalay_estimator,
    power_to_correlation,
)
from .bao_fitting import (
    bao_peak_detect,
    bao_template_fit,
    anisotropic_bao,
    chi2_likelihood,
)

__version__ = "0.1.0"

__all__ = [
    "eisenstein_hu_transfer_nw",
    "linear_power_spectrum",
    "no_wiggle_power_spectrum",
    "wiggle_power_spectrum",
    "landy_szalay_estimator",
    "power_to_correlation",
    "bao_peak_detect",
    "bao_template_fit",
    "anisotropic_bao",
    "chi2_likelihood",
]
