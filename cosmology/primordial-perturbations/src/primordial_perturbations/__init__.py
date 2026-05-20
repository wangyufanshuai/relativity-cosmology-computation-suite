"""Primordial Perturbation Theory and Power Spectrum Calculator.

Solves the Mukhanov-Sasaki equation for scalar and tensor perturbations
during inflation, computes the primordial power spectrum, spectral index,
and tensor-to-scalar ratio.
"""

from primordial_perturbations.constants import G, C, HBAR, M_PL
from primordial_perturbations.mukhanov_sasaki import (
    z_function,
    z_pp_over_z,
    ms_equation,
    bunch_davies_ic,
    integrate_mode,
)
from primordial_perturbations.power_spectrum import (
    curvature_perturbation,
    scalar_power_spectrum,
    tensor_power_spectrum,
    spectral_index,
    tensor_to_scalar_ratio,
)
from primordial_perturbations.transfer import (
    transfer_function,
    matter_power_spectrum,
)

__all__ = [
    "G", "C", "HBAR", "M_PL",
    "z_function", "z_pp_over_z", "ms_equation", "bunch_davies_ic", "integrate_mode",
    "curvature_perturbation", "scalar_power_spectrum", "tensor_power_spectrum",
    "spectral_index", "tensor_to_scalar_ratio",
    "transfer_function", "matter_power_spectrum",
]
