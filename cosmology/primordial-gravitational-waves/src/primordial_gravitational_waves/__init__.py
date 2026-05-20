"""Primordial gravitational wave (stochastic GW background) calculator.

Modules
-------
tensor_perturbation
    Tensor Mukhanov-Sasaki equation solver, tensor power spectrum,
    tensor-to-scalar ratio, and single-mode evolution.
energy_density
    Energy density Omega_GW(f) for inflationary backgrounds,
    transfer functions, and first-order phase transition spectra.
constraints
    Experimental upper limits from Planck, LIGO, LISA, pulsar timing
    arrays, and BBN.
"""

__version__ = "0.1.0"

from .tensor_perturbation import (
    tensor_mukhanov_sasaki,
    tensor_power_spectrum,
    tensor_to_scalar_ratio,
    evolve_tensor_mode,
)
from .energy_density import (
    omega_gw,
    transfer_function,
    frequency_range,
    omega_gw_from_power_spectrum,
    first_order_phase_transition,
)
from .constraints import (
    planck_constraint_r,
    cmb_b_mode_constraint,
    ligo_constraint,
    lisa_constraint,
    bbn_constraint,
    pulsar_timing_constraint,
    combined_constraints,
)

__all__ = [
    "tensor_mukhanov_sasaki",
    "tensor_power_spectrum",
    "tensor_to_scalar_ratio",
    "evolve_tensor_mode",
    "omega_gw",
    "transfer_function",
    "frequency_range",
    "omega_gw_from_power_spectrum",
    "first_order_phase_transition",
    "planck_constraint_r",
    "cmb_b_mode_constraint",
    "ligo_constraint",
    "lisa_constraint",
    "bbn_constraint",
    "pulsar_timing_constraint",
    "combined_constraints",
]
