"""Matter power spectrum calculator.

Provides Eisenstein & Hu (1998) transfer functions, linear growth factor,
primordial and linear power spectra, and sigma_8 computation.
"""

from .constants import (
    A_S_DEFAULT,
    H0_DEFAULT,
    K_PIVOT_DEFAULT,
    N_S_DEFAULT,
    OMEGA_B_DEFAULT,
    OMEGA_M_DEFAULT,
    T_CMB_DEFAULT,
)
from .growth import growth_factor, growth_factor_normalized, growth_rate
from .spectrum import linear_power_spectrum, primordial_power, sigma_8
from .transfer import (
    k_eq_EH98,
    sound_horizon_EH98,
    transfer_EH98,
    transfer_EH98_wiggle,
)

__all__ = [
    "transfer_EH98",
    "transfer_EH98_wiggle",
    "sound_horizon_EH98",
    "k_eq_EH98",
    "growth_factor",
    "growth_factor_normalized",
    "growth_rate",
    "primordial_power",
    "linear_power_spectrum",
    "sigma_8",
    "H0_DEFAULT",
    "OMEGA_M_DEFAULT",
    "OMEGA_B_DEFAULT",
    "T_CMB_DEFAULT",
    "A_S_DEFAULT",
    "N_S_DEFAULT",
    "K_PIVOT_DEFAULT",
]
