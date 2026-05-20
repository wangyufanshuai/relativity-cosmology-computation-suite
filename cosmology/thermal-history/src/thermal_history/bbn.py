"""Big Bang Nucleosynthesis quantities."""

import numpy as np


def bbn_neutron_proton_ratio(T_MeV):
    """Neutron-to-proton ratio at temperature T.

    n/p = exp(-delta_m / T) where delta_m = 1.293 MeV.
    """
    delta_m = 1.293
    return np.exp(-delta_m / T_MeV)


def helium_mass_fraction(T_nuc_MeV=0.073):
    """Estimate the He-4 mass fraction Y_p from BBN."""
    T_weak_freeze = 0.8
    np_ratio = bbn_neutron_proton_ratio(T_weak_freeze)
    Y_p = 2.0 * np_ratio / (1.0 + np_ratio)
    return Y_p
