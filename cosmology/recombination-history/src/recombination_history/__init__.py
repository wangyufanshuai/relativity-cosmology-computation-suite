"""Cosmic Recombination History Precision Calculator.

Computes the recombination history of the universe using the Saha equation
and Peebles Three-Level Atom (TLA) model, along with derived observables
such as Thomson scattering opacity, the visibility function, and the sound
horizon.
"""

from .constants import *
from .background import hubble, temperature, baryon_density
from .saha import saha_xe, saha_helium
from .peebles import (
    alpha_B,
    beta_21,
    peebles_C,
    tla_rhs,
    solve_recombination,
)
from .observables import (
    thomson_opacity,
    thomson_opacity_full,
    visibility_function,
    last_scattering_z,
    sound_horizon,
)
