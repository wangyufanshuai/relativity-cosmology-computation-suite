"""Pulsar Magnetosphere Simulator
===============================

A force-free electrodynamics simulator for rotating neutron star magnetospheres.

Modules
-------
dipole : Rotating magnetic dipole field and co-rotation electric field.
force_free : Force-free electrodynamics equations and constraints.
"""

from .dipole import (
    RotatingDipole,
    light_cylinder_radius,
    spindown_luminosity,
)
from .force_free import ForceFreeSolver

__all__ = [
    "RotatingDipole",
    "light_cylinder_radius",
    "spindown_luminosity",
    "ForceFreeSolver",
]
