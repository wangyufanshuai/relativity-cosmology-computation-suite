"""Dataclasses for stellar and black-hole parameters."""

from dataclasses import dataclass


@dataclass
class StellarParams:
    """Parameters describing the disrupted star."""

    mass: float  # stellar mass [g]
    radius: float  # stellar radius [cm]
    gamma: float = 5.0 / 3.0  # adiabatic index (polytrope)


@dataclass
class BlackHoleParams:
    """Parameters describing the supermassive black hole."""

    mass: float  # black hole mass [g]
    spin: float = 0.0  # dimensionless spin parameter a in [0, 1]
