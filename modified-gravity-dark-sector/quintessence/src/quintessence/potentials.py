"""Quintessence potentials.

Each potential class provides V(phi), dV(phi), and d2V(phi) methods.
All potentials are expressed in natural units with the field phi in units
of the reduced Planck mass M_Pl.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class QuadraticPotential:
    """Quadratic (mass-like) potential: V(phi) = (1/2) * m^2 * phi^2.

    This is the simplest thawing quintessence model. The field starts
    at rest and slowly rolls as H drops below m.

    Parameters
    ----------
    m2 : float
        Mass squared parameter in GeV^4 / M_Pl^2.
    """

    def __init__(self, m2: float) -> None:
        self.m2 = m2

    def V(self, phi: float | NDArray) -> float | NDArray:
        """Potential energy V(phi)."""
        return 0.5 * self.m2 * phi**2

    def dV(self, phi: float | NDArray) -> float | NDArray:
        """First derivative dV/d(phi)."""
        return self.m2 * phi

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        """Second derivative d^2V/d(phi)^2."""
        return np.full_like(np.asarray(phi, dtype=float), self.m2)


class InversePowerPotential:
    """Inverse power-law (Ratra-Peebles tracker) potential:
    V(phi) = M^(4+n) * phi^(-n).

    This potential admits tracker solutions where the field follows
    an attractor trajectory, making the dark energy density insensitive
    to initial conditions.

    Parameters
    ----------
    M : float
        Energy scale parameter in GeV.
    n : float
        Power-law index (n > 0 for tracker behavior).
    """

    def __init__(self, M: float, n: float) -> None:
        if n <= 0:
            raise ValueError("Power-law index n must be positive for tracker behavior.")
        self.M = M
        self.n = n

    def V(self, phi: float | NDArray) -> float | NDArray:
        """Potential energy V(phi)."""
        phi_arr = np.asarray(phi, dtype=float)
        return self.M ** (4 + self.n) * phi_arr ** (-self.n)

    def dV(self, phi: float | NDArray) -> float | NDArray:
        """First derivative dV/d(phi)."""
        phi_arr = np.asarray(phi, dtype=float)
        return -self.n * self.M ** (4 + self.n) * phi_arr ** (-(self.n + 1))

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        """Second derivative d^2V/d(phi)^2."""
        phi_arr = np.asarray(phi, dtype=float)
        return self.n * (self.n + 1) * self.M ** (4 + self.n) * phi_arr ** (-(self.n + 2))


class SUGRAPotential:
    """Supergravity (SUGRA) thawing potential:
    V(phi) = V0 * (1 + cosh(phi / M)).

    A thawing quintessence model motivated by supergravity. The field
    is frozen at early times and thaws as the universe approaches
    dark-energy domination.

    Parameters
    ----------
    V0 : float
        Energy scale in GeV^4.
    M : float
        Characteristic field scale in units of M_Pl.
    """

    def __init__(self, V0: float, M: float) -> None:
        self.V0 = V0
        self.M = M

    def V(self, phi: float | NDArray) -> float | NDArray:
        """Potential energy V(phi)."""
        return self.V0 * (1.0 + np.cosh(phi / self.M))

    def dV(self, phi: float | NDArray) -> float | NDArray:
        """First derivative dV/d(phi)."""
        return self.V0 * np.sinh(phi / self.M) / self.M

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        """Second derivative d^2V/d(phi)^2."""
        return self.V0 * np.cosh(phi / self.M) / (self.M**2)


class ExponentialPotential:
    """Exponential potential: V(phi) = V0 * exp(-lambda * phi / M_Pl).

    Power-law inflation / freezing quintessence. For large lambda the
    field rolls fast and w approaches a constant (tracker).

    Parameters
    ----------
    V0 : float
        Energy scale in GeV^4.
    lam : float
        Dimensionless slope parameter.
    M_Pl : float
        Reduced Planck mass (defaults to constants.M_PL).
    """

    def __init__(self, V0: float, lam: float, M_Pl: float = 2.435e18) -> None:
        self.V0 = V0
        self.lam = lam
        self.M_Pl = M_Pl

    def V(self, phi: float | NDArray) -> float | NDArray:
        """Potential energy V(phi)."""
        return self.V0 * np.exp(-self.lam * phi / self.M_Pl)

    def dV(self, phi: float | NDArray) -> float | NDArray:
        """First derivative dV/d(phi)."""
        return -self.lam / self.M_Pl * self.V0 * np.exp(-self.lam * phi / self.M_Pl)

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        """Second derivative d^2V/d(phi)^2."""
        return (self.lam / self.M_Pl) ** 2 * self.V0 * np.exp(-self.lam * phi / self.M_Pl)
