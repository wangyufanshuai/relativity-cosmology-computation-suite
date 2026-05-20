"""Common inflation potentials.

All potentials take phi in units of Mpl (reduced Planck mass = 2.435e18 GeV).
Each potential class provides V(phi), dV/dphi, d2V/dphi2, and a human-readable name.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "QuadraticPotential",
    "QuarticPotential",
    "StarobinskyPotential",
    "HilltopPotential",
    "NaturalPotential",
    "AlphaAttractorPotential",
]

# We work in reduced-Planck-mass units: Mpl = 1 throughout this module.
MPL = 1.0


class QuadraticPotential:
    r"""Quadratic (chaotic) potential: V = 1/2 m^2 phi^2.

    Parameters
    ----------
    m : float
        Mass parameter in Planck units.
    """

    def __init__(self, m: float) -> None:
        self.m = m
        self.name = r"Quadratic $V = \\frac{1}{2}m^2\\phi^2$"

    def V(self, phi: float | NDArray) -> float | NDArray:
        return 0.5 * self.m**2 * phi**2

    def dV(self, phi: float | NDArray) -> float | NDArray:
        return self.m**2 * phi

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        return self.m**2 * np.ones_like(phi) if isinstance(phi, np.ndarray) else self.m**2


class QuarticPotential:
    r"""Quartic potential: V = 1/4 lambda phi^4.

    Parameters
    ----------
    lambda_ : float
        Self-coupling.
    """

    def __init__(self, lambda_: float) -> None:
        self.lambda_ = lambda_
        self.name = r"Quartic $V = \\frac{1}{4}\\lambda\\phi^4$"

    def V(self, phi: float | NDArray) -> float | NDArray:
        return 0.25 * self.lambda_ * phi**4

    def dV(self, phi: float | NDArray) -> float | NDArray:
        return self.lambda_ * phi**3

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        return 3.0 * self.lambda_ * phi**2


class StarobinskyPotential:
    r"""Starobinsky (R^2) potential: V = M^4 (1 - exp(-sqrt(2/3) phi))^2.

    Parameters
    ----------
    M : float
        Mass scale parameter.
    """

    def __init__(self, M: float) -> None:
        self.M = M
        self._sq23 = np.sqrt(2.0 / 3.0)
        self.name = r"Starobinsky $V = M^4(1 - e^{-\\sqrt{2/3}\\phi})^2$"

    def V(self, phi: float | NDArray) -> float | NDArray:
        x = np.exp(-self._sq23 * phi)
        return self.M**4 * (1.0 - x) ** 2

    def dV(self, phi: float | NDArray) -> float | NDArray:
        x = np.exp(-self._sq23 * phi)
        return 2.0 * self.M**4 * self._sq23 * (1.0 - x) * x

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        x = np.exp(-self._sq23 * phi)
        return 2.0 * self.M**4 * self._sq23**2 * x * (2.0 * x - 1.0)


class HilltopPotential:
    r"""Hilltop potential: V = V0 (1 - (phi/phi_c)^n).

    Parameters
    ----------
    V0 : float
        Energy scale.
    n : int
        Power-law index (must be even for symmetry, but odd is also accepted).
    phi_c : float
        Scale at which the potential drops significantly.
    """

    def __init__(self, V0: float, n: int, phi_c: float) -> None:
        self.V0 = V0
        self.n = n
        self.phi_c = phi_c
        self.name = rf"Hilltop $V = V_0(1 - (\phi/\phi_c)^{{{n}}})$"

    def V(self, phi: float | NDArray) -> float | NDArray:
        return self.V0 * (1.0 - (phi / self.phi_c) ** self.n)

    def dV(self, phi: float | NDArray) -> float | NDArray:
        return -self.V0 * self.n * phi ** (self.n - 1) / self.phi_c**self.n

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        return -self.V0 * self.n * (self.n - 1) * phi ** (self.n - 2) / self.phi_c**self.n


class NaturalPotential:
    r"""Natural inflation potential: V = V0 (1 + cos(phi/f)).

    Parameters
    ----------
    V0 : float
        Overall energy scale.
    f : float
        Axion decay constant in Planck units.
    """

    def __init__(self, V0: float, f: float) -> None:
        self.V0 = V0
        self.f = f
        self.name = r"Natural $V = V_0(1 + \\cos(\\phi/f))$"

    def V(self, phi: float | NDArray) -> float | NDArray:
        return self.V0 * (1.0 + np.cos(phi / self.f))

    def dV(self, phi: float | NDArray) -> float | NDArray:
        return -self.V0 * np.sin(phi / self.f) / self.f

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        return -self.V0 * np.cos(phi / self.f) / self.f**2


class AlphaAttractorPotential:
    r"""Alpha-attractor (T-model) potential: V = V0 tanh^2(phi / sqrt(6 alpha)).

    Parameters
    ----------
    V0 : float
        Overall energy scale.
    alpha : float
        Alpha parameter controlling the plateau width.
    phi0 : float
        Ignored in the T-model parametrisation but kept for API consistency.
    """

    def __init__(self, V0: float, alpha: float, phi0: float = 0.0) -> None:
        self.V0 = V0
        self.alpha = alpha
        self.phi0 = phi0
        self._sq6a = np.sqrt(6.0 * alpha)
        self.name = r"$\\alpha$-attractor $V = V_0\\tanh^2(\\phi/\\sqrt{6\\alpha})$"

    def V(self, phi: float | NDArray) -> float | NDArray:
        y = phi / self._sq6a
        return self.V0 * np.tanh(y) ** 2

    def dV(self, phi: float | NDArray) -> float | NDArray:
        y = phi / self._sq6a
        th = np.tanh(y)
        return 2.0 * self.V0 * th * (1.0 - th**2) / self._sq6a

    def d2V(self, phi: float | NDArray) -> float | NDArray:
        y = phi / self._sq6a
        th = np.tanh(y)
        sech2 = 1.0 - th**2
        return 2.0 * self.V0 * (sech2**2 - 2.0 * th**2 * sech2) / self._sq6a**2
