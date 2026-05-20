"""PPN metric tensor and geodesic equations.

Implements the standard PPN metric (Will & Nordtvedt 1972):

    g00 = -1 + 2U - 2βU² - 2ξΦ_W + (2γ + 2 + α₃ + ζ₁ - 2ξ)Φ₁
          + (2β - 1 - ζ₂ - ξ)Φ₂ + (ζ₁ + 2ξ)A - (2γ + 2α₃ - 2ξ)Φ₃
          + (α₁ - α₂ - α₃)w²U - (2α₂ - 2α₁)wⁱVᵢ + 2α₂wⁱwʲUᵢⱼ
          + (4β - 2γ - 2 - ζ₂ - ξ)Φ₃ + ...
    g0i = -(4γ + 3 + α₁ - α₂ + ζ₁ - 2ξ)/2 · Vᵢ
          - (1 + α₂ - 2ξ)/2 · Wᵢ - (α₁ - 2α₂)/2 · wⁱU - α₂wʲUᵢⱼ
    gij = (1 + 2γU) δᵢⱼ

For most applications only γ and β matter (all others are zero in GR).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .constants import G, C


class PPNParameters:
    """PPN parameters with current experimental constraints."""

    def __init__(
        self,
        gamma: float = 1.0,   # Eddington: space curvature per unit mass
        beta: float = 1.0,    # Eddington: nonlinearity in superposition
        xi: float = 0.0,      # Whitehead potential preference
        alpha1: float = 0.0,  # Preferred-frame effect
        alpha2: float = 0.0,  # Preferred-frame effect
        alpha3: float = 0.0,  # Preferred-frame effect
        zeta1: float = 0.0,   # Violation of conservation of momentum
        zeta2: float = 0.0,
        zeta3: float = 0.0,
        zeta4: float = 0.0,
    ):
        self.gamma = gamma
        self.beta = beta
        self.xi = xi
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.zeta1 = zeta1
        self.zeta2 = zeta2
        self.zeta3 = zeta3
        self.zeta4 = zeta4

    def is_gr(self) -> bool:
        """Check if parameters correspond to General Relativity."""
        return (
            self.gamma == 1.0
            and self.beta == 1.0
            and self.xi == 0.0
            and self.alpha1 == 0.0
            and self.alpha2 == 0.0
            and self.alpha3 == 0.0
            and self.zeta1 == 0.0
            and self.zeta2 == 0.0
            and self.zeta3 == 0.0
            and self.zeta4 == 0.0
        )

    @classmethod
    def brans_dicke(cls, omega: float) -> "PPNParameters":
        """Brans-Dicke theory: γ = (ω+1)/(ω+2), β = 1."""
        gamma_bd = (omega + 1.0) / (omega + 2.0)
        return cls(gamma=gamma_bd, beta=1.0)

    def __repr__(self) -> str:
        return (
            f"PPNParameters(γ={self.gamma}, β={self.beta}, "
            f"ξ={self.xi}, α₁={self.alpha1}, α₂={self.alpha2}, α₃={self.alpha3})"
        )


def newtonian_potential(
    positions: ArrayLike,
    masses: ArrayLike,
    eval_point: ArrayLike,
) -> float:
    """Newtonian gravitational potential U = Σ GMᵢ/|x - xᵢ|."""
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    eval_point = np.asarray(eval_point, dtype=float)

    U = 0.0
    for i in range(len(masses)):
        r = np.linalg.norm(eval_point - positions[i])
        if r > 0:
            U += G * masses[i] / r
    return U


def ppn_metric_g00(
    U: float,
    v_squared: float = 0.0,
    ppn: PPNParameters | None = None,
) -> float:
    """PPN metric component g00 to leading PN order.

    g00 = -1 + 2U/c² - 2β(U/c²)² + O(c⁻⁴)
    """
    if ppn is None:
        ppn = PPNParameters()
    u = U / C**2
    return -1.0 + 2.0 * u - 2.0 * ppn.beta * u**2


def ppn_metric_gij(
    U: float,
    ppn: PPNParameters | None = None,
) -> float:
    """PPN metric spatial component gij = δij (1 + 2γU/c²)."""
    if ppn is None:
        ppn = PPNParameters()
    return 1.0 + 2.0 * ppn.gamma * U / C**2
