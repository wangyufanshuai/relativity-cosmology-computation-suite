"""Hu-Sawicki f(R) gravity model.

Implements the Hu-Sawicki form::

    f(R) = -m² c₁ (R/m²)^n / (1 + c₂ (R/m²)^n)

with its first and second derivatives, the scalaron field, and the
scalaron mass.  For *n* = 1 the expressions reduce to the simpler
closed form::

    f(R) = -m² c₁ R / (m² + c₂ R)

References
----------
Hu & Sawicki, Phys. Rev. D 76, 064004 (2007).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class HuSawickiModel:
    """Hu-Sawicki f(R) gravity model.

    Parameters
    ----------
    m2 : float
        Characteristic curvature scale squared (same units as *R*).
    c1 : float
        Dimensionless coupling constant.
    c2 : float
        Dimensionless coupling constant.
    n : int
        Power-law index (positive integer, common choices are 1 or 2).
    """

    def __init__(self, m2: float, c1: float, c2: float, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be a positive integer")
        self.m2 = float(m2)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.n = int(n)

    # ------------------------------------------------------------------
    # Core f(R) and derivatives
    # ------------------------------------------------------------------

    def f(self, R: float | NDArray) -> float | NDArray:
        """Evaluate f(R).

        Parameters
        ----------
        R : float or array_like
            Ricci scalar curvature.

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        x = R / self.m2                         # R/m²
        xn = np.power(x, self.n)                # (R/m²)^n
        return -self.m2 * self.c1 * xn / (1.0 + self.c2 * xn)

    def f_R(self, R: float | NDArray) -> float | NDArray:
        """First derivative df/dR.

        For general *n*::

            df/dR = -c1 n (R/m²)^{n-1} / (1 + c2 (R/m²)^n)²

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        x = R / self.m2
        xn = np.power(x, self.n)
        denom = (1.0 + self.c2 * xn) ** 2
        return -self.c1 * self.n * np.power(x, self.n - 1) / denom

    def f_RR(self, R: float | NDArray) -> float | NDArray:
        """Second derivative d²f/dR².

        For general *n*::

            d²f/dR² = -c1 n / m²  ×
                [(n-1) x^{n-2} (1+c2 x^n)² - 2 n c2 x^{2n-2} (1+c2 x^n)]
                / (1+c2 x^n)^4

        Simplified::

            = -c1 n x^{n-2} / [m² (1+c2 x^n)^3]
              × [(n-1)(1+c2 x^n) - 2 n c2 x^n]

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        x = R / self.m2
        xn = np.power(x, self.n)
        denom = (1.0 + self.c2 * xn) ** 3
        bracket = (self.n - 1) * (1.0 + self.c2 * xn) - 2.0 * self.n * self.c2 * xn
        return -self.c1 * self.n * np.power(x, self.n - 2) * bracket / (self.m2 * denom)

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def scalaron(self, R: float | NDArray) -> float | NDArray:
        """Scalaron field f_R = df/dR.

        This is the extra scalar degree of freedom in f(R) gravity.

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        return self.f_R(R)

    def scalaron_mass(self, R: float | NDArray) -> float | NDArray:
        """Scalaron (Compton) mass m_φ.

        Defined via m_φ² = 1 / (3 f_{RR}).

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
            m_φ (positive real-valued).
        """
        frr = self.f_RR(R)
        m2_phi = 1.0 / (3.0 * frr)
        return np.sqrt(np.abs(m2_phi))

    def scalaron_mass_squared(self, R: float | NDArray) -> float | NDArray:
        """Scalaron mass squared m_φ² = 1 / (3 f_{RR}).

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        frr = self.f_RR(R)
        return 1.0 / (3.0 * frr)

    # ------------------------------------------------------------------
    # ΛCDM limit
    # ------------------------------------------------------------------

    def lcdm_limit_check(self, R: float | NDArray, atol: float = 1e-8) -> bool:
        """Check that f(R) → 0 when f_R → 0 (ΛCDM limit).

        When the coupling constants c₁, c₂ are tuned so that f_R is
        negligible, the theory reduces to ΛCDM with an effective
        cosmological constant.

        Parameters
        ----------
        R : float or array_like
        atol : float
            Absolute tolerance for both f_R and f(R) ≈ 0.

        Returns
        -------
        bool
            True if both f_R and f(R) are below *atol*.
        """
        return bool(np.all(np.abs(self.f_R(R)) < atol) and np.all(np.abs(self.f(R)) < atol))

    def effective_cosmological_constant(self, R: float | NDArray) -> float | NDArray:
        """Effective cosmological constant in the high-curvature limit.

        For R >> m²::

            Λ_eff ≈ m² c₁ / (2 c₂)

        Parameters
        ----------
        R : float or array_like
            Background Ricci scalar (only used for display; the result
            is independent of *R* in the high-curvature regime).

        Returns
        -------
        float or ndarray
        """
        return self.m2 * self.c1 / (2.0 * self.c2)
