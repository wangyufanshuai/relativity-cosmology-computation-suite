"""Starobinsky R² inflation model.

Implements the model f(R) = R + α R² with slow-roll inflation predictions:

* n_s ≈ 1 − 2/N
* r   ≈ 12/N²

for *N* e-folds of inflation (typically N ~ 50–60).

References
----------
Starobinsky, Phys. Lett. B 91, 99 (1980).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class StarobinskyModel:
    """Starobinsky R² inflation model.

    Parameters
    ----------
    alpha : float
        Coupling constant in f(R) = R + α R².  Positive α drives
        inflation at high curvature.
    """

    def __init__(self, alpha: float) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive for inflation")
        self.alpha = float(alpha)

    # ------------------------------------------------------------------
    # Core f(R) and derivatives
    # ------------------------------------------------------------------

    def f(self, R: float | NDArray) -> float | NDArray:
        """Evaluate f(R) = R + α R².

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        return R + self.alpha * R ** 2

    def f_R(self, R: float | NDArray) -> float | NDArray:
        """First derivative df/dR = 1 + 2αR.

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        return 1.0 + 2.0 * self.alpha * R

    def f_RR(self, R: float | NDArray) -> float | NDArray:
        """Second derivative d²f/dR² = 2α.

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        return np.full_like(R, 2.0 * self.alpha)

    # ------------------------------------------------------------------
    # Inflationary observables
    # ------------------------------------------------------------------

    @staticmethod
    def spectral_index(N: float | NDArray) -> float | NDArray:
        """Scalar spectral index n_s ≈ 1 − 2/N.

        Parameters
        ----------
        N : float or array_like
            Number of e-folds (typically 50–60).

        Returns
        -------
        float or ndarray
        """
        N = np.asarray(N, dtype=float)
        return 1.0 - 2.0 / N

    @staticmethod
    def tensor_to_scalar_ratio(N: float | NDArray) -> float | NDArray:
        """Tensor-to-scalar ratio r ≈ 12/N².

        Parameters
        ----------
        N : float or array_like
            Number of e-folds (typically 50–60).

        Returns
        -------
        float or ndarray
        """
        N = np.asarray(N, dtype=float)
        return 12.0 / N ** 2

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def scalaron_mass_squared(self, R: float | NDArray) -> float | NDArray:
        """Scalaron mass squared m_φ² = 1/(3 f_{RR}) = 1/(6α).

        Parameters
        ----------
        R : float or array_like
            Ricci scalar (unused for this model since f_{RR} is constant).

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        return np.full_like(R, 1.0 / (6.0 * self.alpha))

    def scalaron_mass(self, R: float | NDArray) -> float | NDArray:
        """Scalaron mass m_φ = 1/√(6α).

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        return np.sqrt(self.scalaron_mass_squared(R))

    def slow_roll_epsilon(self, R: float | NDArray) -> float | NDArray:
        """First slow-roll parameter ε = (1/2)(f_R − Rf_{RR})²/f_{RR}²
        evaluated in the Einstein-frame potential approximation.

        For Starobinsky at large R: ε ≈ 3/(4αR²).

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        return 3.0 / (4.0 * self.alpha * R ** 2)

    def slow_roll_eta(self, R: float | NDArray) -> float | NDArray:
        """Second slow-roll parameter |η| ≈ 1/(αR).

        Parameters
        ----------
        R : float or array_like

        Returns
        -------
        float or ndarray
        """
        R = np.asarray(R, dtype=float)
        return 1.0 / (self.alpha * R)

    def e_folds(self, R_end: float, R_start: float) -> float:
        """Approximate number of e-folds between two curvature values.

        N ≈ (3/4) α (R_start² − R_end²) / (R_start R_end) for large R.

        A simpler estimate using the Starobinsky potential is::

            N ≈ (3/4) α R_start²

        Parameters
        ----------
        R_end : float
            Ricci scalar at end of inflation.
        R_start : float
            Ricci scalar at start of inflation (CMB pivot scale).

        Returns
        -------
        float
        """
        return 0.75 * self.alpha * R_start ** 2
