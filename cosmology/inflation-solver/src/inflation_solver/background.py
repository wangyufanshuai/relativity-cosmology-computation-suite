"""Exact background evolution (beyond slow-roll approximation).

Integrates the full Klein-Gordon equation using e-folding N as the
independent time variable so that the Hubble parameter cancels nicely.

System of ODEs (N = ln a as time variable):
    dphi/dN    = phi_dot / H
    d(phi_dot)/dN = -3 phi_dot - V'(phi) / H

where H^2 = (1/3 Mpl^2)(1/2 phi_dot^2 + V(phi))  with Mpl = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy import integrate

__all__ = ["InflationBackground"]

MPL = 1.0


class InflationBackground:
    """Exact inflationary background evolution.

    Parameters
    ----------
    potential : potential object
        Must provide V(phi), dV(phi), d2V(phi).
    phi0 : float
        Initial field value (in Mpl units).
    dphidt0 : float, optional
        Initial field velocity dphi/dt. Default 0 (slow-roll start).
    N_max : float, optional
        Maximum number of e-folds to integrate. Default 100.
    """

    def __init__(
        self,
        potential,
        phi0: float,
        dphidt0: float = 0.0,
        N_max: float = 100.0,
    ) -> None:
        self.potential = potential
        self.phi0 = phi0
        self.dphidt0 = dphidt0
        self.N_max = N_max

        self._N: np.ndarray | None = None
        self._phi: np.ndarray | None = None
        self._dphidt: np.ndarray | None = None
        self._H: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hubble(self, phi: float, dphidt: float) -> float:
        """Hubble parameter from the Friedmann equation."""
        rho = 0.5 * dphidt**2 + self.potential.V(phi)
        return np.sqrt(max(rho, 0.0) / 3.0)

    def _rhs(self, N: float, y: List[float]) -> List[float]:
        """Right-hand side of the coupled ODE system.

        y = [phi, dphidt]
        dphi/dN = dphidt / H
        d(dphidt)/dN = -3 dphidt - V'(phi) / H
        """
        phi, dphidt = y
        H = self._hubble(phi, dphidt)
        if H == 0.0:
            return [0.0, 0.0]
        dphi_dN = dphidt / H
        dphidt_dN = -3.0 * dphidt - self.potential.dV(phi) / H
        return [dphi_dN, dphidt_dN]

    def _end_event(self, N: float, y: List[float]) -> float:
        """Event function: epsilon = 1 => end of inflation."""
        phi, dphidt = y
        H = self._hubble(phi, dphidt)
        if H == 0.0:
            return -1.0
        # epsilon = (1/2)(dphi/dN)^2 / Mpl^2, with Mpl=1
        eps = 0.5 * (dphidt / H) ** 2
        return eps - 1.0

    _end_event.terminal = True  # type: ignore[attr-defined]
    _end_event.direction = 1  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve(self) -> None:
        """Integrate the Klein-Gordon equation from N=0 to N_max or end of inflation."""
        y0 = [self.phi0, self.dphidt0]

        # Dense output via t_eval for smooth interpolation
        N_eval = np.linspace(0.0, self.N_max, 5000)

        sol = integrate.solve_ivp(
            self._rhs,
            (0.0, self.N_max),
            y0,
            method="RK45",
            t_eval=N_eval,
            events=self._end_event,
            rtol=1e-10,
            atol=1e-12,
            max_step=0.1,
        )

        self._N = sol.t
        self._phi = sol.y[0]
        self._dphidt = sol.y[1]
        self._H = np.array([self._hubble(p, d) for p, d in zip(self._phi, self._dphidt)])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def _ensure_evolved(self) -> None:
        if self._N is None:
            raise RuntimeError("Call evolve() first.")

    @property
    def N(self) -> np.ndarray:
        """E-fold array."""
        self._ensure_evolved()
        return self._N

    @property
    def phi(self) -> np.ndarray:
        """Field value phi(N)."""
        self._ensure_evolved()
        return self._phi

    @property
    def phi_dot(self) -> np.ndarray:
        """Field velocity dphi/dt(N)."""
        self._ensure_evolved()
        return self._dphidt

    @property
    def H(self) -> np.ndarray:
        """Hubble parameter H(N)."""
        self._ensure_evolved()
        return self._H

    def epsilon(self) -> np.ndarray:
        """First Hubble slow-roll parameter epsilon(N) = (1/2)(dphi/dN)^2."""
        self._ensure_evolved()
        return 0.5 * (self._dphidt / self._H) ** 2

    def eta(self) -> np.ndarray:
        """Second Hubble slow-roll parameter eta(N) = epsilon - d ln epsilon / dN / 2."""
        self._ensure_evolved()
        eps = self.epsilon()
        # eta_H = eps - d(ln eps)/dN / 2   but the standard definition is:
        # eta_H = d(dphi/dN)/dN / (dphi/dN)   computed numerically
        dphi_dN = self._dphidt / self._H
        # Numerical derivative of dphi_dN w.r.t. N
        d2phi_dN2 = np.gradient(dphi_dN, self._N)
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            eta_H = np.where(np.abs(dphi_dN) > 1e-30, d2phi_dN2 / dphi_dN, 0.0)
        return eta_H

    def n_s(self) -> np.ndarray:
        """Scalar spectral index n_s(N) computed from exact background."""
        self._ensure_evolved()
        eps = self.epsilon()
        eta_H = self.eta()
        return 1.0 - 6.0 * eps + 2.0 * eta_H

    def r(self) -> np.ndarray:
        """Tensor-to-scalar ratio r(N) = 16 epsilon."""
        return 16.0 * self.epsilon()
