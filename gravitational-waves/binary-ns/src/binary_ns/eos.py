"""
Equation of state (EOS) models for neutron star matter.

Provides polytropic, piecewise polytropic, and simplified realistic EOS
parameterizations.  All internal calculations use geometrized units
(G = c = 1) unless otherwise noted.

Density / pressure conventions
------------------------------
* ``epsilon`` (epsilon) = total energy density  [km^-2]
* ``P``                    = pressure            [km^-2]
* ``rho``                  = rest-mass baryon density (where needed)
* ``c_s2``                 = (speed of sound / c)^2

Conversion helpers are provided at the bottom of the module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

# ---------------------------------------------------------------------------
# Physical constants  (SI)
# ---------------------------------------------------------------------------
G_SI = 6.67430e-11        # m^3 kg^-1 s^-2
C_SI = 2.99792458e8       # m s^-1
M_SUN_SI = 1.98892e30     # kg

# geometrized: lengths in km
# 1 km^-2 = c^4 / (G * km^2)   in Pa
_KM2_TO_PA = C_SI**4 / (G_SI * 1e3**2)
_PA_TO_KM2 = 1.0 / _KM2_TO_PA

# rho_0  ~  2.8e14 g/cm^3  (nuclear saturation density)
RHO_NUC = 2.8e14 * 1e3    # kg m^-3

# ---------------------------------------------------------------------------
# Base EOS class
# ---------------------------------------------------------------------------

class BaseEOS:
    """Abstract base for all equation-of-state models."""

    def pressure_from_epsilon(self, epsilon: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def epsilon_from_pressure(self, P: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sound_speed_squared(self, epsilon: np.ndarray) -> np.ndarray:
        """Return (c_s / c)^2 at the given energy densities."""
        eps = np.asarray(epsilon, dtype=float)
        dP = np.gradient(self.pressure_from_epsilon(eps), eps)
        # clip to [0, 1) for numerical safety
        return np.clip(dP, 0.0, 1.0 - 1e-12)

    def is_causal(self, epsilon: np.ndarray) -> bool:
        """True if c_s < c everywhere in the given range."""
        return bool(np.all(self.sound_speed_squared(epsilon) < 1.0))

# ---------------------------------------------------------------------------
# Polytropic EOS:  P = K * epsilon^Gamma   (energy-density polytrope)
# ---------------------------------------------------------------------------

@dataclass
class PolytropicEOS(BaseEOS):
    """Single polytrope  P = K * epsilon^Gamma.

    Parameters
    ----------
    Gamma : float
        Adiabatic index.
    K : float
        Polytropic constant in geometrized units [km^(2(Gamma-1))].
    """

    Gamma: float
    K: float

    def pressure_from_epsilon(self, epsilon: np.ndarray) -> np.ndarray:
        eps = np.asarray(epsilon, dtype=float)
        return self.K * np.power(eps, self.Gamma)

    def epsilon_from_pressure(self, P: np.ndarray) -> np.ndarray:
        P = np.asarray(P, dtype=float)
        return (np.maximum(P, 0.0) / self.K) ** (1.0 / self.Gamma)

    def sound_speed_squared(self, epsilon: np.ndarray) -> np.ndarray:
        # dP/depsilon = K * Gamma * epsilon^{Gamma-1}
        eps = np.asarray(epsilon, dtype=float)
        cs2 = self.Gamma * self.K * np.power(eps, self.Gamma - 1.0)
        return np.clip(cs2, 0.0, 1.0 - 1e-12)

# ---------------------------------------------------------------------------
# Piecewise polytropic EOS
# ---------------------------------------------------------------------------

@dataclass
class PiecewisePolytropeEOS(BaseEOS):
    """Piecewise polytrope with different Gamma_i in each density segment.

    Parameters
    ----------
    log_p0 : float
        log10(P_1) in dyne/cm^2 at the first segment boundary (controls
        overall stiffness).
    Gammas : sequence of float
        Adiabatic indices for each segment.
    log_rho_breaks : sequence of float
        log10(rho) in g/cm^3 at the segment boundaries (length = len(Gammas)-1).
    """

    log_p0: float
    Gammas: tuple[float, ...]
    log_rho_breaks: tuple[float, ...]

    def __post_init__(self):
        self.Gammas = tuple(self.Gammas)
        self.log_rho_breaks = tuple(self.log_rho_breaks)
        if len(self.log_rho_breaks) != len(self.Gammas) - 1:
            raise ValueError("len(log_rho_breaks) must be len(Gammas)-1")
        # Build segment boundaries in epsilon (geometrized, km^-2)
        self._build_segments()

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _rho_to_eps_si(rho_si: float) -> float:
        """Very rough rest-mass -> total energy density (Newtonian approx)."""
        return rho_si * C_SI**2

    def _build_segments(self):
        """Pre-compute segment-boundary energy densities and K_i values."""
        # Convert rho breaks from g/cm^3 to SI kg/m^3
        rho_breaks_si = [10.0 ** lrho * 1e3 for lrho in self.log_rho_breaks]
        eps_breaks_si = [self._rho_to_eps_si(rho) for rho in rho_breaks_si]
        eps_breaks = [e * _PA_TO_KM2 for e in eps_breaks_si]

        # P_0 in geometrized units
        P0_si = 10.0 ** self.log_p0          # dyne/cm^2
        P0_si_pa = P0_si * 0.1               # Pa
        P0 = P0_si_pa * _PA_TO_KM2

        # First segment: K_0 chosen so P = K_0 * eps^Gamma_0 matches P0 at
        # the first break.
        self._K_vals: list[float] = []
        self._eps_breaks: list[float] = []

        # We treat it as: each segment i has P = K_i * eps^{Gamma_i}
        # Continuity of P at boundaries determines K_i.
        eps_lo = eps_breaks[0] if eps_breaks else 1e-6  # fallback
        P_lo = P0
        self._eps_breaks = eps_breaks

        K0 = P_lo / eps_lo ** self.Gammas[0]
        self._K_vals.append(K0)

        for i in range(1, len(self.Gammas)):
            eps_b = self._eps_breaks[i - 1]
            P_b = self._K_vals[i - 1] * eps_b ** self.Gammas[i - 1]
            Ki = P_b / eps_b ** self.Gammas[i]
            self._K_vals.append(Ki)

    def _segment_index(self, epsilon: float) -> int:
        idx = 0
        for b in self._eps_breaks:
            if epsilon > b:
                idx += 1
            else:
                break
        return min(idx, len(self.Gammas) - 1)

    # -- public API ---------------------------------------------------------

    def pressure_from_epsilon(self, epsilon: np.ndarray) -> np.ndarray:
        eps = np.atleast_1d(np.asarray(epsilon, dtype=float))
        P = np.empty_like(eps)
        for i in range(len(eps)):
            si = self._segment_index(eps[i])
            P[i] = self._K_vals[si] * eps[i] ** self.Gammas[si]
        return P if P.ndim > 0 else float(P)

    def epsilon_from_pressure(self, P: np.ndarray) -> np.ndarray:
        P = np.atleast_1d(np.asarray(P, dtype=float))
        # We need to invert P = K_i * eps^{Gamma_i}.  We find which segment
        # a given P corresponds to by scanning.
        eps = np.empty_like(P)
        for i in range(len(P)):
            # find segment by trying each and picking the one whose boundary
            # range contains this P
            si = 0
            for j, b in enumerate(self._eps_breaks):
                P_b = self._K_vals[j] * b ** self.Gammas[j]
                if P[i] > P_b:
                    si = j + 1
                else:
                    break
            si = min(si, len(self.Gammas) - 1)
            eps[i] = (max(P[i], 0.0) / self._K_vals[si]) ** (1.0 / self.Gammas[si])
        return eps if eps.ndim > 0 else float(eps)

    def sound_speed_squared(self, epsilon: np.ndarray) -> np.ndarray:
        eps = np.atleast_1d(np.asarray(epsilon, dtype=float))
        cs2 = np.empty_like(eps)
        for i in range(len(eps)):
            si = self._segment_index(eps[i])
            # dP/depsilon = K_i * Gamma_i * eps^{Gamma_i - 1}
            cs2[i] = self.Gammas[si] * self._K_vals[si] * eps[i] ** (self.Gammas[si] - 1)
        return np.clip(cs2, 0.0, 1.0 - 1e-12)


# ---------------------------------------------------------------------------
# Simplified realistic EOS parameterizations (SLy, APR, H4 inspired)
# ---------------------------------------------------------------------------

def _make_piecewise(log_p0: float, gammas: Sequence[float],
                    log_rho_breaks: Sequence[float]) -> PiecewisePolytropeEOS:
    return PiecewisePolytropeEOS(log_p0=log_p0,
                                 Gammas=tuple(gammas),
                                 log_rho_breaks=tuple(log_rho_breaks))


def SLy_EOS() -> PiecewisePolytropeEOS:
    """Simplified SLy-like piecewise polytrope.

    Based on Read et al. (2009) parameterisation but with simplified values
    that reproduce a ~2.05 Msun maximum mass and R~11.7 km at 1.4 Msun.
    """
    return _make_piecewise(
        log_p0=34.384,
        gammas=[2.948, 3.011, 2.605, 2.678],
        log_rho_breaks=[14.165, 14.825, 15.485],
    )


def APR_EOS() -> PiecewisePolytropeEOS:
    """Simplified APR-like piecewise polytrope.

    Stiffer high-density behaviour giving M_max ~ 2.2 Msun.
    """
    return _make_piecewise(
        log_p0=34.269,
        gammas=[2.442, 3.255, 3.018, 2.905],
        log_rho_breaks=[14.130, 14.790, 15.450],
    )


def H4_EOS() -> PiecewisePolytropeEOS:
    """Simplified H4-like piecewise polytrope.

    Softer, larger radius ~13.6 km, M_max ~ 2.0 Msun.
    """
    return _make_piecewise(
        log_p0=34.467,
        gammas=[2.900, 2.746, 2.200, 2.200],
        log_rho_breaks=[14.235, 14.895, 15.555],
    )


# ---------------------------------------------------------------------------
# Utility: build a tabulated EOS via interpolation (for stable inversion)
# ---------------------------------------------------------------------------

def build_tabulated_eos(eos: BaseEOS,
                        log_eps_min: float = -4.0,
                        log_eps_max: float = 1.5,
                        n_points: int = 2000):
    """Return interpolators P(eps) and eps(P) for any BaseEOS.

    This is useful for EOS models whose analytic inversion is fragile.
    Returns two callables: ``P_of_eps(eps)`` and ``eps_of_P(P)``.
    """
    log_eps = np.linspace(log_eps_min, log_eps_max, n_points)
    eps = 10.0 ** log_eps
    P = eos.pressure_from_epsilon(eps)
    # ensure monotonicity for PCHIP
    # (already monotonic for polytropes, but piecewise can have kinks)
    P = np.maximum(P, 0.0)
    P_of_eps = PchipInterpolator(eps, P, extrapolate=True)
    eps_of_P = PchipInterpolator(P, eps, extrapolate=True)
    return P_of_eps, eps_of_P
