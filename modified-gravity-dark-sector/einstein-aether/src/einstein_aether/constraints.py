"""Experimental constraints on Einstein-aether coupling constants.

Implements bounds from:
  - Solar system tests (Cassini experiment: gamma - 1 < 2.3e-5)
  - Gravitational-wave speed from GW170817
  - Preferred-frame effects (binary pulsar timing)
  - Viable parameter space identification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize

from .ppn_parameters import ppn_gamma, ppn_beta, preferred_frame_params
from .cosmology import gw_speed


# ------------------------------------------------------------------
# Data classes for constraint results
# ------------------------------------------------------------------

@dataclass
class SolarSystemBounds:
    """Bounds from solar-system tests of gravity."""
    gamma_minus_1: float          # |gamma - 1| bound
    gamma_bound: float            # Cassini bound: 2.3e-5
    beta_minus_1: float           # |beta - 1| bound (perihelion precession)
    beta_bound: float             # Typical bound ~ 1e-3
    alpha1_bound: float           # Preferred-frame alpha_1 bound
    alpha2_bound: float           # Preferred-frame alpha_2 bound
    satisfied: bool               # Whether all bounds are satisfied


@dataclass
class GWConstraint:
    """Constraint from gravitational-wave speed measurements."""
    c_T: float                    # Computed GW speed
    c_T_minus_1: float            # |c_T/c - 1|
    bound: float                  # |c_T/c - 1| < bound from GW170817
    satisfied: bool


# ------------------------------------------------------------------
# Solar system constraints
# ------------------------------------------------------------------

# Cassini bound on gamma_PPN - 1 (Bertotti et al. 2003)
CASSINI_GAMMA_BOUND = 2.3e-5

# Bounds on PPN beta from perihelion precession
BETA_BOUND = 1e-3

# Preferred-frame bounds (from binary pulsar observations)
ALPHA1_BOUND = 1e-4
ALPHA2_BOUND = 1e-7


def solar_system_constraints(
    c1: float,
    c2: float,
    c3: float,
    c4: float,
    gamma_bound: float = CASSINI_GAMMA_BOUND,
    beta_bound: float = BETA_BOUND,
    alpha1_bound: float = ALPHA1_BOUND,
    alpha2_bound: float = ALPHA2_BOUND,
) -> SolarSystemBounds:
    """Evaluate solar-system constraints for given coupling constants.

    Parameters
    ----------
    c1, c2, c3, c4 : float
        Aether coupling constants.
    gamma_bound : float
        Upper limit on |gamma_PPN - 1|.
    beta_bound : float
        Upper limit on |beta_PPN - 1|.
    alpha1_bound, alpha2_bound : float
        Bounds on preferred-frame parameters.

    Returns
    -------
    SolarSystemBounds
    """
    gamma = ppn_gamma(c1, c2, c3, c4)
    beta = ppn_beta(c1, c2, c3, c4)
    a1, a2 = preferred_frame_params(c1, c2, c3, c4)

    g_dev = abs(gamma - 1.0)
    b_dev = abs(beta - 1.0)

    satisfied = (
        g_dev < gamma_bound
        and b_dev < beta_bound
        and abs(a1) < alpha1_bound
        and abs(a2) < alpha2_bound
    )

    return SolarSystemBounds(
        gamma_minus_1=g_dev,
        gamma_bound=gamma_bound,
        beta_minus_1=b_dev,
        beta_bound=beta_bound,
        alpha1_bound=alpha1_bound,
        alpha2_bound=alpha2_bound,
        satisfied=satisfied,
    )


# ------------------------------------------------------------------
# GW speed constraint
# ------------------------------------------------------------------

# GW170817: |c_T / c - 1| < ~1e-15  (combined EM + GW observation)
GW170817_BOUND = 1e-15


def gw_speed_constraint(
    c1: float,
    c3: float,
    bound: float = GW170817_BOUND,
) -> GWConstraint:
    """Apply the gravitational-wave speed constraint from GW170817.

    Parameters
    ----------
    c1, c3 : float
        Coupling constants entering c_13 = c_1 + c_3.
    bound : float
        Upper limit on |c_T/c - 1|.

    Returns
    -------
    GWConstraint
    """
    c_T = gw_speed(c1, c3)
    deviation = abs(c_T - 1.0)
    return GWConstraint(
        c_T=c_T,
        c_T_minus_1=deviation,
        bound=bound,
        satisfied=(deviation < bound),
    )


# ------------------------------------------------------------------
# Parameter priors for Bayesian analysis
# ------------------------------------------------------------------

@dataclass
class ParameterPriors:
    """Prior ranges for the coupling constants c_i."""
    c1_range: tuple[float, float] = (-0.1, 0.1)
    c2_range: tuple[float, float] = (-0.1, 0.1)
    c3_range: tuple[float, float] = (-0.1, 0.1)
    c4_range: tuple[float, float] = (-0.1, 0.1)

    def in_range(self, c1: float, c2: float, c3: float, c4: float) -> bool:
        """Check whether the given (c1, c2, c3, c4) lie within prior ranges."""
        return (
            self.c1_range[0] <= c1 <= self.c1_range[1]
            and self.c2_range[0] <= c2 <= self.c2_range[1]
            and self.c3_range[0] <= c3 <= self.c3_range[1]
            and self.c4_range[0] <= c4 <= self.c4_range[1]
        )


def parameter_priors(
    c1_range: tuple[float, float] = (-0.1, 0.1),
    c2_range: tuple[float, float] = (-0.1, 0.1),
    c3_range: tuple[float, float] = (-0.1, 0.1),
    c4_range: tuple[float, float] = (-0.1, 0.1),
) -> ParameterPriors:
    """Create a ParameterPriors object with the given ranges.

    Parameters
    ----------
    c1_range, c2_range, c3_range, c4_range : tuple
        (lower, upper) bounds for each coupling constant.

    Returns
    -------
    ParameterPriors
    """
    return ParameterPriors(
        c1_range=c1_range,
        c2_range=c2_range,
        c3_range=c3_range,
        c4_range=c4_range,
    )


# ------------------------------------------------------------------
# Viable parameter space
# ------------------------------------------------------------------

@dataclass
class ViableRegion:
    """Description of a viable parameter-space region."""
    c1: float
    c2: float
    c3: float
    c4: float
    gamma: float
    beta: float
    alpha1: float
    alpha2: float
    c_T: float
    all_satisfied: bool


def viable_parameter_space(
    c1_range: tuple[float, float] = (-0.01, 0.01),
    c2_range: tuple[float, float] = (-0.01, 0.01),
    c3_range: tuple[float, float] = (-0.01, 0.01),
    c4_range: tuple[float, float] = (-0.01, 0.01),
    n_samples: int = 1000,
    gamma_bound: float = CASSINI_GAMMA_BOUND,
    beta_bound: float = BETA_BOUND,
    gw_bound: float = GW170817_BOUND,
    seed: Optional[int] = 42,
) -> list[ViableRegion]:
    """Monte-Carlo scan of the parameter space to identify viable regions.

    Parameters
    ----------
    c1_range, c2_range, c3_range, c4_range : tuple
        Ranges to scan.
    n_samples : int
        Number of random samples.
    gamma_bound, beta_bound : float
        PPN bounds.
    gw_bound : float
        GW speed bound.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list of ViableRegion
        Parameter combinations satisfying all constraints.
    """
    rng = np.random.default_rng(seed)

    viable: list[ViableRegion] = []

    c1_samples = rng.uniform(*c1_range, size=n_samples)
    c2_samples = rng.uniform(*c2_range, size=n_samples)
    c3_samples = rng.uniform(*c3_range, size=n_samples)
    c4_samples = rng.uniform(*c4_range, size=n_samples)

    for i in range(n_samples):
        c1 = float(c1_samples[i])
        c2 = float(c2_samples[i])
        c3 = float(c3_samples[i])
        c4 = float(c4_samples[i])

        # Skip singular configurations
        c13 = c1 + c3
        c14 = c1 + c4
        if abs(1.0 - c13) < 1e-10:
            continue
        D = 2.0 - c14 - c13 * (2.0 + c2) / (1.0 - c13)
        if abs(D) < 1e-10:
            continue

        try:
            gamma = ppn_gamma(c1, c2, c3, c4)
            beta = ppn_beta(c1, c2, c3, c4)
            a1, a2 = preferred_frame_params(c1, c2, c3, c4)
            c_T = gw_speed(c1, c3)
        except (ZeroDivisionError, FloatingPointError):
            continue

        if np.any(np.isnan([gamma, beta, a1, a2, c_T])):
            continue
        if np.any(np.isinf([gamma, beta, a1, a2, c_T])):
            continue

        ss_ok = (
            abs(gamma - 1.0) < gamma_bound
            and abs(beta - 1.0) < beta_bound
        )
        gw_ok = abs(c_T - 1.0) < gw_bound

        if ss_ok and gw_ok:
            viable.append(ViableRegion(
                c1=c1, c2=c2, c3=c3, c4=c4,
                gamma=gamma, beta=beta,
                alpha1=a1, alpha2=a2,
                c_T=c_T,
                all_satisfied=True,
            ))

    return viable
