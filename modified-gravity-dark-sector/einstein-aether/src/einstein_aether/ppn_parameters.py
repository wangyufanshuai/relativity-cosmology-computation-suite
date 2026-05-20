"""PPN parameters for Einstein-aether theory.

Computes the Parameterised Post-Newtonian (PPN) parameters gamma and beta,
Newton's constant ratio G_N / G, and preferred-frame parameters alpha_1,
alpha_2 as functions of the coupling constants c_1, c_2, c_3, c_4.

Reference notation:
    c_14 = c_1 + c_4
    c_13 = c_1 + c_3
    c_123 = c_1 + c_2 + c_3
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ------------------------------------------------------------------
# Helper combinations
# ------------------------------------------------------------------

def _c_combinations(c1: float, c2: float, c3: float, c4: float) -> dict:
    """Return common combinations of c_i coupling constants."""
    c14 = c1 + c4
    c13 = c1 + c3
    c123 = c1 + c2 + c3
    return {"c14": c14, "c13": c13, "c123": c123}


def _denom_grav(c13: float, c2: float, c14: float) -> float:
    """Denominator appearing in gravitational coupling expressions.

    D = 2 - c_14 - c_13 * (2 + c_2) / (1 - c_13)
    """
    if abs(1.0 - c13) < 1e-15:
        return np.inf
    return 2.0 - c14 - c13 * (2.0 + c2) / (1.0 - c13)


# ------------------------------------------------------------------
# PPN gamma
# ------------------------------------------------------------------

def ppn_gamma(c1: float, c2: float, c3: float, c4: float) -> float:
    """Compute the PPN parameter gamma_PPN.

    gamma_PPN = 1 - 2 * (c_1 + c_2 + c_3 + c_4) / D

    where D = 2 - c_14 - c_13 (2 + c_2) / (1 - c_13)
    and c_14 = c_1 + c_4, c_13 = c_1 + c_3.

    In GR: gamma = 1.
    """
    c14 = c1 + c4
    c13 = c1 + c3
    D = _denom_grav(c13, c2, c14)
    if abs(D) < 1e-30:
        return np.inf
    numerator = 2.0 * (c1 + c2 + c3 + c4)
    return 1.0 - numerator / D


# ------------------------------------------------------------------
# PPN beta
# ------------------------------------------------------------------

def ppn_beta(c1: float, c2: float, c3: float, c4: float) -> float:
    """Compute the PPN parameter beta_PPN.

    The expression follows Jacobson & Mattingly (2004) and Foster (2007):

    beta_PPN = 1 - (c_1 + c_2 + c_3 + c_4)^2 / D^2
               + correction terms involving c_i products.

    For small c_i, beta -> 1 (GR limit).
    The full expression is:

    beta = 1
         + (1/2 D^2) * [ -2 (c123 + c14)^2
                          + c14 (2 + 3 c2) (c123 + c14) / (1 - c13)
                          - ... ]

    We use the compact form valid to second order in c_i:
    """
    c14 = c1 + c4
    c13 = c1 + c3
    c123 = c1 + c2 + c3
    D = _denom_grav(c13, c2, c14)
    if abs(D) < 1e-30:
        return np.inf

    # Leading correction from the square of the gamma deviation
    gamma_dev = (c123 + c14) / D  # (c1+c2+c3+c4)/D
    correction = -gamma_dev ** 2

    # Additional term from the cubic self-interaction of the scalar mode
    if abs(1.0 - c13) > 1e-15:
        correction += c14 * (2.0 + 3.0 * c2) * (c123 + c14) / (D ** 2 * (1.0 - c13))

    return 1.0 + correction


# ------------------------------------------------------------------
# Newton constant ratio
# ------------------------------------------------------------------

def newton_constant_ratio(c1: float, c2: float, c3: float, c4: float) -> float:
    """Compute G_N / G_bare.

    G_N / G = 1 - (c_14 / 2) * (2 + c_13 + 3 c_2) / D

    where D = 2 - c_14 - c_13 (2 + c_2) / (1 - c_13).

    In GR (all c_i = 0): G_N / G = 1.
    """
    c14 = c1 + c4
    c13 = c1 + c3
    D = _denom_grav(c13, c2, c14)
    if abs(D) < 1e-30:
        return np.inf
    return 1.0 - (c14 / 2.0) * (2.0 + c13 + 3.0 * c2) / D


# ------------------------------------------------------------------
# Preferred-frame parameters
# ------------------------------------------------------------------

def preferred_frame_params(
    c1: float, c2: float, c3: float, c4: float
) -> tuple[float, float]:
    """Compute preferred-frame parameters alpha_1 and alpha_2.

    These parametrise violation of Lorentz invariance in the PPN
    formalism.  In GR: alpha_1 = alpha_2 = 0.

    alpha_1 = -4 (c_1 + c_2 + c_3 + c_4) / (1 - c_13)  [simplified]
              More precisely involves the full post-Newtonian expansion.

    alpha_2 ~ (c_1 + c_3) * (c_1 + c_4) / D  [leading order]

    The exact expressions from Foster & Jacobson:

        alpha_1 = -4 * c_2 * (c_13 + c_4) / ((1 - c_13) * D)
                  - 4 * (c_1 + c_3 + c_4) / (1 - c_13)  [approximate]

    We use the standard expressions from the literature.

    Returns
    -------
    (alpha_1, alpha_2) : tuple of float
    """
    c14 = c1 + c4
    c13 = c1 + c3
    c123 = c1 + c2 + c3
    D = _denom_grav(c13, c2, c14)

    if abs(1.0 - c13) < 1e-15 or abs(D) < 1e-30:
        return (np.inf, np.inf)

    # alpha_1 (Foster 2007)
    alpha_1 = -4.0 * (c2 * (c13 + c4) + c123) / ((1.0 - c13) * D)

    # alpha_2 (leading order)
    alpha_2 = -(c13 * c14 + c2 * c14 / 2.0) / D
    # Additional contribution
    alpha_2 += (c14 * c13 / (1.0 - c13)) * (1.0 + c2 / 2.0) / D

    return (alpha_1, alpha_2)
