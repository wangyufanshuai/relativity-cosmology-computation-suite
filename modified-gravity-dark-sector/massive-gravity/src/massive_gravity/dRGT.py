"""
dRGT (de Rham-Gabadadze-Tolley) massive gravity potentials and equations of motion.

Implements the ghost-free nonlinear massive gravity theory of de Rham, Gabadadze
and Tolley (2011). The action contains an Einstein-Hilbert term for the dynamical
metric g_{\\mu\\nu} plus a potential constructed from the reference (typically flat)
metric f_{\\mu\\nu}.

Key objects:
  - Interaction matrix  gamma^\\mu_\\nu = (sqrt(g^{-1} f))^\\mu_\\nu
  - dRGT potentials      U_n[e] = \\delta^{\\mu_1..\\mu_n}_{\\nu_1..\\nu_n}
                               e^{\\nu_1}_{\\mu_1} ... e^{\\nu_n}_{\\mu_n}
  - Full potential       V(g,f) = -m^2 \\sum_{n=0}^{4} \\beta_n U_n
  - Modified Einstein eq G_{\\mu\\nu} + m^2 X_{\\mu\\nu} = 8\\pi G T_{\\mu\\nu}

References:
  - de Rham, Gabadadze, Tolley, Phys.Rev.Lett. 106 (2011) 231101
  - Hassan, Rosen, JHEP 1202 (2012) 126
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants (SI-like natural units with c = 1)
# ---------------------------------------------------------------------------
G_NEWTON = 6.67430e-11  # m^3 kg^{-1} s^{-2}
C_LIGHT = 2.99792458e8  # m/s

# ---------------------------------------------------------------------------
# Elementary symmetric polynomials / Kronecker-delta contractions
# ---------------------------------------------------------------------------

def _elementary_symmetric(eigenvalues: NDArray, n: int) -> float:
    """Return the n-th elementary symmetric polynomial of the given eigenvalues.

    For a 4x4 matrix this is equivalent to the generalized Kronecker-delta
    contraction  U_n = \\delta^{\\mu_1..\\mu_n}_{\\nu_1..\\nu_n}
    e^{\\nu_1}_{\\mu_1} ... e^{\\nu_n}_{\\mu_n}.
    """
    k = len(eigenvalues)
    if n < 0 or n > k:
        return 0.0
    if n == 0:
        return 1.0
    # Iterative computation using Newton-like recursion
    # e_n = sum over subsets of size n of product of eigenvalues
    result = 0.0
    from itertools import combinations
    for combo in combinations(range(k), n):
        result += float(np.prod([eigenvalues[i] for i in combo]))
    return result


def _identity_4d() -> NDArray:
    """Return the 4x4 Minkowski metric diag(-1, 1, 1, 1)."""
    return np.diag([-1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Interaction matrix gamma
# ---------------------------------------------------------------------------

def interaction_matrix(g: NDArray, f: NDArray) -> NDArray:
    """Compute the interaction matrix  gamma^\\mu_\\nu = (sqrt(g^{-1} f))^\\mu_\\nu.

    Parameters
    ----------
    g : (4,4) array
        Dynamical metric g_{\\mu\\nu} (signature -+++).
    f : (4,4) array
        Reference metric f_{\\mu\\nu} (same signature).

    Returns
    -------
    gamma : (4,4) array
        Mixed-index interaction matrix.
    """
    g_inv = np.linalg.inv(g)
    X = g_inv @ f  # (g^{-1} f)^\mu_\nu in mixed form with one up, one down
    # Compute matrix square root via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(X)
    # Guard against negative eigenvalues (should not happen for physical metrics
    # with same signature, but protect numerically)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_eigvals = np.sqrt(eigvals)
    gamma = eigvecs @ np.diag(sqrt_eigvals) @ np.linalg.inv(eigvecs)
    return gamma


# ---------------------------------------------------------------------------
# dRGT potential terms U_n
# ---------------------------------------------------------------------------

def compute_Un(g: NDArray, f: NDArray, n: int) -> float:
    """Compute the n-th dRGT potential term U_n.

    U_n = \\delta^{\\mu_1..\\mu_n}_{\\nu_1..\\nu_n}
          (\\sqrt{g^{-1} f})^{\\nu_1}_{\\mu_1} ...
          (\\sqrt{g^{-1} f})^{\\nu_n}_{\\mu_n}

    This equals the n-th elementary symmetric polynomial of the eigenvalues
    of the interaction matrix gamma.
    """
    gamma = interaction_matrix(g, f)
    eigvals = np.linalg.eigvalsh(gamma)
    return _elementary_symmetric(eigvals, n)


def compute_all_Un(g: NDArray, f: NDArray) -> list[float]:
    """Return all five dRGT potential terms U_0 ... U_4."""
    gamma = interaction_matrix(g, f)
    eigvals = np.linalg.eigvalsh(gamma)
    return [_elementary_symmetric(eigvals, n) for n in range(5)]


# ---------------------------------------------------------------------------
# Full dRGT potential  V(g,f) = -m^2 \\sum \\beta_n U_n
# ---------------------------------------------------------------------------

def dRGT_potential(
    g: NDArray,
    f: NDArray,
    m_grav: float,
    betas: tuple[float, ...] | list[float] | NDArray = (1.0, 1.0, 1.0, 1.0, 1.0),
) -> float:
    """Evaluate the full dRGT potential.

    V(g,f) = -m^2 \\sum_{n=0}^{4} \\beta_n U_n

    Parameters
    ----------
    g : (4,4) array
        Dynamical metric.
    f : (4,4) array
        Reference metric.
    m_grav : float
        Graviton mass parameter.
    betas : length-5 sequence
        Coupling constants (\\beta_0 ... \\beta_4).
    """
    Un_vals = compute_all_Un(g, f)
    betas = np.asarray(betas, dtype=float)
    return -m_grav**2 * float(np.dot(betas, Un_vals))


# ---------------------------------------------------------------------------
# Variation of potential w.r.t. g^{\\mu\\nu}  (X_{\\mu\\nu})
# ---------------------------------------------------------------------------

def potential_variation(
    g: NDArray,
    f: NDArray,
    m_grav: float,
    betas: tuple[float, ...] | list[float] | NDArray = (1.0, 1.0, 1.0, 1.0, 1.0),
    epsilon: float = 1e-6,
) -> NDArray:
    """Compute V_{\\mu\\nu} = \\delta V / \\delta g^{\\mu\\nu} numerically.

    Uses a symmetric finite-difference approximation:
      V_{\\mu\\nu} \\approx [V(g + \\epsilon e_{\\mu\\nu}) - V(g - \\epsilon e_{\\mu\\nu})]
                            / (2 \\epsilon)

    Parameters
    ----------
    g, f : (4,4) arrays
        Metrics.
    m_grav : float
        Graviton mass.
    betas : length-5 sequence
        Coupling constants.
    epsilon : float
        Finite-difference step size.

    Returns
    -------
    X_munu : (4,4) array
        Potential variation tensor (lower index).
    """
    V0 = dRGT_potential(g, f, m_grav, betas)
    X = np.zeros_like(g)

    # We vary g^{\\mu\\nu} (the inverse metric).  g^{\\mu\\nu} and g_{\\mu\\nu}
    # are related by matrix inversion, so to vary g^{\\mu\\nu} by a small amount
    # we perturb the *inverse* of g, then recompute g_pert = inv(g_inv_pert).
    g_inv = np.linalg.inv(g)

    for mu in range(4):
        for nu in range(4):
            # Symmetric perturbation to g^{\\mu\\nu}
            delta = np.zeros((4, 4))
            delta[mu, nu] = epsilon
            if mu != nu:
                # Keep the perturbation symmetric
                delta[nu, mu] = epsilon

            g_inv_plus = g_inv + delta
            g_plus = np.linalg.inv(g_inv_plus)
            g_inv_minus = g_inv - delta
            g_minus = np.linalg.inv(g_inv_minus)

            V_plus = dRGT_potential(g_plus, f, m_grav, betas)
            V_minus = dRGT_potential(g_minus, f, m_grav, betas)

            X[mu, nu] = (V_plus - V_minus) / (2.0 * epsilon)

    return X


# ---------------------------------------------------------------------------
# Modified Einstein tensor
# ---------------------------------------------------------------------------

def eom_lhs(
    g: NDArray,
    f: NDArray,
    m_grav: float,
    betas: tuple[float, ...] | list[float] | NDArray = (1.0, 1.0, 1.0, 1.0, 1.0),
    epsilon: float = 1e-6,
) -> NDArray:
    """Return the left-hand side of the modified Einstein equations.

    G_{\\mu\\nu} + m^2 V_{\\mu\\nu}

    For simplicity G_{\\mu\\nu} is computed in the flat-space limit where
    it vanishes, so this returns just V_{\\mu\\nu} when the background is flat.

    Parameters
    ----------
    g, f : (4,4) arrays
        Dynamical and reference metrics.
    m_grav : float
        Graviton mass.
    betas : length-5 sequence
        Coupling constants.
    epsilon : float
        Finite-difference step size.
    """
    V_munu = potential_variation(g, f, m_grav, betas, epsilon)
    # In full GR we would add the Einstein tensor here.
    # For flat background G_munu = 0.
    return V_munu


# ---------------------------------------------------------------------------
# Flat-space limit check
# ---------------------------------------------------------------------------

def flat_space_potential_vanishes(
    betas: tuple[float, ...] | list[float] | NDArray = (1.0, 1.0, 1.0, 1.0, 1.0),
    m_grav: float = 1.0,
    tol: float = 1e-8,
) -> bool:
    """Check that V_{\\mu\\nu} vanishes (up to tolerance) when g = f = Minkowski.

    This is a necessary condition for flat space to be a solution.
    """
    eta = _identity_4d()
    V_munu = potential_variation(eta, eta, m_grav, betas)
    return bool(np.allclose(V_munu, 0.0, atol=tol))
