"""Slow-roll analysis for inflationary potentials.

All quantities are computed in reduced-Planck-mass units (Mpl = 1).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import integrate, optimize

from .potentials import QuadraticPotential

__all__ = [
    "epsilon_V",
    "eta_V",
    "n_s",
    "r_tensor",
    "dn_s_dlnk",
    "N_efolds",
    "phi_end",
    "phi_start",
    "planck_constraints",
]

MPL2 = 1.0  # Mpl^2 = 1 in our units


# ---------------------------------------------------------------------------
# Slow-roll parameters
# ---------------------------------------------------------------------------

def epsilon_V(potential, phi: float) -> float:
    """First Hubble slow-roll parameter from the potential: eps_V = (Mpl^2/2)(V'/V)^2."""
    V = potential.V(phi)
    dV = potential.dV(phi)
    if V == 0.0:
        return np.inf
    return 0.5 * MPL2 * (dV / V) ** 2


def eta_V(potential, phi: float) -> float:
    """Second slow-roll parameter: eta_V = Mpl^2 V''/V."""
    V = potential.V(phi)
    d2V = potential.d2V(phi)
    if V == 0.0:
        return np.inf
    return MPL2 * d2V / V


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def n_s(potential, phi: float) -> float:
    """Scalar spectral index: n_s - 1 = -6 eps_V + 2 eta_V."""
    return 1.0 - 6.0 * epsilon_V(potential, phi) + 2.0 * eta_V(potential, phi)


def r_tensor(potential, phi: float) -> float:
    """Tensor-to-scalar ratio: r = 16 eps_V."""
    return 16.0 * epsilon_V(potential, phi)


def dn_s_dlnk(potential, phi: float) -> float:
    """Running of the scalar spectral index (first-order slow-roll).

    dn_s/dln k = -24 eps_V^2 + 16 eps_V eta_V - 2 xi_V
    where xi_V = Mpl^4 V' V''' / V^2.
    """
    eps = epsilon_V(potential, phi)
    eta = eta_V(potential, phi)

    # Compute xi_V numerically via finite differences
    delta = 1e-4
    dV_plus = potential.dV(phi + delta)
    dV_minus = potential.dV(phi - delta)
    d3V = (dV_plus - dV_minus) / (2.0 * delta)

    V = potential.V(phi)
    if V == 0.0:
        return np.inf
    xi_V = MPL2**2 * potential.dV(phi) * d3V / V**2

    return -24.0 * eps**2 + 16.0 * eps * eta - 2.0 * xi_V


# ---------------------------------------------------------------------------
# e-fold counting
# ---------------------------------------------------------------------------

def N_efolds(potential, phi_start: float, phi_end: float) -> float:
    """Number of e-folds between phi_start and phi_end.

    N = (1/Mpl^2) int_{phi_end}^{phi_start} (V / V') dphi

    The integral is positive when phi_start > phi_end (inflation rolls
    from large phi towards smaller values for most models).
    """
    if np.isclose(phi_start, phi_end):
        return 0.0

    def integrand(phi):
        dV = potential.dV(phi)
        if abs(dV) < 1e-30:
            return np.inf
        return potential.V(phi) / dV

    result, _ = integrate.quad(integrand, phi_end, phi_start, limit=200)
    return result / MPL2


def phi_end(potential, phi_guess: float | None = None) -> float:
    """Find phi where eps_V = 1 (end of inflation).

    Parameters
    ----------
    potential : potential object
    phi_guess : float, optional
        Initial guess. If None, a heuristic search is performed.

    Returns
    -------
    float
        phi at end of inflation (closest to origin for plateau models).
    """

    def eq(phi):
        return epsilon_V(potential, phi) - 1.0

    if phi_guess is not None:
        sol = optimize.root_scalar(eq, x0=phi_guess, method="newton")
        if sol.converged:
            return sol.root

    # Heuristic: scan a range and find sign changes
    phi_test = np.logspace(-3, 3, 5000)
    vals = np.array([eq(p) for p in phi_test])
    # Find sign changes
    sign_changes = np.where(np.diff(np.sign(vals)))[0]
    if len(sign_changes) == 0:
        # Try negative range too
        phi_test_neg = -phi_test[::-1]
        vals_neg = np.array([eq(p) for p in phi_test_neg])
        sign_changes = np.where(np.diff(np.sign(vals_neg)))[0]
        if len(sign_changes) == 0:
            raise ValueError("Could not find phi_end for this potential.")
        phi_a = phi_test_neg[sign_changes[0]]
        phi_b = phi_test_neg[sign_changes[0] + 1]
    else:
        phi_a = phi_test[sign_changes[0]]
        phi_b = phi_test[sign_changes[0] + 1]

    sol = optimize.brentq(eq, phi_a, phi_b)
    return sol


def phi_start(potential, N_target: float = 60.0, phi_end_val: float | None = None) -> float:
    """Find phi giving N_target e-folds before end of inflation.

    Parameters
    ----------
    potential : potential object
    N_target : float
        Desired number of e-folds (default 60).
    phi_end_val : float, optional
        Value of phi at end of inflation. Computed if not given.

    Returns
    -------
    float
        phi at horizon exit.
    """
    if phi_end_val is None:
        phi_end_val = phi_end(potential)

    def eq(log_phi):
        phi_s = np.exp(log_phi)
        N = N_efolds(potential, phi_s, phi_end_val)
        return N - N_target

    # Search for the starting phi
    # For most models, phi_start > phi_end
    log_phi_min = np.log(max(phi_end_val * 1.001, 1e-10))
    log_phi_max = np.log(phi_end_val * 1e5) if phi_end_val > 0 else np.log(1e3)

    try:
        result = optimize.brentq(eq, log_phi_min, log_phi_max)
        return np.exp(result)
    except ValueError:
        # Wider search
        log_phi_max = np.log(max(abs(phi_end_val) * 1e8, 1e6))
        result = optimize.brentq(eq, log_phi_min, log_phi_max)
        return np.exp(result)


# ---------------------------------------------------------------------------
# Planck constraints
# ---------------------------------------------------------------------------

def planck_constraints() -> Dict[str, Tuple[float, float, float]]:
    """Return Planck 2018 TT,TE,EE+lowE+lensing constraints.

    Returns
    -------
    dict
        Keys: 'n_s', 'r'. Values are (central, lower_1sigma, upper_1sigma).
    """
    return {
        "n_s": (0.9649, 0.9649 - 0.0042, 0.9649 + 0.0042),
        "r": (0.0, 0.0, 0.036),
    }
