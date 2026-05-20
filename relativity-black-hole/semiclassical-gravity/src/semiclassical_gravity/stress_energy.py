"""
Stress-Energy Tensor Computation for Semiclassical Gravity.

Provides routines for computing the renormalized expectation value of the
stress-energy tensor <T_mu_nu> in curved spacetime, with emphasis on 1+1D
conformal field theory where the computation is analytically tractable.

Key concepts:
- Trace anomaly: <T^mu_mu> = hbar * R / (24*pi) in 1+1D
- Point-splitting regularization for <T_mu_nu>
- Vacuum stress computation using the Schwarzian derivative

References:
    Birrell & Davies, "Quantum Fields in Curved Space", Cambridge (1982)
    Christensen, Phys. Rev. D 17, 946 (1978)
"""

import numpy as np
from .constants import HBAR, PI


def conformal_anomaly_1d(rho, coords):
    """
    Compute the trace anomaly in 1+1D: <T^mu_mu> = hbar * R / (24*pi).

    In 1+1D, the Ricci scalar for the metric ds^2 = -e^{2*rho} dx^+ dx^- is:
        R = -8 * e^{-2*rho} * partial_+ partial_- rho

    The trace anomaly is universal (state-independent) for conformal fields:
        <T^mu_mu> = hbar * R / (24*pi)  (per scalar field)

    Parameters
    ----------
    rho : array_like or callable
        Conformal factor. If callable, rho(coords). If array, assumed
        to be evaluated at the grid points coords.
    coords : array_like
        Coordinate grid points (1D array for a single light-cone coordinate,
        or 2D array of shape (n, 2) for (x^+, x^-) pairs).

    Returns
    -------
    anomaly : ndarray
        Trace anomaly <T^mu_mu> at each grid point.
    """
    coords = np.asarray(coords, dtype=float)

    if callable(rho):
        rho_vals = rho(coords)
    else:
        rho_vals = np.asarray(rho, dtype=float)

    rho_vals = np.atleast_1d(rho_vals)

    if rho_vals.size < 3:
        # Cannot compute curvature with fewer than 3 points
        return np.zeros_like(rho_vals)

    # Compute the Ricci scalar: R = -8 * e^{-2*rho} * d^2 rho / (dx^+ dx^-)
    # On a 1D grid, we approximate d^2 rho as the second derivative
    drho = np.gradient(rho_vals)
    ddrho = np.gradient(drho)

    # Ricci scalar
    exp_2rho = np.exp(2.0 * rho_vals)
    R = -8.0 * exp_2rho**(-1) * ddrho

    # Trace anomaly
    anomaly = HBAR * R / (24.0 * PI)

    return anomaly


def point_split_wightman(x1, x2):
    """
    Compute the point-split Wightman function for a massless scalar in 1+1D.

    The Wightman function (two-point function) for a massless scalar field in
    flat 1+1D spacetime is:

        G^+(x1, x2) = -(1/4*pi) * ln[(x1^+ - x2^+)(x1^- - x2^-)]
                     + (1/4*pi) * ln(mu^2)

    where mu is a renormalization scale and x^+/- = t +/- x are light-cone
    coordinates.

    The renormalized stress tensor is obtained from:
        <T_mu_nu> = lim_{x2->x1} D_mu_nu [G^+(x1,x2) - G^+(x1,x2)_singular]

    Parameters
    ----------
    x1 : array_like
        First spacetime point(s). Shape (n, 2) for n points in (t, x)
        coordinates, or (2,) for a single point.
    x2 : array_like
        Second spacetime point(s). Same shape as x1.

    Returns
    -------
    G_plus : ndarray
        Wightman function values G^+(x1, x2).
    """
    x1 = np.atleast_2d(np.asarray(x1, dtype=float))
    x2 = np.atleast_2d(np.asarray(x2, dtype=float))

    # Convert to light-cone coordinates: x^+ = t + x, x^- = t - x
    x1_plus = x1[:, 0] + x1[:, 1]
    x1_minus = x1[:, 0] - x1[:, 1]
    x2_plus = x2[:, 0] + x2[:, 1]
    x2_minus = x2[:, 0] - x2[:, 1]

    # Coordinate differences
    delta_plus = x1_plus - x2_plus
    delta_minus = x1_minus - x2_minus

    # Wightman function: G^+ = -(1/4pi) * ln[delta_+ * delta_-]
    # Regularize to avoid log(0) when points coincide
    eps = 1e-30
    argument = delta_plus * delta_minus + eps

    G_plus = -(1.0 / (4.0 * PI)) * np.log(np.abs(argument))

    return G_plus


def vacuum_stress_1d(rho, drho, ddrho):
    """
    Compute the vacuum expectation value <T_mu_nu> for a 1+1D conformal field.

    For a 1+1D conformal field on a curved background with metric
    ds^2 = -e^{2*rho} dx^+ dx^-, the renormalized stress tensor is:

        <T_{++}> = (hbar/12*pi) * [partial_+^2 rho - (partial_+ rho)^2]
                  + state_dependent_term

        <T_{--}> = (hbar/12*pi) * [partial_-^2 rho - (partial_- rho)^2]
                  + state_dependent_term

    For the Boulware vacuum (state appropriate for a static star),
    the state-dependent terms vanish.

    Note: The formula can also be written in terms of the Schwarzian
    derivative of the conformal factor.

    Parameters
    ----------
    rho : array_like
        Conformal factor values on the grid.
    drho : array_like
        First derivative of rho (numerical gradient).
    ddrho : array_like
        Second derivative of rho.

    Returns
    -------
    dict
        Dictionary with:
        - 'T_plus_plus': <T_{++}> component
        - 'T_minus_minus': <T_{--}> component
        - 'trace': <T^mu_mu> = trace anomaly
    """
    rho = np.asarray(rho, dtype=float)
    drho = np.asarray(drho, dtype=float)
    ddrho = np.asarray(ddrho, dtype=float)

    # Coefficient
    coeff = HBAR / (12.0 * PI)

    # For Boulware vacuum:
    # <T_{++}> = coeff * (ddrho - drho^2)
    # <T_{--}> = coeff * (ddrho - drho^2)
    # These are equal by symmetry for the diagonal components when computed
    # on a single grid (no distinction between + and - directions).

    T_plus_plus = coeff * (ddrho - drho**2)
    T_minus_minus = coeff * (ddrho - drho**2)

    # Trace: <T^mu_mu> = <T_{++}> g^{++} + <T_{--}> g^{-->}
    # In conformal gauge: g^{++} = -2 e^{-2rho}, g^{--} = -2 e^{-2rho}
    # Trace anomaly: <T^mu_mu> = hbar * R / (24*pi)
    # R = -8 * e^{-2rho} * d2rho (for diagonal computation)
    # So trace = hbar * (-8 * e^{-2rho} * d2rho) / (24*pi)
    trace = HBAR * (-8.0 * np.exp(-2.0 * rho) * ddrho) / (24.0 * PI)

    return {
        'T_plus_plus': T_plus_plus,
        'T_minus_minus': T_minus_minus,
        'trace': trace,
    }
