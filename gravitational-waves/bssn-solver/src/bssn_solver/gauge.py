"""
Gauge conditions for BSSN evolution.

Implements:
  - 1+log slicing: d_t alpha = -2 alpha K + beta^i d_i alpha
  - Gamma-driver shift:
      d_t beta^i = (3/4) B^i
      d_t B^i = d_t Lambda_tilde^i - eta B^i

These are the standard moving-puncture gauge conditions used in
binary black hole simulations.
"""

import numpy as np


def one_plus_log_rhs(alpha, K, beta, d_alpha, dx):
    """
    Right-hand side of the 1+log slicing condition.

    d_t alpha = -2 alpha K + beta^i d_i alpha

    Parameters
    ----------
    alpha : ndarray (N,N,N)
        Lapse function.
    K : ndarray (N,N,N)
        Trace of extrinsic curvature.
    beta : ndarray (3,N,N,N)
        Shift vector.
    d_alpha : ndarray (3,N,N,N)
        Spatial derivatives of alpha.
    dx : float
        Grid spacing.

    Returns
    -------
    rhs_alpha : ndarray (N,N,N)
    """
    advection = np.zeros_like(alpha)
    for i in range(3):
        advection += beta[i] * d_alpha[i]

    rhs_alpha = -2.0 * alpha * K + advection
    return rhs_alpha


def gamma_driver_rhs(beta, B, Lambda_tilde_dot, eta=0.75):
    """
    Right-hand side of the Gamma-driver shift condition.

    d_t beta^i = (3/4) B^i
    d_t B^i = d_t Lambda_tilde^i - eta B^i

    Parameters
    ----------
    beta : ndarray (3,N,N,N)
        Shift vector.
    B : ndarray (3,N,N,N)
        Auxiliary variable for shift evolution.
    Lambda_tilde_dot : ndarray (3,N,N,N)
        Time derivative of conformal connection functions.
    eta : float
        Damping parameter (typical value ~0.5-1.0 / M).

    Returns
    -------
    rhs_beta : ndarray (3,N,N,N)
    rhs_B : ndarray (3,N,N,N)
    """
    rhs_beta = 0.75 * B
    rhs_B = Lambda_tilde_dot - eta * B
    return rhs_beta, rhs_B


def compute_alpha_rhs(alpha, K, beta, dx, order=2):
    """
    Full RHS for lapse evolution using 1+log slicing.

    Parameters
    ----------
    alpha : ndarray (N,N,N)
    K : ndarray (N,N,N)
    beta : ndarray (3,N,N,N)
    dx : float
    order : int
        Finite difference order (2 or 4).

    Returns
    -------
    rhs_alpha : ndarray (N,N,N)
    """
    d_alpha = gradient(alpha, dx, order=order)
    return one_plus_log_rhs(alpha, K, beta, d_alpha, dx)


def compute_shift_rhs(beta, B, Lambda_tilde_dot, eta=0.75):
    """
    Full RHS for shift evolution using Gamma-driver.

    Parameters
    ----------
    beta : ndarray (3,N,N,N)
    B : ndarray (3,N,N,N)
    Lambda_tilde_dot : ndarray (3,N,N,N)
    eta : float

    Returns
    -------
    rhs_beta : ndarray (3,N,N,N)
    rhs_B : ndarray (3,N,N,N)
    """
    return gamma_driver_rhs(beta, B, Lambda_tilde_dot, eta)


def gradient(f, dx, order=2):
    """
    Compute the gradient of a scalar field using finite differences.

    Parameters
    ----------
    f : ndarray (N,N,N)
    dx : float
    order : int
        2 for second-order, 4 for fourth-order centered differences.

    Returns
    -------
    grad : ndarray (3,N,N,N)
    """
    N = f.shape[0]
    grad = np.zeros((3,) + f.shape)

    if order == 2:
        for d in range(3):
            # Centered second-order finite difference
            # Use np.roll for periodic-like stencil; apply boundary fix
            grad[d] = (np.roll(f, -1, axis=d) - np.roll(f, 1, axis=d)) / (2.0 * dx)
    elif order == 4:
        for d in range(3):
            grad[d] = (-np.roll(f, -2, axis=d) + 8.0 * np.roll(f, -1, axis=d)
                       - 8.0 * np.roll(f, 1, axis=d) + np.roll(f, 2, axis=d)) / (12.0 * dx)
    else:
        raise ValueError(f"Unsupported finite difference order: {order}")

    return grad


def divergence(vec, dx, order=2):
    """
    Compute the divergence of a vector field.

    Parameters
    ----------
    vec : ndarray (3,N,N,N)
    dx : float
    order : int

    Returns
    -------
    div : ndarray (N,N,N)
    """
    div = np.zeros_like(vec[0])
    N = vec.shape[1]

    if order == 2:
        for d in range(3):
            div += (np.roll(vec[d], -1, axis=d) - np.roll(vec[d], 1, axis=d)) / (2.0 * dx)
    elif order == 4:
        for d in range(3):
            div += (-np.roll(vec[d], -2, axis=d) + 8.0 * np.roll(vec[d], -1, axis=d)
                    - 8.0 * np.roll(vec[d], 1, axis=d) + np.roll(vec[d], 2, axis=d)) / (12.0 * dx)

    return div


def partial_i(f, d, dx, order=2):
    """
    Partial derivative of f with respect to coordinate direction d.

    Parameters
    ----------
    f : ndarray (...,N,N,N)
    d : int (0, 1, or 2)
    dx : float
    order : int

    Returns
    -------
    df : same shape as f
    """
    if order == 2:
        return (np.roll(f, -1, axis=d) - np.roll(f, 1, axis=d)) / (2.0 * dx)
    elif order == 4:
        return (-np.roll(f, -2, axis=d) + 8.0 * np.roll(f, -1, axis=d)
                - 8.0 * np.roll(f, 1, axis=d) + np.roll(f, 2, axis=d)) / (12.0 * dx)
