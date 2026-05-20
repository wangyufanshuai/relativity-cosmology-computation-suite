"""
Hamiltonian and momentum constraints for the BSSN formulation.

Hamiltonian constraint:
    H = R + (2/3) K^2 - A_tilde_ij A_tilde^ij - 16 pi rho = 0

Momentum constraint:
    M_i = D^j A_{ij} - (2/3) d_i K - 8 pi S_i = 0

In vacuum (rho = 0, S_i = 0), the constraints simplify.
"""

import numpy as np
from .bssn_variables import _inverse_3x3, _determinant_3x3
from .gauge import partial_i, gradient, divergence


def hamiltonian_constraint(state, dx, order=2):
    """
    Compute the Hamiltonian constraint violation.

    H = R_tilde - 8 A_tilde^ij d_i d_j phi - 8 d^i phi d_i (e^{-4phi} R_tilde / 8)
        ... simplified to the standard BSSN form:
    H = e^{-4phi} (R_tilde - 8 A_tilde^ij d_i d_j phi
        - 8 d^i phi d_j A_tilde^j_i + ...
        + (2/3) K^2 - A_tilde_ij A_tilde^ij)

    For conformally flat data (R_tilde = 0) with K = 0 and A_tilde = 0,
    the constraint is trivially satisfied.

    Parameters
    ----------
    state : BSSNState
    dx : float
    order : int

    Returns
    -------
    H : ndarray (N,N,N)
        Hamiltonian constraint violation.
    """
    phi = state.phi
    gamma_tilde = state.gamma_tilde
    K = state.K
    A_tilde = state.A_tilde

    e4phi = np.exp(4.0 * phi)

    # Inverse conformal metric
    gamma_tilde_U = _inverse_3x3(gamma_tilde)

    # Conformal Ricci scalar (simplified for near-flat conformal metric)
    R_tilde = _compute_R_tilde(gamma_tilde, gamma_tilde_U, dx, order)

    # A_tilde_ij A_tilde^ij
    A2 = np.zeros_like(phi)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A2 += gamma_tilde_U[i, k] * gamma_tilde_U[j, l] * A_tilde[i, j] * A_tilde[k, l]

    # Gradient terms for phi
    d_phi = gradient(phi, dx, order=order)
    d2_phi = np.zeros_like(phi)
    for i in range(3):
        for j in range(3):
            d2_phi += gamma_tilde_U[i, j] * partial_i(
                partial_i(phi, i, dx, order), j, dx, order
            ) * dx  # This is a simplification

    # Simplified Hamiltonian constraint
    # For the full expression we need:
    # H = e^{-4phi} (R_tilde - 8 A_tilde^ij d_i d_j phi
    #     + (2/3) K^2 - A_tilde_ij A_tilde^ij) + source terms

    # Cross term: A_tilde^ij d_i d_j phi
    cross = np.zeros_like(phi)
    for i in range(3):
        for j in range(3):
            d2_phi_ij = partial_i(partial_i(phi, i, dx, order), j, dx, order)
            for k in range(3):
                for l in range(3):
                    cross += gamma_tilde_U[i, k] * gamma_tilde_U[j, l] * A_tilde[k, l] * d2_phi_ij

    H = e4phi * (R_tilde - 8.0 * cross + 2.0 * K**2 / 3.0 - A2)

    return H


def momentum_constraint(state, dx, order=2):
    """
    Compute the momentum constraint violation.

    M_i = D^j A_tilde_{ij} - (2/3) d_i K - 8 pi S_i

    In vacuum, the source term vanishes.

    Parameters
    ----------
    state : BSSNState
    dx : float
    order : int

    Returns
    -------
    M : ndarray (3,N,N,N)
        Momentum constraint violation.
    """
    phi = state.phi
    gamma_tilde = state.gamma_tilde
    K = state.K
    A_tilde = state.A_tilde

    gamma_tilde_U = _inverse_3x3(gamma_tilde)
    e4phi = np.exp(4.0 * phi)

    M = np.zeros((3,) + phi.shape)

    # D^j A_tilde_{ij} = gamma_tilde^{jk} d_k A_tilde_{ij} + connection terms
    # For conformally flat metric, connection terms vanish
    for i in range(3):
        # d_j K term
        dK_i = partial_i(K, i, dx, order)
        M[i] = -2.0 / 3.0 * dK_i

        # D^j A_tilde_{ij}
        for j in range(3):
            for k in range(3):
                dA = partial_i(A_tilde[i, j], k, dx, order)
                M[i] += gamma_tilde_U[j, k] * dA * e4phi

    return M


def hamiltonian_constraint_simplified(state, dx, order=2):
    """
    Simplified Hamiltonian constraint for near-conformally-flat data.

    For conformally flat data with K=0, A_tilde=0, the Hamiltonian
    constraint reduces to checking that the Laplacian of psi satisfies
    the Lichnerowicz equation.

    H_simplified = Lap(psi) + psi^5 K_ij K^ij / 8 - psi A_tilde^2 / 8

    For time-symmetric conformally flat data: H = 0 identically.

    Parameters
    ----------
    state : BSSNState
    dx : float
    order : int

    Returns
    -------
    H : ndarray (N,N,N)
    """
    phi = state.phi
    K = state.K
    A_tilde = state.A_tilde
    gamma_tilde = state.gamma_tilde

    gamma_tilde_U = _inverse_3x3(gamma_tilde)

    # A_tilde^ij A_tilde_ij
    A2 = np.zeros_like(phi)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    A2 += (gamma_tilde_U[i, k] * gamma_tilde_U[j, l]
                           * A_tilde[i, j] * A_tilde[k, l])

    # Laplacian of phi with respect to conformal metric
    lap_phi = np.zeros_like(phi)
    for i in range(3):
        for j in range(3):
            lap_phi += gamma_tilde_U[i, j] * partial_i(
                partial_i(phi, i, dx, order), j, dx, order
            )

    # e^{4phi} factor
    e4phi = np.exp(4.0 * phi)

    # Simplified Hamiltonian: dominant terms only
    H = e4phi * (2.0 / 3.0 * K**2 - A2) + lap_phi

    return H


def _compute_R_tilde(gamma_tilde, gamma_tilde_U, dx, order=2):
    """
    Compute the Ricci scalar of the conformal metric.

    R_tilde = gamma_tilde^{ij} R_tilde_{ij}

    For a conformal metric that is a small perturbation of flat space,
    we compute this using finite differences of the metric.

    Parameters
    ----------
    gamma_tilde : ndarray (3,3,N,N,N)
    gamma_tilde_U : ndarray (3,3,N,N,N)
    dx : float
    order : int

    Returns
    -------
    R_tilde : ndarray (N,N,N)
    """
    R_tilde = np.zeros_like(gamma_tilde[0, 0])

    # Christoffel symbols of the conformal metric
    # Gamma_tilde^i_{jk} = (1/2) gamma_tilde^{il} (d_j gamma_tilde_{lk} + d_k gamma_tilde_{jl} - d_l gamma_tilde_{jk})

    # For simplicity, compute R_tilde from second derivatives of the metric:
    # R_{ij} = (1/2) (d_k d_i gamma^k_j + d_k d_j gamma^k_i - d_k d^k gamma_ij - d_i d_j (ln det gamma_tilde))
    # But for conformally flat metric, all of these vanish.

    # Compute using the standard formula for a perturbed metric
    # h_ij = gamma_tilde_ij - delta_ij (perturbation)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                d2_g_kk = partial_i(
                    partial_i(gamma_tilde[k, k], i, dx, order), j, dx, order
                )
                d2_g_ij = partial_i(
                    partial_i(gamma_tilde[i, j], k, dx, order), k, dx, order
                )
                d2_g_ik = partial_i(
                    partial_i(gamma_tilde[i, k], j, dx, order), k, dx, order
                )
                d2_g_jk = partial_i(
                    partial_i(gamma_tilde[j, k], i, dx, order), k, dx, order
                )
                R_tilde += gamma_tilde_U[i, j] * (
                    d2_g_ik + d2_g_jk - d2_g_ij - d2_g_kk
                ) / 2.0

    return R_tilde


def compute_constraint_norm(H, M, dx):
    """
    Compute L2 norms of the Hamiltonian and momentum constraints.

    Parameters
    ----------
    H : ndarray (N,N,N)
    M : ndarray (3,N,N,N)
    dx : float

    Returns
    -------
    H_norm : float
        ||H||_2
    M_norm : float
        sqrt(sum_i ||M_i||_2^2)
    """
    H_norm = np.sqrt(np.sum(H**2) * dx**3)
    M_norm = np.sqrt(np.sum(M**2) * dx**3)
    return H_norm, M_norm
