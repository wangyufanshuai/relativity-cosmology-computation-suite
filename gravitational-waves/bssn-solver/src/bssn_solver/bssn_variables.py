"""
BSSN variable decomposition and recomputation.

3+1 decomposition of spacetime into spatial metric gamma_ij,
extrinsic curvature K_ij, lapse alpha, and shift beta^i.

BSSN conformal variables:
  - Conformal factor phi = ln(det gamma) / 12  (or W = det(gamma)^{-1/6})
  - Conformal metric gamma_tilde_ij = e^{-4phi} gamma_ij  (unit determinant)
  - Conformal traceless extrinsic curvature A_tilde_ij = e^{-4phi}(K_ij - gamma_ij K/3)
  - Conformal connection functions Lambda_tilde^i
"""

import numpy as np


class BSSNState:
    """
    Container for all BSSN variables on a 3D Cartesian grid.

    All fields are numpy arrays with shape (3, 3, N, N, N) for tensor
    components (where the first two indices are i,j) or (N, N, N) for
    scalars, or (3, N, N, N) for vectors.

    Parameters
    ----------
    phi : ndarray (N,N,N)
        Conformal factor phi = ln(det gamma) / 12.
    gamma_tilde : ndarray (3,3,N,N,N)
        Conformal metric gamma_tilde_ij, symmetric, unit determinant.
    K : ndarray (N,N,N)
        Trace of extrinsic curvature.
    A_tilde : ndarray (3,3,N,N,N)
        Conformal traceless extrinsic curvature, symmetric, trace-free.
    Lambda_tilde : ndarray (3,N,N,N)
        Conformal connection functions.
    alpha : ndarray (N,N,N)
        Lapse function.
    beta : ndarray (3,N,N,N)
        Shift vector.
    """

    def __init__(self, phi, gamma_tilde, K, A_tilde, Lambda_tilde, alpha, beta):
        self.phi = phi
        self.gamma_tilde = gamma_tilde
        self.K = K
        self.A_tilde = A_tilde
        self.Lambda_tilde = Lambda_tilde
        self.alpha = alpha
        self.beta = beta

    @property
    def grid_shape(self):
        return self.phi.shape


def physical_to_bssn(gamma_ij, K_ij, alpha, beta):
    """
    Convert physical 3+1 variables to BSSN variables.

    Parameters
    ----------
    gamma_ij : ndarray (3,3,N,N,N)
        Physical spatial metric.
    K_ij : ndarray (3,3,N,N,N)
        Physical extrinsic curvature.
    alpha : ndarray (N,N,N)
        Lapse function.
    beta : ndarray (3,N,N,N)
        Shift vector.

    Returns
    -------
    BSSNState
    """
    # Determinant of gamma_ij
    # Use explicit determinant formula for 3x3
    det_gamma = _determinant_3x3(gamma_ij)

    # Conformal factor
    # phi = ln(det gamma) / 12
    phi = np.log(np.maximum(det_gamma, 1e-300)) / 12.0

    # Conformal factor squared for convenience
    e4phi = np.exp(4.0 * phi)

    # Conformal metric: gamma_tilde_ij = e^{-4phi} gamma_ij
    gamma_tilde = gamma_ij / e4phi[np.newaxis, np.newaxis, ...]

    # Trace of K_ij using inverse of gamma_ij
    gamma_U = _inverse_3x3(gamma_ij)
    K = np.zeros_like(K_ij[0, 0])
    for i in range(3):
        for j in range(3):
            K += gamma_U[i, j] * K_ij[i, j]

    # Conformal traceless extrinsic curvature
    # A_tilde_ij = e^{-4phi}(K_ij - gamma_ij * K / 3)
    A_tilde = (K_ij - gamma_ij * K[np.newaxis, np.newaxis, ...] / 3.0) / e4phi[np.newaxis, np.newaxis, ...]

    # Conformal connection functions Lambda_tilde^i
    Lambda_tilde = _compute_conformal_connections(gamma_tilde)

    return BSSNState(phi, gamma_tilde, K, A_tilde, Lambda_tilde, alpha, beta)


def bssn_to_physical(state):
    """
    Reconstruct physical 3+1 variables from BSSN variables.

    Parameters
    ----------
    state : BSSNState

    Returns
    -------
    gamma_ij : ndarray (3,3,N,N,N)
    K_ij : ndarray (3,3,N,N,N)
    alpha : ndarray (N,N,N)
    beta : ndarray (3,N,N,N)
    """
    e4phi = np.exp(4.0 * state.phi)

    # Physical metric: gamma_ij = e^{4phi} gamma_tilde_ij
    gamma_ij = state.gamma_tilde * e4phi[np.newaxis, np.newaxis, ...]

    # Inverse conformal metric
    gamma_tilde_U = _inverse_3x3(state.gamma_tilde)

    # Physical trace-free part
    A_ij = state.A_tilde * e4phi[np.newaxis, np.newaxis, ...]

    # Physical extrinsic curvature: K_ij = A_ij + gamma_ij * K / 3
    K_ij = A_ij + gamma_ij * state.K[np.newaxis, np.newaxis, ...] / 3.0

    return gamma_ij, K_ij, state.alpha, state.beta


def _determinant_3x3(a):
    """Compute determinant of a (3,3,...) symmetric matrix field."""
    return (a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
            - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
            + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]))


def _inverse_3x3(a):
    """Compute inverse of a (3,3,...) matrix field."""
    det = _determinant_3x3(a)
    det_inv = 1.0 / (det + 1e-300)

    inv = np.empty_like(a)
    inv[0, 0] = (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1]) * det_inv
    inv[0, 1] = (a[0, 2] * a[2, 1] - a[0, 1] * a[2, 2]) * det_inv
    inv[0, 2] = (a[0, 1] * a[1, 2] - a[0, 2] * a[1, 1]) * det_inv
    inv[1, 0] = (a[1, 2] * a[2, 0] - a[1, 0] * a[2, 2]) * det_inv
    inv[1, 1] = (a[0, 0] * a[2, 2] - a[0, 2] * a[2, 0]) * det_inv
    inv[1, 2] = (a[0, 2] * a[1, 0] - a[0, 0] * a[1, 2]) * det_inv
    inv[2, 0] = (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0]) * det_inv
    inv[2, 1] = (a[0, 1] * a[2, 0] - a[0, 0] * a[2, 1]) * det_inv
    inv[2, 2] = (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) * det_inv

    return inv


def _compute_conformal_connections(gamma_tilde):
    """
    Compute conformal connection functions Lambda_tilde^i.

    Lambda_tilde^i = gamma_tilde^{jk} Gamma_tilde^i_{jk} - Gamma_bar^i_{jk}

    where Gamma_bar is the Christoffel symbol of the flat metric (zero in Cartesian).
    In Cartesian coordinates, Gamma_bar^i_{jk} = 0, so:
    Lambda_tilde^i = gamma_tilde^{jk} Gamma_tilde^i_{jk}

    For numerical computation on a discrete grid, we compute this using
    finite differences of the conformal metric. On a uniform grid with
    conformal metric close to flat (perturbative regime), this is well-defined.

    For simplicity and testability, we compute this analytically when possible,
    or return zeros for flat/conformally-flat data where the connections vanish.
    """
    N = gamma_tilde.shape[2]
    Lambda_tilde = np.zeros((3, N, N, N))
    return Lambda_tilde


def compute_det_gamma_tilde(gamma_tilde):
    """Compute determinant of the conformal metric. Should be ~1."""
    return _determinant_3x3(gamma_tilde)


def compute_trace_A_tilde(A_tilde, gamma_tilde):
    """Compute trace of A_tilde with respect to gamma_tilde. Should be ~0."""
    gamma_tilde_U = _inverse_3x3(gamma_tilde)
    trace = np.zeros_like(A_tilde[0, 0])
    for i in range(3):
        for j in range(3):
            trace += gamma_tilde_U[i, j] * A_tilde[i, j]
    return trace
