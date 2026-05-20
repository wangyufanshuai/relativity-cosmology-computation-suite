"""
Initial data generators for BSSN simulations.

Provides:
  - Schwarzschild puncture data in isotropic coordinates
  - Brill wave data on moment of time symmetry
  - Binary puncture data (simplified conformal thin-sandwich)
  - Flat (Minkowski) data
  - Constraint verification utilities
"""

import numpy as np
from .bssn_variables import BSSNState, physical_to_bssn, _determinant_3x3


def make_grid(N, x_range, y_range=None, z_range=None):
    """
    Create a uniform 3D Cartesian grid.

    Parameters
    ----------
    N : int
        Number of grid points per dimension.
    x_range : tuple (x_min, x_max)
    y_range : tuple or None
    z_range : tuple or None

    Returns
    -------
    X, Y, Z : ndarray (N,N,N)
        Coordinate arrays.
    dx : float
        Grid spacing.
    """
    if y_range is None:
        y_range = x_range
    if z_range is None:
        z_range = x_range

    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    z = np.linspace(z_range[0], z_range[1], N)
    dx = x[1] - x[0] if N > 1 else 1.0

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z, dx


def flat_data(N, x_range=(-1.0, 1.0)):
    """
    Flat (Minkowski) spacetime initial data.

    Returns BSSNState with all fields set to flat-space values.
    """
    X, Y, Z, dx = make_grid(N, x_range)
    shape = (N, N, N)

    phi = np.zeros(shape)
    gamma_tilde = np.zeros((3, 3, N, N, N))
    for i in range(3):
        gamma_tilde[i, i] = 1.0  # flat metric = identity

    K = np.zeros(shape)
    A_tilde = np.zeros((3, 3, N, N, N))
    Lambda_tilde = np.zeros((3, N, N, N))
    alpha = np.ones(shape)
    beta = np.zeros((3, N, N, N))

    return BSSNState(phi, gamma_tilde, K, A_tilde, Lambda_tilde, alpha, beta), X, Y, Z, dx


def schwarzschild_puncture(N, M, x_range=(-5.0, 5.0)):
    """
    Schwarzschild puncture data in isotropic coordinates.

    The isotropic Schwarzschild metric gives:
        phi = -M / (2r)
        K_ij = 0
        alpha = 1  (will be evolved with 1+log)
        beta = 0

    Parameters
    ----------
    N : int
        Grid points per dimension.
    M : float
        Mass parameter.
    x_range : tuple
        Coordinate range.

    Returns
    -------
    BSSNState, X, Y, Z, dx
    """
    X, Y, Z, dx = make_grid(N, x_range)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    # Avoid division by zero at the puncture
    r_safe = np.maximum(r, 1e-10)

    # Isotropic Schwarzschild: conformal factor psi = 1 + M/(2r)
    # In BSSN: phi = ln(psi^4) / 4 ... actually we use:
    # The physical metric is gamma_ij = psi^4 delta_ij
    # So det(gamma) = psi^12
    # phi = ln(det gamma) / 12 = ln(psi^12) / 12 = ln(psi)
    # Wait, let's be careful:
    # gamma_ij = psi^4 delta_ij  =>  det(gamma) = psi^12
    # phi = ln(det gamma) / 12 = 12 ln(psi) / 12 = ln(psi)
    # But standard convention uses phi = ln(psi)/4 or
    # the "phi" such that e^{4phi} = psi^4, so phi = ln(psi).
    # Actually the standard BSSN definition is:
    # phi = (1/12) ln(det gamma_ij)
    # For Schwarzschild: det gamma = psi^12, so phi = ln(psi)

    psi = 1.0 + M / (2.0 * r_safe)
    phi = np.log(psi)

    # Physical metric: gamma_ij = psi^4 delta_ij
    e4phi = np.exp(4.0 * phi)  # = psi^4
    gamma_ij = np.zeros((3, 3, N, N, N))
    for i in range(3):
        gamma_ij[i, i] = e4phi

    # Time-symmetric: K_ij = 0
    K_ij = np.zeros((3, 3, N, N, N))

    alpha = np.ones((N, N, N))
    beta = np.zeros((3, N, N, N))

    return physical_to_bssn(gamma_ij, K_ij, alpha, beta), X, Y, Z, dx


def schwarzschild_puncture_bssn_direct(N, M, x_range=(-5.0, 5.0)):
    """
    Schwarzschild puncture data constructed directly in BSSN variables.

    This avoids potential numerical issues with the roundtrip through
    physical_to_bssn for the analytic solution.

    Parameters
    ----------
    N : int
    M : float
    x_range : tuple

    Returns
    -------
    BSSNState, X, Y, Z, dx
    """
    X, Y, Z, dx = make_grid(N, x_range)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_safe = np.maximum(r, 1e-10)

    # Conformal factor
    psi = 1.0 + M / (2.0 * r_safe)
    phi = np.log(psi)

    # Conformal metric is flat (delta_ij) for isotropic Schwarzschild
    gamma_tilde = np.zeros((3, 3, N, N, N))
    for i in range(3):
        gamma_tilde[i, i] = 1.0

    # Time symmetry: K = 0, A_tilde = 0
    K = np.zeros((N, N, N))
    A_tilde = np.zeros((3, 3, N, N, N))

    # Conformal connections vanish for conformally flat metric
    Lambda_tilde = np.zeros((3, N, N, N))

    alpha = np.ones((N, N, N))
    beta = np.zeros((3, N, N, N))

    return BSSNState(phi, gamma_tilde, K, A_tilde, Lambda_tilde, alpha, beta), X, Y, Z, dx


def brill_wave(N, amplitude=0.1, x_range=(-2.0, 2.0)):
    """
    Brill wave initial data on a moment of time symmetry.

    Uses the Brill ansatz with a conformal factor:
        psi = 1 + A * q(r) * exp(-r^2/sigma^2)
    where q(r) encodes the quadrupole distortion.

    On the moment of time symmetry: K_ij = 0.

    Parameters
    ----------
    N : int
    amplitude : float
        Wave amplitude.
    x_range : tuple

    Returns
    -------
    BSSNState, X, Y, Z, dx
    """
    X, Y, Z, dx = make_grid(N, x_range)
    shape = (N, N, N)

    r2 = X**2 + Y**2 + Z**2
    r = np.sqrt(r2)
    rho2 = X**2 + Y**2

    sigma = 1.0

    # Quadrupole function
    q = (rho2 - 2.0 * Z**2) / (r2 + 0.01)
    q /= (r2 + 1.0)

    psi = 1.0 + amplitude * q * np.exp(-r2 / sigma**2)
    psi = np.maximum(psi, 0.1)  # Ensure positivity

    phi = np.log(psi)

    # Physical metric with perturbation
    e4phi = np.exp(4.0 * phi)
    gamma_ij = np.zeros((3, 3, N, N, N))
    for i in range(3):
        gamma_ij[i, i] = e4phi

    # Time symmetry
    K_ij = np.zeros((3, 3, N, N, N))

    alpha = np.ones(shape)
    beta = np.zeros((3, N, N, N))

    return physical_to_bssn(gamma_ij, K_ij, alpha, beta), X, Y, Z, dx


def binary_puncture(N, M1, M2, separation, x_range=(-10.0, 10.0)):
    """
    Simplified binary puncture initial data using superposition of two
    Schwarzschild punctures (conformal thin-sandwich approximation).

    This is a simplified version that superposes two punctures without
    solving the full constraint equations. It is valid for well-separated
    black holes.

    Parameters
    ----------
    N : int
    M1, M2 : float
        Masses of the two punctures.
    separation : float
        Coordinate separation along x-axis.
    x_range : tuple

    Returns
    -------
    BSSNState, X, Y, Z, dx
    """
    X, Y, Z, dx = make_grid(N, x_range)

    # Positions of the two punctures
    x1 = separation / 2.0
    x2 = -separation / 2.0

    r1 = np.sqrt((X - x1)**2 + Y**2 + Z**2)
    r2 = np.sqrt((X - x2)**2 + Y**2 + Z**2)
    r1_safe = np.maximum(r1, 1e-10)
    r2_safe = np.maximum(r2, 1e-10)

    # Superposed conformal factor (Bowen-York type, simplified)
    psi = 1.0 + M1 / (2.0 * r1_safe) + M2 / (2.0 * r2_safe)
    psi = np.maximum(psi, 0.01)
    phi = np.log(psi)

    # Conformally flat
    gamma_tilde = np.zeros((3, 3, N, N, N))
    for i in range(3):
        gamma_tilde[i, i] = 1.0

    # Time symmetry (no linear momentum for simplicity)
    K = np.zeros((N, N, N))
    A_tilde = np.zeros((3, 3, N, N, N))
    Lambda_tilde = np.zeros((3, N, N, N))

    alpha = np.ones((N, N, N))
    beta = np.zeros((3, N, N, N))

    return BSSNState(phi, gamma_tilde, K, A_tilde, Lambda_tilde, alpha, beta), X, Y, Z, dx
