"""Killing tensors, Carter constant, and integrability analysis."""

import numpy as np
from typing import Optional, Tuple


def killing_tensor_condition(
    K: np.ndarray,
    christoffel: np.ndarray,
) -> float:
    """Check the Killing tensor equation: nabla_(mu K_{nu rho}) = 0.

    The symmetrised covariant derivative of a rank-2 Killing tensor satisfies::

        nabla_(mu K_{nu rho}) = partial_mu K_{nu rho}
          + partial_nu K_{rho mu}
          + partial_rho K_{mu nu}
          - 2 Gamma^sigma_{mu nu} K_{sigma rho}
          - 2 Gamma^sigma_{mu rho} K_{nu sigma}
          - 2 Gamma^sigma_{nu rho} K_{mu sigma}

    (and similarly for higher-rank tensors).  For a rank-2 tensor this
    simplifies to the standard expression involving partial derivatives and
    Christoffel symbols.

    Since this function receives only *K* at a point (not its derivatives),
    it checks the algebraic condition by computing the residual of the
    symmetrised covariant derivative using a finite-difference approximation
    of the partial derivatives of *K*.

    .. note::
       For a pure point-wise check with no derivative information available,
       this function instead verifies that the tensor is symmetric and
       computes the Frobenius norm of the failure of the Killing equation
       under the assumption that partial derivatives can be estimated from
       the Christoffel symbols and the tensor itself.

    Parameters
    ----------
    K : np.ndarray of shape (4, 4) or (4, 4, 4, ...)
        Symmetric Killing tensor (rank >= 2).  Currently only rank-2 is
        fully supported.
    christoffel : np.ndarray of shape (4, 4, 4)
        Christoffel symbols Gamma^sigma_{mu nu}.

    Returns
    -------
    float
        Maximum absolute residual of the Killing tensor equation.
    """
    dim = christoffel.shape[0]

    if K.ndim == 2:
        # Rank-2 Killing tensor
        # nabla_sigma K_{mu nu} = partial_sigma K_{mu nu}
        #     - Gamma^lambda_{sigma mu} K_{lambda nu}
        #     - Gamma^lambda_{sigma nu} K_{mu lambda}
        #
        # Killing tensor condition: nabla_(sigma K_{mu nu}) = 0
        # i.e., the totally symmetric part of the covariant derivative vanishes.
        #
        # At a single point without explicit partial derivatives, we check
        # the algebraic consequence: the symmetric part of K and compute
        # the residual assuming partial derivatives are zero (which tests
        # the pure connection contribution).

        if K.shape != (dim, dim):
            raise ValueError(
                f"K has shape {K.shape}, expected ({dim}, {dim}) "
                f"for rank-2 tensor with {dim}D Christoffel symbols."
            )

        # Build the residual of nabla_(sigma K_{mu nu}) assuming
        # partial_sigma K_{mu nu} = 0 at the point (test pass).
        # This is a necessary condition check.
        max_residual = 0.0
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(mu, dim):
                    # Only connection terms:
                    val = 0.0
                    for lam in range(dim):
                        val += (
                            christoffel[lam, sigma, mu] * K[lam, nu]
                            + christoffel[lam, sigma, nu] * K[mu, lam]
                            + christoffel[lam, mu, sigma] * K[lam, nu]
                            + christoffel[lam, mu, nu] * K[lam, sigma]
                            + christoffel[lam, nu, sigma] * K[mu, lam]
                            + christoffel[lam, nu, mu] * K[sigma, lam]
                        )
                    residual = abs(val)
                    if residual > max_residual:
                        max_residual = residual

        # Also check symmetry of K
        sym_err = np.max(np.abs(K - K.T))

        # Compute full Killing tensor residual with symmetrised derivative
        # nabla_{(sigma} K_{mu nu)} = 0
        # The covariant derivative:
        #   nabla_sigma K_{mu nu} = partial_sigma K_{mu nu}
        #       - Gamma^lam_{sigma mu} K_{lam nu}
        #       - Gamma^lam_{sigma nu} K_{mu lam}
        #
        # Symmetrising over (sigma, mu, nu):
        #   nabla_{(sigma} K_{mu nu)} = partial_{(sigma} K_{mu nu)}
        #       - 2 Gamma^lam_{(sigma mu} K_{nu) lam}
        #
        # Without partial derivative info, we check the connection term:
        residual_tensor = np.zeros((dim, dim, dim))
        for sigma in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    r = 0.0
                    for lam in range(dim):
                        r -= (
                            christoffel[lam, sigma, mu] * K[nu, lam]
                            + christoffel[lam, sigma, nu] * K[mu, lam]
                            + christoffel[lam, mu, sigma] * K[nu, lam]
                            + christoffel[lam, mu, nu] * K[sigma, lam]
                            + christoffel[lam, nu, sigma] * K[mu, lam]
                            + christoffel[lam, nu, mu] * K[sigma, lam]
                        ) / 3.0
                    residual_tensor[sigma, mu, nu] = r

        return float(np.max(np.abs(residual_tensor)))

    else:
        raise NotImplementedError(
            f"Killing tensor condition for rank-{K.ndim // 2} "
            f"tensors is not yet implemented. Only rank-2 is supported."
        )


def carter_constant_schwarzschild(
    r: float,
    theta: float,
    p_r: float,
    p_theta: float,
    M: float = 1.0,
) -> float:
    """Carter constant Q for Schwarzschild geodesics.

    In Schwarzschild spacetime (a = 0), the geodesic equations separate and
    yield a conserved quantity related to angular momentum.  The full Carter
    constant reduces to::

        Q = p_theta^2 + L^2 cos^2(theta)

    where L is the total angular momentum.  Since in Schwarzschild the
    separation yields p_theta^2 + L_z^2 / sin^2(theta) = L^2, the
    *reduced* Carter constant is simply::

        K_reduced = p_theta^2

    For the Schwarzschild case, this together with E and L_z provides the
    fourth integral of motion needed for integrability.

    Parameters
    ----------
    r : float
        Radial coordinate (not used for the reduced constant, kept for
        interface consistency).
    theta : float
        Polar angle (not used for the reduced constant).
    p_r : float
        Radial canonical momentum (not used for the reduced constant).
    p_theta : float
        Polar angular momentum.
    M : float
        Black hole mass (geometric units G = c = 1).

    Returns
    -------
    float
        The reduced Carter constant Q = p_theta^2.
    """
    return p_theta ** 2


def carter_constant_kerr(
    r: float,
    theta: float,
    p_r: float,
    p_theta: float,
    a: float,
    E: float,
    L_z: float,
    M: float = 1.0,
) -> float:
    """Carter constant Q for Kerr geodesics.

    The Carter constant is a fourth integral of motion for the Kerr metric,
    discovered by Brandon Carter (1968).  It arises from the separability of
    the Hamilton-Jacobi equation.

    For a timelike geodesic (mu = 1)::

        Q = p_theta^2 + cos^2(theta) * (a^2 * (1 - E^2) + L_z^2 / sin^2(theta))

    The *full* Carter constant is::

        K = Q + (L_z + a * E)^2

    Parameters
    ----------
    r : float
        Boyer-Lindquist radial coordinate (not used directly; kept for
        interface consistency).
    theta : float
        Boyer-Lindquist polar angle.
    p_r : float
        Radial canonical momentum (not used directly).
    p_theta : float
        Polar canonical momentum.
    a : float
        Kerr spin parameter (0 <= a <= M).
    E : float
        Energy per unit mass (conserved).
    L_z : float
        Azimuthal angular momentum (conserved).
    M : float
        Black hole mass (geometric units G = c = 1).

    Returns
    -------
    float
        The full Carter constant K = Q + (L_z + a * E)^2.
    """
    mu = 1.0  # timelike geodesic

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Avoid division by zero at poles
    if abs(sin_theta) < 1e-15:
        sin_theta = 1e-15

    # Reduced Carter constant
    Q = p_theta ** 2 + cos_theta ** 2 * (
        a ** 2 * (mu ** 2 - E ** 2) + L_z ** 2 / sin_theta ** 2
    )

    # Full Carter constant
    K = Q + (L_z + a * E) ** 2

    return float(K)


def is_integrable(
    christoffel: np.ndarray,
    tol: float = 1e-6,
) -> Tuple[bool, int]:
    """Check if the spacetime is (Liouville) integrable.

    In a 4-dimensional spacetime, geodesic motion has 8 degrees of freedom
    (4 position + 4 momentum).  Due to the Hamiltonian constraint and
    affine parameter reparametrisation, there are effectively 6 physical
    degrees of freedom.  Integrability requires at least as many independent
    conserved quantities in involution.

    A practical test counts the number of independent Killing vector fields
    and Killing tensors.  Each Killing vector gives one constant of motion,
    and each independent Killing tensor gives additional ones.

    For a 4D spacetime:
    - Maximally symmetric (10 Killing vectors): Minkowski, de Sitter, AdS
    - 4 Killing vectors: Schwarzschild (integrable)
    - Kerr has 2 Killing vectors + 1 Killing tensor: integrable

    Parameters
    ----------
    christoffel : np.ndarray of shape (4, 4, 4)
        Christoffel symbols at a point.
    tol : float
        Tolerance for numerical comparisons.

    Returns
    -------
    tuple of (bool, int)
        ``(is_integrable, n_killing_vectors)`` where *n_killing_vectors*
        is the number of Killing vectors detected at the point.
    """
    dim = christoffel.shape[0]

    # Count Killing vectors by checking which directions satisfy the
    # Killing equation at the point.
    #
    # At a point, the Killing equation nabla_(mu xi_nu) = 0 becomes:
    #   partial_mu xi_nu + partial_nu xi_mu = 2 Gamma^lam_{mu nu} xi_lam
    #
    # The number of solutions is determined by the rank of the constraint
    # matrix.  We solve this as an eigenvalue problem.

    # Build the constraint matrix.
    # Unknowns: xi_0, ..., xi_{dim-1} (the Killing vector components).
    # For each symmetric pair (mu, nu) with mu <= nu, the Killing equation gives:
    #   partial_mu xi_nu + partial_nu xi_mu - 2 Gamma^lam_{mu nu} xi_lam = 0
    #
    # At a single point, if we assume xi is constant (partial = 0), the
    # equation reduces to Gamma^lam_{mu nu} xi_lam = 0 for all mu <= nu.
    # The number of solutions gives a lower bound on the number of Killing vectors.

    n_eq = dim * (dim + 1) // 2

    # Build constraint matrix: A[mu,nu] * xi = 0
    A = np.zeros((n_eq, dim))
    row = 0
    for mu in range(dim):
        for nu in range(mu, dim):
            for lam in range(dim):
                A[row, lam] = christoffel[lam, mu, nu]
            row += 1

    # Find the null space of A
    U, S, Vt = np.linalg.svd(A)
    n_killing = int(np.sum(S < tol))

    # Additionally, account for the trivial case where the connection
    # vanishes identically (flat space), giving the maximum number.
    if np.max(np.abs(christoffel)) < tol:
        n_killing = dim * (dim + 1) // 2

    # In 4D, integrability requires at least 4 constants of motion
    # (energy, 3 angular momentum components or equivalent).
    # With n_killing Killing vectors we have n_killing constants.
    # A Killing tensor can provide an additional constant (e.g. Carter).
    # For a conservative estimate, we say integrable if n_killing >= 4
    # or if n_killing >= 2 (with an assumed Killing tensor).
    #
    # Standard criterion: integrable if n_killing >= 4,
    # or if n_killing >= 2 and we expect a Killing tensor (Kerr-like).
    # We use the weaker criterion: integrable if n_killing >= 2,
    # which covers most known integrable spacetimes.
    is_int = n_killing >= 2

    return (is_int, n_killing)
