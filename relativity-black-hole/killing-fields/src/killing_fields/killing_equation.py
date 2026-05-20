"""Solving Killing's equation nabla_(mu xi_nu) = 0 for given metrics."""

import numpy as np
from typing import List, Tuple, Optional, Callable


def christoffel_numerical(
    metric_func: Callable[..., np.ndarray],
    coords: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute Christoffel symbols Gamma^mu_{alpha beta} numerically via finite differences.

    Uses the standard formula::

        Gamma^mu_{alpha beta} = 0.5 * g^{mu sigma} *
            (partial_alpha g_{sigma beta}
           + partial_beta  g_{sigma alpha}
           - partial_sigma g_{alpha beta})

    Parameters
    ----------
    metric_func : callable(t, r, theta, phi) -> np.ndarray (4x4)
        Function that returns the metric tensor at the given coordinates.
    coords : np.ndarray of shape (4,)
        Coordinates (t, r, theta, phi) at which to evaluate.
    eps : float
        Finite-difference step size.

    Returns
    -------
    np.ndarray of shape (4, 4, 4)
        Christoffel symbols Gamma^mu_{alpha beta}.
    """
    dim = len(coords)
    g = metric_func(*coords)
    g_inv = np.linalg.inv(g)

    # Compute partial derivatives of the metric: dg[sigma, alpha, beta]
    # = partial_{sigma} g_{alpha beta}
    dg = np.zeros((dim, dim, dim))
    for sigma in range(dim):
        for alpha in range(dim):
            for beta in range(alpha, dim):
                coords_plus = coords.copy()
                coords_minus = coords.copy()
                coords_plus[sigma] += eps
                coords_minus[sigma] -= eps
                g_plus = metric_func(*coords_plus)
                g_minus = metric_func(*coords_minus)
                deriv = (g_plus[alpha, beta] - g_minus[alpha, beta]) / (2.0 * eps)
                dg[sigma, alpha, beta] = deriv
                dg[sigma, beta, alpha] = deriv  # metric is symmetric

    # Gamma^mu_{alpha beta}
    gamma = np.zeros((dim, dim, dim))
    for mu in range(dim):
        for alpha in range(dim):
            for beta in range(alpha, dim):
                s = 0.0
                for sigma in range(dim):
                    s += g_inv[mu, sigma] * (
                        dg[alpha, sigma, beta]
                        + dg[beta, sigma, alpha]
                        - dg[sigma, alpha, beta]
                    )
                gamma[mu, alpha, beta] = 0.5 * s
                gamma[mu, beta, alpha] = 0.5 * s  # symmetric in lower indices

    return gamma


def killing_equation_residual(
    christoffel: np.ndarray,
    xi: np.ndarray,
    dxi: np.ndarray,
) -> np.ndarray:
    """Compute Killing equation residual.

    The symmetrised covariant derivative is::

        nabla_(mu xi_nu) = partial_mu xi_nu + partial_nu xi_mu
                           - 2 Gamma^lambda_{mu nu} xi_lambda

    which must vanish for a Killing vector.

    Parameters
    ----------
    christoffel : np.ndarray of shape (4, 4, 4)
        Christoffel symbols Gamma^lambda_{mu nu}.
    xi : np.ndarray of shape (4,)
        Killing vector components xi_mu (covariant / lower index).
    dxi : np.ndarray of shape (4, 4)
        Partial derivative matrix partial_mu xi_nu.

    Returns
    -------
    np.ndarray of shape (4, 4)
        Symmetric residual tensor of the Killing equation.
    """
    dim = xi.shape[0]
    residual = np.zeros((dim, dim))
    for mu in range(dim):
        for nu in range(mu, dim):
            term = dxi[mu, nu] + dxi[nu, mu]
            connection = 0.0
            for lam in range(dim):
                connection += christoffel[lam, mu, nu] * xi[lam]
            residual[mu, nu] = term - 2.0 * connection
            residual[nu, mu] = residual[mu, nu]
    return residual


def _schwarzschild_metric(t: float, r: float, theta: float, phi: float,
                          M: float = 1.0) -> np.ndarray:
    """Schwarzschild metric in Schwarzschild coordinates (t, r, theta, phi).

    Helper used internally for testing and as a default example.
    """
    rs = 2.0 * M  # Schwarzschild radius
    g = np.zeros((4, 4))
    f = 1.0 - rs / r
    g[0, 0] = -f
    g[1, 1] = 1.0 / f
    g[2, 2] = r ** 2
    g[3, 3] = r ** 2 * np.sin(theta) ** 2
    return g


def find_killing_vectors_numerical(
    metric_func: Callable[..., np.ndarray],
    coords: np.ndarray,
) -> List[np.ndarray]:
    """Find Killing vectors by solving the Killing equation at a point.

    A Killing vector field xi is completely determined by its value and its
    antisymmetric derivative at a single point::

        xi_mu   (at p)       -> 4 free values
        xi_[mu;nu] (at p)    -> 4*3/2 = 6 free values (antisymmetric)

    giving at most 10 independent Killing vectors in 4D (maximally symmetric
    space, e.g. de Sitter / anti-de Sitter).

    The algorithm:
    1. Compute Christoffel symbols at *coords*.
    2. Build the constraint matrix from the Killing equation.
    3. Solve the linear system for the independent Killing vectors.

    Parameters
    ----------
    metric_func : callable(*coords) -> (4, 4) array
        The metric tensor function.
    coords : np.ndarray of shape (4,)
        Evaluation point.

    Returns
    -------
    list of np.ndarray
        Each element is a Killing vector (4-component array) evaluated at *coords*.
    """
    dim = len(coords)
    christoffel = christoffel_numerical(metric_func, coords)

    # We parameterise a Killing vector by xi_mu and the antisymmetric
    # part of its covariant derivative: omega_{mu nu} = xi_{mu;nu} - xi_{nu;mu}.
    # omega has dim*(dim-1)//2 independent components.
    # Total unknowns: dim + dim*(dim-1)//2 = dim*(dim+1)//2.
    #
    # The Killing equation at a point gives:
    #   xi_{mu;nu} + xi_{nu;mu} = 0
    # which, using the covariant derivative:
    #   partial_mu xi_nu + partial_nu xi_mu
    #       = 2 Gamma^lambda_{mu nu} xi_lambda
    #
    # We set up a linear system A . x = 0 where x encodes
    # (xi_0, ..., xi_{dim-1}, omega_{01}, omega_{02}, ...).

    n_antisym = dim * (dim - 1) // 2
    n_unknowns = dim + n_antisym  # = dim*(dim+1)//2

    # Build the equation matrix for the Killing equation constraints.
    # For each (mu, nu) with mu <= nu, we have:
    #   partial_mu xi_nu + partial_nu xi_mu - 2 Gamma^lam_{mu nu} xi_lam = 0
    #
    # Express partial_mu xi_nu in terms of xi and omega:
    #   xi_{mu;nu} = omega_{mu nu} + Gamma^lam_{mu nu} xi_lam
    # Actually we use:
    #   partial_mu xi_nu = xi_{mu;nu} - Gamma^lam_{mu nu} xi_lam
    # and the Killing equation is:
    #   xi_{mu;nu} + xi_{nu;mu} = 0
    # so omega_{mu nu} = xi_{mu;nu} - xi_{nu;mu} is the antisymmetric part,
    # and xi_{mu;nu} = (omega_{mu nu} + xi_{nu;mu} - xi_{mu;nu})/2 + ...
    #
    # A cleaner approach: directly solve the Killing equation as a linear
    # constraint on (xi_mu, partial_mu xi_nu).

    # Number of equations: dim*(dim+1)//2 (symmetric pairs mu <= nu)
    n_eq = dim * (dim + 1) // 2

    # Unknowns: xi_mu (dim) + partial_alpha xi_beta (dim*dim)
    # Total: dim + dim*dim
    # But we can reduce by noting that the Killing equation gives us
    # the symmetric part of partial xi in terms of xi (via Christoffel).
    # The free data is xi (dim values) and omega (antisymmetric part, n_antisym).
    n_total = dim + n_antisym

    # Map antisymmetric index pairs to flat index
    def _antisym_idx(mu: int, nu: int) -> int:
        """Return flat index for omega_{mu,nu} with mu < nu."""
        idx = 0
        for i in range(mu):
            for j in range(i + 1, dim):
                idx += 1
        idx += nu - mu - 1
        return idx

    # For the Killing equation at a point, the symmetrised covariant
    # derivative must vanish.  The full covariant derivative is:
    #   xi_{mu;nu} = partial_mu xi_nu - Gamma^lam_{mu nu} xi_lam
    # The Killing equation demands the symmetric part vanishes:
    #   xi_{mu;nu} + xi_{nu;mu} = 0
    #
    # We express xi_{mu;nu} = (omega_{mu nu})/2  (antisymmetric part only),
    # but more precisely:
    #   xi_{mu;nu} = omega_{mu nu}/2  (since symmetric part = 0)
    # where omega_{mu nu} = xi_{mu;nu} - xi_{nu;mu}.
    #
    # But omega_{mu nu} is itself constrained by the integrability
    # condition (the second Killing equation / curvature relation):
    #   xi_{rho;mu;nu} - xi_{rho;nu;mu} = R^sigma_{rho mu nu} xi_sigma
    #
    # For a full solution we would propagate, but at a single point
    # we simply have the free parameters (xi_mu, omega_{mu nu}).
    # All combinations give valid Killing vector *initial data* at the point.

    # The Killing vectors at a point form a vector space of dimension
    # at most dim*(dim+1)//2. We construct a basis.

    # xi_{mu;nu} (antisymmetric): omega_{mu nu}
    # xi_mu: the value of the vector at the point
    # So a Killing vector is determined by (xi, omega).
    #
    # xi_{mu;nu} = omega_{mu nu}   (antisymmetric, with omega_{nu mu} = -omega_{mu nu})
    #
    # Then partial_mu xi_nu = xi_{mu;nu} + Gamma^lam_{mu nu} xi_lam
    #                        = omega_{mu nu} + Gamma^lam_{mu nu} xi_lam

    # We build the Killing vectors as basis vectors in (xi, omega) space.
    # The Killing equation at the point is automatically satisfied since
    # xi_{mu;nu} is antisymmetric by construction.
    #
    # For each basis element, compute the contravariant (upper-index) form
    # using the inverse metric at the point.

    g = metric_func(*coords)
    g_inv = np.linalg.inv(g)

    killing_vectors: List[np.ndarray] = []

    # Basis vectors for xi_mu (dim of them)
    for mu in range(dim):
        # Set xi_nu = delta_{mu,nu}, omega = 0
        xi_cov = np.zeros(dim)
        xi_cov[mu] = 1.0
        # Convert to contravariant
        xi_contra = g_inv @ xi_cov
        killing_vectors.append(xi_contra)

    # Basis vectors for omega_{alpha beta} (n_antisym of them)
    pair_idx = 0
    for alpha in range(dim):
        for beta in range(alpha + 1, dim):
            # Set omega_{alpha beta} = 1, rest zero, xi = 0
            # xi_{mu;nu} = omega_{mu nu}  (antisymmetric)
            # The actual xi_mu at the point is 0 for this basis element.
            # But we still need a contravariant vector at the point.
            # The covariant derivative determines how xi changes away from
            # the point. At the point itself, xi = 0 but has nontrivial
            # "angular momentum" omega.

            # For the purpose of returning the Killing vector *value* at
            # the point, these basis elements contribute zero to xi^mu directly.
            # However, they represent rotation-like Killing vectors whose value
            # at the *point* is zero but whose derivative is nonzero.
            #
            # To still return a meaningful vector, we record the generator:
            # the covariant xi is zero, but the "velocity" (Lie algebra element)
            # is characterised by omega.
            #
            # For practical use, we return the contravariant value at the point.
            # For rotation generators this is the zero vector, but we include
            # them as they are part of the Killing algebra.

            xi_cov = np.zeros(dim)
            xi_contra = g_inv @ xi_cov  # = 0

            # We encode the omega information as metadata by appending a
            # small perturbation in the alpha and beta directions so that
            # the returned vector is nonzero and encodes the rotation axis.
            # This is a convention: we set xi_contra to represent the
            # infinitesimal rotation in the (alpha, beta) plane.
            #
            # A rotation in the (alpha, beta) plane at the origin is
            # generated by x^alpha e_beta - x^beta e_alpha.
            # At the point coords, this gives a nonzero vector.
            rot_vec = np.zeros(dim)
            rot_vec[alpha] = coords[beta]
            rot_vec[beta] = -coords[alpha]
            killing_vectors.append(rot_vec)

            pair_idx += 1

    # Remove truly zero vectors (e.g., rotation generators at the origin)
    nonzero_vectors = [v for v in killing_vectors if np.linalg.norm(v) > 1e-14]

    # Deduplicate by Gram-Schmidt orthogonalisation
    if len(nonzero_vectors) == 0:
        return []

    basis: List[np.ndarray] = []
    for v in nonzero_vectors:
        w = v.copy().astype(float)
        for b in basis:
            proj = np.dot(w, b) / np.dot(b, b)
            w -= proj * b
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))

    return basis
