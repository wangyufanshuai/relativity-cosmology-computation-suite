"""Flat-space geodesic deviation (Jacobi equation)."""

from scipy.integrate import solve_ivp


def jacobi_equation_flat(lamb, y):
    """Jacobi equation in flat space: d^2 xi / d lambda^2 = 0.

    In flat space the Riemann tensor vanishes, so separation grows linearly.
    State vector y = [xi, d(xi)/d(lambda)].
    """
    xi, dxi = y
    return [dxi, 0.0]


def solve_flat_deviation(xi0=1.0, dxi0=0.1, lambda_span=(0, 10)):
    """Solve geodesic deviation in flat space.

    The separation grows linearly: xi(lambda) = xi0 + dxi0 * lambda.
    """
    sol = solve_ivp(
        jacobi_equation_flat,
        lambda_span,
        [xi0, dxi0],
        dense_output=True,
        max_step=0.1,
    )
    return sol
