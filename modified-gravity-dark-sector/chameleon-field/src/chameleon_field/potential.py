"""Chameleon potential and effective potential."""

import numpy as np
from scipy.optimize import minimize_scalar


# Reduced Planck mass in eV (for natural units)
M_PL = 2.435e27  # eV


def chameleon_potential(phi, Lambda, n, beta, rho):
    """
    Chameleon potential V_eff(phi) = V(phi) + beta * rho * exp(beta * phi / M_Pl).

    The self-interaction potential is V(phi) = Lambda^(4+n) / phi^n,
    and the matter coupling gives an effective potential.

    Parameters
    ----------
    phi : float or array
        Scalar field value (eV).
    Lambda : float
        Energy scale of the self-interaction potential (eV).
    n : float
        Power-law index of the Ratra-Peebles potential.
    beta : float
        Matter coupling constant.
    rho : float
        Ambient matter density (eV^4).

    Returns
    -------
    float or array
        Effective potential value.
    """
    phi = np.asarray(phi, dtype=float)
    # Clamp phi to be positive to avoid division by zero
    phi_safe = np.maximum(phi, 1e-100)
    V_self = (Lambda ** (4 + n)) / (phi_safe ** n)
    V_matter = beta * rho * np.exp(beta * phi_safe / M_PL)
    return V_self + V_matter


def effective_potential(phi, Lambda, n, beta, rho):
    """
    Effective potential V_eff(phi) = Lambda^(4+n) / phi^n + beta*rho*exp(beta*phi/M_Pl).

    This is the same as chameleon_potential but provided as a named function
    for clarity when discussing the effective potential and its minimum.

    Parameters
    ----------
    phi : float or array
        Scalar field value (eV).
    Lambda : float
        Energy scale (eV).
    n : float
        Power-law index.
    beta : float
        Coupling constant.
    rho : float
        Matter density (eV^4).

    Returns
    -------
    float or array
        Effective potential value.
    """
    return chameleon_potential(phi, Lambda, n, beta, rho)


def minimize_effective_potential(rho, Lambda, n, beta):
    """
    Find the field value phi_min that minimizes the effective potential.

    Parameters
    ----------
    rho : float
        Matter density (eV^4).
    Lambda : float
        Energy scale (eV).
    n : float
        Power-law index.
    beta : float
        Coupling constant.

    Returns
    -------
    float
        The field value at the minimum of the effective potential.
    """
    # Analytical minimum from dV_eff/dphi = 0:
    # n * Lambda^(4+n) / phi^(n+1) = beta^2 * rho / M_Pl * exp(beta*phi/M_Pl)
    # For small phi (beta*phi/M_Pl << 1), approximate:
    # phi_min ~ (n * M_Pl * Lambda^(4+n) / (beta^2 * rho))^(1/(n+1))
    phi_approx = (n * M_PL * Lambda ** (4 + n) / (beta ** 2 * rho)) ** (1.0 / (n + 1))

    # Use numerical minimization around the analytical estimate
    log_phi_lo = np.log10(max(phi_approx * 1e-6, 1e-100))
    log_phi_hi = np.log10(phi_approx * 1e6)

    result = minimize_scalar(
        lambda lp: effective_potential(10**lp, Lambda, n, beta, rho),
        bounds=(log_phi_lo, log_phi_hi),
        method='bounded',
    )
    return 10 ** result.x
