"""Chameleon field profiles and thin-shell effect."""

import numpy as np
from .potential import minimize_effective_potential, M_PL


def thin_shell_parameter(R, rho_obj, rho_env, beta, M_Pl=M_PL):
    """
    Compute the thin-shell parameter Delta_R / R for a spherical object.

    The thin-shell parameter determines how much of the object's mass
    contributes to the external chameleon field. When it is << 1, the
    object is in the thin-shell regime (chameleon force is suppressed).

    Parameters
    ----------
    R : float
        Radius of the object (m).
    rho_obj : float
        Density of the object (eV^4).
    rho_env : float
        Density of the environment (eV^4).
    beta : float
        Coupling constant.
    M_Pl : float
        Reduced Planck mass (eV).

    Returns
    -------
    float
        Thin-shell parameter (dimensionless).
    """
    phi_min_obj = minimize_effective_potential(rho_obj, 1e-3, 1, beta)
    phi_min_env = minimize_effective_potential(rho_env, 1e-3, 1, beta)
    delta_phi = abs(phi_min_env - phi_min_obj)
    # Thin-shell parameter: (phi_env - phi_obj) / (beta * M_Pl * rho_obj * R^2 / 3)
    # Simplified form:
    return delta_phi / (beta * M_Pl)


def chameleon_profile(r, R_obj, rho_obj, rho_env, beta, Lambda, n):
    """
    Compute the chameleon field profile phi(r) outside a spherical object.

    Parameters
    ----------
    r : float or array
        Radial distance from center of object.
    R_obj : float
        Radius of the object.
    rho_obj : float
        Object density.
    rho_env : float
        Environment density.
    beta : float
        Coupling constant.
    Lambda : float
        Energy scale.
    n : float
        Power-law index.

    Returns
    -------
    float or array
        Field value at distance r.
    """
    r = np.asarray(r, dtype=float)

    phi_min_env = minimize_effective_potential(rho_env, Lambda, n, beta)
    phi_min_obj = minimize_effective_potential(rho_obj, Lambda, n, beta)

    # Outside the object (r > R), the Yukawa-like profile
    # phi(r) = phi_min_env + (phi_min_obj - phi_min_env) * R/r * exp(-m(r-R))
    # For simplicity, use mass scale from environment:
    # m^2 ~ n(n+1) Lambda^(4+n) / phi_min^(n+2)
    phi_min = phi_min_env
    m_eff = np.sqrt(n * (n + 1) * Lambda ** (4 + n) / (phi_min ** (n + 2) + 1e-200))

    outside = r > R_obj
    result = np.full_like(r, phi_min_obj, dtype=float)

    if np.any(outside):
        r_out = r[outside]
        result[outside] = (
            phi_min_env
            + (phi_min_obj - phi_min_env)
            * (R_obj / r_out)
            * np.exp(-m_eff * (r_out - R_obj))
        )

    return result
