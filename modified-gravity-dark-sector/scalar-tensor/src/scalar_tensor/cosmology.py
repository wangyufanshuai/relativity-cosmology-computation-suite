"""Brans-Dicke cosmology and frame transformations."""

import numpy as np
from scipy.integrate import solve_ivp


def brans_dicke_friedmann(a, phi, omega, rho):
    """
    Brans-Dicke modified Friedmann equations.

    The Hubble parameter in BD theory satisfies:
    H^2 = (8*pi / (3*phi)) * rho
         - omega / 6 * (phi_dot / phi)^2
         + H * phi_dot / phi

    Parameters
    ----------
    a : float
        Scale factor.
    phi : float
        Scalar field value.
    omega : float
        Brans-Dicke coupling.
    rho : float
        Energy density.

    Returns
    -------
    float
        Hubble parameter H.
    """
    # Simplified: for slow roll (phi_dot << H*phi), H^2 ~ 8*pi*rho / (3*phi)
    # Full expression requires phi_dot which we approximate
    H_squared = (8 * np.pi / (3 * phi)) * rho
    if H_squared < 0:
        return 0.0
    return np.sqrt(H_squared)


def conformal_transform(g_mn, phi):
    """
    Conformal transformation from Jordan frame to Einstein frame.

    g_mn^(Einstein) = (phi / phi_0) * g_mn^(Jordan)

    where phi_0 is some reference field value.

    Parameters
    ----------
    g_mn : numpy.ndarray
        Metric tensor in Jordan frame (4x4 matrix).
    phi : float
        Scalar field value.

    Returns
    -------
    numpy.ndarray
        Metric tensor in Einstein frame.
    """
    g_mn = np.asarray(g_mn, dtype=float)
    phi_0 = 1.0  # Reference field value
    omega_factor = phi / phi_0
    return omega_factor * g_mn


def inverse_conformal_transform(g_mn_E, phi):
    """
    Inverse conformal transform: Einstein frame -> Jordan frame.

    Parameters
    ----------
    g_mn_E : numpy.ndarray
        Metric tensor in Einstein frame.
    phi : float
        Scalar field value.

    Returns
    -------
    numpy.ndarray
        Metric tensor in Jordan frame.
    """
    g_mn_E = np.asarray(g_mn_E, dtype=float)
    phi_0 = 1.0
    return g_mn_E / (phi / phi_0)


def solve_background(omega, phi0, a_range):
    """
    Solve the Brans-Dicke background cosmology.

    Parameters
    ----------
    omega : float
        Brans-Dicke coupling parameter.
    phi0 : float
        Initial scalar field value.
    a_range : tuple
        (a_start, a_end) range of scale factor.

    Returns
    -------
    dict
        Solution with 'a', 'phi', 'H' arrays.
    """
    a_start, a_end = a_range

    # ODE system: d(ln a)/dN = 1 (using N = ln a as time variable)
    # d(phi)/dN = phi_dot / H / a
    # Simplified evolution: phi evolves slowly for large omega
    def rhs(N, y):
        a = np.exp(N)
        phi = y[0]
        # Matter-dominated: rho = rho_0 / a^3
        rho_0 = 1.0
        rho = rho_0 / a ** 3
        H = brans_dicke_friedmann(a, phi, omega, rho)
        # phi evolution: phi_ddot + 3H*phi_dot = (8*pi*rho - 2*omega*phi_dot^2/phi) / (2*omega + 3)
        # For large omega, phi is nearly constant
        # Simplified: dphi/dN ~ -phi * 2 / ((2*omega+3) * a) * something small
        dphi_dN = -2 * phi / (2 * omega + 3) * (1.0 / (1.0 + 1e-10))
        return [dphi_dN]

    N_start = np.log(max(a_start, 1e-10))
    N_end = np.log(a_end)
    N_eval = np.linspace(N_start, N_end, 200)

    sol = solve_ivp(rhs, [N_start, N_end], [phi0], t_eval=N_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)

    a_vals = np.exp(sol.t)
    phi_vals = sol.y[0]

    rho_0 = 1.0
    H_vals = np.array([
        brans_dicke_friedmann(a, phi, omega, rho_0 / a ** 3)
        for a, phi in zip(a_vals, phi_vals)
    ])

    return {
        "a": a_vals,
        "phi": phi_vals,
        "H": H_vals,
    }
