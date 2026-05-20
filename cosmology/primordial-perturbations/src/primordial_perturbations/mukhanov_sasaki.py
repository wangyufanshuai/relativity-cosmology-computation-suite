"""Mukhanov-Sasaki equation solver for primordial perturbations.

Solves the equation:
    u_k'' + (k^2 - z''/z) u_k = 0
where z = a * phi_dot / H, primes denote conformal time derivatives.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def z_function(phi_dot: float, a: float, H: float) -> float:
    """Compute the Mukhanov-Sasaki variable z = a * phi_dot / H.

    Parameters
    ----------
    phi_dot : float
        Time derivative of the inflaton field dphi/dt.
    a : float
        Scale factor.
    H : float
        Hubble parameter.

    Returns
    -------
    float
        The variable z = a * phi_dot / H.
    """
    return a * phi_dot / H


def z_pp_over_z(
    a_array: np.ndarray,
    phi_dot_array: np.ndarray,
    H_array: np.ndarray,
    tau_array: np.ndarray,
) -> interp1d:
    """Compute z''/z via numerical differentiation and return an interpolator.

    Given arrays of background quantities evaluated at discrete conformal times,
    compute z(tau) = a(tau) * phi_dot(tau) / H(tau), then numerically differentiate
    twice to obtain z''/z as a function of conformal time.

    Parameters
    ----------
    a_array : np.ndarray
        Scale factor values at each conformal time.
    phi_dot_array : np.ndarray
        Inflaton velocity values at each conformal time.
    H_array : np.ndarray
        Hubble parameter values at each conformal time.
    tau_array : np.ndarray
        Conformal time array (must be sorted in ascending order).

    Returns
    -------
    scipy.interpolate.interp1d
        An interpolating function f(tau) -> z''(tau)/z(tau).
    """
    z_array = a_array * phi_dot_array / H_array

    # Compute first derivative z' using central differences
    dz = np.gradient(z_array, tau_array, edge_order=2)

    # Compute second derivative z'' using central differences
    ddz = np.gradient(dz, tau_array, edge_order=2)

    # Compute z''/z, handling potential zeros in z
    with np.errstate(divide="ignore", invalid="ignore"):
        zpp_over_z = np.where(np.abs(z_array) > 0, ddz / z_array, 0.0)

    # Return cubic interpolator; extrapolate using boundary values
    interp_func = interp1d(
        tau_array, zpp_over_z, kind="cubic", fill_value="extrapolate"
    )
    return interp_func


def ms_equation(
    tau: float,
    y: np.ndarray,
    k: float,
    zpp_over_z_func,
) -> np.ndarray:
    """Right-hand side of the Mukhanov-Sasaki ODE system.

    The MS equation u'' + (k^2 - z''/z) u = 0 is written as a first-order system:
        dy/dtau = [u', u''] = [u', -(k^2 - z''/z) * u]

    Parameters
    ----------
    tau : float
        Conformal time.
    y : np.ndarray
        State vector [u, u'].
    k : float
        Comoving wavenumber.
    zpp_over_z_func : callable
        Function returning z''(tau)/z(tau).

    Returns
    -------
    np.ndarray
        Derivatives [u', u''].
    """
    u, du = y
    zpp_z = zpp_over_z_func(tau)
    ddu = -(k**2 - zpp_z) * u
    return np.array([du, ddu])


def bunch_davies_ic(k: float, tau_i: float) -> np.ndarray:
    """Bunch-Davies initial conditions for the MS equation.

    In the sub-horizon limit (k >> aH, or equivalently k|tau| >> 1),
    the mode function reduces to a positive-frequency plane wave:
        u_k(tau_i) = exp(-i k tau_i) / sqrt(2k)
        u_k'(tau_i) = -i k * u_k(tau_i)

    Parameters
    ----------
    k : float
        Comoving wavenumber.
    tau_i : float
        Initial conformal time (should satisfy k|tau_i| >> 1).

    Returns
    -------
    np.ndarray
        Initial state vector [u, u'] (complex-valued).
    """
    u = np.exp(-1j * k * tau_i) / np.sqrt(2.0 * k)
    du = -1j * k * u
    return np.array([u, du])


def integrate_mode(
    k: float,
    tau_i: float,
    tau_f: float,
    zpp_over_z_func,
    n_points: int = 5000,
) -> tuple:
    """Integrate a single k-mode of the Mukhanov-Sasaki equation.

    Parameters
    ----------
    k : float
        Comoving wavenumber.
    tau_i : float
        Initial conformal time (must be sufficiently early for Bunch-Davies IC).
    tau_f : float
        Final conformal time (sufficiently after horizon crossing).
    zpp_over_z_func : callable
        Interpolator returning z''(tau)/z(tau).
    n_points : int, optional
        Number of evaluation points for the solution. Default 5000.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        tau values, u_k values, u_k' values.
    """
    y0 = bunch_davies_ic(k, tau_i)
    tau_eval = np.linspace(tau_i, tau_f, n_points)

    sol = solve_ivp(
        ms_equation,
        [tau_i, tau_f],
        y0,
        args=(k, zpp_over_z_func),
        t_eval=tau_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(
            f"Integration failed for k={k}: {sol.message}"
        )

    return sol.t, sol.y[0], sol.y[1]
