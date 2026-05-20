"""Tensor perturbation evolution: Mukhanov-Sasaki equation for gravitational waves.

Solves the evolution of primordial tensor perturbations (gravitational waves)
generated during inflation via the tensor Mukhanov-Sasaki equation:

    h_k'' + (k^2 - a''/a) h_k = 0

where primes denote derivatives with respect to conformal time tau,
k is the comoving wavenumber, and a is the scale factor.

References
----------
- Mukhanov, Feldman, Brandenberger, Phys. Rep. 215 (1992) 203
- Baumann, "TASI Lectures on Inflation", arXiv:0907.5424
- Weinberg, "Cosmology", Oxford (2008), Ch. 10
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Tensor Mukhanov-Sasaki equation solver
# ---------------------------------------------------------------------------

def tensor_mukhanov_sasaki(
    k: float,
    tau_span: Tuple[float, float],
    a_func: Optional[callable] = None,
    H_func: Optional[callable] = None,
    n_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve tensor Mukhanov-Sasaki equation: h_k'' + (k^2 - a''/a) h_k = 0.

    During inflation, a''/a = 2/tau^2 (de Sitter). Bunch-Davies initial
    conditions: h_k -> e^{-ik tau} / sqrt(2k) as k tau -> -inf.

    Parameters
    ----------
    k : float
        Comoving wavenumber in inverse Mpc.
    tau_span : tuple of float
        (tau_initial, tau_final) conformal time range. For de Sitter
        inflation tau is negative during inflation and approaches 0 at
        the end. Typical choice: tau_initial ~ -100/k, tau_final ~ 1/k.
    a_func : callable, optional
        Function a(tau) returning the scale factor. If None, de Sitter
        is assumed: a(tau) = -1 / (H_inf * tau) with H_inf = 1.
    H_func : callable, optional
        Function H(tau) returning the Hubble parameter. If None, constant
        H_inf = 1 is used.
    n_points : int
        Number of output evaluation points.

    Returns
    -------
    tau : np.ndarray
        Conformal time array of shape (n_points,).
    h_real : np.ndarray
        Real part of h_k(tau), shape (n_points,).
    h_imag : np.ndarray
        Imaginary part of h_k(tau), shape (n_points,).

    Notes
    -----
    The de Sitter analytic solution is:
        h_k(tau) = (pi / 2) * (1 + i k tau) * exp(-i k tau) / sqrt(2 k)
    which is used here to set Bunch-Davies initial conditions and as a
    fallback when default (de Sitter) parameters are given.
    """
    tau_i, tau_f = tau_span

    # Default de Sitter: a = -1/(H tau), H = const = 1
    H_inf = 1.0

    if a_func is None and H_func is None:
        # Use analytic de Sitter solution for speed and accuracy
        tau = np.linspace(tau_i, tau_f, n_points)
        x = k * tau  # dimensionless variable

        # Analytic Bunch-Davies solution for tensor modes in de Sitter:
        # h_k(tau) = (pi/4) * sqrt(-tau * pi) * H_1^{(1)}(-k tau) / a
        # Equivalently in a convenient normalisation:
        # h_k = (1/(2k))^(1/2) * (1 + 1/(i k tau)) * exp(-i k tau)
        phase = np.exp(-1j * k * tau)
        h_k = (1.0 / np.sqrt(2.0 * k)) * (1.0 + 1.0 / (1j * k * tau)) * phase

        return tau, np.real(h_k), np.imag(h_k)

    # General numerical solution
    if a_func is None:
        a_func = lambda tau: -1.0 / (H_inf * tau) if tau != 0 else 1e30
    if H_func is None:
        H_func = lambda tau: H_inf

    # Compute a''/a numerically via central differences
    def _a_double_prime_over_a(tau, eps=1e-6):
        """Compute a''(tau)/a(tau) using finite differences."""
        if abs(tau) < eps:
            tau = eps * np.sign(tau) if tau != 0 else eps
        a_m = a_func(tau - eps)
        a_0 = a_func(tau)
        a_p = a_func(tau + eps)
        a_pp = (a_p - 2.0 * a_0 + a_m) / eps**2
        return a_pp / a_0

    # ODE system: real and imaginary parts
    # h'' + (k^2 - a''/a) h = 0
    # Split into h = h_r + i h_i:
    #   h_r'' + (k^2 - a''/a) h_r = 0
    #   h_i'' + (k^2 - a''/a) h_i = 0
    def _rhs(tau, y):
        h_r, dh_r, h_i, dh_i = y
        coeff = k**2 - _a_double_prime_over_a(tau)
        return [dh_r, -coeff * h_r, dh_i, -coeff * h_i]

    # Bunch-Davies initial conditions at tau_i
    # h_k(tau) ~ e^{-ik tau} / sqrt(2k)
    # h_k' (tau) ~ -i k e^{-ik tau} / sqrt(2k)
    phase0 = np.exp(-1j * k * tau_i)
    h0 = phase0 / np.sqrt(2.0 * k)
    dh0 = -1j * k * phase0 / np.sqrt(2.0 * k)

    y0 = [np.real(h0), np.real(dh0), np.imag(h0), np.imag(dh0)]

    tau_eval = np.linspace(tau_i, tau_f, n_points)
    sol = solve_ivp(
        _rhs,
        [tau_i, tau_f],
        y0,
        t_eval=tau_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        max_step=abs(tau_f - tau_i) / 100.0,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.t, sol.y[0], sol.y[2]


# ---------------------------------------------------------------------------
# Tensor power spectrum
# ---------------------------------------------------------------------------

def tensor_power_spectrum(
    k_array: np.ndarray,
    A_t: float = 0.01,
    n_t: float = 0.0,
    r: Optional[float] = None,
    A_s: float = 2.1e-9,
    k_pivot: float = 0.05,
) -> np.ndarray:
    """Tensor power spectrum P_t(k) = A_t (k / k_pivot)^{n_t}.

    If r is given, A_t = r * A_s (consistency relation).

    Parameters
    ----------
    k_array : np.ndarray
        Wavenumber array in inverse Mpc.
    A_t : float
        Tensor amplitude at pivot scale k_pivot. Overridden if r is given.
    n_t : float
        Tensor spectral index. The single-field inflationary consistency
        relation gives n_t = -r/8, but this function does not enforce it
        so the user can explore deviations.
    r : float, optional
        Tensor-to-scalar ratio. If provided, A_t = r * A_s.
    A_s : float
        Scalar power spectrum amplitude at pivot scale.
        Planck 2018 best-fit: A_s ~ 2.1e-9.
    k_pivot : float
        Pivot scale in Mpc^{-1}. Planck standard: 0.05 Mpc^{-1}.

    Returns
    -------
    np.ndarray
        Tensor power spectrum P_t(k), same shape as k_array.

    Notes
    -----
    The power-law form is:
        P_t(k) = A_t * (k / k_pivot)^{n_t}
    where a positive n_t means more power on small scales (blue spectrum)
    and a negative n_t means more power on large scales (red spectrum).
    Standard slow-roll inflation predicts n_t < 0 (red).
    """
    k_array = np.asarray(k_array, dtype=float)

    if r is not None:
        A_t = r * A_s

    P_t = A_t * (k_array / k_pivot) ** n_t
    return P_t


# ---------------------------------------------------------------------------
# Tensor-to-scalar ratio
# ---------------------------------------------------------------------------

def tensor_to_scalar_ratio(
    V: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> float:
    """Compute tensor-to-scalar ratio r from slow-roll parameters.

    Two equivalent formulations:
        r = 16 epsilon_V           (from inflaton potential)
        r = 2/(3 pi^2) V / M_Pl^4  (direct from potential energy)

    At least one of V or epsilon must be provided.

    Parameters
    ----------
    V : float, optional
        Inflaton potential energy density in Planck units (V / M_Pl^4).
        If given, r is computed as r = 2/(3 pi^2) * V.
    epsilon : float, optional
        First slow-roll parameter epsilon_V = (M_Pl^2 / 2) (V'/V)^2.
        If given, r is computed as r = 16 * epsilon.

    Returns
    -------
    float
        Tensor-to-scalar ratio r.

    Raises
    ------
    ValueError
        If neither V nor epsilon is provided.
    """
    if epsilon is not None:
        return 16.0 * epsilon
    elif V is not None:
        # r = 2 V / (3 pi^2 M_Pl^4), with V in units of M_Pl^4
        return 2.0 * V / (3.0 * np.pi**2)
    else:
        raise ValueError("At least one of V or epsilon must be provided.")


# ---------------------------------------------------------------------------
# Evolve single tensor mode
# ---------------------------------------------------------------------------

def evolve_tensor_mode(
    k: float,
    tau: np.ndarray,
    a: np.ndarray,
    h0: float = 1e-5,
    dh0: float = 0.0,
) -> np.ndarray:
    """Evolve a single tensor mode through horizon crossing.

    Solves h'' + 2 (a'/a) h' + k^2 h = 0 using RK45, where primes are
    conformal time derivatives. This is the standard equation for the
    evolution of tensor perturbations in an expanding universe.

    Parameters
    ----------
    k : float
        Comoving wavenumber.
    tau : np.ndarray
        Conformal time array, monotonically increasing, shape (N,).
    a : np.ndarray
        Scale factor array evaluated at tau, shape (N,).
    h0 : float
        Initial value h(tau[0]).
    dh0 : float
        Initial conformal time derivative h'(tau[0]).

    Returns
    -------
    np.ndarray
        The tensor mode amplitude h(tau), shape (N,).

    Notes
    -----
    The damping term 2(a'/a) = 2 aH comes from the Hubble friction in
    cosmic time. Inside the horizon (k >> aH) the mode oscillates freely.
    Outside the horizon (k << aH) the mode freezes (h' -> 0).
    """
    tau = np.asarray(tau, dtype=float)
    a = np.asarray(a, dtype=float)

    if tau.shape != a.shape:
        raise ValueError("tau and a must have the same shape.")
    if len(tau) < 2:
        raise ValueError("tau must have at least 2 points.")

    # Compute a'/a = (da/dtau) / a via interpolation and differentiation
    # Use a cubic interpolant for smooth derivatives
    a_interp = interp1d(tau, a, kind="cubic", fill_value="extrapolate")

    def _adot_over_a(t):
        """Compute a'(t)/a(t) via finite differences on the interpolation."""
        eps = max(abs(t) * 1e-8, 1e-12)
        ap = a_interp(t + eps)
        am = a_interp(t - eps)
        return (ap - am) / (2.0 * eps * a_interp(t))

    # ODE system: y = [h, h']
    def _rhs(t, y):
        h, dh = y
        adot_a = _adot_over_a(t)
        # h'' + 2 (a'/a) h' + k^2 h = 0
        ddh = -2.0 * adot_a * dh - k**2 * h
        return [dh, ddh]

    sol = solve_ivp(
        _rhs,
        [tau[0], tau[-1]],
        [h0, dh0],
        t_eval=tau,
        method="RK45",
        rtol=1e-10,
        atol=1e-13,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    return sol.y[0]
