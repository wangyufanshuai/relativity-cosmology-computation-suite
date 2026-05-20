"""Primordial power spectrum computation.

Computes the scalar and tensor power spectra from mode function solutions,
the spectral index n_s, and the tensor-to-scalar ratio r.
"""

import numpy as np
from scipy.interpolate import interp1d

from primordial_perturbations.mukhanov_sasaki import (
    integrate_mode,
    z_function,
    z_pp_over_z,
)


def curvature_perturbation(u_k: complex, du_k: complex, z: float, a: float) -> complex:
    """Compute the comoving curvature perturbation R_k = u_k / z.

    Parameters
    ----------
    u_k : complex
        Mukhanov-Sasaki variable mode function value.
    du_k : complex
        Derivative of u_k with respect to conformal time.
    z : float
        The MS variable z = a * phi_dot / H at the evaluation time.
    a : float
        Scale factor at the evaluation time.

    Returns
    -------
    complex
        The curvature perturbation R_k = u_k / z.
    """
    return u_k / z


def scalar_power_spectrum(
    k_array: np.ndarray,
    V_func,
    phi_array: np.ndarray,
    H_array: np.ndarray,
    a_array: np.ndarray,
    tau_array: np.ndarray,
    phi_dot_array: np.ndarray = None,
    n_eval: int = 3000,
) -> np.ndarray:
    """Compute the scalar primordial power spectrum P_s(k).

    P_s(k) = k^3 |R_k|^2 / (2 pi^2)

    where R_k = u_k / z is the comoving curvature perturbation evaluated
    several e-folds after horizon crossing.

    Parameters
    ----------
    k_array : np.ndarray
        Array of comoving wavenumbers.
    V_func : callable
        Inflaton potential V(phi). (Not directly used; kept for interface consistency.)
    phi_array : np.ndarray
        Inflaton field values at each conformal time.
    H_array : np.ndarray
        Hubble parameter values at each conformal time.
    a_array : np.ndarray
        Scale factor values at each conformal time.
    tau_array : np.ndarray
        Conformal time array.
    phi_dot_array : np.ndarray, optional
        Inflaton velocity array. If None, computed from phi_array numerically.
    n_eval : int, optional
        Number of evaluation points for mode integration. Default 3000.

    Returns
    -------
    np.ndarray
        Scalar power spectrum P_s(k) for each k.
    """
    if phi_dot_array is None:
        phi_dot_array = np.gradient(phi_array, tau_array, edge_order=2)
        # phi_dot = dphi/dtau; convert to physical time: dphi/dt = H * dphi/dtau / a...
        # Actually z = a * (dphi/dt) / H, and dphi/dt is in physical time.
        # With slow-roll: dphi/dt ~ -V'/(3H). We compute dphi/dtau and then convert.
        # dphi/dt = (dphi/dN) * H where dN = Hdt. But dphi/dtau = a * dphi/dt.
        # So dphi/dt = (1/a) * dphi/dtau. Then z = a * (1/a) * dphi/dtau / H = dphi/dtau / H.
        # This is consistent with z = a*(dphi/dt)/H = (dphi/dtau)/H.
        # We store the physical dphi/dt for z computation.
        # Actually let's just use phi_dot_physical = dphi/dtau / a (since dtau = dt/a)
        pass

    # Compute phi_dot in physical time: dphi/dt = (dphi/dtau) * (dtau/dt)^{-1} = (dphi/dtau) / a
    # Wait, dtau = dt/a => dt = a dtau, so dphi/dt = (dphi/dtau) / a ... no.
    # Actually dtau = dt / a => dtau/dt = 1/a => dt/dtau = a => dphi/dt = (dphi/dtau) * (dt/dtau)^{-1}
    # Hmm. phi is a function of t. dphi/dtau = (dphi/dt)(dt/dtau) = (dphi/dt) * a.
    # So dphi/dt = (dphi/dtau) / a.
    # But z = a * (dphi/dt) / H = a * (dphi/dtau / a) / H = (dphi/dtau) / H.
    # So actually z = (dphi/dtau) / H.

    # If phi_dot_array is given, treat it as dphi/dt (physical velocity)
    # Otherwise compute dphi/dtau from phi_array
    if phi_dot_array is not None:
        # User provided physical phi_dot = dphi/dt
        z_arr = z_function(phi_dot_array, a_array, H_array)
    else:
        dphi_dtau = np.gradient(phi_array, tau_array, edge_order=2)
        z_arr = dphi_dtau / H_array

    # Build the z''/z interpolator
    zppz_func = z_pp_over_z(a_array, z_arr, H_array, tau_array)

    P_s = np.zeros_like(k_array, dtype=float)

    for i, k in enumerate(k_array):
        # Find horizon crossing: k = aH (or equivalently k|tau| ~ 1)
        # We need tau_i well before horizon crossing: k|tau_i| >> 1
        # and tau_f well after: k|tau_f| << 1

        # Horizon crossing time: k = a*H
        aH = a_array * H_array
        # Find where k crosses aH (aH decreases during inflation in conformal time)
        # tau is typically negative during inflation
        idx_cross = np.argmin(np.abs(aH - k))

        if idx_cross == 0:
            idx_cross = len(tau_array) // 2

        # Set initial time well before horizon crossing
        n_before = max(int(0.3 * len(tau_array)), 10)
        idx_start = max(0, idx_cross - n_before)
        tau_i = tau_array[idx_start]

        # Ensure k|tau_i| >> 1 for Bunch-Davies
        while k * abs(tau_i) < 10 and idx_start > 0:
            idx_start -= 1
            tau_i = tau_array[idx_start]

        # Final time well after horizon crossing
        n_after = max(int(0.3 * len(tau_array)), 10)
        idx_end = min(len(tau_array) - 1, idx_cross + n_after)
        tau_f = tau_array[idx_end]

        if tau_i >= tau_f:
            tau_i = tau_array[0]
            tau_f = tau_array[-1]

        # Integrate the mode
        try:
            tau_vals, u_vals, du_vals = integrate_mode(
                k, tau_i, tau_f, zppz_func, n_points=n_eval
            )
        except Exception:
            P_s[i] = 0.0
            continue

        # Evaluate u_k at the end (well after horizon crossing)
        u_final = u_vals[-1]
        z_final = z_arr[idx_end] if idx_end < len(z_arr) else z_arr[-1]

        R_k = u_final / z_final
        P_s[i] = k**3 * np.abs(R_k) ** 2 / (2.0 * np.pi**2)

    return P_s


def tensor_power_spectrum(
    k_array: np.ndarray,
    H_array: np.ndarray,
    a_array: np.ndarray,
    tau_array: np.ndarray,
    n_eval: int = 3000,
) -> np.ndarray:
    """Compute the tensor primordial power spectrum P_t(k).

    P_t(k) = 2 k^3 |h_k|^2 / (pi^2 a^2)

    where h_k satisfies h_k'' + (k^2 - a''/a) h_k = 0.
    In the slow-roll limit at horizon crossing: P_t = 2 H^2 / (pi^2 M_pl^2).
    We use the approximate slow-roll formula here.

    Parameters
    ----------
    k_array : np.ndarray
        Array of comoving wavenumbers.
    H_array : np.ndarray
        Hubble parameter values at each conformal time.
    a_array : np.ndarray
        Scale factor values at each conformal time.
    tau_array : np.ndarray
        Conformal time array.
    n_eval : int, optional
        Number of integration points. Default 3000.

    Returns
    -------
    np.ndarray
        Tensor power spectrum P_t(k) for each k.
    """
    aH = a_array * H_array

    # Interpolate H at horizon crossing for each k
    # In slow-roll, P_t = 2 H^2 / (pi^2 M_pl^2)
    # We use the convention M_pl = 1 (reduced Planck mass)
    P_t = np.zeros_like(k_array, dtype=float)

    for i, k in enumerate(k_array):
        # Find horizon crossing
        idx = np.argmin(np.abs(aH - k))
        H_star = H_array[idx]
        # Standard tensor power spectrum in slow-roll (M_pl = 1)
        P_t[i] = 2.0 * H_star**2 / (np.pi**2)

    return P_t


def spectral_index(k_array: np.ndarray, P_k: np.ndarray) -> float:
    """Compute the scalar spectral index n_s - 1 = d ln P_s / d ln k.

    Uses a central finite-difference approximation over the k range.

    Parameters
    ----------
    k_array : np.ndarray
        Comoving wavenumber array (sorted ascending).
    P_k : np.ndarray
        Power spectrum values at each k.

    Returns
    -------
    float
        The spectral tilt n_s (so n_s - 1 = d ln P / d ln k evaluated
        near the middle of the range). Returns n_s directly.
    """
    ln_k = np.log(k_array)
    ln_P = np.log(P_k)

    # Compute numerical derivative at the central point
    n = len(ln_k)
    if n < 3:
        return 1.0

    # Use the central region for a robust estimate
    mid = n // 2
    # Compute d ln P / d ln k using numpy gradient
    dlnP_dlnk = np.gradient(ln_P, ln_k)

    # Return n_s at the midpoint
    n_s = 1.0 + dlnP_dlnk[mid]
    return n_s


def tensor_to_scalar_ratio(
    P_t: np.ndarray,
    P_s: np.ndarray,
    k_pivot: float = None,
    k_array: np.ndarray = None,
) -> float:
    """Compute the tensor-to-scalar ratio r = P_t / P_s at the pivot scale.

    Parameters
    ----------
    P_t : np.ndarray
        Tensor power spectrum values.
    P_s : np.ndarray
        Scalar power spectrum values.
    k_pivot : float, optional
        Pivot scale. If None, uses the midpoint of the arrays.
    k_array : np.ndarray, optional
        Wavenumber array (used if k_pivot is specified).

    Returns
    -------
    float
        The tensor-to-scalar ratio r = P_t(k_pivot) / P_s(k_pivot).
    """
    if k_pivot is not None and k_array is not None:
        idx = np.argmin(np.abs(k_array - k_pivot))
    else:
        idx = len(P_s) // 2

    return P_t[idx] / P_s[idx]
