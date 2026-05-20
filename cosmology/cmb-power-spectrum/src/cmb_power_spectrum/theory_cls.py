"""Parametric C_l^{TT} model for the CMB temperature power spectrum.

Simplified model capturing the key physical features:
- Sachs-Wolfe plateau (l < 30)
- Acoustic peaks with Silk damping (l >= 30)

Parameters: A_s, n_s, Omega_b h^2, Omega_c h^2, theta_s, tau
"""

import numpy as np

# Fiducial cosmological parameters (Planck 2018-like)
fiducial_params = {
    "A_s": 2.1e-9,
    "n_s": 0.9665,
    "omega_b": 0.02242,    # Omega_b h^2
    "omega_c": 0.11933,    # Omega_c h^2
    "theta_s": 0.01041,    # sound horizon ratio (r_s / D_A)
    "tau": 0.0561,
}

PARAM_NAMES = ["omega_b", "omega_c", "theta_s", "tau", "ln10As", "n_s"]

# Physical constants
T_CMB = 2.7255  # CMB temperature in K


def _sachs_wolfe_cl(ells, A_s, n_s, tau, omega_b, omega_c):
    """Sachs-Wolfe plateau: nearly scale-invariant for l < 30.

    C_l ~ (2*pi/25) * A_s * T_CMB^2 * (l(l+1) / (2*pi))^(n_s - 1)
         * exp(-2*tau - 2*tau*(l/800)^2)   [reionization damping]

    This gives the characteristic nearly-flat plateau in l(l+1)C_l/(2*pi).
    """
    x = np.log(ells * (ells + 1) / (2.0 * np.pi))
    # SW amplitude: approximately (2*pi/25) * A_s * T_CMB^2 in uK^2
    # We scale to uK^2 for the power spectrum in temperature units
    amp = (2.0 * np.pi / 25.0) * A_s * (T_CMB * 1e6) ** 2
    cl = amp * np.exp((n_s - 1.0) * x)
    # Reionization optical depth suppression
    cl *= np.exp(-2.0 * tau)
    return cl


def _acoustic_cl(ells, A_s, n_s, tau, omega_b, omega_c, theta_s):
    """Acoustic oscillation regime: l >= 30.

    Model: C_l = A_s * T_CMB^2 * exp(-l^2/l_d^2)
               * [1 + A cos(l * theta_A + phi)]
               * tilt_factor * tau_damping

    Peak positions are set by the acoustic scale theta_A.
    Peak heights are modulated by baryon loading.
    """
    # Silk damping scale
    l_d = 1300.0
    # Acoustic scale in radians: theta_A ~ 300 * theta_s / 0.01041
    # The spacing of peaks in l is Delta_l ~ pi / theta_A
    # For theta_s = 0.01041, the first peak is at l ~ 200
    l_A = 300.0 * (theta_s / 0.01041)

    # Baryon loading: ratio of baryon to photon+CDM density
    # Affects peak heights (odd peaks enhanced, even suppressed)
    omega_m = omega_b + omega_c
    # Baryon fraction affects amplitude of oscillations
    R_b = omega_b / omega_m
    # Oscillation amplitude increases with baryon fraction
    A_osc = 0.85 + 1.5 * R_b

    # Tilt factor: spectral index dependence
    x = np.log(ells * (ells + 1.0) / (2.0 * np.pi))
    tilt = np.exp((n_s - 1.0) * x)

    # Base amplitude in uK^2
    base_amp = A_s * (T_CMB * 1e6) ** 2

    # Silk damping envelope
    damping = np.exp(-(ells / l_d) ** 2)

    # Acoustic oscillation with phase
    # Phase offset phi chosen so first peak appears at l ~ 200
    phi = -np.pi / 2.0
    oscillation = 1.0 + A_osc * np.cos(ells / l_A * np.pi + phi)

    # Smooth shape envelope: rises to first peak then follows damping
    # l^2 factor for the C_l -> D_l conversion shape
    shape = (ells * (ells + 1.0) / (2.0 * np.pi)) * np.exp(-0.0004 * (ells - 220) ** 2)

    # Reionization damping (more pronounced at high l)
    tau_damp = np.exp(-2.0 * tau * (1.0 + 0.5 * (ells / 1500.0) ** 2))

    cl = base_amp * shape * tilt * damping * oscillation * tau_damp

    return cl


def compute_cl_tt(params, lmax=2500):
    """Compute the CMB TT power spectrum C_l^{TT}.

    Parameters
    ----------
    params : dict
        Cosmological parameters. Must contain keys:
        'A_s', 'n_s', 'omega_b', 'omega_c', 'theta_s', 'tau'
    lmax : int
        Maximum multipole (default 2500).

    Returns
    -------
    ells : ndarray
        Multipole moments from 2 to lmax, shape (lmax - 1,).
    cl_tt : ndarray
        C_l^{TT} in uK^2, shape (lmax - 1,).
    """
    ells = np.arange(2, lmax + 1, dtype=np.float64)

    A_s = params["A_s"]
    n_s = params["n_s"]
    omega_b = params["omega_b"]
    omega_c = params["omega_c"]
    theta_s = params["theta_s"]
    tau = params["tau"]

    cl_tt = np.empty_like(ells)

    # Sachs-Wolfe plateau: l < 30
    sw_mask = ells < 30
    if np.any(sw_mask):
        cl_tt[sw_mask] = _sachs_wolfe_cl(
            ells[sw_mask], A_s, n_s, tau, omega_b, omega_c
        )

    # Acoustic regime: l >= 30
    ac_mask = ells >= 30
    if np.any(ac_mask):
        cl_tt[ac_mask] = _acoustic_cl(
            ells[ac_mask], A_s, n_s, tau, omega_b, omega_c, theta_s
        )

    # Ensure positivity
    cl_tt = np.maximum(cl_tt, 1e-30)

    return ells, cl_tt


def params_to_vector(params):
    """Convert parameter dict to vector in Fisher/MCMC order."""
    ln10As = np.log(1e10 * params["A_s"])
    return np.array([
        params["omega_b"],
        params["omega_c"],
        params["theta_s"],
        params["tau"],
        ln10As,
        params["n_s"],
    ])


def vector_to_params(vec):
    """Convert parameter vector back to dict."""
    return {
        "omega_b": vec[0],
        "omega_c": vec[1],
        "theta_s": vec[2],
        "tau": vec[3],
        "A_s": np.exp(vec[4]) / 1e10,
        "n_s": vec[5],
    }
