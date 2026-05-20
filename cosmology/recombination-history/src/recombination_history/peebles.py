"""Peebles Three-Level Atom (TLA) model for hydrogen recombination.

Implements the Peebles TLA differential equation for the evolution of the
free electron fraction x_e(z) through the epoch of recombination.

The TLA models hydrogen as a three-level system (ground state, n=2, continuum)
and tracks the net rate of change of the ionization fraction accounting for
recombination, photoionization, and the net rate at which Lyman-alpha photons
redshift out of resonance.

References
----------
Peebles, P.J.E. (1968) "Recombination of the Primeval Plasma", ApJ 153, 1.
Zeldovich et al. (1968) "Recombination of Hydrogen in the Hot Model of the
    Universe", Sov. Phys. Usp. 11, 681.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .constants import (
    K_B,
    M_E,
    HBAR,
    E_ION_H,
    E_21,
    C,
)


def alpha_B(T):
    """Case-B hydrogen recombination coefficient.

    Uses the fitting formula from Pequignot et al. (1991), as commonly
    employed in cosmological recombination codes.

    alpha_B = 4.736e-19 * (T/10^4)^(-0.7) * (1 + 0.56*(T/10^4)^0.7)^(-1)
              ... simplified to a power law for the TLA:

    We use the standard approximation:
        alpha_B ~ 10^{-19.62} * (T/10^4 K)^{-0.7}  m^3/s

    Parameters
    ----------
    T : float or array_like
        Temperature in Kelvin.

    Returns
    -------
    float or array_like
        Case-B recombination coefficient in m^3/s.
    """
    T = np.asarray(T, dtype=float)
    T4 = T / 1.0e4
    # Pequignot, Petitjean & Boisson (1991) fitting formula
    # alpha_B = 10^{-19.62} * (T4)^{-0.7} * ...  (approximate)
    # More accurate form:
    log_alpha = -19.62 + 2.0 * np.log(T4) * (-0.35)
    # Use explicit power law: alpha_B = 2.6e-19 * T4^{-0.7} m^3/s
    return 2.6e-19 * T4**(-0.7)


def beta_21(T):
    """Photoionization rate from the n=2 level of hydrogen.

    beta_21 = alpha_B(T) * (m_e k_B T / (2 pi hbar^2))^{3/2}
              * exp(-E_21 / (k_B T))

    This is derived from detailed balance with the case-B recombination
    coefficient alpha_B, using the principle of microscopic reversibility.

    Parameters
    ----------
    T : float or array_like
        Temperature in Kelvin.

    Returns
    -------
    float or array_like
        Photoionization rate beta_21 in s^-1 (per atom in n=2).
    """
    T = np.asarray(T, dtype=float)
    aB = alpha_B(T)
    # Thermal de Broglie wavelength factor
    lambda_T_factor = (M_E * K_B * T / (2.0 * np.pi * HBAR**2))**1.5
    boltzmann = np.exp(-E_21 / (K_B * T))
    return aB * lambda_T_factor * boltzmann


def peebles_C(alpha_B_val, beta_21_val, T, n_b, H, Y_p=0.24):
    """Peebles C factor: probability that a decay from n=2 reaches the
    ground state without causing reionization.

    C = (Lambda_2s->1s + Lambda_alpha) / (Lambda_2s->1s + Lambda_alpha + beta_21)

    where:
    - Lambda_2s->1s = 8.229 s^-1 is the two-photon decay rate of the 2s level
    - Lambda_alpha = (1 - e^{-tau_Ly_alpha}) * H / (1 - z) is the effective
      Lyman-alpha escape rate (accounting for Sobolev escape probability)

    Parameters
    ----------
    alpha_B_val : float
        Case-B recombination coefficient.
    beta_21_val : float
        Photoionization rate from n=2.
    T : float
        Temperature in Kelvin.
    n_b : float
        Baryon number density in m^-3.
    H : float
        Hubble parameter in s^-1.
    Y_p : float
        Helium mass fraction.

    Returns
    -------
    float
        Peebles C factor (0 < C <= 1).
    """
    # Two-photon decay rate of hydrogen 2s level [s^-1]
    Lambda_2s1s = 8.229

    # Lyman-alpha frequency [Hz]
    nu_Ly_alpha = E_21 / HBAR

    # Sobolev optical depth for Lyman-alpha
    # tau_s = (3 c lambda_alpha^3 n_1s) / (8 pi H)
    # where lambda_alpha = c/nu_Ly_alpha and n_1s ~ n_H (approximately)
    # We approximate n_1s ~ n_H for the purpose of the Sobolev optical depth
    n_H = (1.0 - Y_p) * n_b

    lambda_alpha = C / nu_Ly_alpha
    # n_1s ~ n_H * (1 - x_e), but we don't know x_e here
    # Use the full n_H as an upper-limit approximation for tau_s calculation
    # In practice the caller should pass x_e but for the Peebles C we use
    # a simplified form where we approximate:
    # The escape probability p_alpha = (1 - exp(-tau_s))/tau_s
    # For large tau_s, p_alpha ~ 1/tau_s

    # We use the simplified form commonly seen in textbooks:
    # Lambda_alpha ~ H * 3 * nu_Ly_alpha^3 / (8 pi * n_H * c^2 * (1/(1-x_e)))
    # But actually the standard Peebles C uses:
    # L_alpha = (1 - exp(-tau_S)) * c / (H * lambda_alpha^2)
    # which for large tau_S -> c/(H * lambda_alpha^2)
    # More commonly simplified to: effective escape rate proportional to H

    # Standard Peebles (1968) approximation:
    # The effective Lyman-alpha transition rate:
    #   Lambda_alpha = H / (1 - z)  *  (1 - exp(-tau_S)) / tau_S * tau_S
    # But for very large tau_S, the escape fraction ~ 1/tau_S and
    # Lambda_alpha ~ H * c^3 / (8 pi * nu_alpha^3 * n_1s * (1-z))
    # Simplification: use Lambda_alpha ~ H / ((1-z) * n_1s * lambda_alpha^3 / (3*c))
    # In the end, we get Lambda_alpha ~ H on dimensional grounds

    # Simple standard form used in many textbook implementations:
    # L_alpha ~ (8 pi H) / (3 * n_1s * lambda_alpha^3)
    # where lambda_alpha = c / nu_alpha

    # For the C factor, we use the commonly adopted approximation:
    # C = (Lambda_2s1s + Lambda_alpha) / (Lambda_2s1s + Lambda_alpha + beta_21)
    # where Lambda_alpha accounts for redshifting of Lyman-alpha photons

    # Approximate n_1s ~ n_H (most atoms are in ground state during recomb)
    n_1s = n_H

    # Sobolev escape probability (approximate)
    tau_S = (3.0 * lambda_alpha**3 * n_1s * C) / (8.0 * np.pi * H)

    # Escape fraction
    if tau_S > 1.0e30:
        p_escape = 1.0 / tau_S
    else:
        p_escape = (1.0 - np.exp(-tau_S)) / tau_S

    # Effective Lyman-alpha rate
    Lambda_alpha = p_escape * nu_Ly_alpha * (8.0 * np.pi * H) / (3.0 * n_1s * lambda_alpha**3)

    # Simplify: Lambda_alpha ~ p_escape * H * c / (lambda_alpha * (1-z))
    # But for the standard TLA, we use:
    Lambda_alpha = p_escape * C / (lambda_alpha * (1.0))  # rough

    # Actually, the simplest commonly-used form is:
    # The rate at which Lyman-alpha photons redshift out of the line is:
    #   Lambda_alpha = (1/(1-z)) * dnu/dt * ... ~ H * c/(lambda_alpha) * (1/(1-z))
    # For a more standard form, use:
    Lambda_alpha = H / (1.0)  # dimensional, ~ H at z ~ 1000

    # The correct standard Peebles approach:
    # p_escape = (1 - exp(-tau_S)) / tau_S
    # Lambda_alpha ~ p_escape * (8 pi H)/(3 n_1s lambda_alpha^3) * nu_alpha
    # For very large tau_S: Lambda_alpha ~ H/tau_S * ...
    # = H * (8 pi)/(3 * lambda_alpha^3 * n_1s) / tau_S ... which cancels

    # Let me use the clean textbook result:
    # Lambda_alpha = H * (1-z)^{-1} * (1 - exp(-tau_S)) * ...
    # But actually Peebles showed the net rate is just H for the escape.

    # Clean implementation following Dodelson (2003) / Weinberg (2008):
    # Lambda_alpha (escape rate) ~ H / (1-z) for Sobolev
    # The full expression is complex; the standard approximation is:
    #   L_alpha ~ H(z) (when Sobolev is valid, tau_S >> 1)
    # More precisely:
    #   L_alpha = (1 - e^{-tau_S}) / tau_S * 8 pi / (3 n_1s lambda^3) * H
    # This simplifies because tau_S = 3 c lambda^3 n_1s / (8 pi H)
    # so L_alpha = (1 - e^{-tau_S}) * H
    # For large tau_S: L_alpha -> H

    # Use the exact Sobolev form:
    if tau_S > 500:
        Lambda_alpha = H
    else:
        Lambda_alpha = (1.0 - np.exp(-tau_S)) * H

    numerator = Lambda_2s1s + Lambda_alpha
    denominator = Lambda_2s1s + Lambda_alpha + beta_21_val

    return numerator / denominator


def tla_rhs(z, x_e, H0, Omega_m, Omega_r, Omega_lambda, Omega_b, h, Y_p=0.24):
    """Right-hand side of the Peebles TLA differential equation.

    dx_e/dz = C * [alpha_B * n_H * x_e^2 - beta_21 * (1 - x_e) * exp(-E_21/(kT))]
              * (1 + z) / H(z)

    But the sign convention depends on whether we solve for dx_e/dz or dx_e/dt.
    We solve dx_e/dz:
      dx_e/dz = -(1+z)/H * dx_e/dt

    where dx_e/dt = -C * [alpha_B * x_e^2 * n_H - beta_21 * (1-x_e) * n_{1s,eq}]

    Parameters
    ----------
    z : float
        Redshift.
    x_e : float
        Current free electron fraction.
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_r : float
        Radiation density parameter.
    Omega_lambda : float
        Dark energy density parameter.
    Omega_b : float
        Baryon density parameter.
    h : float
        Reduced Hubble constant.
    Y_p : float
        Helium mass fraction.

    Returns
    -------
    float
        dx_e/dz
    """
    from .background import hubble, temperature, baryon_density

    if x_e <= 0.0:
        return 0.0
    if x_e >= 1.0:
        x_e = 1.0

    H_z = hubble(z, H0, Omega_m, Omega_r, Omega_lambda)
    T_z = temperature(z)
    n_b_z = baryon_density(z, Omega_b, h)
    n_H_z = (1.0 - Y_p) * n_b_z

    aB = alpha_B(T_z)
    b21 = beta_21(T_z)

    # Peebles C factor
    C_factor = peebles_C(aB, b21, T_z, n_b_z, H_z, Y_p)

    # dx_e/dt = -C_factor * [alpha_B * x_e^2 * n_H - beta_21 * (1 - x_e) * exp(-E_21/(kT)) * (some factor)]
    # Standard Peebles TLA:
    # dx_e/dt = -C * alpha_B * n_H * [x_e^2 - S(T)]
    # where S(T) = (beta_21 / (alpha_B * n_H)) * (1 - x_e) * exp(-E_21/(kT))
    # Actually the standard form is:
    # dx_e/dt = -C * [alpha_B * x_e^2 * n_H - beta_21 * (1 - x_e)]

    # Correct Peebles TLA equation:
    # dx_e/dt = -C * alpha_B * n_H * x_e^2 + C * beta_21 * (1 - x_e)
    # (recombination reduces x_e, reionization increases it)

    dxedt = -C_factor * (aB * n_H_z * x_e**2 - b21 * (1.0 - x_e))

    # Convert to dx_e/dz: dx_e/dz = dx_e/dt * dt/dz = dx_e/dt * (-1/((1+z)*H))
    dxedz = dxedt / (-(1.0 + z) * H_z)

    return dxedz


def solve_recombination(z_array, H0, Omega_m, Omega_r, Omega_lambda, Omega_b, h, Y_p=0.24):
    """Solve the Peebles TLA for the free electron fraction x_e(z).

    Integrates the TLA differential equation from high to low redshift,
    using the Saha equation solution as the initial condition at the
    highest redshift.

    Parameters
    ----------
    z_array : array_like
        Redshift array (need not be sorted; will be sorted internally).
    H0 : float
        Hubble constant in km/s/Mpc.
    Omega_m : float
        Matter density parameter.
    Omega_r : float
        Radiation density parameter.
    Omega_lambda : float
        Dark energy density parameter.
    Omega_b : float
        Baryon density parameter.
    h : float
        Reduced Hubble constant.
    Y_p : float
        Helium mass fraction.

    Returns
    -------
    x_e_array : ndarray
        Free electron fraction x_e at each redshift in z_array.
    """
    from .background import temperature, baryon_density
    from .saha import saha_xe

    z_array = np.asarray(z_array, dtype=float)
    z_sorted = np.sort(z_array)
    z_min = z_sorted[0]
    z_max = z_sorted[-1]

    # Start integration from well above recombination
    z_start = z_max + 100.0
    if z_start < 2000.0:
        z_start = 2500.0

    # Initial condition from Saha at z_start
    T_init = temperature(z_start)
    n_b_init = baryon_density(z_start, Omega_b, h)
    x_e_init = float(saha_xe(T_init, n_b_init, Y_p))

    # Ensure initial condition is close to 1 (fully ionized)
    if x_e_init < 0.99:
        x_e_init = 0.999

    # Integrate from high z to low z
    # scipy solve_ivp with z going from z_start down to z_min
    # We integrate dz (going from high z to low z, so z is decreasing)
    # We flip: integrate in variable y = z_start - z, so y goes from 0 upward

    def rhs(z, x_e_vec):
        """RHS wrapper for solve_ivp. z is decreasing."""
        x_e_val = x_e_vec[0]
        return [tla_rhs(z, x_e_val, H0, Omega_m, Omega_r, Omega_lambda,
                        Omega_b, h, Y_p)]

    # We need to go from z_start (high) to z_min (low)
    # solve_ivp integrates forward, so we set t_span = [z_start, z_min]
    # and our RHS naturally has the correct sign (dx_e/dz > 0 as z increases,
    # so going forward from z_start to z_min means z decreases)
    sol = solve_ivp(
        rhs,
        t_span=[z_start, z_min],
        y0=[x_e_init],
        t_eval=z_sorted[::-1],  # reverse because z decreases
        method='BDF',
        rtol=1e-6,
        atol=1e-8,
        max_step=10.0,
    )

    if not sol.success:
        raise RuntimeError(f"TLA integration failed: {sol.message}")

    # sol.t is the z values (decreasing), sol.y[0] is x_e
    x_e_sorted_reversed = sol.y[0]

    # Reverse back to match z_sorted (increasing)
    x_e_sorted = x_e_sorted_reversed[::-1]

    # Interpolate to match the original z_array ordering
    x_e_result = np.interp(z_array, z_sorted, x_e_sorted)

    return np.clip(x_e_result, 1e-12, 10.0)
