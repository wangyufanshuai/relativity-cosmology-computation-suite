"""Eisenstein & Hu (1998) transfer function fitting formulae.

Implements both the no-wiggle (smoothed) and with-wiggle (BAO) versions
of the EH98 transfer function, along with related helper quantities.

Reference: Eisenstein & Hu, "Baryonic Features in the Matter Transfer Function",
ApJ 496:605, 1998. arXiv:astro-ph/9709112
"""

import numpy as np

from .constants import OMEGA_B_DEFAULT, OMEGA_M_DEFAULT, T_CMB_DEFAULT


def k_eq_EH98(h=0.674, Omega_m=OMEGA_M_DEFAULT):
    """Matter-radiation equality wavenumber in 1/Mpc.

    From EH98 Eq. (3):
        k_eq = 7.46e-2 * Omega_m * h^2  [1/Mpc]

    Parameters
    ----------
    h : float
        Dimensionless Hubble parameter.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float
        Wavenumber at matter-radiation equality in 1/Mpc.
    """
    omega_m = Omega_m * h**2
    return 7.46e-2 * omega_m


def sound_horizon_EH98(h=0.674, Omega_m=OMEGA_M_DEFAULT, Omega_b=OMEGA_B_DEFAULT):
    """Sound horizon at the drag epoch (EH98 fitting formula).

    Uses the fitting formula from EH98 Eq. (26):
        r_s = 44.5 * ln(9.83 / omega_m) / sqrt(1 + 10 * omega_b^0.75)  [Mpc]

    Parameters
    ----------
    h : float
        Dimensionless Hubble parameter.
    Omega_m : float
        Matter density parameter.
    Omega_b : float
        Baryon density parameter.

    Returns
    -------
    float
        Sound horizon in Mpc.
    """
    omega_m = Omega_m * h**2
    omega_b = Omega_b * h**2
    return 44.5 * np.log(9.83 / omega_m) / np.sqrt(1.0 + 10.0 * omega_b**0.75)


def transfer_EH98(k, h=0.674, Omega_m=OMEGA_M_DEFAULT, Omega_b=OMEGA_B_DEFAULT,
                  T_CMB=T_CMB_DEFAULT):
    """No-wiggle (smoothed) transfer function from Eisenstein & Hu 1998.

    Uses EH98 Eqs. (28)-(31) which provide a smooth transfer function
    without BAO oscillations.

    Parameters
    ----------
    k : array_like
        Wavenumber in 1/Mpc.
    h : float
        Dimensionless Hubble parameter.
    Omega_m : float
        Matter density parameter.
    Omega_b : float
        Baryon density parameter.
    T_CMB : float
        CMB temperature in Kelvin.

    Returns
    -------
    array_like
        Transfer function T(k).
    """
    k = np.asarray(k, dtype=float)
    scalar_input = k.ndim == 0
    k = np.atleast_1d(k)

    omega_m = Omega_m * h**2
    omega_b = Omega_b * h**2
    f_b = omega_b / omega_m
    f_c = 1.0 - f_b

    theta = T_CMB / 2.7

    s = sound_horizon_EH98(h, Omega_m, Omega_b)

    # alpha_Gamma from Eq. (31)
    alpha_Gamma = (
        1.0
        - 0.328 * np.log(431.0 * omega_m) * f_b
        + 0.38 * np.log(22.3 * omega_m) * f_b**2
    )

    # Gamma_eff from Eq. (30)
    Gamma_eff = (
        omega_m / h
        * (alpha_Gamma + (1.0 - alpha_Gamma) / (1.0 + (0.43 * k * s)**4))
    )

    # q from Eq. (28)
    q = k * theta**2 / Gamma_eff

    # Transfer function from Eq. (29)
    L0 = np.log(2.0 * np.e + 1.844 * q)
    C0 = 14.1 + 725.1 / (1.0 + 164.1 * q)
    T0 = L0 / (L0 + C0 * q**2)

    if scalar_input:
        return float(T0[0])
    return T0


def transfer_EH98_wiggle(k, h=0.674, Omega_m=OMEGA_M_DEFAULT, Omega_b=OMEGA_B_DEFAULT,
                         T_CMB=T_CMB_DEFAULT):
    """With-wiggle (BAO) transfer function from Eisenstein & Hu 1998.

    Uses the full EH98 fitting formula Eqs. (16)-(26) including baryon
    acoustic oscillations.

    Parameters
    ----------
    k : array_like
        Wavenumber in 1/Mpc.
    h : float
        Dimensionless Hubble parameter.
    Omega_m : float
        Matter density parameter.
    Omega_b : float
        Baryon density parameter.
    T_CMB : float
        CMB temperature in Kelvin.

    Returns
    -------
    array_like
        Transfer function T(k) with BAO wiggles.
    """
    k = np.asarray(k, dtype=float)
    scalar_input = k.ndim == 0
    k = np.atleast_1d(k)

    omega_m = Omega_m * h**2
    omega_b = Omega_b * h**2
    f_b = omega_b / omega_m
    f_c = 1.0 - f_b

    theta = T_CMB / 2.7

    keq = k_eq_EH98(h, Omega_m)
    s = sound_horizon_EH98(h, Omega_m, Omega_b)

    # q in units of keq: q = k / keq
    q = k / keq

    # --- CDM transfer function, Eqs. (17)-(19) ---
    a1 = (46.9 * omega_m)**0.670 * (1.0 + (32.1 * omega_m)**(-0.532))
    a2 = (12.0 * omega_m)**0.424 * (1.0 + (45.0 * omega_m)**(-0.582))
    alpha_c = a1**(-f_b) * a2**(-f_b**3)

    b1_ = 0.944 / (1.0 + (358.0 * omega_m)**(-0.664))
    b2_ = (0.395 * omega_m)**(-0.223)
    beta_c = 1.0 / (1.0 + b1_ * (f_c**(b2_) - 1.0))

    # Ttilde_c from Eq. (17)-(18)
    def _T_tilde_c(q_val, beta_val):
        C = 14.0 / alpha_c + 386.0 / (1.0 + 69.9 * q_val**1.08)
        return np.log(np.e + 1.8 * beta_val * q_val) / (
            np.log(np.e + 1.8 * beta_val * q_val) + C * q_val**2
        )

    # Full CDM piece from Eq. (19): T_c = f_c * T_c(q, beta=1) + f_b * T_c(q, beta=0)
    # Actually from EH98: T_tilde_c uses beta_c and alpha_c
    # T_c = T_tilde_c(q, 1) if beta_c ~1, T_tilde_c(q, 0) if beta_c ~0
    # interpolation via Eq. (19): T_c = q * (T_tilde_c(1, q)/q) ...
    # Simplified: use Eq. (17) with beta_c parameter
    C_c = 14.0 / alpha_c + 386.0 / (1.0 + 69.9 * q**1.08)
    T_c_1 = np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + C_c * q**2)
    T_c_0 = np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + C_c * alpha_c**(-1) * q**2)

    T_c = (1.0 - beta_c) * T_c_0 + beta_c * T_c_1

    # --- Baryon transfer function, Eqs. (20)-(24) ---
    # Redshift of drag epoch, Eq. (4)
    z_d = (
        1291.0
        * omega_m**0.251
        / (1.0 + 0.659 * omega_m**0.828)
        * (1.0 + 0.213 * omega_b**0.636)
    )

    # R_eq and R_d (baryon/photon ratio), Eq. (2)
    # R = (3 * rho_b) / (4 * rho_gamma)
    # R = 31.5 * Omega_b * h^2 * (T_CMB / 2.7 K)^(-4) / (1 + z)
    z_eq = 2.5e4 * omega_m * theta**(-4)
    R_eq = 31.5 * omega_b * theta**(-4) / (1.0 + z_eq)
    R_d = 31.5 * omega_b * theta**(-4) / (1.0 + z_d)

    # Silk damping scale, Eq. (7)
    k_Silk = (
        1.6 * omega_b**0.52 * omega_m**0.73
        * (1.0 + (10.4 * omega_m)**(-2.93))
    )

    # alpha_b from Eq. (20)
    # G(y) from text below Eq. (20)
    y = (1.0 + z_eq) / (1.0 + z_d)
    sqrt_y = np.sqrt(y)
    sqrt_1py = np.sqrt(1.0 + y)
    G_y = y * (
        -6.0 * sqrt_1py
        + (2.0 + 3.0 * y) * np.log((sqrt_1py + sqrt_y) / (sqrt_1py - sqrt_y + 1e-30))
    )
    alpha_b = 2.07 * keq * s * (1.0 + R_d)**(-0.75) * G_y

    # beta_b from Eq. (21)
    beta_b = (
        0.5 + f_b
        + (3.0 - 2.0 * f_b) * np.sqrt((17.2 * omega_m)**2 + 1.0)
    )

    # Baryon transfer function, Eq. (22)
    C_b = 14.0 + 725.0 / (1.0 + 164.0 * q)
    T_b_base = np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + C_b * q**2)

    # BAO oscillation term
    # sin(k*s) / (k*s) * damping
    ks = k * s
    # Use np.sinc which computes sin(pi*x)/(pi*x), so we compute sin(x)/x directly
    bao_osc = np.where(ks > 1e-30, np.sin(ks) / ks, 1.0)

    # Silk damping
    silk_damp = np.exp(-((k / k_Silk)**1.4))

    T_b = (
        T_b_base
        + alpha_b * silk_damp * bao_osc
        * np.exp(-((k / (keq * 8.41))**2))
    )

    # --- Total transfer function, Eq. (16) ---
    T = f_b * T_b + f_c * T_c

    if scalar_input:
        return float(T[0])
    return T
