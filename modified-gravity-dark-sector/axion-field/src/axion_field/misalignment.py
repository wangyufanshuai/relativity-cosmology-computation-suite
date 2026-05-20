"""Axion dark matter via misalignment mechanism.

Computes the axion energy density from the vacuum misalignment angle theta_i.
"""

import numpy as np


def axion_mass_temperature(T, m_a, Lambda_QCD=1e9):
    """Axion mass as function of temperature.

    m_a(T) = m_a * (Lambda_QCD / T)^n for T > Lambda_QCD, else m_a.
    n ~ 3.7 for T >> Lambda_QCD (dilute instanton gas).

    Parameters
    ----------
    T : float or array
        Temperature in eV.
    m_a : float
        Zero-temperature axion mass in eV.
    Lambda_QCD : float
        QCD scale in eV.

    Returns
    -------
    float or array
        Temperature-dependent axion mass in eV.
    """
    T = np.asarray(T, dtype=float)
    m_T = np.where(T > Lambda_QCD, m_a * (Lambda_QCD / T) ** 3.7, m_a)
    return m_T


def oscillation_temperature(m_a, Lambda_QCD=1e9):
    """Temperature when axion field starts oscillating (m_a(T) ~ 3H(T)).

    For radiation domination: H ~ T^2 / M_Pl.
    So m_a * (Lambda_QCD/T_osc)^3.7 ~ 3 * T_osc^2 / M_Pl.

    Simplified: T_osc ~ (m_a * Lambda_QCD^3.7 * M_Pl)^(1/5.7)

    Parameters
    ----------
    m_a : float
        Zero-temperature axion mass in eV.
    Lambda_QCD : float
        QCD scale in eV.

    Returns
    -------
    float
        Oscillation temperature in eV.
    """
    M_Pl_eV = 1.22e28  # Planck mass in eV
    T_osc = (m_a * Lambda_QCD**3.7 * M_Pl_eV) ** (1.0 / 5.7)
    return T_osc


def axion_density(m_a, theta_i=1.0, T_gamma=2.7255, h=0.674):
    """Axion relic density from misalignment mechanism.

    Omega_a h^2 ~ 0.12 * (theta_i)^2 * (m_a / 5e-6 eV)^(-7/6)

    Simplified expression from Preskill, Wise, Wilczek (1983).

    Parameters
    ----------
    m_a : float
        Axion mass in eV.
    theta_i : float
        Initial misalignment angle.
    T_gamma : float
        CMB temperature in K.
    h : float
        Reduced Hubble constant.

    Returns
    -------
    float
        Omega_a * h^2.
    """
    m_a_ref = 5e-6  # Reference mass in eV
    Omega_a_h2 = 0.12 * theta_i**2 * (m_a / m_a_ref) ** (-7.0 / 6.0)
    return Omega_a_h2


def axion_mass_from_density(Omega_h2_target=0.12, theta_i=1.0):
    """Infer axion mass from target relic density.

    Parameters
    ----------
    Omega_h2_target : float
        Target Omega_a h^2.
    theta_i : float
        Initial misalignment angle.

    Returns
    -------
    float
        Axion mass in eV.
    """
    m_a_ref = 5e-6
    m_a = m_a_ref * (Omega_h2_target / (0.12 * theta_i**2)) ** (6.0 / 7.0)
    return m_a
