"""Saha equation for hydrogen and helium ionization fractions.

The Saha equation provides the equilibrium ionization fraction assuming
thermal equilibrium.  It is accurate at high redshift (z >> 1000) but
breaks down during recombination because the Lyman-alpha photons cannot
escape fast enough to maintain equilibrium.
"""

import numpy as np

from .constants import (
    K_B,
    M_E,
    M_H,
    M_P,
    HBAR,
    E_ION_H,
    EV_TO_J,
)


# Helium ionization energies [J]
E_ION_HE1 = 24.587 * EV_TO_J   # He I  -> He II (24.587 eV)
E_ION_HE2 = 54.418 * EV_TO_J   # He II -> He III (54.418 eV)


def _saha_general(T, n_e, g_ratio, ionization_energy, reduced_mass_factor=1.0):
    """General Saha equation: ratio of ionized to neutral states.

    Parameters
    ----------
    T : float or array_like
        Temperature in Kelvin.
    n_e : float or array_like
        Electron number density in m^-3.
    g_ratio : float
        Ratio of statistical weights g_ionized / g_neutral.
    ionization_energy : float
        Ionization energy in Joules.
    reduced_mass_factor : float
        Correction factor for reduced mass in the partition function ratio.

    Returns
    -------
    float or array_like
        Ratio n_ionized / n_neutral.
    """
    T = np.asarray(T, dtype=float)
    # Saha constant factor: (2 pi m_e k_B T / h^2)^(3/2) / n_e
    saha_prefactor = (
        2.0 * np.pi * M_E * K_B * T / HBAR**2
    )**1.5 / n_e * 2.0 * g_ratio * reduced_mass_factor
    boltzmann = np.exp(-ionization_energy / (K_B * T))
    return saha_prefactor * boltzmann


def saha_xe(T, n_b, Y_p=0.24):
    """Hydrogen Saha ionization fraction x_e.

    Solves the Saha equation for hydrogen ionization assuming helium is
    fully ionized (valid at z >> 2000).

    x_e = n_e / n_H  where n_H = (1 - Y_p) n_b

    With the constraint n_e = x_e n_H + contributions from helium, we
    simplify by treating the Saha equation for hydrogen only and assuming
    helium contributes n_He = Y_p/(4(1-Y_p)) * n_H electrons when fully ionized.

    Parameters
    ----------
    T : float or array_like
        Temperature in Kelvin.
    n_b : float or array_like
        Baryon number density in m^-3.
    Y_p : float
        Helium mass fraction.

    Returns
    -------
    float or array_like
        Free electron fraction x_e = n_e / n_H.
    """
    T = np.asarray(T, dtype=float)
    n_b = np.asarray(n_b, dtype=float)

    # Hydrogen number density
    n_H = (1.0 - Y_p) * n_b

    # Saha equation: n_p n_e / n_H = S(T)
    # where S(T) is the Saha coefficient
    # Assuming helium is fully ionized: n_e = x_e n_H + n_He_full
    # n_He = (Y_p / 4) n_b / m_p  (number density of helium nuclei)
    n_He = Y_p / (4.0 * (1.0 - Y_p)) * n_H  # = Y_p n_b / (4 m_p) / 1

    # For full treatment: x_e = n_e/n_H
    # n_e = x_e * n_H + 2 * n_He (both He electrons ionized at high z)
    # Saha: x_e^2 n_H / (1 - x_e) = S(T)  (simplified, no He correction in Saha)
    # We use the standard form: x_e^2 / (1 - x_e) = S(T) / n_H

    saha_ratio = (
        2.0 * (2.0 * np.pi * M_E * K_B * T / HBAR**2)**1.5
        * np.exp(-E_ION_H / (K_B * T))
        / n_H
    )

    # Solve quadratic: x_e^2 + saha_ratio * x_e - saha_ratio = 0
    # (from x_e^2 / (1 - x_e) = saha_ratio)
    # x_e = [-saha_ratio + sqrt(saha_ratio^2 + 4*saha_ratio)] / 2
    discriminant = saha_ratio**2 + 4.0 * saha_ratio
    x_e = (-saha_ratio + np.sqrt(discriminant)) / 2.0

    # At high T, x_e should approach ~1 + small corrections from He
    # The Saha equation above gives x_e for hydrogen
    # Total electron fraction includes helium contributions
    # x_e_total = x_e_H + f_He * (x_HeI + 2*x_HeII)
    # where f_He = n_He/n_H = Y_p/(4*(1-Y_p))
    f_He = Y_p / (4.0 * (1.0 - Y_p))

    # At high redshift where Saha gives x_e ~ 1 for H,
    # total x_e ~ 1 + 2*f_He (He fully doubly ionized)
    # Return x_e for hydrogen ionization fraction
    return np.clip(x_e, 0.0, 1.0)


def saha_helium(T, n_b, Y_p=0.24):
    """Helium ionization fractions from the Saha equation.

    Computes x_HeI (fraction of He that is singly ionized) and
    x_HeII (fraction of He that is doubly ionized).

    Parameters
    ----------
    T : float or array_like
        Temperature in Kelvin.
    n_b : float or array_like
        Baryon number density in m^-3.
    Y_p : float
        Helium mass fraction.

    Returns
    -------
    tuple of (x_HeII_to_HeI, x_HeIII_to_HeII)
        Ratios of ionized to neutral for each stage.
    """
    T = np.asarray(T, dtype=float)
    n_b = np.asarray(n_b, dtype=float)

    n_He = Y_p / 4.0 * n_b / M_P * M_H  # approximate: n_He nuclei
    n_e_approx = n_b  # rough approximation for n_e at high z

    # He I -> He II (statistical weight ratio g_HeII/g_HeI = 2/1)
    saha_He1 = _saha_general(T, n_e_approx, 2.0, E_ION_HE1)

    # He II -> He III (statistical weight ratio g_HeIII/g_HeII = 1/2)
    saha_He2 = _saha_general(T, n_e_approx, 0.5, E_ION_HE2)

    return saha_He1, saha_He2
