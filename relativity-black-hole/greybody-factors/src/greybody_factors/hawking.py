"""Hawking radiation calculations with greybody factor corrections.

This module computes Hawking radiation spectra, emission rates, power spectra,
evaporation rates, and the Page curve for black hole evaporation, including
the greybody modification to the ideal Planck spectrum.

Key formulas:
    Hawking temperature: T_H = hbar * c^3 / (8 * pi * G * M * k_B)
    Emission rate: dN/dt = Gamma(omega,l,s) / (2*pi) * 1 / (exp(omega/T_H) - 1)
    Power spectrum: dE/dtdomega = sum_l (hbar*omega) * Gamma(omega,l,s) * (2l+1) / (2*pi) * 1/(exp(omega/T_H)-1)

All internal computations use geometric units (G = c = hbar = k_B = 1).
Conversion to SI is performed at the interface level.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import zeta

from .constants import G, C, HBAR, K_B, M_SUN, SIGMA_SB
from .wkb import greybody_factor_wkb


def hawking_temperature(M):
    """Compute the Hawking temperature of a Schwarzschild black hole.

    T_H = hbar * c^3 / (8 * pi * G * M * k_B)

    In geometric units (G=c=hbar=k_B=1): T_H = 1/(8*pi*M)

    Parameters
    ----------
    M : float
        Black hole mass in SI units (kg).

    Returns
    -------
    float
        Hawking temperature in Kelvin.
    """
    return HBAR * C ** 3 / (8.0 * np.pi * G * M * K_B)


def _to_geometric_mass(M_si):
    """Convert mass from SI (kg) to geometric units (meters).

    In geometric units with G=c=1, mass has dimensions of length:
    M_geo = G * M_si / c^2

    Parameters
    ----------
    M_si : float
        Mass in kilograms.

    Returns
    -------
    float
        Mass in geometric units (meters).
    """
    return G * M_si / C ** 2


def _to_geometric_omega(omega_si):
    """Convert angular frequency from SI (rad/s) to geometric units (1/meters).

    omega_geo = omega_si * (geometric time unit) = omega_si * (G * M / c^3)
    But actually omega has units of 1/time = c^3/(G*M) * omega_geo
    So omega_geo = omega_si * G / c^3 ... no.

    In geometric units with G=c=1, omega has dimensions of 1/length.
    omega_geo = omega_si / c  (since c=1, omega = frequency in m^-1)

    Wait, let's be careful. omega [rad/s] in SI.
    In geometric units, omega has units of 1/length.
    omega_geo = omega_si * (length unit) / (time unit)
    With length = G*M/c^2 and time = G*M/c^3:
    omega_geo = omega_si * G*M/c^3 ... this gives dimensionless omega*M.

    Actually for our purposes, the dimensionless ratio omega/T_H is what matters.
    omega/T_H = omega * 8*pi*G*M / c^2 ... no.

    Let's use natural geometric units consistently:
    Length unit: L = G*M/c^2
    Time unit: T = G*M/c^3
    omega_geo = omega_si * T = omega_si * G*M/c^3
    T_H_geo = 1/(8*pi*M_geo) where M_geo = L = G*M/c^2
    T_H_geo = c^2/(8*pi*G*M)
    omega_geo/T_H_geo = omega_si * G*M/c^3 * 8*pi*G*M/c^2

    This is getting convoluted. Let's use a cleaner approach.
    Use units where the black hole mass M sets the scale.
    Define omega_tilde = omega / (c^3 / (G*M)) = omega * G*M / c^3  (dimensionless)
    T_H = hbar*c^3/(8*pi*G*M*k_B)
    omega / T_H = omega * 8*pi*G*M*k_B / (hbar*c^3)
                = omega_tilde * 8*pi * k_B * c^3/(G*M) * G*M/c^3 / (hbar*c^3/(G*M))

    Actually, the simplest approach: work entirely in geometric units where
    G = c = hbar = k_B = 1, so M is in meters and omega in 1/meters.
    The Hawking temperature in these units is T_H = 1/(8*pi*M).

    Parameters
    ----------
    omega_si : float
        Angular frequency in SI (rad/s).

    Returns
    -------
    float
        Frequency in geometric units (1/meters).
    """
    return omega_si / C


def emission_rate(omega, l, s, M):
    """Compute the Hawking emission rate for a given mode.

    dN/(dt * domega) = Gamma(omega,l,s) / (2*pi) * 1 / (exp(omega/T_H) - 1)

    This gives the number of quanta emitted per unit time per unit frequency
    for the specific mode (omega, l, s), accounting for the greybody factor.

    Parameters
    ----------
    omega : float
        Frequency in geometric units (1/length).
    l : int
        Angular momentum quantum number.
    s : int
        Spin of the field.
    M : float
        Black hole mass in geometric units.

    Returns
    -------
    float
        Emission rate in geometric units.
    """
    T_H = 1.0 / (8.0 * np.pi * M)

    if omega <= 0:
        return 0.0

    # Bose-Einstein distribution (for integer spin fields)
    x = omega / T_H
    if x > 500:
        return 0.0
    thermal = 1.0 / (np.exp(x) - 1.0)

    # Greybody factor
    Gamma = greybody_factor_wkb(omega, l, s, M, order=3)

    return Gamma / (2.0 * np.pi) * thermal


def power_spectrum(M, s, omega_range, l_max=10):
    """Compute the total power spectrum dE/(dt*domega) summed over l.

    P(omega) = sum_{l=l_min}^{l_max} (2l+1) * hbar*omega * Gamma(omega,l,s) / (2*pi)
               * 1 / (exp(omega/T_H) - 1)

    The factor (2l+1) accounts for the degeneracy of each l mode (m = -l, ..., l).
    In geometric units, hbar = 1.

    Parameters
    ----------
    M : float
        Black hole mass in geometric units.
    s : int
        Spin of the field.
    omega_range : array-like
        Array of frequencies in geometric units.
    l_max : int, optional
        Maximum l value to sum over. Default is 10.

    Returns
    -------
    numpy.ndarray
        Power spectrum at each frequency, in geometric units.
    """
    T_H = 1.0 / (8.0 * np.pi * M)
    l_min = s  # Minimum l for spin s

    power = np.zeros_like(omega_range, dtype=float)

    for i, omega in enumerate(omega_range):
        if omega <= 0:
            continue

        x = omega / T_H
        if x > 500:
            continue

        thermal = 1.0 / (np.exp(x) - 1.0)

        mode_power = 0.0
        for l in range(l_min, l_max + 1):
            Gamma = greybody_factor_wkb(omega, l, s, M, order=3)
            # (2l+1) degeneracy from magnetic quantum numbers
            mode_power += (2 * l + 1) * omega * Gamma / (2.0 * np.pi)

        power[i] = mode_power * thermal

    return power


def evaporation_rate(M):
    """Compute the mass loss rate dM/dt due to Hawking radiation.

    dM/dt = -integral domega sum_l (hbar*omega) * (2l+1) * Gamma(omega,l,s) / (2*pi)
            * 1/(exp(omega/T_H) - 1)

    summed over all particle species (s=0,1,2,...) and integrated over frequency.

    For a Schwarzschild black hole emitting only massless particles:
    dM/dt = -alpha / M^2

    where alpha depends on the particle content. For the Standard Model,
    alpha ~ 3.8e-5 (Hawking's original estimate gives alpha = 1/(15360*pi)
    for a single scalar field).

    Parameters
    ----------
    M : float
        Black hole mass in geometric units.

    Returns
    -------
    float
        Mass loss rate dM/dt in geometric units (negative).
    """
    T_H = 1.0 / (8.0 * np.pi * M)
    # The peak of the Planck distribution is at omega ~ 3 * T_H
    # Integrate from small omega to ~ 20 * T_H
    omega_min = T_H * 0.01
    omega_max = T_H * 20.0

    # Include scalar (s=0), EM (s=1), and gravitational (s=2) fields
    # with appropriate degrees of freedom
    # Scalar: 1 dof, 2 polarizations for EM (s=1), 2 for gravitational (s=2)
    species = [
        (0, 1),    # 1 scalar field
        (1, 2),    # 2 EM polarizations
        (2, 2),    # 2 gravitational polarizations
    ]

    total_rate = 0.0

    for s, n_pol in species:
        l_min = max(s, 0)
        l_max = max(10, s + 8)

        def integrand(omega, s=s, l_min=l_min, l_max=l_max, n_pol=n_pol):
            if omega <= 0:
                return 0.0
            x = omega / T_H
            if x > 500:
                return 0.0
            thermal = 1.0 / (np.exp(x) - 1.0)

            mode_sum = 0.0
            for l in range(l_min, l_max + 1):
                Gamma = greybody_factor_wkb(omega, l, s, M, order=3)
                mode_sum += (2 * l + 1) * n_pol * omega * Gamma / (2.0 * np.pi)

            return mode_sum * thermal

        result, _ = quad(integrand, omega_min, omega_max,
                         limit=200, epsrel=1e-6)
        total_rate += result

    return -total_rate


def page_curve(M_initial_si, n_steps=100):
    """Compute the Page curve: entropy evolution during black hole evaporation.

    The black hole entropy (Bekenstein-Hawking entropy) is:
        S_BH = A / (4 * l_Pl^2) = 4 * pi * G * M^2 / (hbar * c)

    As the black hole evaporates, M decreases and so does S_BH.
    The Page curve plots the entanglement entropy, which initially follows S_BH
    but then turns around at the Page time (when half the entropy has been emitted)
    and decreases to zero.

    For this calculation, we track the Bekenstein-Hawking entropy of the
    remaining black hole mass over the evaporation timeline.

    Parameters
    ----------
    M_initial_si : float
        Initial black hole mass in SI units (kg).
    n_steps : int, optional
        Number of time steps. Default is 100.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'time': array of proper times (seconds)
        - 'mass': array of black hole masses (kg)
        - 'entropy': array of Bekenstein-Hawking entropies (dimensionless)
        - 'temperature': array of Hawking temperatures (Kelvin)
    """
    M_geo_initial = _to_geometric_mass(M_initial_si)

    # Convert back to SI factors
    # Geometric time unit: t_geo = G * M_si / c^3
    # dt_si = dt_geo * c^3 / (G * M) ... but M changes, so we need care.
    # We work in geometric units and convert at the end.

    # In geometric units: dM/dt = -alpha / M^2
    # => M^2 dM = -alpha dt
    # => M^3/3 = M_initial^3/3 - alpha * t
    # => M(t) = (M_initial^3 - 3*alpha*t)^{1/3}
    # => t_evap = M_initial^3 / (3*alpha)

    # First compute alpha by evaluating the evaporation rate at the initial mass
    alpha = -evaporation_rate(M_geo_initial) * M_geo_initial ** 2

    if alpha <= 0:
        alpha = 1.0 / (15360.0 * np.pi)  # Fallback: single scalar field value

    t_evap = M_geo_initial ** 3 / (3.0 * alpha)

    # Time array in geometric units
    t_geo = np.linspace(0, t_evap * 0.999, n_steps)

    # Mass evolution: M(t) = (M_initial^3 - 3*alpha*t)^{1/3}
    M_geo = (M_geo_initial ** 3 - 3.0 * alpha * t_geo) ** (1.0 / 3.0)

    # Bekenstein-Hawking entropy: S = 4*pi*M^2 (geometric units with hbar=G=c=1)
    # In full SI: S = 4*pi*G*M^2 / (hbar*c)
    entropy = 4.0 * np.pi * M_geo ** 2  # in units of (l_Pl)^2

    # Convert to proper SI values
    # S_SI = entropy * k_B  (entropy is dimensionless, S/k_B = 4*pi*(M_geo/l_Pl)^2)
    # Actually S/k_B = 4*pi*(M*G/c^2)^2 / l_Pl^2 = 4*pi*(M_geo/l_Pl)^2
    # l_Pl = sqrt(hbar*G/c^3), so M_geo/l_Pl = M_geo * c^(3/2) / sqrt(hbar*G)
    l_pl = np.sqrt(HBAR * G / C ** 3)
    entropy_kb = 4.0 * np.pi * (M_geo / l_pl) ** 2

    # Temperature in SI
    temperature = np.array([hawking_temperature(m * C ** 2 / G) for m in M_geo])

    # Convert time to SI
    # t_geo is in meters (geometric units with c=1)
    # t_si = t_geo / c
    time_si = t_geo / C

    # Convert mass to SI
    mass_si = M_geo * C ** 2 / G

    return {
        'time': time_si,
        'mass': mass_si,
        'entropy': entropy_kb,  # S/k_B
        'temperature': temperature,
    }
