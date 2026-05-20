"""Binary pulsar timing models and relativistic parameter extraction.

Provides tools for computing orbital parameters of binary pulsars,
including Shklovskii effect, Einstein delay, and a catalogue of
observed/predicted values for PSR B1913+16 (Hulse-Taylor pulsar).
"""

from __future__ import annotations

import numpy as np

from .constants import G, c, M_SUN

__all__ = [
    "pulsar_binary_orbit",
    "shklovskii_effect",
    "einstein_delay",
    "hulse_taylor_parameters",
]


# ---------------------------------------------------------------------------
# Binary orbit parameters
# ---------------------------------------------------------------------------

def pulsar_binary_orbit(
    Pb: float,
    ecc: float,
    M1: float,
    M2: float,
    omega_dot: float | None = None,
) -> dict:
    """Compute derived orbital parameters for a binary pulsar system.

    Parameters
    ----------
    Pb : float
        Orbital period [s].
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    M1 : float
        Pulsar mass [kg].
    M2 : float
        Companion mass [kg].
    omega_dot : float or None
        Observed periastron advance rate [rad/s].  If given, the GR
        prediction is also computed for comparison.

    Returns
    -------
    dict with keys:
        ``a1``        : semi-major axis of pulsar orbit [m]
        ``a2``        : semi-major axis of companion orbit [m]
        ``a_total``   : total semi-major axis (a1 + a2) [m]
        ``Mt``        : total mass [kg]
        ``mu``        : reduced mass [kg]
        ``omega_dot_gr`` : GR-predicted periastron advance [rad/s]
        ``Pb_yr``     : orbital period in years
        ``v_orb``     : characteristic orbital velocity [m/s]
    """
    Mt = M1 + M2
    mu = M1 * M2 / Mt

    # Total semi-major axis from Kepler's third law: Pb^2 = 4 pi^2 a^3 / (G Mt)
    a_total = (G * Mt * Pb**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
    a1 = a_total * M2 / Mt  # pulsar orbit
    a2 = a_total * M1 / Mt  # companion orbit

    # GR periastron advance: d(omega)/dt = 3 (2 pi / Pb)^{5/3} (G Mt)^{2/3} / (c^2 (1-e^2))
    # (Will, 1993; Damour & Deruelle 1985)
    n = 2.0 * np.pi / Pb  # mean motion
    omega_dot_gr = (
        3.0
        * n ** (5.0 / 3.0)
        * (G * Mt) ** (2.0 / 3.0)
        / (c**2 * (1.0 - ecc**2))
    )

    # Characteristic orbital velocity (at periastron is too high; use circular equivalent)
    v_orb = 2.0 * np.pi * a_total / Pb  # mean velocity

    return {
        "a1": a1,
        "a2": a2,
        "a_total": a_total,
        "Mt": Mt,
        "mu": mu,
        "omega_dot_gr": omega_dot_gr,
        "Pb_yr": Pb / (365.25 * 86400.0),
        "v_orb": v_orb,
    }


# ---------------------------------------------------------------------------
# Shklovskii effect
# ---------------------------------------------------------------------------

def shklovskii_effect(
    v_transverse: float,
    distance: float,
    P: float,
) -> float:
    """Apparent period derivative due to the Shklovskii effect (transverse
    motion of the pulsar).

    When a pulsar has a non-zero transverse velocity, its distance increases
    with time, producing an apparent spin-down::

        P_dot_shk = P * v_T^2 / (c * d)

    Parameters
    ----------
    v_transverse : float
        Transverse velocity of the pulsar [m/s].
    distance : float
        Distance to the pulsar [m].
    P : float
        Spin period of the pulsar [s].

    Returns
    -------
    P_dot_shk : float
        Apparent period derivative [s/s] (dimensionless ratio P_dot/P).
    """
    return P * v_transverse**2 / (c * distance)


# ---------------------------------------------------------------------------
# Einstein delay
# ---------------------------------------------------------------------------

def einstein_delay(
    Pb: float,
    ecc: float,
    gamma: float | None = None,
) -> dict:
    """Einstein timing delay parameter for a binary pulsar.

    The Einstein delay gamma is the amplitude of the combined
    gravitational-redshift + time-dilation variation over the orbit::

        gamma = e Pb / (2 pi c^2) * (G Mt)^{2/3} * (M2 (M1 + 2 M2)) / Mt^{4/3}

    but it is customary to treat gamma as an observed (fitted) parameter
    and use it as a mass constraint.  Here we compute the maximum timing
    excursion.

    Parameters
    ----------
    Pb : float
        Orbital period [s].
    ecc : float
        Orbital eccentricity.
    gamma : float or None
        Observed Einstein delay parameter [s].  If None the function
        returns a placeholder.

    Returns
    -------
    dict with keys:
        ``gamma``        : the Einstein delay parameter [s]
        ``max_delay``    : maximum timing excursion = gamma * ecc [s]
    """
    if gamma is None:
        return {
            "gamma": None,
            "max_delay": None,
        }

    return {
        "gamma": gamma,
        "max_delay": gamma * ecc,
    }


# ---------------------------------------------------------------------------
# PSR B1913+16 (Hulse-Taylor pulsar) catalogue
# ---------------------------------------------------------------------------

def hulse_taylor_parameters() -> dict:
    """Return a dictionary of observed and GR-predicted parameters for
    PSR B1913+16 (the Hulse-Taylor binary pulsar).

    Values are taken from Weisberg & Taylor (2005, ASPC 328, 25) and
    Weisberg, Nice & Taylor (2010, ApJ 722, 1030).

    Returns
    -------
    dict with the following keys (all floats in SI unless noted):

        Orbital parameters:
            Pb            – orbital period [s]
            ecc           – eccentricity
            omega         – longitude of periastron [deg]
            Pb_dot_obs    – observed orbital period derivative (dimensionless)
            omega_dot_obs – observed periastron advance [deg/yr]
            gamma_obs     – observed Einstein delay [s]

        Mass measurements:
            M1            – pulsar mass [kg]
            M2            – companion mass [kg]

        GR predictions:
            omega_dot_gr  – predicted periastron advance [deg/yr]
            Pb_dot_gr     – predicted orbital period derivative (dimensionless)

        Derived:
            a_total       – total semi-major axis [m]
            v_orb         – characteristic orbital velocity [m/s]
    """
    # Observed orbital parameters
    Pb = 7.751938773864 * 3600.0  # orbital period [s]
    ecc = 0.6171334                # eccentricity
    omega_deg = 292.54487          # longitude of periastron [deg]
    Pb_dot_obs = -2.4211e-12       # observed P_dot (dimensionless)
    omega_dot_obs = 4.226595       # observed periastron advance [deg/yr]
    gamma_obs = 4.2992e-3          # Einstein delay [s]

    # Best-fit masses (from DDGR model)
    M1 = 1.4398 * M_SUN  # pulsar [kg]
    M2 = 1.3886 * M_SUN  # companion [kg]

    # Compute derived orbital parameters
    Mt = M1 + M2
    a_total = (G * Mt * Pb**2 / (4.0 * np.pi**2)) ** (1.0 / 3.0)
    v_orb = 2.0 * np.pi * a_total / Pb

    # GR prediction for periastron advance
    n = 2.0 * np.pi / Pb
    omega_dot_gr_rad_s = (
        3.0
        * n ** (5.0 / 3.0)
        * (G * Mt) ** (2.0 / 3.0)
        / (c**2 * (1.0 - ecc**2))
    )
    omega_dot_gr = omega_dot_gr_rad_s * (180.0 / np.pi) * (
        365.25 * 86400.0
    )  # deg/yr

    # GR prediction for orbital period derivative (Peters 1964)
    # Pb_dot/Pb = -(192 pi / 5 c^5) (G Mt n)^{5/3} (M1 M2 / Mt^{1/3}) f(e)
    # f(e) = (1 + 73/24 e^2 + 37/96 e^4) / (1-e^2)^{7/2}
    f_e = (
        (1.0 + 73.0 / 24.0 * ecc**2 + 37.0 / 96.0 * ecc**4)
        / (1.0 - ecc**2) ** 3.5
    )
    Pb_dot_gr = (
        -(192.0 * np.pi / (5.0 * c**5))
        * (G * Mt * n) ** (5.0 / 3.0)
        * (M1 * M2 / Mt ** (1.0 / 3.0))
        * f_e
    )

    return {
        # Orbital parameters
        "Pb": Pb,
        "ecc": ecc,
        "omega": omega_deg,
        "Pb_dot_obs": Pb_dot_obs,
        "omega_dot_obs": omega_dot_obs,
        "gamma_obs": gamma_obs,
        # Masses
        "M1": M1,
        "M2": M2,
        # GR predictions
        "omega_dot_gr": omega_dot_gr,
        "Pb_dot_gr": Pb_dot_gr,
        # Derived
        "a_total": a_total,
        "v_orb": v_orb,
    }
