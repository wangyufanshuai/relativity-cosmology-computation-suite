"""Tidal heating: energy dissipation and gravitational-wave phase corrections.

Tidal heating arises from the coupling of a compact object's tidal field to
the spinning black hole.  In the Kerr spacetime the horizon acts as a
viscous membrane (Hartle-Hawking, Damour), dissipating energy at a rate
that depends on the spin and the orbital separation.
"""

from __future__ import annotations

import numpy as np


def tidal_heating_rate(M: float, a: float, r: float, m: float) -> float:
    """Energy dissipation rate from tidal coupling (horizon dissipation).

    Following Poisson & Sasaki (1995) and Alvi (2001), the leading-order
    tidal-heating power for a body of mass m orbiting a Kerr black hole of
    mass M at separation r (circular orbit, equatorial) is:

        dE/dt = -(32/5) * eta^2 * (M/r)^5 * (M * omega_orb)^2 * M / r^2

    where eta = m*M / (m+M)^2 is the symmetric mass ratio and
    omega_orb = sqrt(M / r^3) is the Keplerian angular frequency.

    For simplicity we use:

        dE/dt ~ (32/5) * mu^2 * M^3 / r^5

    where mu = m is the small-body mass (test-particle limit).

    This quantity is always positive (energy is dissipated at the horizon).

    Parameters
    ----------
    M : float
        Black hole mass (geometric units).
    a : float
        Spin parameter (enters at higher order, included as a correction factor).
    r : float
        Orbital separation (Boyer-Lindquist coordinate).
    m : float
        Companion mass (test particle limit: m << M).

    Returns
    -------
    float
        Tidal heating rate (positive definite).
    """
    if r <= 0 or M <= 0 or m < 0:
        return 0.0

    # Leading-order quadrupolar dissipation
    P0 = (32.0 / 5.0) * m**2 * M**3 / r**5

    # Spin correction factor (enhances dissipation for prograde orbits)
    # At leading order the correction is 1 + O(a/M); we include a mild
    # spin-dependent enhancement.
    if M > 0:
        a_star = a / M
    else:
        a_star = 0.0

    spin_factor = 1.0 + 0.5 * a_star**2
    return P0 * spin_factor


def tidal_phase_shift(M: float, a: float, r: float, m: float, omega_orb: float) -> float:
    """Tidal-heating phase correction to the gravitational-waveform.

    The accumulated GW phase is modified by tidal heating at 5PN relative
    order.  The leading correction is:

        delta_Psi ~ -(dE/dt)_tidal * t_obs / (dE/dt)_GW

    For a circular orbit the gravitational-wave luminosity is

        L_GW = (32/5) * mu^2 * M^3 / r^5  (leading Newtonian)

    and the tidal correction introduces a spin-dependent phase shift:

        delta_Psi = -(5/64) * (a/M) * (M/r)^(7/2) * (M * omega_orb)

    This is a simplified estimate; a full treatment requires Teukolsky
    formalism.

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    r : float
        Orbital separation.
    m : float
        Companion mass.
    omega_orb : float
        Orbital angular frequency.

    Returns
    -------
    float
        Phase shift in radians.
    """
    if M <= 0 or r <= 0:
        return 0.0

    a_star = a / M if M > 0 else 0.0
    # Phase correction from horizon dissipation
    delta_psi = (5.0 / 64.0) * a_star * (M / r) ** 3.5 * (M * omega_orb)
    return delta_psi
