"""Nonlinear (Christodoulou) gravitational wave memory.

Nonlinear memory arises from the contribution of the gravitational waves
themselves to the total energy flux at null infinity. Unlike linear memory,
which comes from the change in the source's mass quadrupole, nonlinear memory
is produced by the GW energy carrying energy-momentum away from the system.

This is also known as the Christodoulou memory (Christodoulou 1991) or the
nonlinear memory effect (Blanchet & Damour 1992).

References:
    Christodoulou (1991), Blanchet & Damour (1992),
    Favata (2009, 2010), Thornburg (2018)
"""

import numpy as np
from scipy import integrate

from .constants import G, C, M_SUN, MPC


def christodoulou_memory(h_plus, h_cross, times):
    """Extract nonlinear (Christodoulou) memory from GW strain data.

    The nonlinear memory is computed by integrating the squared time derivative
    of the oscillatory GW strain over time. The key formula is:

        h_memory(t) = (1 / (2 * pi)) * integral_0^t dt'
            * [ (dh_+/dt')^2 + (dh_x/dt')^2 ]

    More precisely, using the Favata (2009) formulation:

        h_memory(t) = (r / (4 * pi)) * integral_0^t dt'
            * (1/r^2) * [ (dh_+/dt')^2 + (dh_x/dt')^2 ]

    For numerical computation we use the dimensionless strain directly:

        h_mem(t) = (1 / (4 * pi)) * integral_0^t
            [ (dh_+/dt')^2 + (dh_x/dt')^2 ] dt'

    Parameters
    ----------
    h_plus : array_like
        Plus polarization of the GW strain (dimensionless).
    h_cross : array_like
        Cross polarization of the GW strain (dimensionless).
    times : array_like
        Time array corresponding to the strain data [s].

    Returns
    -------
    numpy.ndarray
        The nonlinear memory signal h_memory(t) at each time sample.
        Monotonically increasing (or constant), always >= 0.
    """
    h_plus = np.asarray(h_plus, dtype=float)
    h_cross = np.asarray(h_cross, dtype=float)
    times = np.asarray(times, dtype=float)

    # Compute time derivatives using central differences
    dt = np.gradient(times)

    hdot_plus = np.gradient(h_plus, times)
    hdot_cross = np.gradient(h_cross, times)

    # Integrand: sum of squared time derivatives
    integrand = hdot_plus**2 + hdot_cross**2

    # Cumulative integration using trapezoidal rule
    h_memory = integrate.cumulative_trapezoid(integrand, times, initial=0.0)

    # Apply the (1 / 4*pi) prefactor
    h_memory /= (4.0 * np.pi)

    return h_memory


def nonlinear_memory_buried(chirp_mass, total_mass, distance, f_low, f_high):
    """Theoretical estimate of nonlinear memory amplitude for a compact binary.

    Uses the Favata (2009) fitting formula for the nonlinear memory amplitude
    accumulated during the inspiral from f_low to f_high:

        h_mem ~ (5 / (24 * pi)) * eta * (G * M * pi * f)^(2/3) / (c^2 * r)
              * (G * M * pi * f / c^3)^(2/3)

    More practically, following Favata (2009) Eq. (3.1):

        h_mem = (5 / 24*pi) * (eta / r) * (G*M_c / c^2)^(5/3)
              * (pi * f_low / c)^(2/3) * [1 - (f_low / f_high)^(2/3)] * (1 / c^2)

    Simplified for the total memory accumulated over the full coalescence
    (assuming f_high >> f_low, so the bracketed term approaches 1):

        h_mem ~ (5 / (24*pi)) * (G*M_c/c^3)^(5/3) * (pi*f_low)^(2/3)
              * eta / (r * c^(1/3))

    We use a more standard form from the literature. The Christodoulou memory
    amplitude for the full inspiral-merger-ringdown is approximately:

        h_mem ~ eta * (G*M)/(c^2 * r) * (G*M*pi*f_low/c^3)^(2/3)

    Parameters
    ----------
    chirp_mass : float
        Chirp mass of the binary [kg].
    total_mass : float
        Total mass of the binary [kg].
    distance : float
        Luminosity distance to source [m].
    f_low : float
        Lower frequency cutoff of the inspiral [Hz].
    f_high : float
        Upper frequency cutoff (roughly the ISCO/ringdown frequency) [Hz].

    Returns
    -------
    float
        Estimated nonlinear memory amplitude (dimensionless strain). Always >= 0.
    """
    # Symmetric mass ratio from chirp mass and total mass
    # M_c = eta^(3/5) * M  =>  eta = (M_c / M)^(5/3)
    eta = (chirp_mass / total_mass) ** (5.0 / 3.0)

    # Dimensionless characteristic velocity parameter
    # v/c at f_low: v = (pi * G * M * f_low)^(1/3)
    v_low = (np.pi * G * total_mass * f_low) ** (1.0 / 3.0) / C
    v_high = (np.pi * G * total_mass * f_high) ** (1.0 / 3.0) / C

    # Favata (2009) formula for memory amplitude (Eq 3.1 simplified)
    # h_mem = (5/24*pi) * eta * (G*M)/(c^2 * r) * [v_high^(7/3) - v_low^(7/3)] * (1/eta^(1/3))
    # But more commonly expressed as:
    # h_mem = eta * (G*M)/(c^2 * r) * v_high^2  (approximate, for full coalescence)
    #
    # Using the full Favata (2009) Eq (3.1):
    # h_mem = (5/(24*pi)) * (1/eta^(1/3)) * eta * (G*M)/(c^2*r) * (v^2) evaluated between limits
    # We use a simpler but accurate form:
    prefactor = eta * (G * total_mass) / (C**2 * distance)

    # Memory accumulated between v_low and v_high
    memory_amplitude = prefactor * (v_high**2 - v_low**2)

    return abs(memory_amplitude)
