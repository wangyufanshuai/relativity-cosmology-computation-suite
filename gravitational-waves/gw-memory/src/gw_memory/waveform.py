"""Memory waveform generation.

Provides functions to generate gravitational wave memory waveforms for
compact binary coalescences. The memory signal is a slowly rising,
non-oscillatory component that settles to a permanent offset after the
merger is complete.
"""

import numpy as np

from .constants import G, C, M_SUN, MPC


def memory_waveform(times, chirp_mass, total_mass, eta, distance):
    """Generate a full memory waveform h_memory(t) for a compact binary merger.

    The memory signal rises during the inspiral as the GW amplitude (and thus
    the energy flux) increases, then saturates at merger. We model it using
    a phenomenological approach:

    1. Compute the characteristic time of coalescence (merger time).
    2. Model the memory as a monotonic rise proportional to the accumulated
       GW energy flux integral.
    3. The rise follows a tanh-like sigmoid centered at the merger time.

    The amplitude is estimated from the Favata (2009) nonlinear memory
    formula for the full coalescence.

    Parameters
    ----------
    times : array_like
        Time array [s]. Assumed centered such that t=0 is near merger.
    chirp_mass : float
        Chirp mass of the binary [kg].
    total_mass : float
        Total mass of the binary [kg].
    eta : float
        Symmetric mass ratio (0 < eta <= 0.25).
    distance : float
        Luminosity distance to source [m].

    Returns
    -------
    numpy.ndarray
        Memory strain h_memory(t) at each time. Non-negative, monotonically
        rising to a final offset.
    """
    times = np.asarray(times, dtype=float)

    # Estimate merger time: roughly the midpoint of the time array
    # (assumes times span the inspiral through ringdown)
    t_merger = 0.0

    # Characteristic frequency at ISCO (innermost stable circular orbit)
    f_isco = C**3 / (6.0**1.5 * 2.0 * np.pi * G * total_mass)

    # Characteristic time scale: the GW period at ISCO
    tau = 1.0 / f_isco

    # Memory amplitude from the nonlinear memory estimate
    # Using the formula: h_mem ~ eta * (G*M)/(c^2*r) * v_isco^2
    v_isco = (np.pi * G * total_mass * f_isco) ** (1.0 / 3.0) / C
    amplitude = eta * (G * total_mass) / (C**2 * distance) * v_isco**2

    # Model the memory rise as a sigmoid (tanh) function
    # The memory builds up during inspiral and saturates after ringdown
    # Use a smooth transition width of ~ tau
    h_memory = 0.5 * amplitude * (1.0 + np.tanh((times - t_merger) / (2.0 * tau)))

    return h_memory


def step_function_memory(t, t_merger, amplitude):
    """Simplified step-function model for gravitational wave memory.

    Models the memory as an instantaneous step at the merger time:

        h_memory(t) = 0            for t < t_merger
        h_memory(t) = amplitude    for t >= t_merger

    This is the simplest possible model, useful for order-of-magnitude
    estimates and pedagogical purposes.

    Parameters
    ----------
    t : float or array_like
        Time(s) at which to evaluate the memory [s].
    t_merger : float
        Time of the merger (step location) [s].
    amplitude : float
        Memory amplitude (dimensionless strain). Should be > 0.

    Returns
    -------
    float or numpy.ndarray
        Memory strain at the given time(s).
    """
    t = np.asarray(t, dtype=float)
    result = np.where(t >= t_merger, amplitude, 0.0)
    return result
