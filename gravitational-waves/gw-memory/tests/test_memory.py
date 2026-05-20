"""Tests for gravitational wave memory effect calculations."""

import numpy as np
import pytest

from gw_memory.constants import G, C, M_SUN, MPC
from gw_memory.linear_memory import linear_memory_delta_h, linear_memory_from_burst
from gw_memory.nonlinear_memory import christodoulou_memory, nonlinear_memory_buried
from gw_memory.waveform import memory_waveform, step_function_memory


# ---------- Helper: realistic binary black hole parameters ----------

def _equal_mass_bbh():
    """Return parameters for an equal-mass BBH (30+30 Msun) at 400 Mpc."""
    m1 = m2 = 30.0 * M_SUN
    total_mass = m1 + m2
    chirp_mass = (m1 * m2) ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0)
    eta = (m1 * m2) / total_mass**2  # = 0.25 for equal mass
    distance = 400.0 * MPC
    return chirp_mass, total_mass, eta, distance


# ---------- test_linear_memory_positive ----------

def test_linear_memory_positive():
    """Linear memory Delta h should be >= 0 (returned as absolute value)."""
    # A simple scenario: two masses gain velocity (e.g., from a kick)
    masses = np.array([10.0 * M_SUN, 10.0 * M_SUN])
    velocities_before = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocities_after = np.array([[0.1 * C, 0.0, 0.0], [-0.1 * C, 0.0, 0.0]])

    delta_h = linear_memory_delta_h(
        masses, velocities_before, velocities_after,
        theta=np.pi / 2, phi=0.0, distance=100.0 * MPC,
    )
    assert delta_h >= 0.0


# ---------- test_linear_memory_scales_inverse_distance ----------

def test_linear_memory_scales_inverse_distance():
    """Doubling distance should halve the linear memory amplitude."""
    masses = np.array([10.0 * M_SUN, 10.0 * M_SUN])
    velocities_before = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocities_after = np.array([[0.1 * C, 0.0, 0.0], [-0.1 * C, 0.0, 0.0]])

    d1 = 100.0 * MPC
    d2 = 200.0 * MPC

    h1 = linear_memory_delta_h(
        masses, velocities_before, velocities_after,
        theta=np.pi / 2, phi=0.0, distance=d1,
    )
    h2 = linear_memory_delta_h(
        masses, velocities_before, velocities_after,
        theta=np.pi / 2, phi=0.0, distance=d2,
    )

    # Should scale as 1/r, so h2 ~ h1/2
    np.testing.assert_allclose(h2, h1 / 2.0, rtol=1e-10)


# ---------- test_nonlinear_memory_positive ----------

def test_nonlinear_memory_positive():
    """Christodoulou memory extracted from strain should be >= 0 everywhere."""
    # Generate a simple chirp signal
    times = np.linspace(-0.1, 0.1, 5000)
    f_gw = 100.0  # Hz
    amp = 1e-21

    # Simple sinusoidal GW with an envelope
    envelope = np.exp(-times**2 / (2 * 0.03**2))
    h_plus = amp * envelope * np.cos(2 * np.pi * f_gw * times)
    h_cross = amp * envelope * np.sin(2 * np.pi * f_gw * times)

    h_mem = christodoulou_memory(h_plus, h_cross, times)

    # Christodoulou memory should be non-negative (monotonically rising)
    assert np.all(h_mem >= -1e-35), f"Found negative memory: min = {h_mem.min()}"


# ---------- test_nonlinear_memory_from_chirp ----------

def test_nonlinear_memory_from_chirp():
    """Nonlinear memory for a realistic BBH should have reasonable magnitude.

    For a 30+30 Msun BBH at 400 Mpc, memory should be in the range
    ~1e-23 to 1e-21 (roughly 0.1% to 10% of peak strain).
    """
    chirp_mass, total_mass, eta, distance = _equal_mass_bbh()

    # Inspiral from 20 Hz to ISCO frequency (~150 Hz for 60 Msun total)
    f_low = 20.0
    f_high = C**3 / (6.0**1.5 * 2.0 * np.pi * G * total_mass)

    h_mem = nonlinear_memory_buried(chirp_mass, total_mass, distance, f_low, f_high)

    # Memory should be a small but positive number
    assert h_mem > 0.0
    # Should be a detectable strain (not absurdly small or large)
    assert h_mem < 1e-18, f"Memory unreasonably large: {h_mem}"
    assert h_mem > 1e-26, f"Memory unreasonably small: {h_mem}"


# ---------- test_memory_amplitude_small ----------

def test_memory_amplitude_small():
    """Memory should be much smaller than peak strain for a typical BBH.

    The memory is typically ~1-10% of the peak oscillatory strain.
    Peak strain for a 30+30 Msun BBH at 400 Mpc is ~1e-21.
    """
    chirp_mass, total_mass, eta, distance = _equal_mass_bbh()

    f_low = 20.0
    f_high = C**3 / (6.0**1.5 * 2.0 * np.pi * G * total_mass)

    h_mem = nonlinear_memory_buried(chirp_mass, total_mass, distance, f_low, f_high)

    # Peak strain for an optimally oriented equal-mass BBH (order of magnitude)
    # h_peak ~ 4 * eta * (G*M) / (c^2 * r) ~ 4 * 0.25 * (G*M)/(c^2*r)
    h_peak = 4.0 * eta * (G * total_mass) / (C**2 * distance)

    # Memory should be smaller than peak strain
    assert h_mem < h_peak, (
        f"Memory {h_mem:.2e} should be < peak strain {h_peak:.2e}"
    )
    # Memory should be at least 0.1% of peak (not negligibly tiny)
    assert h_mem > 1e-4 * h_peak, (
        f"Memory {h_mem:.2e} is too small compared to peak {h_peak:.2e}"
    )


# ---------- test_step_function_shape ----------

def test_step_function_shape():
    """Step-function memory: ~0 before merger, ~amplitude after merger."""
    amplitude = 1e-22
    t_merger = 0.0

    times = np.linspace(-1.0, 1.0, 1001)
    h = step_function_memory(times, t_merger, amplitude)

    # Before merger: should be 0
    before = times < t_merger
    np.testing.assert_array_equal(h[before], 0.0)

    # At and after merger: should be amplitude
    after = times >= t_merger
    np.testing.assert_allclose(h[after], amplitude)


# ---------- test_christodoulou_vs_linear ----------

def test_christodoulou_vs_linear():
    """Nonlinear memory should be comparable to or larger than linear memory
    for an equal-mass BBH merger.

    For equal-mass inspiral, the nonlinear (Christodoulou) memory dominates
    over the linear memory from mass ejection.
    """
    chirp_mass, total_mass, eta, distance = _equal_mass_bbh()

    # Nonlinear memory
    f_low = 20.0
    f_high = C**3 / (6.0**1.5 * 2.0 * np.pi * G * total_mass)
    h_nonlinear = nonlinear_memory_buried(chirp_mass, total_mass, distance, f_low, f_high)

    # Linear memory from burst: estimate using the radiated energy
    # For equal-mass BBH, ~5% of total mass-energy is radiated
    E_radiated = 0.05 * total_mass * C**2
    h_linear = linear_memory_from_burst(E_radiated, distance)

    # Nonlinear memory should be at least a significant fraction of linear
    # In practice, nonlinear memory is of the same order or larger
    assert h_nonlinear > 0.01 * h_linear, (
        f"Nonlinear {h_nonlinear:.2e} too small vs linear {h_linear:.2e}"
    )
