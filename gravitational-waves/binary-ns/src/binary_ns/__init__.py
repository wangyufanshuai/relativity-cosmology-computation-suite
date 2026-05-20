"""
binary_ns: Binary neutron star merger simulation.

Modules
-------
eos : Equation of state models (polytropic, piecewise polytrope).
tov  : Tolman-Oppenheimer-Volkoff equation solver.
tidal: Tidal deformability and Love numbers.
inspiral: Inspiral waveform with tidal corrections.
"""

from . import eos, tov, tidal, inspiral

__all__ = ["eos", "tov", "tidal", "inspiral"]
