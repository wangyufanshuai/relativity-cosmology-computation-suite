"""
Physical constants for semiclassical gravity calculations.

All values in SI units unless otherwise noted. Geometrized/natural units
can be constructed from these for specific computations.
"""

import numpy as np

# Newton's gravitational constant [m^3 kg^-1 s^-2]
G = 6.67430e-11

# Speed of light in vacuum [m/s]
C = 2.99792458e8

# Reduced Planck constant [J s]
HBAR = 1.054571817e-34

# Planck mass [kg]: M_PL = sqrt(hbar * c / G)
M_PL = np.sqrt(HBAR * C / G)

# Planck length [m]: L_PL = sqrt(hbar * G / c^3)
L_PL = np.sqrt(HBAR * G / C**3)

# Planck time [s]: T_PL = sqrt(hbar * G / c^5)
T_PL = np.sqrt(HBAR * G / C**5)

# Planck energy [J]: E_PL = sqrt(hbar * c^5 / G)
E_PL = np.sqrt(HBAR * C**5 / G)

# Useful dimensionless combination for 1+1D: kappa = (N/24pi)
# where N is the number of scalar fields. Default: N=1.
DEFAULT_KAPPA = 1.0 / (24.0 * np.pi)

# Pi
PI = np.pi
