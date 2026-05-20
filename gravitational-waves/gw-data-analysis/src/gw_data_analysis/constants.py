"""Physical constants used in gravitational wave data analysis.

All values in SI units.
"""

import numpy as np

# Gravitational constant [m^3 kg^-1 s^-2]
G = 6.67430e-11

# Speed of light [m/s]
C = 299792458.0

# Solar mass [kg]
M_SUN = 1.98892e30

# Megaparsec [m]
MPC = 3.0856775814913673e22

# Planck's constant (reduced) [J s]
HBAR = 1.054571817e-34

# Geometric factor: G * M_SUN / C^2 [m] — one solar mass in geometric units
M_SUN_GEOM = G * M_SUN / C**2

# Reference frequency for LIGO [Hz]
F_REF_LIGO = 100.0
