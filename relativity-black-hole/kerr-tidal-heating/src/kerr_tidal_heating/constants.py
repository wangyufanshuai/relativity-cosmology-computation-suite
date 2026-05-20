"""Physical constants in SI units and geometric units."""

import numpy as np

# --- SI constants ---
G_SI: float = 6.67430e-11  # m^3 kg^-1 s^-2
C_SI: float = 2.99792458e8  # m/s
M_SUN_SI: float = 1.98892e30  # kg

# Gravitational radius per solar mass: G*M_sun / c^2
R_G_PER_MSUN: float = G_SI * M_SUN_SI / C_SI**2  # ~1.477 km

# Speed of light (convenience alias in geometric units where c = 1)
C: float = 1.0

# Gravitational constant (geometric units G = 1)
G: float = 1.0

# Solar mass in geometric mass units (metres)
M_SUN: float = R_G_PER_MSUN  # G * M_sun / c^2 in metres
