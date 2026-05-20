"""Physical constants in SI units and geometric units.

In geometric units (G = c = 1), mass has dimensions of length.
We provide both SI constants and convenient conversion factors.
"""

# SI constants
G = 6.67430e-11        # gravitational constant [m^3 kg^-1 s^-2]
C = 2.99792458e8       # speed of light [m/s]
HBAR = 1.054571817e-34 # reduced Planck constant [J s]
M_SUN = 1.98892e30     # solar mass [kg]

# Derived: gravitational radius for 1 solar mass
# r_g = G * M_SUN / C^2
R_G_SUN = G * M_SUN / C**2  # ~1.477 km

# Planck mass
M_PL = (HBAR * C / G) ** 0.5  # ~2.176e-8 kg
