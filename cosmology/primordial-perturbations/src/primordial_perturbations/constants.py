"""Physical constants in SI units."""

# Gravitational constant [m^3 kg^-1 s^-2]
G = 6.67430e-11

# Speed of light [m/s]
C = 2.99792458e8

# Reduced Planck constant [J s]
HBAR = 1.054571817e-34

# Reduced Planck mass [kg]
# M_Pl = sqrt(hbar * c / (8 pi G))
M_PL = (HBAR * C / (8.0 * G)) ** 0.5
