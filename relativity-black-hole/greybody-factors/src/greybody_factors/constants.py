"""Physical constants in SI units."""

# Gravitational constant [m^3 kg^-1 s^-2]
G = 6.67430e-11

# Speed of light in vacuum [m/s]
C = 2.99792458e8

# Reduced Planck constant [J s]
HBAR = 1.054571817e-34

# Boltzmann constant [J/K]
K_B = 1.380649e-23

# Solar mass [kg]
M_SUN = 1.98892e30

# Stefan-Boltzmann constant [W m^-2 K^-4]
SIGMA_SB = 5.670374419e-8

# Planck mass [kg]
M_PL = (HBAR * C / G) ** 0.5

# Planck length [m]
L_PL = (HBAR * G / C ** 3) ** 0.5

# Planck time [s]
T_PL = (HBAR * G / C ** 5) ** 0.5

# Planck energy [J]
E_PL = (HBAR * C ** 5 / G) ** 0.5

# Planck temperature [K]
T_PL = E_PL / K_B
