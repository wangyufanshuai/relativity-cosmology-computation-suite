"""Physical constants and astronomical units.

All values in SI units unless otherwise noted.
"""

# Fundamental constants
G = 6.67430e-11          # Gravitational constant [m^3 kg^-1 s^-2]
C = 2.99792458e8         # Speed of light [m/s]
HBAR = 1.054571817e-34   # Reduced Planck constant [J s]
K_B = 1.380649e-23       # Boltzmann constant [J/K]

# Solar mass
M_SUN = 1.98892e30       # Solar mass [kg]

# CMB and neutrino temperatures
T_CMB = 2.7255           # CMB temperature today [K]
T_NU = 1.95              # Neutrino temperature after e+e- annihilation [K]

# Unit conversions
MPC_IN_M = 3.0857e22     # 1 Mpc in meters

# Planck mass
PLANCK_MASS = 2.1764e-8  # Planck mass [kg]

# Derived convenience
KM_PER_MPC_IN_S = C / MPC_IN_M * 1e-3  # c in km/s per Mpc ~ 1/(H0 in s) conversion factor
