"""Physical constants for neutrino cosmology calculations.

All values in SI units unless otherwise noted.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------

# Gravitational constant [m^3 kg^-1 s^-2]
G = 6.67430e-11

# Speed of light [m/s]
C = 2.99792458e8

# Boltzmann constant [J/K]
K_B = 1.380649e-23

# Reduced Planck constant [J s]
HBAR = 1.054571817e-34

# ---------------------------------------------------------------------------
# Cosmological / particle-physics constants
# ---------------------------------------------------------------------------

# CMB temperature today [K]
T_CMB0 = 2.7255

# Thomson cross-section [m^2]
SIGMA_T = 6.6524587321e-29

# Electron mass [kg]
m_e = 9.1093837015e-31

# Proton mass [kg]
m_p = 1.67262192369e-27

# ---------------------------------------------------------------------------
# Neutrino-specific constants
# ---------------------------------------------------------------------------

# Effective number of neutrino species (standard model)
N_EFF_STANDARD = 3.044

# Temperature ratio T_nu / T_gamma after e+e- annihilation
T_NU_OVER_T_GAMMA = (4.0 / 11.0) ** (1.0 / 3.0)

# Conversion: 1 eV in Joules
EV_TO_J = 1.602176634e-19

# Neutrino mass density relation: Omega_nu = Sigma_m_nu / (93.14 eV * h^2)
OMEGA_NU_DENOMINATOR_EV = 93.14  # eV

# Fermi-Dirac constant: g * pi^2 / 30 for a single Weyl fermion
# rho = (7/8) * (pi^2/30) * g * T^4 in natural units
# For a single neutrino species (g = 1 for Weyl):
#   rho_nu = (7 pi^2 / 240) * T^4   (per species, massless, natural units)
FD_CONSTANT = 7.0 * np.pi**2 / 240.0

# Stefan-Boltzmann constant [W m^-2 K^-4]
STEFAN_BOLTZMANN = 5.670374419e-8

# Photon energy density constant: a_rad = pi^2 / (15 hbar^3 c^3) k_B^4
# rho_gamma = a_rad * T^4
A_RAD = np.pi**2 / 15.0  # in natural units (hbar=c=k_B=1)
