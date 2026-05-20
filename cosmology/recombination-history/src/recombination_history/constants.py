"""Physical constants in SI units used throughout the recombination history calculator."""

import numpy as np

# Gravitational constant [m^3 kg^-1 s^-2]
G = 6.67430e-11

# Speed of light [m/s]
C = 2.99792458e8

# Boltzmann constant [J/K]
K_B = 1.380649e-23

# Reduced Planck constant [J s]
HBAR = 1.054571817e-34

# Planck mass [kg]
M_PL = 2.176434e-8

# Thomson scattering cross-section [m^2]
SIGMA_T = 6.6524587321e-29

# Electron mass [kg]
M_E = 9.1093837015e-31

# Proton mass [kg]
M_P = 1.67262192369e-27

# Hydrogen ionization energy from n=1: 13.6 eV [J]
E_ION_H = 2.1798723611035e-18

# Hydrogen Lyman-alpha energy (n=2 -> n=1): 10.2 eV [J]
E_LYMAN_ALPHA = 1.634049027e-18

# Hydrogen energy difference between n=2 and n=1 [J]
E_21 = E_LYMAN_ALPHA

# Fine-structure constant
ALPHA_FS = 7.2973525693e-3

# Stefan-Boltzmann constant [W m^-2 K^-4]
SIGMA_SB = 5.670374419e-8

# Radiation constant a [J m^-3 K^-4]
A_RAD = 4.0 * SIGMA_SB / C

# CMB temperature today [K]
T_CMB0 = 2.7255

# Critical density [kg m^-3], computed from H0 in SI
# H0 = 100 h km/s/Mpc
H0_SI = 100.0e3 / (3.0856775814913673e22)  # 100 km/s/Mpc in s^-1
RHO_CRIT_FACTOR = 3.0 * H0_SI**2 / (8.0 * np.pi * G)

# Mean molecular weight per baryon (approximately proton mass)
M_H = M_P

# Helium-4 mass [kg] (approximately 4 * proton mass for our purposes)
M_HE4 = 4.0 * M_P

# Atomic unit of energy (Hartree) [J]
E_HARTREE = 4.3597447222071e-18

# Number of neutrino species (standard)
N_eff = 3.046

# Conversion: 1 eV = 1.602176634e-19 J
EV_TO_J = 1.602176634e-19
