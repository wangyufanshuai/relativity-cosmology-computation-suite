"""Standard cosmological parameters (Planck 2018 baseline)."""

import numpy as np

# Hubble constant in km/s/Mpc
H0_DEFAULT = 67.4

# Dimensionless Hubble parameter h = H0 / 100
H_DEFAULT = H0_DEFAULT / 100.0

# Matter density parameter
OMEGA_M_DEFAULT = 0.315

# Baryon density parameter
OMEGA_B_DEFAULT = 0.049

# Dark energy density parameter (flat universe: Omega_lambda = 1 - Omega_m)
OMEGA_LAMBDA_DEFAULT = 1.0 - OMEGA_M_DEFAULT

# CMB temperature in Kelvin
T_CMB_DEFAULT = 2.7255

# Primordial scalar amplitude
A_S_DEFAULT = 2.1e-9

# Scalar spectral index
N_S_DEFAULT = 0.965

# Pivot scale in 1/Mpc
K_PIVOT_DEFAULT = 0.05  # 1/Mpc

# Photon density parameter today
# rho_gamma = pi^2 / 15 * T_CMB^4 * k_B^4 / (hbar^3 c^5)
# Omega_gamma h^2 = 2.469e-5 * (T_CMB / 2.7255)^4
OMEGA_GAMMA_H2 = 2.469e-5  # for T_CMB = 2.7255 K

# Radiation density parameter (including 3 massless neutrino species)
# Omega_nu = 3.046 * 7/8 * (4/11)^(4/3) * Omega_gamma
# Omega_r h^2 = Omega_gamma h^2 * (1 + 0.2271 * N_eff)
# with N_eff = 3.046
OMEGA_R_H2 = OMEGA_GAMMA_H2 * (1.0 + 0.2271 * 3.046)

# Speed of light in km/s
C_KM_S = 2.998e5

# Newton's constant in Mpc^3 / (Msun * s^2)
# G = 4.3009e-9 Mpc (km/s)^2 / Msun
G_NEWT = 4.3009e-9  # Mpc (km/s)^2 / Msun

# Boltzmann constant in eV/K
K_BOLTZMANN_EV = 8.617e-5

# Critical density in Msun / Mpc^3 (for h=1)
RHO_CRIT = 2.775e11  # Msun / Mpc^3 (h=1)
