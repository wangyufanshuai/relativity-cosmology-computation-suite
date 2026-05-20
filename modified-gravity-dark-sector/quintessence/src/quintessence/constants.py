"""Physical constants in natural units (GeV)."""

# Newton's gravitational constant in GeV^-2
# G = 1 / M_Pl^2 where M_Pl is the reduced Planck mass
# In SI: G = 6.67430e-11 m^3 kg^-1 s^-2
G = 6.7083e-39  # GeV^-2

# Speed of light (natural units)
C = 1.0  # dimensionless in natural units

# Boltzmann constant (natural units)
K_B = 1.0  # dimensionless in natural units

# Reduced Planck constant (natural units)
HBAR = 1.0  # GeV^-1 in natural units (hbar * c)

# Reduced Planck mass M_Pl = sqrt(8 pi G)^{-1} ~ 2.435e18 GeV
M_PL = 2.435e18  # GeV

# Planck mass squared (useful for normalizing potentials)
M_PL_SQ = M_PL ** 2

# Critical density today (approximately, in GeV^4)
# rho_c,0 ~ 1.054e-5 h^2 GeV cm^-3, with h ~ 0.674
# Converted to natural units: rho_c,0 ~ 3.7e-47 GeV^4
RHO_CRIT_0 = 3.7e-47  # GeV^4 (approximate)

# 8 pi G
EIGHT_PI_G = 8.0 * 3.141592653589793 * G
