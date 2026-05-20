"""Physical constants."""

G = 6.67430e-11       # m^3 kg^-1 s^-2
C = 2.99792458e8      # m/s
M_SUN = 1.98892e30    # kg
AU = 1.49598e11       # m
R_SUN = 6.9634e8      # m
J2_SUN = 2.2e-7

# Planets (mass kg, semi-major axis m, eccentricity)
MERCURY = {"mass": 3.3011e23, "a": 5.7909e10, "e": 0.205630, "T_days": 87.969}
VENUS = {"mass": 4.8675e24, "a": 1.0821e11, "e": 0.006772, "T_days": 224.701}
EARTH = {"mass": 5.9722e24, "a": 1.49598e11, "e": 0.016709, "T_days": 365.256}
MARS = {"mass": 6.4171e23, "a": 2.2794e11, "e": 0.0934, "T_days": 686.980}
JUPITER = {"mass": 1.8982e27, "a": 7.7857e11, "e": 0.0489, "T_days": 4332.59}
SATURN = {"mass": 5.6834e26, "a": 1.4335e12, "e": 0.0565, "T_days": 10759.22}
