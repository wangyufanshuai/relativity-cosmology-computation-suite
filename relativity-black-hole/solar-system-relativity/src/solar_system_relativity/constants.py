"""Solar System body data and physical constants."""

G = 6.67430e-11
C = 2.99792458e8
AU = 1.495978707e11
YEAR = 365.25 * 86400.0

# Body data: mass (kg), semi-major axis (m), eccentricity, inclination (rad),
# longitude of ascending node (rad), argument of perihelion (rad), mean anomaly (rad)
# Epoch J2000.0
BODIES = {
    "Sun": {
        "mass": 1.98892e30,
        "a": 0.0,
        "e": 0.0,
        "incl": 0.0,
        "Omega": 0.0,
        "omega": 0.0,
        "M0": 0.0,
        "radius": 6.9634e8,
        "J2": 2.2e-7,
        "color": "#FFD700",
    },
    "Mercury": {
        "mass": 3.3011e23,
        "a": 5.7909e10,
        "e": 0.205630,
        "incl": 7.005 * 3.14159 / 180,
        "Omega": 48.331 * 3.14159 / 180,
        "omega": 29.124 * 3.14159 / 180,
        "M0": 174.796 * 3.14159 / 180,
        "radius": 2.4397e6,
        "color": "#A0522D",
    },
    "Venus": {
        "mass": 4.8675e24,
        "a": 1.0821e11,
        "e": 0.006772,
        "incl": 3.3947 * 3.14159 / 180,
        "Omega": 76.680 * 3.14159 / 180,
        "omega": 54.884 * 3.14159 / 180,
        "M0": 50.115 * 3.14159 / 180,
        "radius": 6.0518e6,
        "color": "#DEB887",
    },
    "Earth": {
        "mass": 5.9722e24,
        "a": 1.49598e11,
        "e": 0.016709,
        "incl": 0.0,
        "Omega": -11.26064 * 3.14159 / 180,
        "omega": 114.20783 * 3.14159 / 180,
        "M0": 358.617 * 3.14159 / 180,
        "radius": 6.371e6,
        "color": "#4169E1",
    },
    "Mars": {
        "mass": 6.4171e23,
        "a": 2.2794e11,
        "e": 0.0934,
        "incl": 1.850 * 3.14159 / 180,
        "Omega": 49.558 * 3.14159 / 180,
        "omega": 286.502 * 3.14159 / 180,
        "M0": 19.373 * 3.14159 / 180,
        "radius": 3.3895e6,
        "color": "#CD5C5C",
    },
    "Jupiter": {
        "mass": 1.8982e27,
        "a": 7.7857e11,
        "e": 0.0489,
        "incl": 1.303 * 3.14159 / 180,
        "Omega": 100.464 * 3.14159 / 180,
        "omega": 273.867 * 3.14159 / 180,
        "M0": 20.020 * 3.14159 / 180,
        "radius": 6.9911e7,
        "color": "#DAA520",
    },
    "Saturn": {
        "mass": 5.6834e26,
        "a": 1.4335e12,
        "e": 0.0565,
        "incl": 2.489 * 3.14159 / 180,
        "Omega": 113.665 * 3.14159 / 180,
        "omega": 339.392 * 3.14159 / 180,
        "M0": 317.020 * 3.14159 / 180,
        "radius": 5.8232e7,
        "color": "#F4A460",
    },
}

PLANETS = [name for name in BODIES if name != "Sun"]
