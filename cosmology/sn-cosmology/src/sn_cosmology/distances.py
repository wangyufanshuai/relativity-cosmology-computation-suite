"""Cosmological distance calculations."""

import numpy as np
from scipy.integrate import quad

from .constants import C_KM_S, H0_DEFAULT


def hubble_distance(H0=H0_DEFAULT):
    """Hubble distance d_H = c / H0 in Mpc."""
    return C_KM_S / H0


def E(z, Omega_m=0.315, Omega_Lambda=0.685, w=-1.0):
    """Dimensionless Hubble parameter E(z) = H(z)/H0."""
    z = np.asarray(z, dtype=float)
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda * (1 + z)**(3 * (1 + w)))


def comoving_distance(z, Omega_m=0.315, Omega_Lambda=0.685, w=-1.0, H0=H0_DEFAULT):
    """Line-of-sight comoving distance in Mpc."""
    d_H = hubble_distance(H0)
    integrand = lambda zp: 1.0 / E(zp, Omega_m, Omega_Lambda, w)
    integral, _ = quad(integrand, 0, z)
    return d_H * integral


def luminosity_distance(z, Omega_m=0.315, Omega_Lambda=0.685, w=-1.0, H0=H0_DEFAULT):
    """Luminosity distance d_L = (1+z) * d_C in Mpc."""
    return (1 + z) * comoving_distance(z, Omega_m, Omega_Lambda, w, H0)


def angular_diameter_distance(z, Omega_m=0.315, Omega_Lambda=0.685, w=-1.0, H0=H0_DEFAULT):
    """Angular diameter distance d_A = d_C / (1+z) in Mpc."""
    return comoving_distance(z, Omega_m, Omega_Lambda, w, H0) / (1 + z)


def distance_modulus(z, Omega_m=0.315, Omega_Lambda=0.685, w=-1.0, H0=H0_DEFAULT):
    """Distance modulus mu = 5 * log10(d_L) + 25."""
    d_L = luminosity_distance(z, Omega_m, Omega_Lambda, w, H0)
    return 5.0 * np.log10(d_L) + 25.0


def mu_at_z0():
    """Distance modulus at near-zero redshift (z=1e-10)."""
    return distance_modulus(1e-10)
