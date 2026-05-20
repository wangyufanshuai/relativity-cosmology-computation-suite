from __future__ import annotations

from dataclasses import dataclass
from math import log10

import numpy as np

C_KM_S = 299792.458


@dataclass(frozen=True)
class Cosmology:
    h0: float = 70.0
    omega_m: float = 0.3
    w0: float = -1.0
    wa: float = 0.0

    @property
    def omega_de(self) -> float:
        return 1.0 - self.omega_m


def c_over_h0(cosmo: Cosmology) -> float:
    return C_KM_S / cosmo.h0


def dark_energy_density_scale(z: np.ndarray | float, cosmo: Cosmology) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    return a ** (-3.0 * (1.0 + cosmo.w0 + cosmo.wa)) * np.exp(-3.0 * cosmo.wa * (1.0 - a))


def e_z(z: np.ndarray | float, cosmo: Cosmology) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    matter = cosmo.omega_m * (1.0 + z) ** 3
    dark_energy = cosmo.omega_de * dark_energy_density_scale(z, cosmo)
    return np.sqrt(matter + dark_energy)


def transverse_comoving_distance(z: float, cosmo: Cosmology, steps: int = 512) -> float:
    if z < 0:
        raise ValueError("redshift must be non-negative")
    grid = np.linspace(0.0, z, steps)
    integral = np.trapezoid(1.0 / e_z(grid, cosmo), grid)
    return c_over_h0(cosmo) * float(integral)


def luminosity_distance(z: float, cosmo: Cosmology) -> float:
    return (1.0 + z) * transverse_comoving_distance(z, cosmo)


def distance_modulus(z: float, cosmo: Cosmology) -> float:
    dl_mpc = luminosity_distance(z, cosmo)
    if dl_mpc <= 0:
        raise ValueError("luminosity distance must be positive")
    return 5.0 * log10(dl_mpc) + 25.0


def bao_dv_over_rd(z: float, cosmo: Cosmology, rd_mpc: float = 147.1) -> float:
    dm = transverse_comoving_distance(z, cosmo)
    hz_over_c = e_z(z, cosmo)[()] / c_over_h0(cosmo)
    dv = (dm * dm * z / hz_over_c) ** (1.0 / 3.0)
    return float(dv / rd_mpc)
