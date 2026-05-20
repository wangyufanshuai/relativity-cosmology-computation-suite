"""Effective degrees of freedom g_*(T) and g_*s(T)."""

import numpy as np

# Particle data: (name, mass_MeV, g_internal, fermion=1/boson=0)
_SM_PARTICLES = [
    ("photon",       0.0,     2, 0),
    ("gluon",        0.0,    16, 0),
    ("W+",          80379.0,  2, 0),
    ("W-",          80379.0,  2, 0),
    ("Z0",          91188.0,  3, 0),
    ("higgs",      125000.0,  1, 0),
    ("u",             2.2,    6, 1),
    ("d",             4.7,    6, 1),
    ("s",            95.0,    6, 1),
    ("c",         1_275_000,  6, 1),
    ("b",         4_180_000,  6, 1),
    ("t",       173_000_000,  6, 1),
    ("e-",            0.511,  2, 1),
    ("e+",            0.511,  2, 1),
    ("mu-",         105.66,   2, 1),
    ("mu+",         105.66,   2, 1),
    ("tau-",       1776.86,   2, 1),
    ("tau+",       1776.86,   2, 1),
    ("nu_e",          0.0,    1, 1),
    ("nu_e_bar",      0.0,    1, 1),
    ("nu_mu",         0.0,    1, 1),
    ("nu_mu_bar",     0.0,    1, 1),
    ("nu_tau",        0.0,    1, 1),
    ("nu_tau_bar",    0.0,    1, 1),
]


def _boltzmann_factor(T_MeV, mass_MeV, fermion):
    """Contribution of a particle species to g_* (Boltzmann suppression)."""
    T_MeV = np.asarray(T_MeV, dtype=float)
    if mass_MeV == 0.0:
        return 1.0
    ratio = mass_MeV / T_MeV
    return np.where(ratio < 50, np.exp(-ratio), 0.0)


def g_star(T_MeV):
    """Effective relativistic d.o.f. for energy density g_*(T).

    At T >> 173 GeV: g_* = 106.75 (full SM).
    """
    T_MeV = np.asarray(T_MeV, dtype=float)
    scalar = T_MeV.ndim == 0
    T_MeV = np.atleast_1d(T_MeV)

    result = np.zeros_like(T_MeV, dtype=float)
    for name, mass, g_int, fermion in _SM_PARTICLES:
        fermion_factor = 7.0 / 8.0 if fermion else 1.0
        if mass == 0.0:
            contribution = g_int * fermion_factor
        else:
            contribution = g_int * fermion_factor * _boltzmann_factor(T_MeV, mass, fermion)
        result += contribution

    return float(result[0]) if scalar else result


def g_star_s(T_MeV):
    """Effective relativistic d.o.f. for entropy density g_*s(T)."""
    return g_star(T_MeV)
