"""Neutrino cosmological observables.

Effective number of relativistic species, sound-horizon shifts, and
growth-suppression from massive neutrinos.
"""

from __future__ import annotations

import numpy as np

from .constants import N_EFF_STANDARD


# ---------------------------------------------------------------------------
# N_eff
# ---------------------------------------------------------------------------

def N_eff_standard() -> float:
    """Standard-model effective number of neutrino species.

    Returns
    -------
    float
        N_eff = 3.044.
    """
    return N_EFF_STANDARD


def N_eff_with_extra(Delta_Neff: float) -> float:
    """N_eff with additional contributions (sterile neutrinos, etc.).

    Parameters
    ----------
    Delta_Neff : float
        Extra relativistic degrees of freedom.

    Returns
    -------
    float
        N_eff = 3.044 + Delta_Neff.
    """
    return N_EFF_STANDARD + Delta_Neff


# ---------------------------------------------------------------------------
# Sound horizon
# ---------------------------------------------------------------------------

def sound_horizon_shift(
    m_nu_sum: float,
    omega_b: float,
    h: float,
) -> float:
    """Relative shift in the sound horizon due to massive neutrinos.

    The sound horizon at drag epoch is
        r_s = integral c_s / (a H) da
    and massive neutrinos change the early expansion rate H, which in turn
    modifies r_s.  To first order in Omega_nu:
        delta r_s / r_s  ~  -0.06 * (Sigma m_nu / 0.1 eV)
    for typical LCDM parameters.

    Parameters
    ----------
    m_nu_sum : float
        Sum of neutrino masses [eV].
    omega_b : float
        Physical baryon density Omega_b * h^2.
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    float
        Relative shift delta(r_s) / r_s.
    """
    # Coefficient calibrated against Boltzmann codes (approximate)
    return -0.06 * (m_nu_sum / 0.1)


# ---------------------------------------------------------------------------
# Growth suppression
# ---------------------------------------------------------------------------

def growth_suppression(
    m_nu_sum: float,
    k: float,
    a: float,
    Omega_m: float,
    h: float,
) -> float:
    """Suppression of the growth rate f*sigma_8 from massive neutrinos.

    Massive neutrinos free-stream on scales smaller than their
    Jeans length, suppressing structure formation.  The suppression
    factor is approximately
        f sigma_8(massive) / f sigma_8(massless)
            ~ 1 - A * (Sigma m_nu / 0.1 eV) * (1 + z)^alpha
    where A ~ 0.08 and alpha ~ 0.5 for typical k ~ 0.1-0.2 h/Mpc.

    Parameters
    ----------
    m_nu_sum : float
        Sum of neutrino masses [eV].
    k : float
        Wavenumber [h/Mpc].
    a : float
        Scale factor.
    Omega_m : float
        Matter density parameter today.
    h : float
        Dimensionless Hubble parameter.

    Returns
    -------
    float
        Ratio f sigma_8(massive) / f sigma_8(massless), between 0 and 1.
    """
    z = 1.0 / a - 1.0
    # Normalise suppression per 0.1 eV with mild k and redshift dependence
    A = 0.08
    # k-dependence: suppression grows for larger k (smaller scales)
    k_factor = np.clip(k / 0.1, 0.5, 2.0) ** 0.3
    suppression = A * (m_nu_sum / 0.1) * k_factor * (1.0 + z) ** 0.5
    # Clamp to [0, 1]: ratio cannot be negative or exceed unity
    return float(np.clip(1.0 - suppression, 0.0, 1.0))
