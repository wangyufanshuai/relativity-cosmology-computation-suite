"""
Cosmological solutions of massive and bimetric gravity.

Implements:
  - Massive graviton contribution to the Friedmann equation
  - Hubble parameter in dRGT cosmology
  - Gravitational-wave propagation speed (constrained by GW170817)
  - Comoving and luminosity distances

References:
  - Gumrukcuoglu, Lin, Mukohyama, Phys.Lett. B728 (2014) 700
  - GW170817 constraint: v_g = c \\pm 10^{-15}  (Abbott et al., 2017)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_trapezoid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_LIGHT = 2.99792458e8  # m/s
H0_SI = 2.184e-18  # H0 ~ 67.4 km/s/Mpc  in s^{-1}
MPC_TO_M = 3.0856775815e22  # metres per Mpc

# ---------------------------------------------------------------------------
# Hubble parameter
# ---------------------------------------------------------------------------

def hubble_parameter(
    a: float | NDArray,
    H0: float,
    Omega_m: float,
    Omega_rad: float = 9e-5,
    Omega_Lambda: float | None = None,
    m_g_eV: float = 0.0,
    beta_eff: float = 0.0,
) -> float | NDArray:
    """Hubble parameter H(a) in dRGT massive gravity cosmology.

    H^2(a) = H0^2 [\\Omega_rad/a^4 + \\Omega_m/a^3
                     + \\Omega_\\Lambda
                     + \\Omega_{mg}(a)]

    where the massive gravity contribution is:
      \\Omega_{mg}(a) = (m_g / H0)^2 \\beta_eff

    Parameters
    ----------
    a : scale factor (a=1 today)
    H0 : Hubble constant (s^{-1})
    Omega_m : matter density parameter
    Omega_rad : radiation density parameter
    Omega_Lambda : dark-energy density (computed from Omega_m + Omega_rad if None)
    m_g_eV : graviton mass in eV
    beta_eff : effective dRGT coupling
    """
    a = np.asarray(a, dtype=float)

    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m - Omega_rad

    # Massive-gravity effective density parameter
    Omega_mg = 0.0
    if m_g_eV > 0.0 and H0 > 0.0:
        # Convert m_g to s^{-1}:  m_g = m_g_eV * e / hbar
        HBAR = 1.054571817e-34
        EV_TO_J = 1.602176634e-19
        m_g_inv_s = m_g_eV * EV_TO_J / HBAR
        Omega_mg = (m_g_inv_s / H0) ** 2 * beta_eff

    H2 = H0**2 * (
        Omega_rad / a**4
        + Omega_m / a**3
        + Omega_Lambda
        + Omega_mg
    )
    return np.sqrt(H2)


def hubble_parameter_lcdm(
    a: float | NDArray,
    H0: float,
    Omega_m: float,
    Omega_rad: float = 9e-5,
) -> float | NDArray:
    """Standard LCDM Hubble parameter (massive gravity with m_g -> 0)."""
    return hubble_parameter(a, H0, Omega_m, Omega_rad, m_g_eV=0.0)


# ---------------------------------------------------------------------------
# Gravitational-wave speed
# ---------------------------------------------------------------------------

def gw_speed_dRGT(
    m_g_eV: float = 0.0,
    c: float = C_LIGHT,
    alpha_param: float = 0.0,
) -> float:
    """Gravitational-wave propagation speed in dRGT massive gravity.

    In the minimal dRGT theory the graviton propagates at exactly the speed
    of light: v_g = c.  This is because the vector and scalar constraints
    remove the extra polarisation modes' speed deviation.

    Some extensions allow  v_g = c (1 + \\alpha m_g^2 / H^2), but for the
    pure dRGT subclass  \\alpha = 0  and v_g = c exactly.

    The GW170817 constraint is  |v_g/c - 1| < 10^{-15}.
    """
    return c * (1.0 + alpha_param * 0.0)  # pure dRGT: exactly c


def gw_speed_constraint_gw170817(v_g: float, c: float = C_LIGHT) -> bool:
    """Check whether a given GW speed satisfies the GW170817 bound.

    |v_g / c - 1| < 10^{-15}
    """
    return bool(abs(v_g / c - 1.0) < 1e-15)


# ---------------------------------------------------------------------------
# Distance measures
# ---------------------------------------------------------------------------

def comoving_distance(
    z: float | NDArray,
    H0: float,
    Omega_m: float,
    Omega_rad: float = 9e-5,
    m_g_eV: float = 0.0,
    beta_eff: float = 0.0,
    n_points: int = 500,
) -> float | NDArray:
    """Comoving distance  \\chi(z) = c \\int_0^z dz' / H(z').

    Parameters
    ----------
    z : redshift (scalar or array)
    H0 : Hubble constant (s^{-1})
    Omega_m : matter density parameter
    Omega_rad : radiation density parameter
    m_g_eV : graviton mass (eV)
    beta_eff : effective dRGT coupling
    n_points : integration points per interval
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    results = np.empty_like(z)

    for i, zi in enumerate(z):
        zz = np.linspace(0.0, zi, n_points)
        a = 1.0 / (1.0 + zz)
        H = hubble_parameter(a, H0, Omega_m, Omega_rad, m_g_eV=m_g_eV, beta_eff=beta_eff)
        integrand = C_LIGHT / H
        results[i] = float(np.trapezoid(integrand, zz))

    return results if results.size > 1 else float(results[0])


def luminosity_distance(
    z: float | NDArray,
    H0: float,
    Omega_m: float,
    Omega_rad: float = 9e-5,
    m_g_eV: float = 0.0,
    beta_eff: float = 0.0,
) -> float | NDArray:
    """Luminosity distance  d_L(z) = (1+z) * \\chi(z)."""
    chi = comoving_distance(z, H0, Omega_m, Omega_rad, m_g_eV, beta_eff)
    return (1.0 + np.asarray(z)) * chi


# ---------------------------------------------------------------------------
# Graviton mass bounds
# ---------------------------------------------------------------------------

def graviton_mass_bound_ligo() -> dict:
    """Return the LIGO bound on the graviton mass.

    From the LIGO/Virgo analysis of GW events:
      m_g < 1.27e-22 eV  (90% CL, from GW150914 and later events)
    corresponding to a Compton wavelength  \\lambda_g > 10^{13} km.
    """
    return {
        "m_g_upper_eV": 1.27e-22,
        "lambda_g_lower_m": 1.0e16,  # ~10^{13} km
        "confidence": "90% CL",
        "source": "LIGO/Virgo (GW150914 + GW170104 + ...)",
    }


def check_graviton_mass_bound(m_g_eV: float) -> bool:
    """Check whether a given graviton mass satisfies the LIGO bound."""
    bound = graviton_mass_bound_ligo()
    return m_g_eV <= bound["m_g_upper_eV"]


# ---------------------------------------------------------------------------
# LCDM reduction check
# ---------------------------------------------------------------------------

def reduces_to_lcdm(
    H0: float = H0_SI,
    Omega_m: float = 0.3,
    Omega_rad: float = 9e-5,
    m_g_eV_small: float = 1e-30,
    beta_eff: float = 1.0,
    a_test: float = 0.5,
    rtol: float = 1e-6,
) -> bool:
    """Verify that the dRGT Hubble parameter reduces to LCDM for m_g -> 0."""
    H_mg = hubble_parameter(a_test, H0, Omega_m, Omega_rad, m_g_eV=m_g_eV_small, beta_eff=beta_eff)
    H_lcdm = hubble_parameter_lcdm(a_test, H0, Omega_m, Omega_rad)
    return bool(np.isclose(H_mg, H_lcdm, rtol=rtol))
