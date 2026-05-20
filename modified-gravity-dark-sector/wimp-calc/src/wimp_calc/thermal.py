"""Thermal relic density calculation via Boltzmann equation."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import kv

def k2(x):
    """Modified Bessel function of the second kind, order 2."""
    return kv(2, x)

from . import constants as _c

# ---------------------------------------------------------------------------
# Helper quantities
# ---------------------------------------------------------------------------

def _entropy_density(T: float, g_star_s: float) -> float:
    """Relativistic entropy density s = (2 pi^2 / 45) g_*s T^3."""
    return (2.0 * np.pi**2 / 45.0) * g_star_s * T**3


def _hubble_rate(T: float, g_star: float) -> float:
    """Radiation-dominated Hubble rate H = pi / sqrt(90) * sqrt(g_*) T^2 / M_Pl."""
    M_pl_kg = _c.M_PL * _c.GEV_TO_KG  # Planck mass in kg
    T_J = T * _c.GEV_TO_JOULE          # T in Joules (T in GeV)
    return np.pi / np.sqrt(90.0) * np.sqrt(g_star) * T_J**2 / (
        _c.HBAR * _c.C * M_pl_kg
    )


def _g_star_SM(T_gev: float) -> float:
    """Approximate effective relativistic degrees of freedom g_*(T) in the SM."""
    if T_gev > 300.0:
        return 106.25
    elif T_gev > 100.0:
        return 86.0
    elif T_gev > 1.0:
        return 75.0
    elif T_gev > 0.2:
        return 61.75
    elif T_gev > 0.1:
        return 43.0
    else:
        return 10.75


def _g_star_s_SM(T_gev: float) -> float:
    """Approximate g_*s(T) in the SM."""
    if T_gev > 300.0:
        return 106.25
    elif T_gev > 100.0:
        return 86.0
    elif T_gev > 1.0:
        return 75.0
    elif T_gev > 0.2:
        return 61.75
    elif T_gev > 0.1:
        return 43.0
    else:
        return 10.75


# ---------------------------------------------------------------------------
# Equilibrium abundance
# ---------------------------------------------------------------------------

def equilibrium_abundance(
    m_chi: float,
    T: float,
    g_chi: int = 2,
    g_star_s: float | None = None,
) -> float:
    """Equilibrium co-moving number density Y_eq = n_eq / s.

    Parameters
    ----------
    m_chi : float
        WIMP mass in GeV.
    T : float
        Temperature in GeV.
    g_chi : int
        Internal degrees of freedom (2 for fermion, 1 for scalar).
    g_star_s : float or None
        Entropy degrees of freedom. If *None*, uses SM approximation.

    Returns
    -------
    float
        Y_eq = (45 / (4 pi^4)) * (g_chi / g_*s) * (m/T)^2 * K_2(m/T)

    Notes
    -----
    For m/T >> 1 this is Boltzmann-suppressed ~ exp(-m/T).
    """
    if g_star_s is None:
        g_star_s = _g_star_s_SM(T)
    z = m_chi / T
    prefactor = 45.0 / (4.0 * np.pi**4) * g_chi / g_star_s
    return prefactor * z**2 * float(k2(z))


# ---------------------------------------------------------------------------
# Thermal average
# ---------------------------------------------------------------------------

def thermal_average_sv(
    m_chi: float,
    T: float,
    sv_func: callable,
) -> float:
    r"""Thermally averaged cross section <sigma v>.

    .. math::
        \langle\sigma v\rangle =
            \frac{\int_0^\infty \sigma v \, E^2 e^{-E/T} dE}
                 {\int_0^\infty E^2 e^{-E/T} dE}

    where E is the centre-of-mass energy.  The normalisation integral equals
    2 T^3.  For the numerator we integrate over x = E / T from threshold
    (2 m_chi) to infinity.

    Parameters
    ----------
    m_chi : float
        WIMP mass in GeV.
    T : float
        Temperature in GeV.
    sv_func : callable
        Function sv_func(E_cm) returning sigma*v in cm^3/s.
    """
    x_min = 2.0 * m_chi / T

    def integrand(x):
        E = x * T
        return sv_func(E) * x**2 * np.exp(-x)

    num, _ = quad(integrand, x_min, np.inf, limit=200)
    den = 2.0  # integral of x^2 e^{-x} from 0 to inf = 2
    return num / den


# ---------------------------------------------------------------------------
# Boltzmann equation
# ---------------------------------------------------------------------------

def boltzmann_equation(
    x: float,
    Y: float,
    sv_func: callable,
    m_chi: float,
    g_star: float = 80.0,
    g_star_s: float = 80.0,
) -> float:
    r"""Right-hand side of the Boltzmann equation for Y = n/s.

    .. math::
        \frac{dY}{dx} = -\frac{s}{H x} \langle\sigma v\rangle (Y^2 - Y_{eq}^2)

    Parameters
    ----------
    x : float
        x = m_chi / T.
    Y : float
        Current yield n/s.
    sv_func : callable
        sv_func(E_cm) in cm^3/s.
    m_chi : float
        WIMP mass in GeV.
    g_star : float
        Energy degrees of freedom.
    g_star_s : float
        Entropy degrees of freedom.
    """
    T = m_chi / x
    s = _entropy_density(T, g_star_s)
    H = _hubble_rate(T, g_star)

    # Thermal average in natural units (cm^3/s).  We need to convert to
    # GeV^-2 so the yield equation is dimensionless.  Conversion factor:
    # 1 cm = 5.068e13 GeV^-1, 1 s = 1.519e24 GeV^-1
    CM3S_TO_GEV2 = 1.0 / (1.0 / (5.068e13)**3 * 1.519e24)  # ~ 4.07e-24 GeV^-2 per cm^3/s
    # Simpler: just use numerical prefactor from standard treatment
    # sigma_eff in cm^3/s, s in GeV^3, H in GeV, x dimensionless
    # dY/dx = -(s <sv>)/(H x) (Y^2 - Yeq^2)   [all in GeV units]
    # We keep <sv> in cm^3/s and convert s and H to matching units.

    # Convert s from J^3 to GeV^3: divide by (GEV_TO_JOULE)^3
    s_gev3 = _entropy_density(T, g_star_s) / _c.GEV_TO_JOULE**3
    H_gev = _hubble_rate(T, g_star) * _c.HBAR  # H in GeV (hbar * H_SI gives energy)
    # Actually H in s^-1; H * hbar = GeV
    H_gev2 = _hubble_rate(T, g_star) / (1.0 / _c.HBAR)  # s^-1 * GeV*s = GeV
    # Better: H_SI [s^-1], hbar [GeV s], so H_gev = H_SI * hbar
    H_gev = _hubble_rate(T, g_star) * _c.HBAR

    # sigma_v in cm^3/s -> natural units: 1 GeV^-2 = hbar^2 c^5 / (GeV^2 ... )
    # 1 cm^3/s = (1.973e-14 GeV^-1)^3 * (1/6.582e-25 GeV^-1) = ...
    # Use: 1 cm^3/s = 1.167e-17 GeV^-2  (standard cosmology conversion)
    GEV2_PER_CM3S = 1.167e-17
    sv_gev2 = sv_func(m_chi) * GEV2_PER_CM3S  # approximate: evaluate at E~m_chi

    Y_eq = equilibrium_abundance(m_chi, T, g_star_s=g_star_s)

    dYdx = -(s_gev3 * sv_gev2) / (H_gev * x) * (Y**2 - Y_eq**2)
    return dYdx


# ---------------------------------------------------------------------------
# Freeze-out solver
# ---------------------------------------------------------------------------

def solve_freezeout(
    m_chi: float,
    sv_func: callable,
    g_star: float = 80.0,
    g_star_s: float = 80.0,
    T_start: float | None = None,
    T_end: float = 1e-3,
    x_start: float = 1.0,
    x_end: float = 1000.0,
) -> dict:
    """Integrate the Boltzmann equation through freeze-out.

    Parameters
    ----------
    m_chi : float
        WIMP mass in GeV.
    sv_func : callable
        sigma*v as function of E_cm in cm^3/s.
    T_start, T_end : float
        Temperature range in GeV (overridden by x_start, x_end).
    x_start : float
        Start integration at x = m/T (default 1, i.e. T ~ m).
    x_end : float
        End integration at x = m/T (default 1000, well after freeze-out).

    Returns
    -------
    dict with keys 'x', 'Y', 'Y_inf' (asymptotic yield).
    """
    T_initial = m_chi / x_start
    Y_eq_start = equilibrium_abundance(m_chi, T_initial, g_star_s=g_star_s)

    def rhs(x, y):
        return [boltzmann_equation(x, y[0], sv_func, m_chi, g_star, g_star_s)]

    sol = solve_ivp(
        rhs,
        (x_start, x_end),
        [Y_eq_start],
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
        max_step=1.0,
    )

    Y_inf = sol.y[0, -1]
    return {
        "x": sol.t,
        "Y": sol.y[0],
        "Y_inf": Y_inf,
        "sol": sol,
    }


# ---------------------------------------------------------------------------
# Relic density
# ---------------------------------------------------------------------------

def relic_density(Y_inf: float, m_chi: float) -> float:
    r"""Compute Omega h^2 from the asymptotic yield.

    .. math::
        \Omega h^2 = 2.742 \times 10^8 \; m_\chi(\mathrm{GeV}) \; Y_\infty

    Parameters
    ----------
    Y_inf : float
        Asymptotic yield n/s at T << m_chi.
    m_chi : float
        WIMP mass in GeV.

    Returns
    -------
    float
        Omega h^2.
    """
    return 2.742e8 * m_chi * Y_inf
