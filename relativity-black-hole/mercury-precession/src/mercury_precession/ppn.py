"""Post-Newtonian perturbation theory for planetary orbits.

Implements the Lagrange planetary equations with post-Newtonian
corrections for computing perihelion precession from perturbation theory.

References:
    - Will (1993), "Theory and Experiment in Gravitational Physics"
    - Soffel et al. (2003), "The IAU 2000 Resolutions"
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .constants import A_MERCURY, C, E_MERCURY, G, M_SUN


def lagrange_planetary_eqs(
    t: float,
    y: np.ndarray,
    mu: float,
    pn_order: int = 1,
) -> np.ndarray:
    """Lagrange planetary equations with PN corrections for (a, e, ω, M₀).

    State vector y = [a, e, varpi, lambda_mean] where:
        a : semi-major axis
        e : eccentricity
        varpi : longitude of perihelion (Ω + ω)
        lambda_mean : mean longitude (M + varpi)

    We use the Gauss form of Lagrange's equations with the
    1PN Schwarzschild perturbing acceleration:
        R = -3GM²/(c²r³) · [v² - (GM/r) - (dr/dt)²]  ... not separable simply

    Instead, we use the direct approach: the 1PN perihelion precession rate is:
        dω/dt = 3(GM)^{3/2} / [c² a^{5/2} (1-e²)]

    For higher orders we add the 2PN term.
    """
    a, e, varpi, lam = y

    if e < 1e-15:
        e = 1e-15

    n = np.sqrt(mu / a**3)  # mean motion

    # Keplerian: mean longitude advances at rate n
    dlam = n
    da = 0.0
    de = 0.0

    # 1PN Schwarzschild precession of perihelion
    # dω/dt = 3(GM)^{3/2} / [c² a^{5/2} (1-e²)]
    gm_c2 = mu / C**2
    dvarpi_1pn = 3.0 * mu * n / (C**2 * a * (1.0 - e**2))

    dvarpi = dvarpi_1pn

    if pn_order >= 2:
        # 2PN correction
        # dω/dt|_2PN = (3/4) · (GM/c²)² · n / [a² (1-e²)²] · (10 + 3e²)
        dvarpi_2pn = (3.0 / 4.0) * (mu / C**2)**2 * n / (a**2 * (1.0 - e**2)**2) * (10.0 + 3.0 * e**2)
        dvarpi += dvarpi_2pn

    # In Schwarzschild, a and e are conserved (only ω precesses)
    return np.array([da, de, dvarpi, dlam])


def integrate_perturbation(
    a0: float = A_MERCURY,
    e0: float = E_MERCURY,
    M: float = M_SUN,
    n_orbits: int = 100,
    pn_order: int = 1,
    method: str = "DOP853",
    rtol: float = 1e-13,
    atol: float = 1e-15,
) -> dict:
    """Integrate the Lagrange planetary equations with PN corrections.

    Returns the accumulated perihelion precession over the integration time.
    """
    mu = G * M
    T_orbit = 2.0 * np.pi * np.sqrt(a0**3 / mu)
    t_span = (0.0, n_orbits * T_orbit)
    t_eval = np.linspace(*t_span, n_orbits * 100 + 1)

    y0 = np.array([a0, e0, 0.0, 0.0])

    sol = solve_ivp(
        lagrange_planetary_eqs,
        t_span,
        y0,
        args=(mu, pn_order),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    varpi = sol.y[2]  # longitude of perihelion

    # Precession per orbit
    total_precession = varpi[-1]
    precession_per_orbit = total_precession / n_orbits

    seconds_per_century = 100.0 * 365.25 * 86400.0
    orbits_per_century = seconds_per_century / T_orbit
    precession_arcsec_per_century = precession_per_orbit * orbits_per_century * (180.0 * 3600.0 / np.pi)

    return {
        "t": sol.t,
        "varpi": varpi,
        "precession_per_orbit_rad": precession_per_orbit,
        "precession_arcsec_per_century": precession_arcsec_per_century,
        "total_precession_rad": total_precession,
    }


def ppn_parameters() -> dict[str, dict[str, float]]:
    """Current experimental constraints on PPN parameters.

    From Will (2014) Living Reviews in Relativity.
    """
    return {
        "gamma": {
            "value": 1.0,
            "uncertainty": 2.3e-5,
            "source": "Cassini tracking (Bertotti et al. 2003)",
            "formula": "γ_PPN = (ω_BD + 1)/(ω_BD + 2) for Brans-Dicke",
        },
        "beta": {
            "value": 1.0,
            "uncertainty": 8e-5,
            "source": "Mercury perihelion shift + γ constraint",
        },
        "alpha1": {
            "value": 0.0,
            "uncertainty": 1e-4,
            "source": "Lunar laser ranging (Hofmann & Müller 2018)",
        },
        "alpha2": {
            "value": 0.0,
            "uncertainty": 2e-9,  # actually much tighter from spin alignment
            "source": "Solar spin alignment with galactic frame",
        },
        "xi": {"value": 0.0, "uncertainty": 1e-3},
        "alpha3": {"value": 0.0, "uncertainty": 4e-20, "source": "Pulsar acceleration"},
        "zeta1": {"value": 0.0, "uncertainty": 2e-2},
        "zeta2": {"value": 0.0, "uncertainty": 4e-5, "source": "Pulsar timing"},
        "zeta3": {"value": 0.0, "uncertainty": 1e-8},
        "zeta4": {"value": 0.0, "uncertainty": "not independent (≡ α3)"},
    }


def precession_ppn(
    gamma: float = 1.0,
    beta: float = 1.0,
    a: float = A_MERCURY,
    e: float = E_MERCURY,
    M: float = M_SUN,
) -> float:
    """Perihelion precession in PPN formalism (rad/orbit).

    Δφ = [6πGM/(ac²)] · [(2 + 2γ - β)/3] / (1 - e²)

    For GR: γ = β = 1 → factor = (2+2-1)/3 = 1
    For Brans-Dicke: γ = (ω+1)/(ω+2), β = 1
    """
    return (
        6.0 * np.pi * G * M / (a * C**2 * (1.0 - e**2))
        * (2.0 + 2.0 * gamma - beta) / 3.0
    )
