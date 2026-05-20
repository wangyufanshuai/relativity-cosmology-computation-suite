"""PPN observable predictions: light deflection, Shapiro delay, perihelion precession.

All classical Solar System tests expressed in the PPN formalism.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .constants import G, C, AU
from .metric import PPNParameters


def light_deflection(
    impact_parameter: float,
    M: float,
    ppn: PPNParameters | None = None,
) -> float:
    """Light deflection angle in PPN formalism.

    Δθ = (1 + γ)/2 · 4GM/(c²b)

    For GR (γ=1): Δθ = 4GM/(c²b)
    For light grazing the Sun (b ≈ R_☉): Δθ ≈ 1.75 arcsec

    Parameters
    ----------
    impact_parameter : closest approach distance b [m]
    M : deflecting mass [kg]
    ppn : PPN parameters

    Returns
    -------
    deflection angle [radians]
    """
    if ppn is None:
        ppn = PPNParameters()
    return (1.0 + ppn.gamma) / 2.0 * 4.0 * G * M / (C**2 * impact_parameter)


def shapiro_delay(
    r_emitter: float,
    r_receiver: float,
    b: float,
    M: float,
    ppn: PPNParameters | None = None,
) -> float:
    """Shapiro (gravitational) time delay in PPN formalism.

    Δt = (1 + γ)/2 · (4GM/c³) · ln[(r_e + r_r + d)/(r_e + r_r - d)]

    Parameters
    ----------
    r_emitter : distance from mass to emitter [m]
    r_receiver : distance from mass to receiver [m]
    b : impact parameter [m]
    M : mass of central body [kg]
    ppn : PPN parameters

    Returns
    -------
    time delay [seconds]
    """
    if ppn is None:
        ppn = PPNParameters()
    d = np.sqrt(r_emitter**2 + r_receiver**2 - 2 * r_emitter * r_receiver * np.cos(0))
    # Simplified for straight-line approximation
    d = r_emitter + r_receiver
    return (1.0 + ppn.gamma) / 2.0 * (4.0 * G * M / C**3) * np.log(
        (r_emitter + r_receiver + d) / (r_emitter + r_receiver - d + 1e-30)
    )


def perihelion_precession(
    a: float,
    e: float,
    M: float,
    ppn: PPNParameters | None = None,
) -> float:
    """Perihelion precession rate in PPN formalism.

    dω/dt = (2 + 2γ - β)/3 · 3nGM/(c²a(1-e²))

    For GR (γ=β=1): (2+2-1)/3 = 1
    """
    if ppn is None:
        ppn = PPNParameters()
    n = np.sqrt(G * M / a**3)
    return (2.0 + 2.0 * ppn.gamma - ppn.beta) / 3.0 * 3.0 * n * G * M / (C**2 * a * (1.0 - e**2))


def nordtvedt_effect(
    M1: float,
    M2: float,
    r12: float,
    ppn: PPNParameters | None = None,
) -> float:
    """Nordtvedt effect: anomalous acceleration in gravitational binding energy.

    a_Nordtvedt = (4β - γ - 3 - α₁/4) · GM/(c²r²) · Ω

    where Ω is the gravitational self-energy. Non-zero only if β ≠ 1 or α₁ ≠ 0.
    Tests strong equivalence principle.
    """
    if ppn is None:
        ppn = PPNParameters()
    coeff = 4.0 * ppn.beta - ppn.gamma - 3.0 - ppn.alpha1 / 4.0
    return coeff * G * M1 * M2 / (C**2 * r12**2)


def lensethirring_precession(
    J: float,
    r: float,
    theta: float,
    ppn: PPNParameters | None = None,
) -> float:
    """Lense-Thirring (frame-dragging) precession rate.

    Ω_LT = (γ + 1 + α₁/4) · GJ/(c²r³) · [terms in θ]

    For GR: Ω_LT = 2GJ/(c²r³)
    """
    if ppn is None:
        ppn = PPNParameters()
    return (ppn.gamma + 1.0 + ppn.alpha1 / 4.0) * G * J / (C**2 * r**3)


class ExperimentalConstraints:
    """Current experimental bounds on PPN parameters."""

    @staticmethod
    def cassini_gamma() -> dict:
        """Cassini radio tracking (Bertotti et al. 2003)."""
        return {"parameter": "gamma", "value": 1.0, "sigma": 2.3e-5, "method": "Shapiro delay (Cassini)"}

    @staticmethod
    def mercury_perihelion() -> dict:
        """Mercury perihelion precession constraint on (2γ - β + 2)."""
        return {"parameter": "2γ - β", "value": 3.0, "sigma": 3e-3, "method": "MESSENGER ranging"}

    @staticmethod
    def lunar_nordtvedt() -> dict:
        """Lunar laser ranging Nordtvedt effect constraint."""
        return {"parameter": "4β - γ - 3", "value": 0.0, "sigma": 1e-3, "method": "LLR (Hofmann & Müller 2018)"}

    @staticmethod
    def all_constraints() -> list[dict]:
        """All current PPN constraints."""
        return [
            ExperimentalConstraints.cassini_gamma(),
            ExperimentalConstraints.mercury_perihelion(),
            ExperimentalConstraints.lunar_nordtvedt(),
        ]
