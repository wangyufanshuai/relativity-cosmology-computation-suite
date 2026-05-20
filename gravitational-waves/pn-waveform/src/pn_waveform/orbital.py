"""Binary orbital dynamics in the post-Newtonian approximation.

Provides chirp mass, symmetric mass ratio, ISCO frequency,
PN phase coefficients, and inspiral frequency evolution.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .constants import C, G


# ---------------------------------------------------------------------------
# Basic binary parameters
# ---------------------------------------------------------------------------

def chirp_mass(m1: float, m2: float) -> float:
    """Compute the chirp mass M_c of a binary system.

    Parameters
    ----------
    m1, m2 : float
        Component masses [kg].

    Returns
    -------
    float
        Chirp mass M_c = (m1 m2)^{3/5} / (m1 + m2)^{1/5} [kg].
    """
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def symmetric_mass_ratio(m1: float, m2: float) -> float:
    """Compute the symmetric mass ratio eta of a binary system.

    Parameters
    ----------
    m1, m2 : float
        Component masses [kg].

    Returns
    -------
    float
        Symmetric mass ratio eta = m1 m2 / (m1 + m2)^2.
    """
    total = m1 + m2
    return m1 * m2 / (total * total)


# ---------------------------------------------------------------------------
# Characteristic frequencies
# ---------------------------------------------------------------------------

def schwarzschild_isco_freq(m_total: float) -> float:
    """Innermost stable circular orbit frequency for a Schwarzschild BH.

    Parameters
    ----------
    m_total : float
        Total mass of the binary [kg].

    Returns
    -------
    float
        ISCO orbital frequency f_isco = c^3 / (6^{3/2} pi G M) [Hz].
    """
    return C ** 3 / (6.0 ** 1.5 * np.pi * G * m_total)


# ---------------------------------------------------------------------------
# PN phase / frequency coefficients
# ---------------------------------------------------------------------------

def pn_phase_coefficients(eta: float) -> dict[int, float]:
    """Return the PN phase coefficients psi_0 … psi_7 for a given eta.

    These are the coefficients in the stationary-phase approximation
    Psi(f) = 2 pi f t_c - phi_c - pi/4 + sum_k psi_k f^{(k-5)/3}
    where the psi_k depend on eta (symmetric mass ratio).

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.

    Returns
    -------
    dict[int, float]
        Phase coefficients {0: psi_0, 1: psi_1, ..., 7: psi_7}.
    """
    psi: dict[int, float] = {}

    # 0PN (leading order)
    psi[0] = 3.0 / (128.0 * eta)

    # 1PN
    psi[1] = (3715.0 / 756.0 + 55.0 * eta / 9.0)

    # 1.5PN  (tail term)
    psi[2] = -16.0 * np.pi

    # 2PN
    psi[3] = (15293365.0 / 508032.0
              + 27145.0 * eta / 504.0
              + 3085.0 * eta ** 2 / 72.0)

    # 2.5PN
    psi[4] = (1.0 + np.log(4.0)) * np.pi * (
        38645.0 / 756.0 - 65.0 * eta / 9.0
    )

    # 3PN
    psi[5] = (11583231236531.0 / 4694215680.0
              - 640.0 * (3.0 / 4.0) ** 2
              + (-15335597827.0 / 3048192.0
                 + 457.0 * (743.0 / 252.0 + 11.0 * eta / 3.0) / 12.0) * eta
              + 76055.0 * eta ** 2 / 1728.0
              - (127825.0 / 1296.0) * eta ** 3)

    # 3.5PN
    psi[6] = np.pi * (
        77096675.0 / 254016.0
        + 378515.0 * eta / 1512.0
        - 74045.0 * eta ** 2 / 756.0
    )

    # Higher-order placeholder (log term at 3.5PN+)
    psi[7] = 0.0

    return psi


# ---------------------------------------------------------------------------
# Inspiral frequency evolution
# ---------------------------------------------------------------------------

def inspiral_frequency(
    Mc: float,
    eta: float,
    t_ref: float | NDArray[np.floating],
    f_ref: float,
    pn_order: float = 3.5,
) -> NDArray[np.floating]:
    """Compute the inspiral GW frequency as a function of time-to-coalescence.

    Uses the PN expansion of df/dt and integrates backwards from f_ref
    at t_ref = 0 (coalescence time).  The result gives f(t_ref - t).

    Parameters
    ----------
    Mc : float
        Chirp mass [kg].
    eta : float
        Symmetric mass ratio.
    t_ref : float or array_like
        Time(s) before coalescence [s].
    f_ref : float
        Reference frequency at t_ref = 0 [Hz].
    pn_order : float
        PN order (multiples of 0.5).  Supported: 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5.

    Returns
    -------
    ndarray
        Frequency array [Hz].
    """
    t_ref = np.asarray(t_ref, dtype=float)

    # Leading-order chirp: df/dt = (96/5) pi^{8/3} (G Mc)^{5/3} f^{11/3} / c^5
    v_Mc = (G * Mc) / C ** 3  # dimensionless chirp-mass parameter

    # Leading-order frequency evolution:
    # f(t) = f_ref * (1 - 256/5 * pi^{8/3} v_Mc^{5/3} f_ref^{8/3} * t)^(-3/8)
    tau_coeff = (256.0 / 5.0) * np.pi ** (8.0 / 3.0) * v_Mc ** (5.0 / 3.0)

    # Include higher PN corrections as a simple multiplicative factor on df/dt
    # For orders > 0, we compute the PN correction factor at f_ref
    x_ref = (np.pi * G * Mc / C ** 3 * f_ref) ** (2.0 / 3.0)  # PN expansion parameter

    correction = 1.0
    if pn_order >= 0.5:
        correction += (743.0 / 336.0 + 11.0 * eta / 4.0) * x_ref / 4.0  # 1PN
    if pn_order >= 1.0:
        correction += (-32.0 * np.pi / 5.0) * x_ref ** 1.5  # 1.5PN
    if pn_order >= 1.5:
        correction += (34103.0 / 18144.0 + 13661.0 * eta / 2016.0
                       + 59.0 * eta ** 2 / 18.0) * x_ref ** 2  # 2PN
    if pn_order >= 2.0:
        correction += (-4159.0 * np.pi / 672.0 - 189.0 * np.pi * eta / 8.0) * x_ref ** 2.5  # 2.5PN

    tau_coeff *= correction

    # f(t) from leading-order formula with PN correction
    bracket = 1.0 - tau_coeff * f_ref ** (8.0 / 3.0) * t_ref

    # Avoid negative/zero values (beyond coalescence)
    bracket = np.maximum(bracket, 1e-30)

    return f_ref * bracket ** (-3.0 / 8.0)
