"""
Poincare section computation for Schwarzschild geodesics.

A Poincare section samples the (r, dr/dtau) phase space each time the
orbit crosses phi = 2 n pi (i.e. each full azimuthal revolution).
For integrable systems (like Schwarzschild) this produces a smooth curve;
deviations would signal chaos in more complex spacetimes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .metric import G, c, schwarzschild_radius
from .integrator import _M_geo


def poincare_section(
    E: float,
    L: float,
    M: float,
    n_orbits: int = 20,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute a Poincare section at phi = 2 n pi.

    Integrates the geodesic and records (r, dr/dtau) each time the orbit
    completes a full revolution (phi crosses a multiple of 2 pi).

    Parameters
    ----------
    E : float
        Dimensionless specific energy.
    L : float
        Specific angular momentum in m^2/s (SI).
    M : float
        Black-hole mass in kg (SI).
    n_orbits : int, optional
        Number of complete orbits to integrate.

    Returns
    -------
    tuple[ndarray, ndarray]
        (r_section, vr_section) — arrays of radial coordinate (metres) and
        radial velocity (m/s) at each Poincare piercing.
    """
    from .integrator import integrate_geodesic

    M_g = _M_geo(M)
    rs = schwarzschild_radius(M)
    l = L / (c * M_g)

    # Estimate a good starting radius: near the stable circular orbit
    # radius for the given L, or use a default.
    # For l^2 > 12, stable circular orbit radius (in M_g units):
    if l**2 > 12.0:
        disc = l**4 - 12.0 * l**2
        r_stable = (l**2 + np.sqrt(disc)) / 2.0  # in units M_g
    else:
        r_stable = 10.0

    r0 = r_stable * M_g  # convert to metres

    # Estimate proper time for n_orbits
    r_circ_n = r_stable
    period_est = 2.0 * np.pi * r_circ_n**2 / l   # in units of M_g/c
    total_tau_est = n_orbits * period_est

    # Use enough steps for the requested number of orbits
    n_steps = max(50000, n_orbits * 5000)

    # Increase the default tau span by passing n_steps large enough.
    # The integrator estimates tau_span internally, so we may need to
    # tweak the approach: integrate with a large n_steps.
    result = integrate_geodesic(E, L, M, r0, phi0=0.0, n_steps=n_steps)

    phi = result["phi"]
    r = result["r"]
    tau = result["tau"]

    # Compute dr/dtau numerically from the r array
    dtau = np.diff(tau)
    # Avoid division by zero
    mask = dtau > 0
    vr = np.zeros_like(r)
    vr[:-1][mask] = np.diff(r)[mask] / dtau[mask]
    vr[-1] = vr[-2]

    # Find crossings of phi = 2 n pi
    # We detect where phi crosses a multiple of 2 pi from below.
    phi_mod = phi % (2.0 * np.pi)
    # Detect where phi_mod wraps around (goes from near 2pi to near 0)
    crossings = np.where(np.diff(phi_mod) < -np.pi)[0]

    r_section = r[crossings]
    vr_section = vr[crossings]

    # Trim to requested number of orbits
    if len(r_section) > n_orbits:
        r_section = r_section[:n_orbits]
        vr_section = vr_section[:n_orbits]

    return r_section, vr_section
