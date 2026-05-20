"""
Core Schwarzschild geometry functions.

Physical constants and geometric quantities for the Schwarzschild metric:
    ds^2 = -(1 - r_s/r) c^2 dt^2 + (1 - r_s/r)^{-1} dr^2
           + r^2 dtheta^2 + r^2 sin^2(theta) dphi^2

All functions work in SI units unless otherwise noted.
Geometrized quantities use units where G = c = 1, so M is in metres.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
G: float = 6.67430e-11       # m^3 kg^-1 s^-2
c: float = 2.99792458e8      # m s^-1
M_SUN: float = 1.98892e30    # kg


# ---------------------------------------------------------------------------
# Characteristic radii
# ---------------------------------------------------------------------------

def schwarzschild_radius(M: float) -> float:
    """Schwarzschild radius  r_s = 2 G M / c^2.

    Parameters
    ----------
    M : float
        Mass in kilograms (SI).

    Returns
    -------
    float
        Schwarzschild radius in metres.
    """
    return 2.0 * G * M / c**2


def photon_sphere(M: float) -> float:
    """Radius of the photon sphere  r_ph = 3 G M / c^2 = 1.5 r_s.

    Parameters
    ----------
    M : float
        Mass in kilograms (SI).

    Returns
    -------
    float
        Photon sphere radius in metres.
    """
    return 3.0 * G * M / c**2


def isco_radius(M: float, prograde: bool = True) -> float:
    """Innermost stable circular orbit radius  r_isco = 6 G M / c^2 = 3 r_s.

    For the Schwarzschild (non-spinning) metric the ISCO is the same
    for prograde and retrograde orbits.  The *prograde* parameter is
    kept for API compatibility with Kerr extensions.

    Parameters
    ----------
    M : float
        Mass in kilograms (SI).
    prograde : bool, optional
        Ignored for Schwarzschild; kept for interface consistency.

    Returns
    -------
    float
        ISCO radius in metres.
    """
    return 6.0 * G * M / c**2


def marginally_bound_orbit(M: float) -> float:
    """Marginally bound orbit radius  r_mb = 4 G M / c^2 = 2 r_s.

    Parameters
    ----------
    M : float
        Mass in kilograms (SI).

    Returns
    -------
    float
        Marginally bound orbit radius in metres.
    """
    return 4.0 * G * M / c**2


# ---------------------------------------------------------------------------
# Metric tensor components
# ---------------------------------------------------------------------------

def metric_components(
    r: float | NDArray[np.floating],
    M: float,
) -> tuple:
    """Schwarzschild metric components g_{\\mu\\nu} in Schwarzschild coordinates.

    Returns (g_tt, g_rr, g_{\\theta\\theta}, g_{\\phi\\phi}) evaluated at
    coordinate radius *r* for a black hole of mass *M*.

    Parameters
    ----------
    r : float or array_like
        Radial coordinate(s) in metres.
    M : float
        Mass in kilograms (SI).

    Returns
    -------
    tuple of float or ndarray
        (g_tt, g_rr, g_thth, g_phph)
    """
    rs = schwarzschild_radius(M)
    f = 1.0 - rs / r
    g_tt = -f * c**2          # time-time component (with c^2)
    g_rr = 1.0 / f            # radial-radial
    g_thth = r**2              # theta-theta
    g_phph = r**2              # phi-phphi (equatorial, sin(theta)=1)
    return g_tt, g_rr, g_thth, g_phph


# ---------------------------------------------------------------------------
# Christoffel symbols
# ---------------------------------------------------------------------------

def christoffel_schwarzschild(
    r: float,
    M: float,
) -> dict[str, float]:
    """Non-zero Christoffel symbols of the Schwarzschild metric (equatorial plane).

    Uses geometrized-unit convention where positions are in units of *M*
    (i.e. set G = c = 1 so that r_s = 2M).  The returned symbols are
    labelled with coordinate indices t=0, r=1, theta=2, phi=3.

    Parameters
    ----------
    r : float
        Radial coordinate in metres.
    M : float
        Mass in kilograms (SI).

    Returns
    -------
    dict[str, float]
        Dictionary mapping 'Gamma^i_{jk}' strings to their values.
        Only the independent non-zero components are returned.
    """
    rs = schwarzschild_radius(M)
    f = 1.0 - rs / r       # metric function

    Gamma = {}

    # Gamma^t_{tr} = Gamma^t_{rt} = M / (r * (r - 2M))  = rs / (2r * f)
    Gamma["t_tr"] = rs / (2.0 * r * f)
    Gamma["t_rt"] = Gamma["t_tr"]

    # Gamma^r_{tt} = M*(r - 2M) / r^3  = c^2 * rs * f / (2 r^2)
    # In SI units the geodesic equation mixes t (seconds) and r (metres),
    # so we keep the dimensionful form:
    Gamma["r_tt"] = c**2 * rs * f / (2.0 * r**2)

    # Gamma^r_{rr} = -M / (r*(r - 2M))  = -rs / (2 r f)
    Gamma["r_rr"] = -rs / (2.0 * r * f)

    # Gamma^r_{thth} = -(r - 2M) = -r * f
    Gamma["r_thth"] = -r * f

    # Gamma^r_{phph} = -(r - 2M) sin^2(theta), equatorial => sin=1
    Gamma["r_phph"] = -r * f

    # Gamma^th_{r th} = Gamma^th_{th r} = 1/r
    Gamma["th_rth"] = 1.0 / r
    Gamma["th_thr"] = 1.0 / r

    # Gamma^th_{ph ph} = -sin(theta) cos(theta), equatorial (theta=pi/2) => 0
    # (omitted because it vanishes in the equatorial plane)

    # Gamma^ph_{r ph} = Gamma^ph_{ph r} = 1/r
    Gamma["ph_rph"] = 1.0 / r
    Gamma["ph_phr"] = 1.0 / r

    # Gamma^ph_{th ph} = Gamma^ph_{ph th} = cot(theta), equatorial => 0
    # (omitted)

    return Gamma


# ---------------------------------------------------------------------------
# Curvature invariant
# ---------------------------------------------------------------------------

def kretschner_scalar(
    r: float | NDArray[np.floating],
    M: float,
) -> float | NDArray[np.floating]:
    """Kretschner curvature scalar  K = 48 G^2 M^2 / (c^4 r^6).

    This is the only algebraically independent curvature invariant for
    the Schwarzschild vacuum solution.

    Equivalently, K = 12 r_s^2 / r^6, where r_s = 2GM/c^2.

    Parameters
    ----------
    r : float or array_like
        Radial coordinate(s) in metres.
    M : float
        Mass in kilograms (SI).

    Returns
    -------
    float or ndarray
        Kretschner scalar in m^{-4}.
    """
    # K = 48 G^2 M^2 / (c^4 r^6)
    return 48.0 * G**2 * M**2 / (c**4 * r**6)
