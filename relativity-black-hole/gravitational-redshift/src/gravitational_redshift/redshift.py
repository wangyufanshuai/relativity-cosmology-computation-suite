"""Core gravitational redshift and relativistic time-dilation calculations.

All functions accept numpy arrays where applicable and return results in
physically meaningful units documented in each docstring.
"""

from __future__ import annotations

import numpy as np

from .constants import G, c, EARTH_MASS, EARTH_RADIUS

__all__ = [
    "schwarzschild_redshift",
    "kerr_redshift",
    "gps_clock_correction",
    "doppler_shift",
    "sagnac_effect",
    "lense_thirring_rate",
]


# ---------------------------------------------------------------------------
# Schwarzschild redshift
# ---------------------------------------------------------------------------

def schwarzschild_redshift(r: float | np.ndarray, M: float) -> float | np.ndarray:
    """Gravitational redshift factor for the Schwarzschild metric.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate of the observer [m].  Must satisfy r > 2GM/c^2.
    M : float
        Mass of the gravitating body [kg].

    Returns
    -------
    z : float or ndarray
        Dimensionless redshift factor  z = (1 - r_s/r)^{-1/2} - 1,
        where r_s = 2GM/c^2 is the Schwarzschild radius.
    """
    r_s = 2.0 * G * M / c**2
    ratio = r_s / r
    if np.any(ratio >= 1.0):
        raise ValueError("r must be outside the Schwarzschild radius (r > 2GM/c^2).")
    return 1.0 / np.sqrt(1.0 - ratio) - 1.0


# ---------------------------------------------------------------------------
# Kerr redshift (equatorial, circular orbit approximation)
# ---------------------------------------------------------------------------

def kerr_redshift(
    r: float | np.ndarray,
    theta: float | np.ndarray,
    M: float,
    a: float,
) -> float | np.ndarray:
    """Approximate gravitational redshift in Kerr spacetime for circular
    equatorial orbits.

    The formula used is the first-order approximation for the combined
    gravitational + frame-dragging redshift seen by a distant observer::

        z = (1 - 3M/r + 2a*sqrt(M)/r^{3/2})^{-1/2} - 1

    where *a* = J/(Mc) is the dimensionless spin parameter (in geometrised
    units with G=c=1 the mass M is expressed in metres).

    **Important:** this function uses *geometrised units* internally for M
    and a (i.e. M and a are lengths in metres, already divided by G/c^2 and
    c respectively if converting from SI).  For convenience, if you pass
    *a* in the range [-1, 1] it is interpreted as a dimensionless spin and
    the mass is converted internally.

    Parameters
    ----------
    r : float or ndarray
        Boyer-Lindquist radial coordinate [geometrised units, i.e. metres
        with G=c=1].
    theta : float or ndarray
        Inclination angle [radians].  Kept for API consistency; the formula
        is evaluated on the equatorial plane (theta = pi/2).
    M : float
        Mass parameter in geometrised units (length).
    a : float
        Spin parameter.  If |a| <= 1 it is treated as dimensionless
        (a = Jc/(GM^2)) and multiplied by M internally to get the
        geometrised spin a* = J/(Mc).

    Returns
    -------
    z : float or ndarray
        Dimensionless redshift.
    """
    # If |a| <= 1 treat as dimensionless spin
    if np.abs(a) <= 1.0:
        a_star = a * M  # geometrised spin length
    else:
        a_star = a

    term = 1.0 - 3.0 * M / r + 2.0 * a_star * np.sqrt(M) / r**1.5
    if np.any(term <= 0.0):
        raise ValueError(
            "Kerr redshift denominator is non-positive; "
            "r may be inside the innermost stable circular orbit."
        )
    return 1.0 / np.sqrt(term) - 1.0


# ---------------------------------------------------------------------------
# GPS clock correction
# ---------------------------------------------------------------------------

def gps_clock_correction(
    altitude: float,
    M: float = EARTH_MASS,
    R: float = EARTH_RADIUS,
) -> dict:
    """Combined gravitational and special-relativistic clock correction for
    an orbiting clock (e.g. a GPS satellite).

    Parameters
    ----------
    altitude : float
        Orbital altitude above the body's surface [m].
    M : float
        Mass of the central body [kg].  Defaults to Earth.
    R : float
        Radius of the central body [m].  Defaults to Earth mean radius.

    Returns
    -------
    dict with keys:
        ``gravitational_us_day``  : float  – gravitational blueshift contribution [us/day]
        ``special_us_day``        : float  – special-relativistic (velocity) dilation [us/day]
        ``total_us_day``          : float  – net correction [us/day]
    """
    r_orbit = R + altitude

    # Orbital speed for a circular orbit
    v_orbit = np.sqrt(G * M / r_orbit)

    # Gravitational potential difference (orbit vs surface)
    # Gravitational frequency ratio: f_surface / f_orbit = sqrt((1 - r_s/r_orbit)/(1 - r_s/R))
    r_s = 2.0 * G * M / c**2
    grav_ratio = np.sqrt((1.0 - r_s / r_orbit) / (1.0 - r_s / R))
    # Positive means orbit clock runs faster (blueshift) relative to surface
    gravitational_fraction = grav_ratio - 1.0

    # Special-relativistic time dilation (orbit clock runs slower)
    special_fraction = np.sqrt(1.0 - v_orbit**2 / c**2) - 1.0  # negative

    # Convert to microseconds per day
    seconds_per_day = 86400.0
    grav_us_per_day = gravitational_fraction * seconds_per_day * 1e6
    spec_us_per_day = special_fraction * seconds_per_day * 1e6
    total_us_per_day = grav_us_per_day + spec_us_per_day

    return {
        "gravitational_us_day": grav_us_per_day,
        "special_us_day": spec_us_per_day,
        "total_us_day": total_us_per_day,
    }


# ---------------------------------------------------------------------------
# Relativistic Doppler shift
# ---------------------------------------------------------------------------

def doppler_shift(
    v: float | np.ndarray,
    angle: float | np.ndarray,
    include_transverse: bool = True,
) -> float | np.ndarray:
    """Relativistic Doppler frequency ratio.

    Parameters
    ----------
    v : float or ndarray
        Speed of the source [m/s].
    angle : float or ndarray
        Angle between the source velocity and the line of sight *from the
        observer to the source* [radians].  angle=0 means the source moves
        directly away from the observer.
    include_transverse : bool
        If True (default) include the transverse (time-dilation) factor.
        If False, return the purely longitudinal Doppler factor.

    Returns
    -------
    z : float or ndarray
        Dimensionless redshift  z = f_emitted / f_observed - 1.
        Positive z means redshift (receding), negative means blueshift.
    """
    beta = v / c
    if np.any(np.abs(beta) >= 1.0):
        raise ValueError("Speed must be less than c.")
    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    if include_transverse:
        # Full relativistic Doppler: f_obs = f_emit / [gamma (1 + beta cos theta)]
        # where theta is measured in the observer frame
        freq_ratio = gamma * (1.0 + beta * np.cos(angle))
    else:
        freq_ratio = 1.0 + beta * np.cos(angle)

    return freq_ratio - 1.0


# ---------------------------------------------------------------------------
# Sagnac effect
# ---------------------------------------------------------------------------

def sagnac_effect(
    omega: float,
    R: float,
    area_vector: np.ndarray,
    direction: int,
) -> float:
    """Sagnac time difference for a rotating interferometer.

    Parameters
    ----------
    omega : float
        Angular velocity of the rotating platform [rad/s].
    R : float
        Radius of the interferometer loop [m].
    area_vector : ndarray, shape (3,)
        Unit-normal vector defining the orientation of the interferometer
        plane (e.g. ``[0, 0, 1]`` for the xy-plane).
    direction : int
        +1 for co-rotating beam, -1 for counter-rotating beam.

    Returns
    -------
    dt : float
        Time difference between the two counter-propagating beams [s].
        The *magnitude* of the full round-trip Sagnac delay is
        4 * omega * A / c^2 where A is the enclosed area.  This function
        returns the signed single-trip contribution.
    """
    area_vector = np.asarray(area_vector, dtype=float)
    area = np.pi * R**2  # circular loop
    # Full Sagnac round-trip delay
    dt_round = 4.0 * omega * area / c**2
    # Signed single-trip contribution
    dot = np.dot(area_vector, area_vector)
    norm = np.sqrt(dot) if dot > 0 else 1.0
    sign = np.dot(area_vector / norm, [0, 0, 1])  # project onto z
    return direction * sign * dt_round / 2.0


# ---------------------------------------------------------------------------
# Lense-Thirring (frame-dragging) precession
# ---------------------------------------------------------------------------

def lense_thirring_rate(
    J: float,
    M: float,
    r: float,
    theta: float,
) -> float:
    """Lense-Thirring frame-dragging precession rate.

    The precession rate of a test gyroscope at position (r, theta) around a
    body of mass *M* with angular momentum *J* is::

        Omega_LT = (2 G J) / (c^2 r^3) * [1 + (3 cos^2 theta) / 2 .. ]

    We use the leading-order expression which already captures the dominant
    behaviour::

        Omega_LT = 2 G J / (c^2 r^3)

    with a theta-dependent enhancement factor.

    Parameters
    ----------
    J : float
        Angular momentum of the central body [kg m^2 s^-1].
    M : float
        Mass of the central body [kg].
    r : float
        Radial distance [m].
    theta : float
        Colatitude angle [radians] (measured from the spin axis).

    Returns
    -------
    Omega : float
        Precession rate [rad/s].
    """
    # Leading order Lense-Thirring precession
    Omega = 2.0 * G * J / (c**2 * r**3)
    # Theta-dependent correction (first-order multipole)
    Omega *= 1.0 + 1.5 * np.cos(theta) ** 2
    return Omega
