"""
Fluorescent iron K-alpha line profile from an accretion disk around a Kerr black hole.

The iron Kalpha line at E_0 = 6.4 keV is produced by fluorescence in the
inner accretion disk. The observed line profile is broadened and skewed by:

1. Gravitational redshift: photons lose energy climbing out of the potential well.
   g_grav = sqrt(1 - 2*M*r/Sigma)  (simplified; full expression below)

2. Doppler shift from orbital motion of the disk material.
   The orbital velocity is Keplerian.

3. Relativistic beaming (aberration).

The total redshift factor:
    g = E_obs / E_em = sqrt(-g^{tt} - 2*Omega*g^{tphi} - Omega^2*g^{phiphi}) / (1 + v_r * ...)  [simplified]

For circular Keplerian orbits (v_r = 0):
    g = sqrt(-g_tt - 2*Omega*g_tphi - Omega^2 * g_phiphi)

The line profile:
    I(E_obs) = integral over disk of epsilon(r) * delta(E_obs - g * E_line) * g^4 * dA

where:
    epsilon(r) is the line emissivity (often a power law: epsilon ~ r^{-q})
    g^4 accounts for radiation beaming + redshift
    delta ensures energy conservation
"""

import numpy as np
from .kerr_rays import compute_isco, metric_coefficients
from .accretion_disk import keplerian_angular_velocity


# ---------------------------------------------------------------------------
# Redshift factor
# ---------------------------------------------------------------------------

def redshift_factor(
    r: float,
    phi: float,
    M: float = 1.0,
    a: float = 0.0,
) -> float:
    """
    Compute the total redshift factor g = E_obs / E_em for photons emitted
    from a Keplerian circular orbit at (r, phi) in the equatorial plane
    and received by a distant observer at inclination theta_obs.

    For simplicity, this computes the local redshift factor (excluding
    light-bending). The full transfer function is in transfer_function.py.

    For a stationary emitter on a Keplerian orbit:
    g = sqrt(-g_tt - 2*Omega*g_tphi - Omega^2 * g_phiphi)

    This combines gravitational redshift and transverse Doppler effect.

    Parameters
    ----------
    r : float
        Emission radius (Boyer-Lindquist).
    phi : float
        Azimuthal angle (not used for axisymmetric metrics, included for interface).
    M : float
        Black hole mass.
    a : float
        Spin parameter.

    Returns
    -------
    float
        Redshift factor g = E_obs / E_em.
        g < 1 means redshift (photon loses energy).
        g > 1 means blueshift (approaching side of disk).
    """
    theta = np.pi / 2.0  # equatorial disk
    gc = metric_coefficients(r, theta, M, a)
    Omega = keplerian_angular_velocity(r, M, a)

    g_squared = -(
        gc["g_tt"]
        + 2.0 * Omega * gc["g_tphi"]
        + Omega**2 * gc["g_phiphi"]
    )

    # g_squared should be positive for physical orbits
    if g_squared <= 0:
        return 0.01  # fallback for pathological cases

    return np.sqrt(g_squared)


def redshift_factor_schwarzschild(r: float, M: float = 1.0) -> float:
    """
    Redshift factor for Schwarzschild (a=0), equatorial Keplerian orbit.

    g = sqrt(1 - 3M/r)  (combines gravitational + transverse Doppler)

    This is the standard result: at r=3M (photon sphere), g=0.
    At r=6M (ISCO), g = sqrt(1/2) ~ 0.707.
    """
    val = 1.0 - 3.0 * M / r
    if val <= 0:
        return 0.01
    return np.sqrt(val)


# ---------------------------------------------------------------------------
# Iron K-alpha line profile
# ---------------------------------------------------------------------------

def iron_line_energy() -> float:
    """Rest energy of the iron Kalpha fluorescence line in keV."""
    return 6.4


def line_emissivity(
    r: float,
    r_isco: float = 6.0,
    q: float = 3.0,
) -> float:
    """
    Line emissivity as a function of radius.

    Standard power-law model: epsilon(r) = r^{-q} for r >= r_isco.

    Parameters
    ----------
    r : float or array
        Radius.
    r_isco : float
        Inner edge of emission.
    q : float
        Power-law index (typically q=2..3).

    Returns
    -------
    float or array
        Emissivity (zero inside ISCO).
    """
    r = np.asarray(r, dtype=float)
    eps = np.where(r >= r_isco, r ** (-q), 0.0)
    return eps


def compute_iron_line_profile(
    E_obs_array: np.ndarray,
    M: float = 1.0,
    a: float = 0.0,
    r_inner: float = None,
    r_outer: float = 20.0,
    inclination: float = np.pi / 4.0,
    q: float = 3.0,
    n_r: int = 100,
    n_phi: int = 100,
    E_line: float = 6.4,
) -> np.ndarray:
    """
    Compute the observed iron Kalpha line profile from a Kerr accretion disk.

    Integrates over the disk surface:
    I(E_obs) = integral epsilon(r) * g^4 * delta(E_obs - g*E_line) * r dr dphi

    The delta function is approximated by a narrow Gaussian for numerical integration.

    Parameters
    ----------
    E_obs_array : array
        Observed energy grid (keV).
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    r_inner : float or None
        Inner emission radius (default: ISCO).
    r_outer : float
        Outer emission radius.
    inclination : float
        Observer inclination (0 = face-on, pi/2 = edge-on).
    q : float
        Emissivity power-law index.
    n_r, n_phi : int
        Number of radial and azimuthal grid points.
    E_line : float
        Rest-frame line energy (keV), default 6.4 for Fe Kalpha.

    Returns
    -------
    array
        Line flux I(E_obs) on the energy grid.
    """
    r_isco = compute_isco(M, a)
    if r_inner is None:
        r_inner = r_isco

    # Radial grid (log-spaced for better resolution near ISCO)
    r_grid = np.linspace(r_inner, r_outer, n_r)
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    dr = r_grid[1] - r_grid[0] if n_r > 1 else 0.1
    dphi = phi_grid[1] - phi_grid[0] if n_phi > 1 else 2 * np.pi / n_phi

    # Line width for delta-function approximation (keV)
    sigma_E = 0.05  # narrow Gaussian width

    profile = np.zeros_like(E_obs_array, dtype=float)

    for ri in r_grid:
        eps_r = line_emissivity(ri, r_isco, q)
        if eps_r <= 0:
            continue

        for phi_i in phi_grid:
            # Compute redshift factor including Doppler from inclination
            g_local = redshift_factor(ri, phi_i, M, a)

            # Doppler boost from inclination:
            # The approaching side of the disk is blueshifted,
            # the receding side is redshifted.
            # v_orbital = Omega * r  (coordinate velocity)
            Omega = keplerian_angular_velocity(ri, M, a)
            v_proj = Omega * ri * np.sin(inclination) * np.sin(phi_i)

            # First-order Doppler factor: 1 + v/c projected along line of sight
            # In the weak-field limit: g_total ~ g_local * (1 - v_proj)
            # For consistency with the GR calculation, we use the full metric:
            # g already includes the orbital motion via Omega in the metric.
            # The inclination effect on the observed energy comes from the
            # projection of the 4-momentum. For a simple model:
            g_total = g_local * (1.0 - v_proj / (1.0 + 0.0))  # simplified projection

            if g_total <= 0:
                continue

            E_shifted = g_total * E_line

            # g^4 factor (radiation beaming + redshift)
            g4 = g_total**4

            # Add to profile via Gaussian-smeared delta function
            contribution = (
                eps_r * g4 * ri * dr * dphi
                * np.exp(-0.5 * ((E_obs_array - E_shifted) / sigma_E) ** 2)
                / (sigma_E * np.sqrt(2 * np.pi))
            )

            profile += contribution

    return profile


def compute_iron_line_profile_simple(
    E_obs_array: np.ndarray,
    M: float = 1.0,
    a: float = 0.0,
    r_outer: float = 20.0,
    q: float = 3.0,
    n_r: int = 200,
    E_line: float = 6.4,
) -> np.ndarray:
    """
    Simplified iron line profile using only gravitational redshift (no Doppler asymmetry).

    Useful for testing and for face-on (inclination=0) viewing.

    I(E_obs) = integral_r epsilon(r) * g(r)^4 * delta(E_obs - g(r)*E_line) * r dr

    Parameters
    ----------
    E_obs_array : array
        Observed energy grid (keV).
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    r_outer : float
        Outer disk radius.
    q : float
        Emissivity index.
    n_r : int
        Radial grid points.
    E_line : float
        Rest-frame line energy (keV).

    Returns
    -------
    array
        Line flux profile.
    """
    r_isco = compute_isco(M, a)
    r_grid = np.linspace(r_isco + 0.01, r_outer, n_r)
    dr = r_grid[1] - r_grid[0] if n_r > 1 else 0.1

    sigma_E = 0.03
    profile = np.zeros_like(E_obs_array, dtype=float)

    for ri in r_grid:
        eps_r = line_emissivity(ri, r_isco, q)
        if eps_r <= 0:
            continue

        g = redshift_factor(ri, 0.0, M, a)
        if g <= 0:
            continue

        E_shifted = g * E_line
        g4 = g**4

        contribution = (
            eps_r * g4 * ri * dr
            * np.exp(-0.5 * ((E_obs_array - E_shifted) / sigma_E) ** 2)
            / (sigma_E * np.sqrt(2 * np.pi))
        )
        profile += contribution

    return profile
