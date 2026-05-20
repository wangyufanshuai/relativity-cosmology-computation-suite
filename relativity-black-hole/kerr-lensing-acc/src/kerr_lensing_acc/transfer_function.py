"""
Disk-to-observer transfer function for Kerr black hole accretion disks.

Maps each pixel of the observer's image plane to disk coordinates (r, phi),
accounting for gravitational lensing and multiple images (direct, secondary).

The transfer function encodes:
    f(r, phi) = g^4

where g = E_obs/E_em is the redshift factor, and g^4 accounts for:
- Gravitational redshift (g)
- Doppler shift (g)
- Relativistic beaming / solid angle transformation (g^3)
Combined: g^4 (for bolometric flux from a moving emitter in curved spacetime).

The observed flux is:
    F_obs = integral f(r, phi) * I_em(r, phi) * r dr dphi

This module uses ray tracing from kerr_rays.py to build the transfer function.
"""

import numpy as np
from .kerr_rays import trace_ray, compute_isco
from .iron_line import redshift_factor
from .accretion_disk import keplerian_angular_velocity, disk_temperature


# ---------------------------------------------------------------------------
# Transfer function computation
# ---------------------------------------------------------------------------

def compute_transfer_function(
    alpha: float,
    beta: float,
    r_obs: float,
    theta_obs: float,
    M: float = 1.0,
    a: float = 0.0,
    r_disk_outer: float = 20.0,
) -> dict:
    """
    Compute the transfer function value for a single ray from observer to disk.

    Traces a ray from impact parameters (alpha, beta) on the observer's sky
    back to the disk. If the ray hits the disk, computes the redshift factor
    g and returns g^4.

    Parameters
    ----------
    alpha, beta : float
        Impact parameters (M-units) on the observer's sky.
    r_obs : float
        Observer radial coordinate.
    theta_obs : float
        Observer inclination angle.
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    r_disk_outer : float
        Outer disk radius.

    Returns
    -------
    dict with keys:
        'g': float or None  (redshift factor)
        'f': float or None  (transfer function value = g^4)
        'r_disk': float or None
        'phi_disk': float or None
        'hit': bool
    """
    result = trace_ray(
        alpha=alpha,
        beta=beta,
        r_obs=r_obs,
        theta_obs=theta_obs,
        M=M,
        a=a,
        lambda_max=80.0,
        n_steps=5000,
        r_disk_outer=r_disk_outer,
    )

    if not result["hit_disk"]:
        return {
            "g": None,
            "f": None,
            "r_disk": None,
            "phi_disk": None,
            "hit": False,
        }

    r_disk = result["r_disk"]
    phi_disk = result["phi_disk"]

    # Compute redshift factor at emission point
    g = redshift_factor(r_disk, phi_disk, M, a)

    # Transfer function: g^4
    f = g**4 if g > 0 else 0.0

    return {
        "g": g,
        "f": f,
        "r_disk": r_disk,
        "phi_disk": phi_disk,
        "hit": True,
    }


def compute_disk_image(
    r_obs: float,
    theta_obs: float,
    M: float = 1.0,
    a: float = 0.0,
    n_pixels: int = 50,
    fov: float = 20.0,
    r_disk_outer: float = 20.0,
) -> dict:
    """
    Compute a 2D image of the accretion disk as seen by a distant observer.

    For each pixel on the observer's sky, traces a ray back to the disk
    and records the disk coordinates and transfer function.

    Parameters
    ----------
    r_obs : float
        Observer distance.
    theta_obs : float
        Observer inclination.
    M : float
        Black hole mass.
    a : float
        Spin parameter.
    n_pixels : int
        Image resolution (n_pixels x n_pixels).
    fov : float
        Field of view in M units (half-width).
    r_disk_outer : float
        Outer disk radius.

    Returns
    -------
    dict with keys:
        'alpha_grid': 2D array of alpha coordinates
        'beta_grid': 2D array of beta coordinates
        'g_image': 2D array of redshift factors (0 where no hit)
        'f_image': 2D array of transfer function values
        'r_image': 2D array of disk radii (0 where no hit)
        'phi_image': 2D array of disk azimuthal angles (0 where no hit)
    """
    alpha_range = np.linspace(-fov, fov, n_pixels)
    beta_range = np.linspace(-fov, fov, n_pixels)
    alpha_grid, beta_grid = np.meshgrid(alpha_range, beta_range)

    g_image = np.zeros((n_pixels, n_pixels))
    f_image = np.zeros((n_pixels, n_pixels))
    r_image = np.zeros((n_pixels, n_pixels))
    phi_image = np.zeros((n_pixels, n_pixels))

    for i in range(n_pixels):
        for j in range(n_pixels):
            tf = compute_transfer_function(
                alpha=alpha_grid[i, j],
                beta=beta_grid[i, j],
                r_obs=r_obs,
                theta_obs=theta_obs,
                M=M,
                a=a,
                r_disk_outer=r_disk_outer,
            )
            if tf["hit"]:
                g_image[i, j] = tf["g"]
                f_image[i, j] = tf["f"]
                r_image[i, j] = tf["r_disk"]
                phi_image[i, j] = tf["phi_disk"]

    return {
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "g_image": g_image,
        "f_image": f_image,
        "r_image": r_image,
        "phi_image": phi_image,
    }


def observed_flux(
    r_obs: float,
    theta_obs: float,
    M: float = 1.0,
    a: float = 0.0,
    Mdot: float = 1.0,
    n_r: int = 50,
    n_phi: int = 60,
    r_outer: float = 20.0,
) -> float:
    """
    Compute total observed flux from the accretion disk.

    F_obs = integral f(r,phi) * I_em(r,phi) * r dr dphi

    where f = g^4 is the transfer function and I_em is the local intensity.

    For a thin disk with blackbody emission:
    F_obs = integral g^4 * sigma_SB * T(r)^4 * r dr dphi

    This uses the redshift factor directly (not ray tracing) for speed.

    Parameters
    ----------
    r_obs : float
        Observer distance.
    theta_obs : float
        Observer inclination.
    M, a, Mdot : float
        Black hole / accretion parameters.
    n_r, n_phi : int
        Integration grid sizes.
    r_outer : float
        Outer disk radius.

    Returns
    -------
    float
        Total observed flux.
    """
    r_isco = compute_isco(M, a)
    r_grid = np.linspace(r_isco + 0.01, r_outer, n_r)
    phi_grid = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    dr = r_grid[1] - r_grid[0] if n_r > 1 else 0.1
    dphi = phi_grid[1] - phi_grid[0] if n_phi > 1 else 2 * np.pi / n_phi

    F_total = 0.0

    for ri in r_grid:
        T = disk_temperature(ri, M, a, Mdot)
        if T <= 0:
            continue

        # Local intensity: sigma_SB * T^4
        sigma_sb = np.pi**2 / 60.0
        I_em = sigma_sb * T**4

        for phi_i in phi_grid:
            g = redshift_factor(ri, phi_i, M, a)
            if g <= 0:
                continue

            f = g**4

            F_total += f * I_em * ri * dr * dphi

    return F_total


def transfer_function_radial(
    r: float,
    M: float = 1.0,
    a: float = 0.0,
) -> float:
    """
    Compute the transfer function f = g^4 as a function of radius only.

    Uses the on-axis redshift factor (no inclination-dependent Doppler).

    Parameters
    ----------
    r : float or array
        Disk radius.
    M : float
        Black hole mass.
    a : float
        Spin parameter.

    Returns
    -------
    float or array
        Transfer function value g^4.
    """
    r = np.asarray(r, dtype=float)
    g = np.vectorize(lambda ri: redshift_factor(ri, 0.0, M, a))(r)
    g = np.where(g > 0, g, 0.0)
    return g**4
