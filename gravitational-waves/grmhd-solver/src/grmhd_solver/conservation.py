"""Conservation law checks for GRMHD simulations.

Provides tools to verify:
- Total energy-momentum conservation
- Magnetic field divergence-free condition (div B = 0)
- Particle number conservation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def check_divergence_free(
    B_r: NDArray,
    B_theta: NDArray,
    r: NDArray,
    theta: NDArray,
) -> NDArray:
    """Compute div B in spherical coordinates (should be zero).

    div B = (1/r^2) d(r^2 B_r)/dr + (1/(r sin theta)) d(sin theta B_theta)/dtheta

    Parameters
    ----------
    B_r : array
        Radial magnetic field component at cell centers.
    B_theta : array
        Polar magnetic field component at cell centers.
    r : array
        Radial coordinates.
    theta : array
        Polar angle coordinates.

    Returns
    -------
    array
        div B at each point (should be zero for physical fields).
    """
    # Use 1D coordinate arrays for gradient computation along each axis
    # r and theta may be 2D meshgrids; extract unique 1D arrays
    if r.ndim == 2:
        r_1d = r[0, :]
        theta_1d = theta[:, 0]
    else:
        r_1d = r
        theta_1d = theta

    # Compute gradients numerically along each axis
    # axis=-1 is the r-direction (columns), axis=-2 is the theta-direction (rows)
    d_r2Br_dr = np.gradient(r**2 * B_r, r_1d, axis=-1)
    d_sinBth_dtheta = np.gradient(np.sin(theta) * B_theta, theta_1d, axis=-2)

    div_B = (1.0 / r**2) * d_r2Br_dr + \
            (1.0 / (r * np.sin(theta))) * d_sinBth_dtheta

    return div_B


def total_energy(
    D: NDArray,
    tau: NDArray,
    S: NDArray,
    v: NDArray,
    dV: NDArray,
) -> float:
    """Compute total conserved energy E_tot = sum(tau + D) * dV.

    Parameters
    ----------
    D : array
        Lab-frame density.
    tau : array
        Energy - rest-mass density.
    S : array
        Momentum density.
    v : array
        Velocity.
    dV : array
        Volume element.

    Returns
    -------
    float
        Total energy in the domain.
    """
    return float(np.sum((tau + D) * dV))


def total_momentum(S: NDArray, dV: NDArray) -> float:
    """Compute total momentum P_tot = sum(S * dV)."""
    return float(np.sum(S * dV))


def total_rest_mass(D: NDArray, dV: NDArray) -> float:
    """Compute total rest mass M_tot = sum(D * dV)."""
    return float(np.sum(D * dV))
