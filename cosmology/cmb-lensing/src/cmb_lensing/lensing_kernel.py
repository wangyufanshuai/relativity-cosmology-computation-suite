"""CMB lensing kernel: deflection angle, convergence, and shear."""

import numpy as np
from typing import Tuple


def lensing_potential(
    phi_map: np.ndarray, pixel_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute deflection angle alpha = grad(phi) from lensing potential map.

    Uses centred finite differences with wrap-around boundary conditions
    appropriate for full-sky projections on a flat periodic patch.

    Parameters
    ----------
    phi_map : np.ndarray
        2D lensing potential phi(x, y) in arbitrary potential units.
    pixel_scale : float
        Physical scale per pixel (arcmin/pixel).  The returned deflection
        angles are expressed in the same angular units.

    Returns
    -------
    alpha_x, alpha_y : np.ndarray
        x- and y-components of the deflection angle, same shape as *phi_map*.
    """
    phi = np.asarray(phi_map, dtype=np.float64)
    # centred finite differences  (periodic via np.roll)
    alpha_x = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (
        2.0 * pixel_scale
    )
    alpha_y = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (
        2.0 * pixel_scale
    )
    return alpha_x, alpha_y


def convergence_from_potential(
    phi_map: np.ndarray, pixel_scale: float = 1.0
) -> np.ndarray:
    """Convergence kappa = (1/2) Laplacian(phi) from the lensing potential.

    The Laplacian is evaluated via second-order centred finite differences
    with periodic boundaries.

    Parameters
    ----------
    phi_map : np.ndarray
        2D lensing potential.
    pixel_scale : float
        Arcmin per pixel.

    Returns
    -------
    np.ndarray
        Convergence map kappa(x, y).
    """
    phi = np.asarray(phi_map, dtype=np.float64)
    dx2 = (
        np.roll(phi, -1, axis=1) - 2.0 * phi + np.roll(phi, 1, axis=1)
    ) / pixel_scale**2
    dy2 = (
        np.roll(phi, -1, axis=0) - 2.0 * phi + np.roll(phi, 1, axis=0)
    ) / pixel_scale**2
    kappa = 0.5 * (dx2 + dy2)
    return kappa


def shear_from_potential(
    phi_map: np.ndarray, pixel_scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Shear (gamma_1, gamma_2) from the lensing potential.

    gamma_1 = (1/2)(d^2 phi/dx^2 - d^2 phi/dy^2)
    gamma_2 = d^2 phi/(dx dy)

    Parameters
    ----------
    phi_map : np.ndarray
        2D lensing potential.
    pixel_scale : float
        Arcmin per pixel.

    Returns
    -------
    gamma_1, gamma_2 : np.ndarray
        Two shear component maps.
    """
    phi = np.asarray(phi_map, dtype=np.float64)

    # second partial derivatives via centred differences (periodic)
    d2phi_dx2 = (
        np.roll(phi, -1, axis=1) - 2.0 * phi + np.roll(phi, 1, axis=1)
    ) / pixel_scale**2
    d2phi_dy2 = (
        np.roll(phi, -1, axis=0) - 2.0 * phi + np.roll(phi, 1, axis=0)
    ) / pixel_scale**2

    # cross derivative  d^2 phi / (dx dy)
    # Use the standard stencil:
    #   (phi[i+1,j+1] - phi[i+1,j-1] - phi[i-1,j+1] + phi[i-1,j-1]) / (4 dx dy)
    d2phi_dxdy = (
        np.roll(np.roll(phi, -1, axis=0), -1, axis=1)
        - np.roll(np.roll(phi, -1, axis=0), 1, axis=1)
        - np.roll(np.roll(phi, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(phi, 1, axis=0), 1, axis=1)
    ) / (4.0 * pixel_scale**2)

    gamma_1 = 0.5 * (d2phi_dx2 - d2phi_dy2)
    gamma_2 = d2phi_dxdy
    return gamma_1, gamma_2


def lens_cmb_temperature(
    T_unlensed: np.ndarray,
    phi_map: np.ndarray,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Apply lensing remapping to a CMB temperature map.

    T_lensed(x) = T_unlensed(x + grad(phi)(x))

    The displaced positions are looked up via bilinear interpolation in the
    unlensed map.  Pixels that map outside the array are wrapped periodically.

    Parameters
    ----------
    T_unlensed : np.ndarray
        2D unlensed CMB temperature map.
    phi_map : np.ndarray
        2D lensing potential (same shape as *T_unlensed*).
    pixel_scale : float
        Arcmin per pixel (only affects the gradient scale).

    Returns
    -------
    np.ndarray
        Lensed temperature map of the same shape.
    """
    T_unlensed = np.asarray(T_unlensed, dtype=np.float64)
    phi = np.asarray(phi_map, dtype=np.float64)

    ny, nx = T_unlensed.shape

    # pixel coordinate grids
    iy, ix = np.mgrid[0:ny, 0:nx].astype(np.float64)

    # deflection angles in pixel units
    alpha_x, alpha_y = lensing_potential(phi, pixel_scale=1.0)  # gradient in pixel units
    # scale from pixel-scale gradient to deflection in pixel coordinates
    # alpha is d(phi)/d(pixel), we interpret it as a pixel displacement
    alpha_x *= pixel_scale
    alpha_y *= pixel_scale

    # lensed positions  (continuous pixel coordinates)
    x_lensed = ix + alpha_x
    y_lensed = iy + alpha_y

    # bilinear interpolation with periodic wrapping
    x_lensed_mod = np.mod(x_lensed, nx)
    y_lensed_mod = np.mod(y_lensed, ny)

    x0 = np.floor(x_lensed_mod).astype(np.intp)
    y0 = np.floor(y_lensed_mod).astype(np.intp)
    x1 = np.mod(x0 + 1, nx)
    y1 = np.mod(y0 + 1, ny)

    wx = x_lensed_mod - x0
    wy = y_lensed_mod - y0

    T_lensed = (
        T_unlensed[y0, x0] * (1 - wx) * (1 - wy)
        + T_unlensed[y0, x1] * wx * (1 - wy)
        + T_unlensed[y1, x0] * (1 - wx) * wy
        + T_unlensed[y1, x1] * wx * wy
    )
    return T_lensed


def lensed_positions(
    x: np.ndarray,
    y: np.ndarray,
    phi_map: np.ndarray,
    pixel_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lensed source positions from the deflection field.

    (x_lensed, y_lensed) = (x, y) + (alpha_x, alpha_y)

    Parameters
    ----------
    x, y : np.ndarray
        Unlensed source position arrays (in pixel coordinates).
    phi_map : np.ndarray
        2D lensing potential map.
    pixel_scale : float
        Arcmin per pixel.

    Returns
    -------
    x_lensed, y_lensed : np.ndarray
        Lensed position arrays.
    """
    phi = np.asarray(phi_map, dtype=np.float64)
    ny, nx = phi.shape

    # deflection in pixel units
    alpha_x, alpha_y = lensing_potential(phi, pixel_scale=1.0)
    alpha_x *= pixel_scale
    alpha_y *= pixel_scale

    # interpolate deflection at arbitrary (x, y) positions
    x_mod = np.mod(np.asarray(x, dtype=np.float64), nx)
    y_mod = np.mod(np.asarray(y, dtype=np.float64), ny)

    x0 = np.floor(x_mod).astype(np.intp)
    y0 = np.floor(y_mod).astype(np.intp)
    x1 = np.mod(x0 + 1, nx)
    y1 = np.mod(y0 + 1, ny)

    wx = x_mod - x0
    wy = y_mod - y0

    def _bilinear(field):
        return (
            field[y0, x0] * (1 - wx) * (1 - wy)
            + field[y0, x1] * wx * (1 - wy)
            + field[y1, x0] * (1 - wx) * wy
            + field[y1, x1] * wx * wy
        )

    ax_interp = _bilinear(alpha_x)
    ay_interp = _bilinear(alpha_y)

    return x + ax_interp, y + ay_interp
