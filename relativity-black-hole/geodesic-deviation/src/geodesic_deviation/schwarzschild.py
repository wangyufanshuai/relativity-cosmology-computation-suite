"""Tidal forces and Riemann tensor for Schwarzschild spacetime."""

import numpy as np


def schwarzschild_tidal_radial(r, M=1.0):
    """Radial tidal force coefficient -2M/r^3 in Schwarzschild."""
    r = np.asarray(r, dtype=float)
    return -2.0 * M / r**3


def schwarzschild_tidal_transverse(r, M=1.0):
    """Transverse tidal force coefficient M/r^3 in Schwarzschild."""
    r = np.asarray(r, dtype=float)
    return M / r**3


def jacobi_schwarzschild_radial(lamb, y, M=1.0, r0=10.0, E=1.0, L=0.0):
    """Jacobi equation for radial geodesic deviation in Schwarzschild.

    For radial geodesics (L=0): d^2(xi^r)/dtau^2 = -2M/r^3 * xi^r
    """
    xi_r, dxi_r = y
    r = r0
    R_rtr = -2.0 * M / r**3
    return [dxi_r, R_rtr * xi_r]


def riemann_schwarzschild(r, theta_coord, M=1.0):
    """Non-vanishing independent Riemann tensor components for Schwarzschild."""
    r2 = r * r
    r3 = r2 * r
    return {
        "R_trtr": (2 * M) / r3 * (1 - 2 * M / r),
        "R_tptp": -M / r3 * (1 - 2 * M / r),
        "R_tftf": -M / r3 * (1 - 2 * M / r),
        "R_rprp": -M / r3 / (1 - 2 * M / r),
        "R_rfrf": -M / r3 / (1 - 2 * M / r),
        "R_pfpf": 2 * M / r3,
    }


def tidal_tensor_trace(R_munu_un_un):
    """Trace of the tidal tensor E_ij = R_{0i0j} (should be 0 in vacuum)."""
    return np.sum(R_munu_un_un)
