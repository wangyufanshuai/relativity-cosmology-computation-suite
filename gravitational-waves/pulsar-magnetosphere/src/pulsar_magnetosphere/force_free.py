"""
Force-Free Electrodynamics
============================

Implements the force-free condition  rho E + J x B = 0  and the
constraint  E . B = 0  for pulsar magnetosphere modelling.

In the force-free approximation the plasma inertia is neglected and the
Lorentz force on the charge-current distribution must vanish:

    rho E + J x B = 0          (force-free condition)
    E . B = 0                   (degeneracy condition)

These imply:
    rho_e = (E . (nabla x B)) / (4 pi B^2)          (charge density)
    J     = (c / (4 pi)) nabla x B - (1 / (4 pi)) dE/dt   (Maxwell current)
"""

import numpy as np
from .dipole import RotatingDipole, C_LIGHT


class ForceFreeSolver:
    """Evaluate force-free quantities on a grid using a RotatingDipole field.

    Parameters
    ----------
    dipole : RotatingDipole
        The rotating dipole configuration to evaluate.
    """

    def __init__(self, dipole: RotatingDipole) -> None:
        self.dipole = dipole

    # ------------------------------------------------------------------
    # E . B diagnostic
    # ------------------------------------------------------------------

    def edotb(
        self,
        r: float | np.ndarray,
        theta: float | np.ndarray,
    ) -> np.ndarray:
        """Compute E . B for the co-rotation field.

        In a perfectly force-free magnetosphere E . B = 0.  For the
        co-rotation electric field of a static dipole, E . B vanishes
        exactly because the E-field lies in the (r, theta) plane but
        is perpendicular to B.

        Parameters
        ----------
        r : float or ndarray
            Spherical radius [cm].
        theta : float or ndarray
            Polar angle [rad].

        Returns
        -------
        ndarray
            E . B [G statV/cm].
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)

        B_r, B_theta = self.dipole.magnetic_field(r, theta)
        E_r, E_theta, _ = self.dipole.electric_field(r, theta)

        return E_r * B_r + E_theta * B_theta

    # ------------------------------------------------------------------
    # Charge density
    # ------------------------------------------------------------------

    def charge_density(
        self,
        r: float | np.ndarray,
        theta: float | np.ndarray,
    ) -> np.ndarray:
        """Goldreich-Julian charge density (co-rotation).

        The GJ charge density in the near-zone limit is:
            rho_GJ = -Omega . B / (2 pi c)

        For Omega along z and B in the (r, theta) plane:
            Omega . B = Omega (B_r cos(theta) - B_theta sin(theta))

        Parameters
        ----------
        r : float or ndarray
            Spherical radius [cm].
        theta : float or ndarray
            Polar angle [rad].

        Returns
        -------
        ndarray
            Charge density [esu/cm^3].
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)

        B_r, B_theta = self.dipole.magnetic_field(r, theta)
        Omega_dot_B = self.dipole.omega * (B_r * np.cos(theta) - B_theta * np.sin(theta))

        return -Omega_dot_B / (2.0 * np.pi * C_LIGHT)

    # ------------------------------------------------------------------
    # Current density via curl B  (magnetostatic, near-zone)
    # ------------------------------------------------------------------

    def current_density(
        self,
        r: float | np.ndarray,
        theta: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Current density from curl B (static near-zone approximation).

        In the magnetostatic limit dE/dt = 0, so:
            J = (c / (4 pi)) curl B

        For a dipole field the non-zero curl gives a toroidal (phi)
        current plus poloidal corrections.

        In spherical coordinates, for an axisymmetric field
        B = (B_r, B_theta, 0):

        (curl B)_r = (1/(r sin theta)) d/dtheta (sin theta B_phi)  [= 0]
        (curl B)_theta = -(1/r) d/dr (r B_phi)                    [= 0]
        (curl B)_phi = (1/r) [d/dr (r B_theta) - dB_r/dtheta]

        Parameters
        ----------
        r : float or ndarray
            Spherical radius [cm].  Must be a scalar or 1-D array.
        theta : float or ndarray
            Polar angle [rad].  Must be a scalar or 1-D array.

        Returns
        -------
        (J_r, J_theta, J_phi) : tuple of ndarrays
            Current density components [esu/cm^2/s ~ statA/cm^2].
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)

        mu = self.dipole.mu

        # Analytical curl of a pure dipole field:
        # B_r = 2 mu cos(theta) / r^3
        # B_theta = mu sin(theta) / r^3
        #
        # (curl B)_phi = (1/r)[ d(r B_theta)/dr - dB_r/dtheta ]
        #   d(r B_theta)/dr = d(mu sin(theta) / r^2)/dr = -2 mu sin(theta)/r^3
        #   dB_r/dtheta = -2 mu sin(theta) / r^3
        # => (curl B)_phi = (1/r)[ -2 mu sin(theta)/r^3 + 2 mu sin(theta)/r^3 ] = 0
        #
        # This is the well-known result: a vacuum dipole has zero current.
        # In a force-free magnetosphere the field is NOT a vacuum dipole --
        # it is distorted by the plasma.  We add the toroidal field component
        # that arises from the GJ current (split-monopole-like beyond R_LC).

        J_r = np.zeros_like(r)
        J_theta = np.zeros_like(r)

        # Toroidal current exists in the force-free magnetosphere due to
        # the poloidal current closure.  In the near-zone it is small;
        # we model the return current as:
        #   J_phi ~ rho_GJ * v_phi  where v_phi = Omega * r * sin(theta)
        # but capped at c (only meaningful for r < R_LC).
        rho_gj = self.charge_density(r, theta)
        v_phi = self.dipole.omega * r * np.sin(theta)
        # Cap at light cylinder
        v_phi = np.minimum(v_phi, C_LIGHT)

        J_phi = rho_gj * v_phi

        return J_r, J_theta, J_phi

    # ------------------------------------------------------------------
    # Force-free check
    # ------------------------------------------------------------------

    def check_force_free(
        self,
        r: float | np.ndarray,
        theta: float | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Evaluate how well the force-free conditions are satisfied.

        Returns a dict with:
            - 'edotb'         : E . B (should be ~0)
            - 'edotb_norm'    : |E . B| / (|E| |B|) (should be ~0)
            - 'rho_e'         : charge density

        Parameters
        ----------
        r : float or ndarray
            Spherical radius [cm].
        theta : float or ndarray
            Polar angle [rad].

        Returns
        -------
        dict
            Diagnostic quantities.
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)

        B_r, B_theta = self.dipole.magnetic_field(r, theta)
        E_r, E_theta, E_phi = self.dipole.electric_field(r, theta)

        B_mag = np.sqrt(B_r**2 + B_theta**2)
        E_mag = np.sqrt(E_r**2 + E_theta**2 + E_phi**2)

        edotb_val = E_r * B_r + E_theta * B_theta + E_phi * np.zeros_like(r)

        result = {
            "edotb": edotb_val,
            "edotb_norm": np.where(
                (B_mag * E_mag) > 0,
                np.abs(edotb_val) / (B_mag * E_mag),
                0.0,
            ),
            "rho_e": self.charge_density(r, theta),
        }
        return result
