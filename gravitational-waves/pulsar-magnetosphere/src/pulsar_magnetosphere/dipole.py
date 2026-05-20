"""
Rotating Magnetic Dipole Field
================================

Implements the Deutsch (1955) solution for a rotating magnetic dipole in vacuum,
plus the co-rotation electric field and key magnetospheric quantities.

Physics
-------
- Magnetic dipole field:
    B_r = (2 * B0 * R^3 / r^3) * cos(theta)
    B_theta = (B0 * R^3 / r^3) * sin(theta)

- Co-rotation electric field:
    E = -(Omega x r) x B

- Light cylinder radius:
    R_LC = c / Omega

- Spindown luminosity:
    L_sd = (2 / (3 * c^3)) * mu^2 * Omega^4 * sin^2(alpha)
    where mu = B0 * R^3 is the magnetic dipole moment.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
C_LIGHT = 2.99792458e10       # speed of light  [cm/s]
E_CHARGE = 4.80320425e10      # elementary charge [esu]
M_ELECTRON = 9.1093837e-28    # electron mass [g]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def light_cylinder_radius(omega: float) -> float:
    """Return the light-cylinder radius R_LC = c / Omega.

    Parameters
    ----------
    omega : float
        Angular velocity of the neutron star [rad/s].

    Returns
    -------
    float
        Light-cylinder radius [cm].
    """
    if omega <= 0.0:
        raise ValueError("Angular velocity omega must be positive.")
    return C_LIGHT / omega


def magnetic_dipole_moment(B0: float, R: float) -> float:
    """Return the magnetic dipole moment mu = B0 * R^3.

    Parameters
    ----------
    B0 : float
        Surface magnetic field at the pole [G].
    R : float
        Stellar radius [cm].

    Returns
    -------
    float
        Dipole moment [G cm^3].
    """
    return B0 * R**3


def spindown_luminosity(
    B0: float,
    R: float,
    omega: float,
    alpha: float = np.pi / 2.0,
) -> float:
    """Vacuum dipole spindown luminosity.

    L_sd = (2 / (3 c^3)) * mu^2 * Omega^4 * sin^2(alpha)

    Parameters
    ----------
    B0 : float
        Surface polar magnetic field [G].
    R : float
        Stellar radius [cm].
    omega : float
        Angular velocity [rad/s].
    alpha : float, optional
        Inclination angle between rotation and magnetic axes [rad].
        Default is pi/2 (orthogonal rotator).

    Returns
    -------
    float
        Spindown luminosity [erg/s].
    """
    mu = magnetic_dipole_moment(B0, R)
    return (2.0 / (3.0 * C_LIGHT**3)) * mu**2 * omega**4 * np.sin(alpha)**2


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RotatingDipole:
    """Rotating magnetic dipole in the near-zone (static approximation).

    The near-zone solution ignores retardation effects and uses the
    time-independent vacuum dipole field.  This is an excellent
    approximation for r << R_LC.

    Parameters
    ----------
    B0 : float
        Surface polar magnetic field [G].
    R_star : float
        Stellar radius [cm].
    omega : float
        Angular velocity [rad/s].
    alpha : float
        Magnetic inclination angle [rad].
    """

    def __init__(
        self,
        B0: float,
        R_star: float,
        omega: float,
        alpha: float = np.pi / 2.0,
    ) -> None:
        if B0 <= 0:
            raise ValueError("B0 must be positive.")
        if R_star <= 0:
            raise ValueError("R_star must be positive.")
        if omega <= 0:
            raise ValueError("omega must be positive.")

        self.B0 = B0
        self.R_star = R_star
        self.omega = omega
        self.alpha = alpha
        self.mu = magnetic_dipole_moment(B0, R_star)
        self.R_LC = light_cylinder_radius(omega)

    # ----- field components in spherical coordinates (r, theta, phi) -----

    def magnetic_field(self, r: float | np.ndarray, theta: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Static vacuum dipole magnetic field.

        B_r     = 2 mu cos(theta) / r^3
        B_theta =   mu sin(theta) / r^3

        Parameters
        ----------
        r : float or ndarray
            Spherical radius [cm].
        theta : float or ndarray
            Polar angle [rad].

        Returns
        -------
        (B_r, B_theta) : tuple of ndarrays
            Magnetic field components [G].
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)
        mu = self.mu

        B_r = 2.0 * mu * np.cos(theta) / r**3
        B_theta = mu * np.sin(theta) / r**3
        return B_r, B_theta

    def electric_field(
        self,
        r: float | np.ndarray,
        theta: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Co-rotation electric field E = -(Omega x r) x B.

        In the rotating frame the co-rotation E-field arises from the
        transformation E' = -(v/c) x B where v = Omega x r.

        For a dipole with Omega along z, the co-rotation E has components:
            E_r = -(Omega * r * sin(theta) / c) * B_theta
            E_theta = (Omega * r * sin(theta) / c) * B_r
            E_phi = 0

        Parameters
        ----------
        r : float or ndarray
            Spherical radius [cm].
        theta : float or ndarray
            Polar angle [rad].

        Returns
        -------
        (E_r, E_theta, E_phi) : tuple of ndarrays
            Electric field components [statV/cm].
        """
        r = np.asarray(r, dtype=float)
        theta = np.asarray(theta, dtype=float)

        B_r, B_theta = self.magnetic_field(r, theta)

        # (Omega x r) magnitude in the phi-direction: Omega * r * sin(theta)
        Omega_cross_r = self.omega * r * np.sin(theta)  # phi-direction

        # -(Omega x r) x B  gives components in r and theta:
        #   E_r     = -Omega_cross_r * B_theta   (from phi x theta = -r)
        #   E_theta =  Omega_cross_r * B_r       (from phi x r = theta)
        E_r = -Omega_cross_r * B_theta / C_LIGHT
        E_theta = Omega_cross_r * B_r / C_LIGHT
        E_phi = np.zeros_like(r)

        return E_r, E_theta, E_phi

    def magnetic_field_magnitude(self, r: float | np.ndarray, theta: float | np.ndarray) -> np.ndarray:
        """Total |B| of the dipole field.

        Parameters
        ----------
        r : float or ndarray
            Spherical radius [cm].
        theta : float or ndarray
            Polar angle [rad].

        Returns
        -------
        ndarray
            |B| [G].
        """
        B_r, B_theta = self.magnetic_field(r, theta)
        return np.sqrt(B_r**2 + B_theta**2)

    def luminosity(self) -> float:
        """Spindown luminosity for this dipole configuration.

        Returns
        -------
        float
            L_sd [erg/s].
        """
        return spindown_luminosity(self.B0, self.R_star, self.omega, self.alpha)
