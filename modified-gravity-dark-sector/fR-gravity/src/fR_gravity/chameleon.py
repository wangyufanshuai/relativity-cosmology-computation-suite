"""Chameleon screening mechanism in f(R) gravity.

Implements the thin-shell screening condition, the scalaron (chameleon)
potential, and the fifth-force suppression for massive objects.

The chameleon effective potential is::

    V_eff(φ) = V(φ) + ρ exp(βφ/M_Pl)

where φ = √(3/2) M_Pl ln(1 + f_R) and β = 1/√6 for f(R) gravity.

Key formulae
------------
Thin-shell parameter::

    ΔR / R = 3 β M_Pl φ_∞ / (2 ρ R²)

Screening condition::

    Φ_obj > 3 β² φ_∞ / 2

Fifth force (Yukawa-type)::

    F_5 = 2 β² (M_Pl / ρ) ∇ρ × exp(−m_φ r) / r

References
----------
Khoury & Weltman, Phys. Rev. Lett. 93, 171104 (2004).
Hu & Sawicki, Phys. Rev. D 76, 064004 (2007).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# β = 1/√6 for metric f(R) gravity
BETA_F_R: float = 1.0 / np.sqrt(6.0)


class ChameleonScreening:
    """Chameleon screening calculator for f(R) gravity.

    Parameters
    ----------
    M_Pl : float
        Reduced Planck mass M_Pl = (8πG)^{−1/2} in the chosen unit system.
    beta : float
        Coupling of the scalaron to matter.  Defaults to 1/√6 for
        metric f(R) gravity.
    """

    def __init__(self, M_Pl: float, beta: float = BETA_F_R) -> None:
        self.M_Pl = float(M_Pl)
        self.beta = float(beta)

    # ------------------------------------------------------------------
    # Scalaron / chameleon field
    # ------------------------------------------------------------------

    def scalaron_from_fR(self, f_R: float | NDArray) -> float | NDArray:
        """Compute the scalaron field φ from f_R.

        φ = √(3/2) M_Pl ln(1 + f_R)

        Parameters
        ----------
        f_R : float or array_like
            Derivative of f(R) (dimensionless).

        Returns
        -------
        float or ndarray
        """
        f_R = np.asarray(f_R, dtype=float)
        return np.sqrt(1.5) * self.M_Pl * np.log(1.0 + f_R)

    # ------------------------------------------------------------------
    # Effective potential
    # ------------------------------------------------------------------

    def V_eff(
        self,
        phi: float | NDArray,
        V: float | NDArray,
        rho: float | NDArray,
    ) -> float | NDArray:
        """Chameleon effective potential V_eff = V(φ) + ρ exp(βφ/M_Pl).

        Parameters
        ----------
        phi : float or array_like
            Scalaron field value.
        V : float or array_like
            Self-interaction potential V(φ) of the scalaron.
        rho : float or array_like
            Ambient matter density.

        Returns
        -------
        float or ndarray
        """
        phi = np.asarray(phi, dtype=float)
        V = np.asarray(V, dtype=float)
        rho = np.asarray(rho, dtype=float)
        return V + rho * np.exp(self.beta * phi / self.M_Pl)

    # ------------------------------------------------------------------
    # Thin-shell screening
    # ------------------------------------------------------------------

    def thin_shell_parameter(
        self,
        rho: float,
        R_obj: float,
        phi_inf: float,
    ) -> float:
        """Thin-shell parameter ΔR / R.

        ΔR / R = 3 β M_Pl φ_∞ / (2 ρ R_obj²)

        When ΔR/R << 1 the object is in the thin-shell regime and the
        fifth force is heavily suppressed.

        Parameters
        ----------
        rho : float
            Density of the object.
        R_obj : float
            Radius of the object.
        phi_inf : float
            Asymptotic field value far from the object.

        Returns
        -------
        float
        """
        return 3.0 * self.beta * self.M_Pl * phi_inf / (2.0 * rho * R_obj ** 2)

    def screening_condition(
        self,
        Phi_obj: float,
        phi_inf: float,
    ) -> bool:
        """Check whether an object is screened.

        Screening occurs when::

            Φ_obj > 3 β² φ_∞ / 2

        Parameters
        ----------
        Phi_obj : float
            Newtonian potential at the surface of the object,
            Φ_obj = G M / R_obj.
        phi_inf : float
            Asymptotic field value.

        Returns
        -------
        bool
            True if the object is screened.
        """
        return Phi_obj > 1.5 * self.beta ** 2 * phi_inf

    # ------------------------------------------------------------------
    # Fifth force
    # ------------------------------------------------------------------

    def fifth_force_magnitude(
        self,
        m_phi: float,
        rho: float,
        grad_rho: float,
        r: float,
    ) -> float:
        """Magnitude of the chameleon fifth force (Yukawa).

        F_5 = 2 β² (M_Pl / ρ) |∇ρ| × exp(−m_φ r) / r

        Parameters
        ----------
        m_phi : float
            Scalaron (chameleon) mass at the ambient density.
        rho : float
            Ambient density.
        grad_rho : float
            Magnitude of the density gradient |∇ρ|.
        r : float
            Distance from the source.

        Returns
        -------
        float
        """
        return (
            2.0 * self.beta ** 2 * (self.M_Pl / rho) * grad_rho
            * np.exp(-m_phi * r) / r
        )

    def fifth_force_yukawa_suppression(self, m_phi: float, r: float) -> float:
        """Yukawa suppression factor exp(−m_φ r).

        Parameters
        ----------
        m_phi : float
            Scalaron mass.
        r : float
            Distance.

        Returns
        -------
        float
        """
        return float(np.exp(-m_phi * r))

    def is_screened(
        self,
        Phi_obj: float,
        phi_inf: float,
    ) -> bool:
        """Convenience alias for :meth:`screening_condition`."""
        return self.screening_condition(Phi_obj, phi_inf)

    def effective_coupling(self, Phi_obj: float, phi_inf: float) -> float:
        """Effective coupling β_eff accounting for screening.

        For a thin-shell object::

            β_eff ≈ β × (3 ΔR/R) / 2

        which is << β when the object is screened.

        Parameters
        ----------
        Phi_obj : float
            Newtonian potential Φ = GM/R at the surface.
        phi_inf : float
            Asymptotic field value.

        Returns
        -------
        float
        """
        if not self.is_screened(Phi_obj, phi_inf):
            return self.beta
        # screened: effective coupling suppressed by thin-shell fraction
        ratio = 1.5 * self.beta ** 2 * phi_inf / Phi_obj
        return self.beta * ratio
