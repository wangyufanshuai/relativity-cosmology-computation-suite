"""Aether vector field u^mu with constraint u^mu u_mu = -1.

Implements the aether action and stress-energy tensor computation for
the Einstein-aether theory.  The aether is a unit timelike vector field
defined at every spacetime point.

Action:
    S_ae = -c_1/2 * (nabla_mu u_nu)(nabla^mu u^nu)
           - c_2   * (nabla_mu u^mu)^2
           - c_3/2 * (nabla_mu u_nu)(nabla^nu u^mu)
           + c_4   * u^mu u^nu (nabla_rho u_mu)(nabla^rho u_nu)
           + lambda * (u^mu u_mu + 1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class AetherField:
    """Aether vector field on a flat (Minkowski) background.

    Parameters
    ----------
    u_contra : array_like, shape (4,)
        Contravariant components u^mu of the aether field.
        The field will be projected to satisfy u^mu u_mu = -1
        using the mostly-plus metric convention (-, +, +, +).
    """

    def __init__(self, u_contra: NDArray | list | tuple) -> None:
        self._u = np.asarray(u_contra, dtype=float).reshape(4)
        if self._u.shape != (4,):
            raise ValueError("u_contra must have exactly 4 components")
        self._u = self._project_unit_timelike(self._u)

    # ------------------------------------------------------------------
    # Metric tensor (Minkowski, mostly-plus)
    # ------------------------------------------------------------------
    _ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

    @classmethod
    def eta(cls) -> NDArray:
        """Return the Minkowski metric eta_{mu nu} with signature (-,+,+,+)."""
        return cls._ETA.copy()

    @classmethod
    def eta_inv(cls) -> NDArray:
        """Return the inverse metric eta^{mu nu}."""
        return cls._ETA.copy()

    # ------------------------------------------------------------------
    # Projection to unit timelike
    # ------------------------------------------------------------------
    @classmethod
    def _project_unit_timelike(cls, u: NDArray) -> NDArray:
        """Normalise *u* so that u^mu u_mu = -1.

        The contraction uses the mostly-plus convention:
            u_mu = eta_{mu nu} u^nu
            u^mu u_mu = u^0 u_0 + u^i u_i = -(u^0)^2 + (u^1)^2 + (u^2)^2 + (u^3)^2

        We normalise by dividing by sqrt(|u^mu u_mu|) and ensuring the
        result is timelike (norm^2 < 0) with future-directed time
        component (u^0 > 0).
        """
        u_cov: NDArray = cls._ETA @ u          # u_mu
        norm_sq: float = float(np.dot(u, u_cov))  # u^mu u_mu
        if norm_sq >= 0:
            # Spacelike or null -- flip sign of spatial part is not
            # sufficient; enforce a minimum time component.
            u[0] = max(abs(u[0]), 1.0)
            u_cov = cls._ETA @ u
            norm_sq = float(np.dot(u, u_cov))
        norm: float = np.sqrt(-norm_sq)
        u = u / norm
        # Ensure future-directed
        if u[0] < 0:
            u = -u
        return u

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def u_contra(self) -> NDArray:
        """Contravariant components u^mu."""
        return self._u.copy()

    @property
    def u_cov(self) -> NDArray:
        """Covariant components u_mu = eta_{mu nu} u^nu."""
        return self._ETA @ self._u

    def norm_sq(self) -> float:
        """Return u^mu u_mu.  Should be -1 after projection."""
        return float(np.dot(self._u, self.u_cov))

    # ------------------------------------------------------------------
    # Action density
    # ------------------------------------------------------------------
    @staticmethod
    def action_density(
        du: NDArray,
        c1: float,
        c2: float,
        c3: float,
        c4: float,
    ) -> float:
        """Compute the aether Lagrangian density (per unit volume).

        Parameters
        ----------
        du : array_like, shape (4, 4)
            Partial derivatives du^mu/dx^nu representing nabla_nu u^mu
            in flat spacetime (covariant derivative = partial derivative).
        c1, c2, c3, c4 : float
            Coupling constants of the aether action.

        Returns
        -------
        float
            The aether Lagrangian density L_ae.
        """
        du = np.asarray(du, dtype=float)
        eta = AetherField._ETA

        # Lower first index: nabla_mu u_nu = eta_{nu rho} nabla_mu u^rho
        # du[mu, rho] = nabla_mu u^rho  =>  du_cov[mu, nu] = eta[nu, rho] * du[mu, rho]
        du_cov = du @ eta  # shape (4, 4):  (nabla_mu u^rho) eta_{rho nu}

        # Term 1: -c1/2 * (nabla_mu u_nu)(nabla^mu u^nu)
        #   = -c1/2 * eta^{mu alpha} eta^{nu beta} (nabla_alpha u_beta)(nabla_mu u_nu)
        #   = -c1/2 * (nabla_mu u^nu)(nabla^mu u_nu)
        # Using: sum_{mu,nu} du_cov[mu, nu] * du[mu, nu] * eta[mu, mu] (diagonal)
        # Simpler:  (nabla_mu u_nu)(nabla^mu u^nu) = du_cov[mu,nu] * eta^{mu,mu} * du[mu,nu]
        # Actually: raise both indices of du_cov:
        du_contra_contra = eta @ du_cov @ eta  # nabla^mu u^nu
        term1 = -c1 / 2.0 * np.sum(du_cov * du_contra_contra)

        # Term 2: -c2 * (nabla_mu u^mu)^2
        trace = np.trace(du)  # nabla_mu u^mu
        term2 = -c2 * trace ** 2

        # Term 3: -c3/2 * (nabla_mu u_nu)(nabla^nu u^mu)
        # (nabla_mu u_nu)(nabla^nu u^mu) = sum_{mu,nu} du_cov[mu,nu] * du_contra_contra[nu,mu]
        term3 = -c3 / 2.0 * np.sum(du_cov * du_contra_contra.T)

        # Term 4: c4 * u^mu u^nu (nabla_rho u_mu)(nabla^rho u_nu)
        # This term requires the field itself.  We approximate it as zero
        # when evaluated at a general derivative level (full evaluation
        # needs the field).  For a self-contained static method we accept
        # the derivative matrix only.
        term4 = 0.0

        return term1 + term2 + term3 + term4

    def action_density_full(
        self,
        du: NDArray,
        c1: float,
        c2: float,
        c3: float,
        c4: float,
    ) -> float:
        """Compute the full aether Lagrangian density including the c4 term.

        Parameters
        ----------
        du : array_like, shape (4, 4)
            Partial derivatives du^mu/dx^nu = nabla_nu u^mu.
        c1, c2, c3, c4 : float
            Coupling constants.

        Returns
        -------
        float
            Full aether Lagrangian density L_ae including the c4 term.
        """
        du = np.asarray(du, dtype=float)
        eta = self._ETA
        u = self._u
        u_cov = self.u_cov

        du_cov = du @ eta
        du_contra_contra = eta @ du_cov @ eta

        # Term 1
        term1 = -c1 / 2.0 * np.sum(du_cov * du_contra_contra)

        # Term 2
        trace = np.trace(du)
        term2 = -c2 * trace ** 2

        # Term 3
        term3 = -c3 / 2.0 * np.sum(du_cov * du_contra_contra.T)

        # Term 4: c4 * u^mu u^nu (nabla_rho u_mu)(nabla^rho u_nu)
        #   (nabla_rho u_mu)(nabla^rho u_nu) = sum_rho du_cov[rho, mu] * du_contra_contra[rho, nu]
        # Then contract with u^mu u^nu
        inner = np.einsum("rm,rn->mn", du_cov, du_contra_contra)  # (4,4)
        term4 = c4 * np.einsum("m,n,mn->", u, u, inner)

        return term1 + term2 + term3 + term4

    # ------------------------------------------------------------------
    # Stress-energy tensor
    # ------------------------------------------------------------------
    def stress_energy_tensor(
        self,
        du: NDArray,
        c1: float,
        c2: float,
        c3: float,
        c4: float,
        lam: float = 0.0,
    ) -> NDArray:
        """Compute the aether stress-energy tensor T^{mu nu}_{ae}.

        This is the variation of S_ae with respect to the metric.
        For flat spacetime and constant derivatives the dominant
        contribution is evaluated from the kinetic terms.

        Parameters
        ----------
        du : array_like, shape (4, 4)
            Partial derivatives nabla_nu u^mu.
        c1, c2, c3, c4 : float
            Coupling constants.
        lam : float
            Lagrange multiplier for the unit constraint.

        Returns
        -------
        T : ndarray, shape (4, 4)
            Stress-energy tensor T^{mu nu}.
        """
        du = np.asarray(du, dtype=float)
        eta = self._ETA
        u = self._u
        u_cov = self.u_cov

        du_cov = du @ eta  # nabla_mu u_nu
        du_up = eta @ du_cov @ eta  # nabla^mu u^nu

        T = np.zeros((4, 4))

        # Contribution from the metric variation of the kinetic terms.
        # The general expression is lengthy; we keep the dominant
        # contributions valid in the weak-field / Minkowski limit.

        # c1 term contribution: proportional to (nabla^mu u^alpha)(nabla^nu u_alpha)
        # T1^{mu nu} = c1 * (nabla^mu u^alpha)(nabla^nu u_alpha)
        #            - (c1/2) * eta^{mu nu} * (nabla_rho u_alpha)(nabla^rho u^alpha)
        kinetic_scalar = np.sum(du_cov * du_up)
        T1 = c1 * (du_up @ du_cov.T) - (c1 / 2.0) * eta * kinetic_scalar

        # c2 term contribution
        trace = np.trace(du)
        T2 = 2.0 * c2 * trace * eta * trace * 0.5 - c2 * (trace ** 2) * eta * 0.5
        # More precisely: T2^{mu nu} ~ 2 c2 (nabla.u) (nabla^mu u^nu + ...) - ...
        # Simplified Minkowski contribution:
        T2 = -c2 * (trace ** 2) * eta / 2.0 + 2.0 * c2 * trace * (
            np.outer(eta[0], du[0]) + np.outer(du[:, 0], eta[0]) - np.trace(du) * eta
        ) * 0.0  # In flat-space limit c2 contributes through trace structure

        # c3 term contribution
        T3 = (c3 / 2.0) * (du_up @ du_up.T) - (c3 / 2.0) * eta * np.sum(du_cov * du_up.T)

        # c4 term: u^rho u^sigma (nabla_mu u_rho)(nabla_nu u_sigma) structure
        inner_c4 = np.einsum("ra,rb->ab", du_cov, du_cov)
        T4_c4_part = np.einsum("a,b,ab->", u_cov, u_cov, inner_c4)
        T4 = c4 * np.einsum("r,a,s,b,ra,sb->rs", u, u_cov, u, u_cov, du_cov, du_cov)
        T4 -= (c4 / 2.0) * eta * T4_c4_part

        # Lagrange multiplier term
        T_lam = lam * np.outer(u, u)

        T = T1 + T2 + T3 + T4 + T_lam
        return T

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_spherical(
        cls,
        theta: float = 0.0,
        phi: float = 0.0,
    ) -> "AetherField":
        """Create a unit-timelike aether field from angular direction.

        The aether points in the time direction boosted by angles
        theta (polar from t-axis) and phi (azimuthal).

        Parameters
        ----------
        theta : float
            Polar angle from the time axis (radians).  0 gives pure
            time-like aether.
        phi : float
            Azimuthal angle in the spatial hyperplane.

        Returns
        -------
        AetherField
        """
        u = np.array([
            1.0 / np.cos(theta) if np.cos(theta) != 0 else 1.0,
            np.tan(theta) * np.cos(phi),
            np.tan(theta) * np.sin(phi),
            0.0,
        ])
        return cls(u)

    @classmethod
    def pure_time(cls) -> "AetherField":
        """Create a pure time-directed aether: u^mu = (1, 0, 0, 0)."""
        return cls(np.array([1.0, 0.0, 0.0, 0.0]))
