"""Core cosmology module: Cosmology class and Friedmann equation solver."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate

from . import constants as const


class Cosmology:
    """Cosmological parameters container and Friedmann equation solver.

    Parameters
    ----------
    H0 : float
        Hubble constant today [km/s/Mpc]. Default: 67.4 (Planck18).
    Omega_m : float
        Matter density parameter today. Default: 0.315.
    Omega_r : float
        Radiation density parameter today (photons + neutrinos).
        If 0, computed from T_CMB and standard neutrino species.
    Omega_lambda : float
        Dark energy density parameter today. Default: 0.685.
    Omega_k : float
        Curvature density parameter. If None, set so that sum = 1.
    w0 : float
        Dark energy equation of state parameter w0 (CPL). Default: -1.
    wa : float
        Dark energy equation of state parameter wa (CPL). Default: 0.
    N_eff : float
        Effective number of relativistic neutrino species. Default: 3.046.
    T_cmb : float
        CMB temperature today [K]. Default: 2.7255.
    """

    def __init__(
        self,
        H0: float = 67.4,
        Omega_m: float = 0.315,
        Omega_r: float = 0.0,
        Omega_lambda: float = 0.685,
        Omega_k: float | None = None,
        w0: float = -1.0,
        wa: float = 0.0,
        N_eff: float = 3.046,
        T_cmb: float = const.T_CMB,
    ):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda
        self.w0 = w0
        self.wa = wa
        self.N_eff = N_eff
        self.T_cmb = T_cmb

        # Compute Omega_r from first principles if not provided
        if Omega_r == 0.0:
            self.Omega_r = self._compute_omega_r()
        else:
            self.Omega_r = Omega_r

        # Enforce flatness or use provided curvature
        if Omega_k is None:
            # Flat universe: set Omega_k = 0, adjust Omega_lambda for exact flatness
            self.Omega_k = 0.0
            self.Omega_lambda = 1.0 - self.Omega_m - self.Omega_r
        else:
            self.Omega_k = Omega_k

        # H0 in SI [1/s]
        self.H0_si = H0 * 1e3 / const.MPC_IN_M  # km/s/Mpc -> 1/s

        # Hubble distance [Mpc]
        self.dH = const.C / (H0 * 1e3) * 1e-6 * const.MPC_IN_M  # in Mpc
        self.dH_mpc = const.C / (H0 * 1e3 / const.MPC_IN_M) / const.MPC_IN_M * 1e6
        # Simpler: dH = c/H0 in Mpc
        self.dH_mpc = const.C / self.H0_si / const.MPC_IN_M  # Mpc

    def _compute_omega_r(self) -> float:
        """Compute radiation density parameter from T_CMB and N_eff.

        Omega_gamma = (8 pi^5 / 15) * (k_B T)^4 / (2 pi hbar)^3 * 1/(c^2) / (3 H0^2 / (8 pi G))
        Simplified: Omega_gamma * h^2 = 2.469e-5 * (T_CMB/2.7255)^4  for T in K
        Omega_nu = N_eff * (7/8) * (4/11)^(4/3) * Omega_gamma
        """
        h = self.H0 / 100.0
        Omega_gamma_h2 = 2.469e-5 * (self.T_cmb / 2.7255) ** 4
        Omega_gamma = Omega_gamma_h2 / h ** 2
        Omega_nu = self.N_eff * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * Omega_gamma
        return Omega_gamma + Omega_nu

    def w_de(self, a: ArrayLike) -> NDArray:
        """Dark energy equation of state (CPL parameterization).

        w(a) = w0 + wa * (1 - a)

        Parameters
        ----------
        a : array_like
            Scale factor.

        Returns
        -------
        w : ndarray
        """
        a = np.asarray(a, dtype=float)
        return self.w0 + self.wa * (1.0 - a)

    def rho_de(self, a: ArrayLike) -> NDArray:
        """Dark energy density evolution.

        Omega_DE(a) = Omega_Lambda * a^{-3(1+w0+wa)} * exp(-3*wa*(1-a))

        This comes from integrating the fluid equation with w(a) = w0 + wa*(1-a).

        Parameters
        ----------
        a : array_like
            Scale factor.

        Returns
        -------
        Omega_DE_a : ndarray
        """
        a = np.asarray(a, dtype=float)
        exponent = -3.0 * (1.0 + self.w0 + self.wa)
        return self.Omega_lambda * a ** exponent * np.exp(-3.0 * self.wa * (1.0 - a))

    def H(self, a: ArrayLike) -> NDArray:
        """Hubble parameter H(a) in km/s/Mpc.

        H^2(a) = H0^2 * [Omega_r/a^4 + Omega_m/a^3 + Omega_k/a^2 + Omega_DE(a)]

        Parameters
        ----------
        a : array_like
            Scale factor.

        Returns
        -------
        H : ndarray
            Hubble parameter in km/s/Mpc.
        """
        a = np.asarray(a, dtype=float)
        a = np.maximum(a, 1e-30)  # avoid division by zero

        H2 = (
            self.Omega_r / a ** 4
            + self.Omega_m / a ** 3
            + self.Omega_k / a ** 2
            + self.rho_de(a)
        )
        return self.H0 * np.sqrt(np.maximum(H2, 0.0))

    def H_si(self, a: ArrayLike) -> NDArray:
        """Hubble parameter H(a) in SI units [1/s]."""
        a = np.asarray(a, dtype=float)
        return self.H(a) * 1e3 / const.MPC_IN_M

    def _E(self, a: ArrayLike) -> NDArray:
        """Dimensionless Hubble parameter E(a) = H(a)/H0."""
        a = np.asarray(a, dtype=float)
        a = np.maximum(a, 1e-30)
        E2 = (
            self.Omega_r / a ** 4
            + self.Omega_m / a ** 3
            + self.Omega_k / a ** 2
            + self.rho_de(a)
        )
        return np.sqrt(np.maximum(E2, 0.0))

    def dt_da(self, a: ArrayLike) -> NDArray:
        """dt/da = 1 / (a * H(a)).

        Parameters
        ----------
        a : array_like
            Scale factor.

        Returns
        -------
        dt_da : ndarray
            dt/da in seconds (SI).
        """
        a = np.asarray(a, dtype=float)
        a = np.maximum(a, 1e-30)
        return 1.0 / (a * self.H_si(a))

    def age(self) -> float:
        """Age of the universe t0 in Gyr.

        t0 = integral_0^1 da / (a * H(a))

        Returns
        -------
        age : float
            Age in Gyr.
        """
        # Integrate in log(a) for numerical stability
        def integrand(ln_a):
            a = np.exp(ln_a)
            return 1.0 / (a * self.H_si(a)) * a  # da = a * d(ln a)

        # ln(a) from very small to 0 (a=1)
        result, _ = integrate.quad(integrand, np.log(1e-15), 0.0, limit=200)
        # Convert seconds to Gyr
        sec_per_gyr = 3.1557e16
        return result / sec_per_gyr

    def _integrand_distance(self, z: float) -> float:
        """Integrand for comoving distance: c / H(z) in Mpc."""
        a = 1.0 / (1.0 + z)
        H_z_si = self.H_si(a)
        return const.C / H_z_si / const.MPC_IN_M  # Mpc

    def comoving_distance(self, z: ArrayLike) -> NDArray:
        """Comoving distance chi(z) in Mpc.

        chi = integral_0^z c/H(z') dz'

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        chi : ndarray
            Comoving distance in Mpc.
        """
        z = np.asarray(z, dtype=float)
        scalar = z.ndim == 0
        z = np.atleast_1d(z)

        result = np.empty_like(z)
        for i, zi in enumerate(z):
            val, _ = integrate.quad(self._integrand_distance, 0.0, zi, limit=200)
            result[i] = val

        if scalar:
            return float(result[0])
        return result

    def _f_k(self, chi: ArrayLike) -> NDArray:
        """Transverse comoving distance f_k(chi) in Mpc.

        Handles curvature:
        - Omega_k = 0: f_k = chi
        - Omega_k > 0 (open): f_k = dH/sqrt(Omega_k) * sinh(sqrt(Omega_k)*chi/dH)
        - Omega_k < 0 (closed): f_k = dH/sqrt(|Omega_k|) * sin(sqrt(|Omega_k|)*chi/dH)
        """
        chi = np.asarray(chi, dtype=float)
        if abs(self.Omega_k) < 1e-10:
            return chi
        elif self.Omega_k > 0:
            sqrt_ok = np.sqrt(self.Omega_k)
            return self.dH_mpc / sqrt_ok * np.sinh(sqrt_ok * chi / self.dH_mpc)
        else:
            sqrt_ok = np.sqrt(abs(self.Omega_k))
            return self.dH_mpc / sqrt_ok * np.sin(sqrt_ok * chi / self.dH_mpc)

    def luminosity_distance(self, z: ArrayLike) -> NDArray:
        """Luminosity distance d_L(z) in Mpc.

        d_L = (1+z) * f_k(chi(z))

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        d_L : ndarray
            Luminosity distance in Mpc.
        """
        z = np.asarray(z, dtype=float)
        chi = self.comoving_distance(z)
        return (1.0 + z) * self._f_k(chi)

    def angular_diameter_distance(self, z: ArrayLike) -> NDArray:
        """Angular diameter distance d_A(z) in Mpc.

        d_A = d_L / (1+z)^2

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        d_A : ndarray
            Angular diameter distance in Mpc.
        """
        z = np.asarray(z, dtype=float)
        return self.luminosity_distance(z) / (1.0 + z) ** 2

    def distance_modulus(self, z: ArrayLike) -> NDArray:
        """Distance modulus mu(z).

        mu = 5 * log10(d_L / 10 pc)
           = 5 * log10(d_L_Mpc) + 25

        since 10 pc = 10 * 3.0857e-18 Mpc = 3.0857e-17 Mpc,
        and 5*log10(1/3.0857e-17) = 5*log10(3.2408e16) = 5*16.5109... ~ 25.

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        mu : ndarray
            Distance modulus [mag].
        """
        z = np.asarray(z, dtype=float)
        dL_mpc = self.luminosity_distance(z)
        return 5.0 * np.log10(dL_mpc) + 25.0

    def deceleration_parameter(self, z: ArrayLike) -> NDArray:
        """Deceleration parameter q(z).

        q(z) = -a*H''/H^2 = (1/2) * sum_i Omega_i(a) * (1 + 3*w_i)

        where the sum is over all energy components:
        - Radiation: w_r = 1/3
        - Matter: w_m = 0
        - Curvature: w_k = -1/3
        - Dark energy: w_de(a)

        Parameters
        ----------
        z : array_like
            Redshift.

        Returns
        -------
        q : ndarray
            Deceleration parameter.
        """
        z = np.asarray(z, dtype=float)
        a = 1.0 / (1.0 + z)

        E2 = (
            self.Omega_r / a ** 4
            + self.Omega_m / a ** 3
            + self.Omega_k / a ** 2
            + self.rho_de(a)
        )

        # Each component's contribution to q:
        # q = (1/2) * sum_i Omega_i(a) * (1 + 3*w_i)
        # Omega_i(a) = Omega_i * a^{-3(1+w_i)} / E^2(a)
        q_r = 0.5 * (self.Omega_r / a ** 4) * (1.0 + 3.0 * (1.0 / 3.0))
        q_m = 0.5 * (self.Omega_m / a ** 3) * (1.0 + 3.0 * 0.0)
        q_k = 0.5 * (self.Omega_k / a ** 2) * (1.0 + 3.0 * (-1.0 / 3.0))
        q_de = 0.5 * self.rho_de(a) * (1.0 + 3.0 * self.w_de(a))

        return (q_r + q_m + q_k + q_de) / E2

    def sound_horizon(self, z_drag: float = 1060.0) -> float:
        """Sound horizon r_s at drag epoch in Mpc.

        r_s = integral_{z_drag}^{infty} c_s(z) / H(z) dz

        where c_s = c / sqrt(3 * (1 + R)) is the sound speed in the photon-baryon fluid,
        and R = (3 * rho_b) / (4 * rho_gamma).

        Parameters
        ----------
        z_drag : float
            Redshift of baryon drag epoch. Default: 1060.

        Returns
        -------
        r_s : float
            Sound horizon in Mpc.
        """
        # Baryon density from Omega_b (assume Omega_b ~ 0.049 from Planck18)
        # We use Omega_b/h^2 ~ 0.0224
        h = self.H0 / 100.0
        Omega_b = 0.0224 / h ** 2
        Omega_gamma = 2.469e-5 * (self.T_cmb / 2.7255) ** 4 / h ** 2

        def integrand(z):
            a = 1.0 / (1.0 + z)
            # R = 3 * rho_b / (4 * rho_gamma) at scale factor a
            R = 3.0 * Omega_b / (4.0 * Omega_gamma) * a
            c_s = const.C / np.sqrt(3.0 * (1.0 + R))  # m/s
            H_z = self.H_si(a)  # 1/s
            return c_s / H_z / const.MPC_IN_M  # Mpc

        result, _ = integrate.quad(integrand, z_drag, 1e8, limit=200)
        return result
