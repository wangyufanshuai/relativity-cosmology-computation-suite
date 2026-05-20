"""
Gravitational radiation reaction for EMRI systems.

Implements energy and angular momentum fluxes due to gravitational wave emission,
using the Peters-Mathews formula as a baseline with Kerr corrections for
spin-orbit coupling effects.

References:
    - Peters & Mathews (1963) "Gravitational Radiation from Point Masses..."
    - Cutler, Kennefick & Poisson (1994) "Gravitational radiation reaction..."
    - Gair, Glampedakis & Hughes (2012) "Improved approximate fluxes for Kerr..."
"""

import numpy as np


class RadiationReaction:
    """
    Gravitational radiation reaction (fluxes) for an EMRI system.

    Computes energy and angular momentum fluxes for a small compact object
    of mass mu orbiting a Kerr black hole of mass M.

    Parameters
    ----------
    M : float
        Primary (massive) black hole mass.
    a : float
        Spin parameter of the primary (|a| <= M).
    mu : float
        Secondary (compact object) mass. Must satisfy mu << M.
    """

    def __init__(self, M=1.0, a=0.0, mu=1e-4):
        if abs(a) > M:
            raise ValueError(f"Spin |a|={abs(a)} must not exceed M={M}")
        self.M = M
        self.a = a
        self.mu = mu

    def peters_mathews_energy_flux(self, p, e):
        """
        Peters-Mathews averaged energy flux (Schwarzschild baseline).

        <dE/dt> = -(32/5) * mu^2 * M^3 / r^5 * f(e)

        For eccentric orbits, the enhancement factor:
        f(e) = (1 + 73/24 e^2 + 37/96 e^4) / (1-e^2)^{7/2}

        Parameters
        ----------
        p : float
            Semi-latus rectum.
        e : float
            Eccentricity.

        Returns
        -------
        float
            Time-averaged energy flux dE/dt (negative = energy loss).
        """
        M = self.M
        mu = self.mu

        # Mean orbital radius
        r = p / (1.0 - e**2)

        # Enhancement factor for eccentricity
        f_e = (1.0 + 73.0 / 24.0 * e**2 + 37.0 / 96.0 * e**4) / (1.0 - e**2) ** 3.5

        dEdt = -(32.0 / 5.0) * mu**2 * M**3 / r**5 * f_e

        return dEdt

    def peters_mathews_angular_momentum_flux(self, p, e):
        """
        Peters-Mathews averaged angular momentum flux.

        <dLz/dt> = -(32/5) * mu^2 * M^{5/2} / r^{7/2} * g(e)

        g(e) = (1 + 7/8 e^2) / (1-e^2)^2

        Parameters
        ----------
        p : float
            Semi-latus rectum.
        e : float
            Eccentricity.

        Returns
        -------
        float
            Time-averaged angular momentum flux dLz/dt (negative = loss).
        """
        M = self.M
        mu = self.mu

        r = p / (1.0 - e**2)

        g_e = (1.0 + 7.0 / 8.0 * e**2) / (1.0 - e**2) ** 2

        dLzdt = -(32.0 / 5.0) * mu**2 * M**2.5 / r**3.5 * g_e

        return dLzdt

    def kerr_correction_factor(self, p, e, iota):
        """
        Kerr correction factor for energy flux due to spin-orbit coupling.

        For prograde orbits (aligned with spin), fluxes are enhanced at small r
        due to the ISCO being closer. For retrograde orbits, fluxes are suppressed.

        Parameters
        ----------
        p : float
            Semi-latus rectum.
        e : float
            Eccentricity.
        iota : float
            Inclination angle.

        Returns
        -------
        float
            Multiplicative correction factor.
        """
        a = self.a
        M = self.M

        if abs(a) < 1e-15:
            return 1.0

        r = p / (1.0 - e**2)

        # Frame-dragging correction: factor depends on spin-orbit alignment
        # prograde (cos(iota) > 0) => enhanced flux
        # retrograde (cos(iota) < 0) => suppressed flux
        chi = a / M  # dimensionless spin

        # Leading-order spin correction to the energy flux
        # Based on the shift in the ISCO and the potential
        # Approximate: factor ~ 1 + alpha * chi * (M/r)^{3/2} * cos(iota)
        alpha = 4.0  # approximate coefficient
        correction = 1.0 + alpha * chi * (M / r) ** 1.5 * np.cos(iota)

        return correction

    def energy_flux(self, p, e, iota=0.0):
        """
        Total energy flux including Kerr corrections.

        Parameters
        ----------
        p : float
            Semi-latus rectum.
        e : float
            Eccentricity.
        iota : float
            Inclination.

        Returns
        -------
        float
            dE/dt (negative for energy loss).
        """
        dEdt_schwarz = self.peters_mathews_energy_flux(p, e)
        correction = self.kerr_correction_factor(p, e, iota)
        return dEdt_schwarz * correction

    def angular_momentum_flux(self, p, e, iota=0.0):
        """
        Total angular momentum flux including Kerr corrections.

        Parameters
        -------
        p : float
            Semi-latus rectum.
        e : float
            Eccentricity.
        iota : float
            Inclination.

        Returns
        -------
        float
            dLz/dt (negative for angular momentum loss).
        """
        dLzdt_schwarz = self.peters_mathews_angular_momentum_flux(p, e)
        correction = self.kerr_correction_factor(p, e, iota)
        return dLzdt_schwarz * correction

    def element_evolution_rates(self, p, e, iota=0.0):
        """
        Compute adiabatic evolution rates dp/dt, de/dt, diota/dt.

        From flux balance: the GW energy and angular momentum loss
        drives the evolution of orbital elements.

        Using the Jacobian d(E,Lz)/d(p,e) inverted.

        Parameters
        ----------
        p : float
            Semi-latus rectum.
        e : float
            Eccentricity.
        iota : float
            Inclination.

        Returns
        -------
        dict
            {'dp_dt': ..., 'de_dt': ..., 'diota_dt': ...}
        """
        M = self.M
        mu = self.mu

        dEdt = self.energy_flux(p, e, iota)
        dLzdt = self.angular_momentum_flux(p, e, iota)

        # For Schwarzschild (leading order):
        # E ~ 1 - (1-e^2)/(2p)  [weak field]
        # Lz ~ sqrt(M*p)  [circular, equatorial]
        # => dE/dp = (1-e^2)/(2p^2)
        # => dE/de = e/p
        # => dLz/dp = sqrt(M)/(2*sqrt(p))
        # => dLz/de = 0 (at leading order for circular)

        # dE/dp, dE/de
        dE_dp = (1.0 - e**2) / (2.0 * p**2)
        dE_de = e / p

        # dLz/dp, dLz/de
        if abs(iota) < 1e-15:
            dLz_dp = np.sqrt(M) / (2.0 * np.sqrt(p))
            dLz_de = 0.0
        else:
            dLz_dp = np.sqrt(M) * np.cos(iota) / (2.0 * np.sqrt(p))
            dLz_de = 0.0

        # Invert the Jacobian:
        # [dEdt]   [dE/dp  dE/de ] [dp/dt]
        # [dLzdt] = [dLz/dp dLz/de] [de/dt]
        det = dE_dp * dLz_de - dE_de * dLz_dp
        if abs(det) < 1e-30:
            # Fallback: use simple circular orbit formulas
            dp_dt = dEdt / max(abs(dE_dp), 1e-30) * np.sign(dE_dp)
            de_dt = 0.0
        else:
            dp_dt = (dLz_de * dEdt - dE_de * dLzdt) / det
            de_dt = (-dLz_dp * dEdt + dE_dp * dLzdt) / det

        # Inclination evolution (from Carter constant flux)
        # For equatorial orbits, dQ/dt ~ 0 at leading order
        if abs(iota) < 1e-15:
            diota_dt = 0.0
        else:
            # Approximate: inclination evolves slowly
            Lz = np.sqrt(M * p) * np.cos(iota)
            Q = M * p * np.sin(iota) ** 2
            # dQ/dt from the fluxes (simplified)
            dQ_dt = -self.mu * Q / (p**2.5) * 0.1  # rough scaling
            diota_dt = dQ_dt / (2.0 * Lz * np.sin(iota) + 1e-30) if abs(np.sin(iota)) > 1e-10 else 0.0

        return {
            "dp_dt": dp_dt,
            "de_dt": de_dt,
            "diota_dt": diota_dt,
        }

    def inspiral_time(self, p, e=0.0, iota=0.0):
        """
        Estimate the inspiral time from a given orbital configuration.

        Uses the Peters formula (quadrupole approximation):
        t_insp ~ (5/256) * (c^5 / G^3) * (M^2 * (M+mu)) / mu * (p/c^2)^4
        In geometric units (G=c=1):
        t_insp ~ (5/256) * M^3 / mu * p^4 / M^4 * f(e)

        More precisely, for quasi-circular orbits:
        t_insp ~ (5/256) * p^4 / (mu * M^2)

        Parameters
        ----------
        p : float
            Initial semi-latus rectum.
        e : float
            Initial eccentricity.
        iota : float
            Initial inclination.

        Returns
        -------
        float
            Estimated inspiral time (in geometric units, M=1).
        """
        M = self.M
        mu = self.mu

        # Eccentricity correction factor
        f_e = (1.0 - e**2) ** 3.5 / (1.0 + 73.0 / 24.0 * e**2 + 37.0 / 96.0 * e**4)

        # Peters (1964) inspiral time
        t_insp = (5.0 / 256.0) * p**4 / (mu * M**2) * f_e

        # Kerr correction
        correction = 1.0 / self.kerr_correction_factor(p, e, iota)
        t_insp *= correction

        return t_insp

    def evolve_orbit(self, p0, e0, iota0, t_span, n_steps=1000):
        """
        Evolve orbital elements under radiation reaction (adiabatic evolution).

        Uses simple forward Euler integration of dp/dt, de/dt, diota/dt.

        Parameters
        ----------
        p0 : float
            Initial semi-latus rectum.
        e0 : float
            Initial eccentricity.
        iota0 : float
            Initial inclination.
        t_span : tuple
            (t_start, t_end) coordinate time span.
        n_steps : int
            Number of integration steps.

        Returns
        -------
        dict
            Arrays of 't', 'p', 'e', 'iota' during inspiral.
        """
        dt = (t_span[1] - t_span[0]) / n_steps
        t_arr = np.linspace(t_span[0], t_span[1], n_steps + 1)

        p_arr = np.zeros(n_steps + 1)
        e_arr = np.zeros(n_steps + 1)
        iota_arr = np.zeros(n_steps + 1)

        p_arr[0] = p0
        e_arr[0] = e0
        iota_arr[0] = iota0

        from .orbital_elements import OrbitalElements

        oe = OrbitalElements(self.M, self.a)
        r_isco = oe.isco_radius()

        for i in range(n_steps):
            p = p_arr[i]
            e = e_arr[i]
            iota = iota_arr[i]

            # Check if we've reached the ISCO
            r_peri = p / (1.0 + e) if e > 0 else p
            if r_peri <= r_isco * 1.01:
                # Truncate the inspiral at ISCO
                p_arr[i + 1:] = p_arr[i]
                e_arr[i + 1:] = e_arr[i]
                iota_arr[i + 1:] = iota_arr[i]
                t_arr = t_arr[: i + 2]
                p_arr = p_arr[: i + 2]
                e_arr = e_arr[: i + 2]
                iota_arr = iota_arr[: i + 2]
                break

            rates = self.element_evolution_rates(p, e, iota)
            p_arr[i + 1] = p + rates["dp_dt"] * dt
            e_arr[i + 1] = e + rates["de_dt"] * dt
            iota_arr[i + 1] = iota + rates["diota_dt"] * dt

            # Ensure physical bounds
            p_arr[i + 1] = max(p_arr[i + 1], r_isco)
            e_arr[i + 1] = max(e_arr[i + 1], 0.0)
            e_arr[i + 1] = min(e_arr[i + 1], 0.99)
            iota_arr[i + 1] = np.clip(iota_arr[i + 1], 0.0, np.pi / 2.0)

        return {"t": t_arr, "p": p_arr, "e": e_arr, "iota": iota_arr}
