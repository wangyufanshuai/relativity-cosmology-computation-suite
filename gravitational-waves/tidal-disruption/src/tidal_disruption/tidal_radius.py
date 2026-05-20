"""Tidal radius and disruption physics for TDEs.

Implements the core gravitational physics of stellar tidal disruption
by a supermassive black hole, including the tidal radius, penetration
factor, and energy spread calculations.
"""

import numpy as np
from dataclasses import dataclass

# Physical constants (CGS)
G = 6.67430e-8        # gravitational constant [cm^3 g^-1 s^-2]
c = 2.99792458e10     # speed of light [cm/s]
M_sun = 1.989e33      # solar mass [g]
R_sun = 6.957e10      # solar radius [cm]
sigma_T = 6.6524e-25  # Thomson cross-section [cm^2]
m_p = 1.6726e-24      # proton mass [g]
sigma_SB = 5.6704e-5  # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
k_B = 1.3807e-16      # Boltzmann constant [erg K^-1]


@dataclass
class StellarParams:
    """Parameters describing the disrupted star."""
    mass: float        # stellar mass [g]
    radius: float      # stellar radius [cm]
    gamma: float = 5.0 / 3.0  # adiabatic index (polytrope)


@dataclass
class BlackHoleParams:
    """Parameters describing the supermassive black hole."""
    mass: float        # black hole mass [g]
    spin: float = 0.0  # dimensionless spin parameter a in [0, 1]


class TidalDisruption:
    """Calculate tidal disruption physics for a star approaching a SMBH.

    Parameters
    ----------
    star : StellarParams
        Stellar parameters (mass, radius, structure).
    bh : BlackHoleParams
        Black hole parameters (mass, spin).
    """

    def __init__(self, star: StellarParams, bh: BlackHoleParams):
        self.star = star
        self.bh = bh

    # ------------------------------------------------------------------
    # Basic radii
    # ------------------------------------------------------------------

    def tidal_radius(self) -> float:
        """Classical tidal radius r_t = R_* (M_BH / M_*)^{1/3}.

        Returns
        -------
        float
            Tidal radius in cm.
        """
        return self.star.radius * (self.bh.mass / self.star.mass) ** (1.0 / 3.0)

    def schwarzschild_radius(self) -> float:
        """Schwarzschild radius r_S = 2 G M_BH / c^2.

        Returns
        -------
        float
            Schwarzschild radius in cm.
        """
        return 2.0 * G * self.bh.mass / c ** 2

    def isco_radius(self) -> float:
        """Inner-most stable circular orbit radius.

        For a non-spinning BH this is 6 GM/c^2 = 3 r_S.

        Returns
        -------
        float
            ISCO radius in cm.
        """
        # Exact expression for Schwarzschild (a=0): r_isco = 6 GM/c^2
        # For Kerr the prograde ISCO shrinks; here we use the Schwarzschild
        # value as the default and correct for spin when spin > 0.
        r_g = G * self.bh.mass / c ** 2
        a = self.bh.spin
        # Kerr ISCO (prograde)
        z1 = 1.0 + (1.0 - a ** 2) ** (1.0 / 3.0) * (
            (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
        )
        z2 = (3.0 * a ** 2 - z1 ** 2) ** 0.5
        r_isco = r_g * (3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)) ** 0.5)
        return r_isco

    def hill_radius(self, semi_major_axis: float, eccentricity: float) -> float:
        """Hill radius for a star on an orbit around the BH.

        r_Hill = a (1 - e) (m_star / (3 M_BH))^{1/3}

        Parameters
        ----------
        semi_major_axis : float
            Semi-major axis of the stellar orbit [cm].
        eccentricity : float
            Orbital eccentricity.

        Returns
        -------
        float
            Hill radius in cm.
        """
        return semi_major_axis * (1.0 - eccentricity) * (
            self.star.mass / (3.0 * self.bh.mass)
        ) ** (1.0 / 3.0)

    # ------------------------------------------------------------------
    # Penetration & disruption criteria
    # ------------------------------------------------------------------

    def penetration_factor(self, r_pericenter: float) -> float:
        """Dimensionless penetration factor beta = r_t / r_p.

        Parameters
        ----------
        r_pericenter : float
            Pericenter distance of the stellar orbit [cm].

        Returns
        -------
        float
            Penetration factor beta (dimensionless).
        """
        return self.tidal_radius() / r_pericenter

    def critical_beta(self) -> float:
        """Critical penetration factor for full disruption.

        For a gamma = 5/3 polytrope beta_crit ~ 1.0-2.0 depending
        on the stellar structure. We adopt beta_crit = 1.85 following
        Mainetti et al. (2017) for a solar-type star.

        Returns
        -------
        float
            Critical beta for full disruption.
        """
        # Simplified: depends on stellar structure parameter
        # For gamma = 5/3 n=3/2 polytrope
        # More massive stars have lower beta_crit
        mass_ratio = self.star.mass / M_sun
        if mass_ratio < 0.5:
            return 2.0
        elif mass_ratio < 1.5:
            return 1.85
        else:
            return 1.0 + 0.85 * (1.5 / mass_ratio)

    def is_full_disruption(self, r_pericenter: float) -> bool:
        """Check whether the encounter leads to full disruption.

        Parameters
        ----------
        r_pericenter : float
            Pericenter distance [cm].

        Returns
        -------
        bool
            True if beta > beta_crit (full disruption).
        """
        return self.penetration_factor(r_pericenter) >= self.critical_beta()

    def partial_disruption_fraction(self, r_pericenter: float) -> float:
        """Mass fraction stripped during a partial disruption.

        For beta < beta_crit only a fraction of the stellar envelope
        is removed; the core survives.

        Parameters
        ----------
        r_pericenter : float
            Pericenter distance [cm].

        Returns
        -------
        float
            Mass fraction stripped (0 to 1). Returns 1.0 for full disruption.
        """
        beta = self.penetration_factor(r_pericenter)
        beta_crit = self.critical_beta()
        if beta >= beta_crit:
            return 1.0
        if beta <= 0.5:
            return 0.0
        # Approximate mass-loss curve from Guillouchon & Ramirez-Ruiz (2013)
        frac = ((beta - 0.5) / (beta_crit - 0.5)) ** 2.5
        return min(frac, 1.0)

    # ------------------------------------------------------------------
    # Energy spread
    # ------------------------------------------------------------------

    def energy_spread(self) -> float:
        """Specific orbital energy spread across the debris stream.

        Delta E ~ G M_BH R_* / r_t^2

        Returns
        -------
        float
            Energy spread [erg/g] (specific energy).
        """
        r_t = self.tidal_radius()
        return G * self.bh.mass * self.star.radius / r_t ** 2

    def is_outside_horizon(self) -> bool:
        """Check that the tidal radius lies outside the event horizon.

        A TDE is only observable if r_t > r_Schwarzschild (or more
        precisely r_t > r_ISCO). For main-sequence stars this requires
        M_BH < ~10^8 M_sun.

        Returns
        -------
        bool
            True if the star is tidally disrupted outside the horizon.
        """
        return self.tidal_radius() > self.schwarzschild_radius()

    def maximum_bh_mass_for_tde(self) -> float:
        """Maximum BH mass [g] for which a TDE is possible (r_t > r_S).

        Solves r_t(M_max) = r_S(M_max) for M_BH, given the stellar
        structure. For a main-sequence star R_* ~ M_*^{0.8} this gives
        M_max ~ 10^8 M_sun.

        Returns
        -------
        float
            Maximum BH mass in grams.
        """
        # r_t = R_* (M/rho_*)^{1/3}, r_S = 2GM/c^2
        # Set equal: R_* (M/m_*)^{1/3} = 2GM/c^2
        # => M^{2/3} = R_* c^2 / (2G m_*^{1/3})
        # => M = (R_* c^2 / (2G))^{3/2} / m_*^{1/2}
        return (self.star.radius * c ** 2 / (2.0 * G)) ** 1.5 / self.star.mass ** 0.5
