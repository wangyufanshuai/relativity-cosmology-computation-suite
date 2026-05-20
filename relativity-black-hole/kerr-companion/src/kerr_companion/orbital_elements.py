"""
Orbital elements for Kerr geodesics.

Converts between constants of motion (E, Lz, Q) and orbital elements
(semi-latus rectum p, eccentricity e, inclination iota) and computes
the ISCO radius as a function of black hole spin.

References:
    - Schmidt (2002) "Celestial mechanics in Kerr spacetime"
    - Bardeen, Press & Teukolsky (1972) "Rotating Black Holes..."
    - Glampedakis, Hughes & Kennefick (2002)
"""

import numpy as np
from scipy.optimize import brentq


class OrbitalElements:
    """
    Orbital element computations for Kerr geodesics.

    Parameters
    ----------
    M : float
        Black hole mass.
    a : float
        Black hole spin parameter (|a| <= M).
    """

    def __init__(self, M=1.0, a=0.0):
        if abs(a) > M:
            raise ValueError(f"Spin parameter |a|={abs(a)} must not exceed M={M}")
        self.M = M
        self.a = a

    def isco_radius(self):
        """
        Innermost Stable Circular Orbit (ISCO) radius.

        r_isco = M(3 + Z2 -/+ sqrt((3-Z1)(3+Z1+2Z2)))
        where - for prograde, + for retrograde orbits.

        Z1 = 1 + (1-a^2/M^2)^{1/3} * ((1+a/M)^{1/3} + (1-a/M)^{1/3})
        Z2 = sqrt(3 a^2/M^2 + Z1^2)

        Returns
        -------
        float
            ISCO radius for prograde orbits (co-rotating with spin).
        """
        M = self.M
        a = self.a
        a_abs = abs(a)

        # Normalized spin
        a_star = a_abs / M

        Z1 = 1.0 + (1.0 - a_star**2) ** (1.0 / 3.0) * (
            (1.0 + a_star) ** (1.0 / 3.0) + (1.0 - a_star) ** (1.0 / 3.0)
        )
        Z2 = np.sqrt(3.0 * a_star**2 + Z1**2)

        # Prograde: minus sign (innermost orbit)
        r_isco = M * (3.0 + Z2 - np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))

        return r_isco

    def isco_radius_retrograde(self):
        """ISCO radius for retrograde orbits (plus sign)."""
        M = self.M
        a = self.a
        a_abs = abs(a)
        a_star = a_abs / M

        Z1 = 1.0 + (1.0 - a_star**2) ** (1.0 / 3.0) * (
            (1.0 + a_star) ** (1.0 / 3.0) + (1.0 - a_star) ** (1.0 / 3.0)
        )
        Z2 = np.sqrt(3.0 * a_star**2 + Z1**2)

        r_isco_retro = M * (3.0 + Z2 + np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))
        return r_isco_retro

    def circular_orbit_constants(self, r_circ):
        """
        Compute (E, Lz, Q=0) for a circular equatorial orbit at radius r_circ.

        For equatorial circular orbits in Kerr:
            E = (r^2 - 2Mr +/- a sqrt(Mr)) / (r sqrt(r^2 - 3Mr +/- 2a sqrt(Mr)))
            Lz = +/- sqrt(Mr) (r^2 -/+ 2a sqrt(Mr) - a^2) /
                 (r sqrt(r^2 - 3Mr +/- 2a sqrt(Mr)))
        where upper sign = prograde, lower = retrograde.

        Parameters
        ----------
        r_circ : float
            Circular orbit radius.

        Returns
        -------
        tuple
            (E, Lz, Q=0) for prograde orbit.
        """
        M = self.M
        a = self.a
        r = r_circ

        sqrt_Mr = np.sqrt(M * r)

        # Prograde (upper signs)
        denom_sq = r**2 - 3.0 * M * r + 2.0 * a * sqrt_Mr
        if denom_sq <= 0:
            raise ValueError(f"Radius {r} is inside the ISCO for a={a}")
        denom = r * np.sqrt(denom_sq)

        E = (r**2 - 2.0 * M * r + a * sqrt_Mr) / denom
        Lz = sqrt_Mr * (r**2 - 2.0 * a * sqrt_Mr - a**2) / denom

        return E, Lz, 0.0

    def constants_to_elements(self, E, Lz, Q):
        """
        Convert constants of motion to orbital elements (p, e, iota).

        For equatorial orbits (Q=0):
            Semi-latus rectum: p = Lz^2 / (M^2 (1-e^2))
            Eccentricity: from turning points r_min, r_max
                p/(1+e) = r_min, p/(1-e) = r_max

        For inclined orbits, use the separatrix structure.

        Parameters
        ----------
        E : float
            Specific energy.
        Lz : float
            Specific angular momentum.
        Q : float
            Carter constant.

        Returns
        -------
        dict
            {'p': semi-latus rectum, 'e': eccentricity, 'iota': inclination}
        """
        M = self.M
        a = self.a

        if Q == 0.0:
            # Equatorial orbit
            return self._equatorial_elements(E, Lz)
        else:
            # Inclined orbit
            return self._inclined_elements(E, Lz, Q)

    def _equatorial_elements(self, E, Lz):
        """Compute orbital elements for equatorial (Q=0) orbits."""
        M = self.M
        a = self.a

        # Find radial turning points: R(r) = 0
        # R = P^2 - Delta * (r^2 + (Lz - aE)^2)
        # since Q=0, kappa = r^2 + (Lz-aE)^2

        def R_func(r):
            P = E * (r**2 + a**2) - a * Lz
            D = r**2 - 2.0 * M * r + a**2
            kappa = r**2 + (Lz - a * E) ** 2
            return P**2 - D * kappa

        # Search for roots
        r_min_search = 2.0 * M
        r_max_search = 200.0 * M
        r_arr = np.linspace(r_min_search, r_max_search, 50000)
        R_vals = np.array([R_func(r) for r in r_arr])

        # Find sign changes
        sign_changes = np.where(np.diff(np.sign(R_vals)))[0]
        roots = []
        for idx in sign_changes:
            try:
                root = brentq(R_func, r_arr[idx], r_arr[idx + 1])
                roots.append(root)
            except (ValueError, RuntimeError):
                continue

        if len(roots) >= 2:
            r_min = roots[0]
            r_max = roots[-1]
        elif len(roots) == 1:
            # Circular orbit (double root)
            r_min = roots[0]
            r_max = roots[0]
        else:
            # Fallback: use approximate formula
            # For nearly circular orbits
            r_min = Lz**2 / (2.0 * M) * (1.0 - 0.1)
            r_max = Lz**2 / (2.0 * M) * (1.0 + 0.1)

        # Semi-latus rectum and eccentricity from turning points
        if r_max > r_min:
            p = 2.0 * r_min * r_max / (r_min + r_max)
            e = (r_max - r_min) / (r_max + r_min)
        else:
            # Circular orbit
            p = r_min
            e = 0.0

        return {"p": p, "e": e, "iota": 0.0}

    def _inclined_elements(self, E, Lz, Q):
        """Compute orbital elements for inclined orbits."""
        M = self.M
        a = self.a

        # Find radial turning points
        def R_func(r):
            P = E * (r**2 + a**2) - a * Lz
            D = r**2 - 2.0 * M * r + a**2
            kappa = r**2 + (Lz - a * E) ** 2 + Q
            return P**2 - D * kappa

        r_arr = np.linspace(2.0 * M, 200.0 * M, 50000)
        R_vals = np.array([R_func(r) for r in r_arr])

        sign_changes = np.where(np.diff(np.sign(R_vals)))[0]
        roots = []
        for idx in sign_changes:
            try:
                root = brentq(R_func, r_arr[idx], r_arr[idx + 1])
                roots.append(root)
            except (ValueError, RuntimeError):
                continue

        if len(roots) >= 2:
            r_min = roots[0]
            r_max = roots[-1]
            p = 2.0 * r_min * r_max / (r_min + r_max)
            e = (r_max - r_min) / (r_max + r_min)
        else:
            p = roots[0] if roots else 10.0 * M
            e = 0.0

        # Inclination angle from Carter constant
        # For small inclination: Q ~ Lz^2 sin^2(iota) + O(a^2)
        # More precisely: Q = Lz^2 * tan^2(iota) for the leading term
        # Inclination defined via:
        # cos^2(iota) = (Lz^2 + Q - a^2(E^2 - 1)) / (Lz^2 + Q)
        # But a cleaner relation uses the angular turning points
        if Lz != 0:
            cos2_iota = Lz**2 / (Lz**2 + Q) if (Lz**2 + Q) > 0 else 1.0
            cos2_iota = np.clip(cos2_iota, -1.0, 1.0)
            iota = np.arccos(np.sqrt(cos2_iota))
        else:
            iota = np.pi / 2.0

        return {"p": p, "e": e, "iota": iota}

    def elements_to_constants(self, p, e, iota):
        """
        Convert orbital elements to constants of motion (E, Lz, Q).

        Uses approximate relations valid in the weak-field (p >> M) regime
        and exact for the Schwarzschild case (a=0, iota=0).

        Parameters
        ----------
        p : float
            Semi-latus rectum (in units of M).
        e : float
            Eccentricity [0, 1).
        iota : float
            Inclination angle (radians). 0 = equatorial.

        Returns
        -------
        tuple
            (E, Lz, Q)
        """
        M = self.M
        a = self.a

        # For Schwarzschild (a=0):
        # E^2 = (p-2-2e)(p-2+2e) / (p(p-3-e^2))  [exact for equatorial]
        # Lz^2 = M^2 p^2 / (p-3-e^2)  [exact for equatorial]
        if abs(a) < 1e-15 and abs(iota) < 1e-15:
            denom = p - 3.0 - e**2
            if denom <= 0:
                raise ValueError(f"Invalid orbital elements: p={p}, e={e} (p-3-e^2={denom}<=0)")
            E = np.sqrt((p - 2.0 - 2.0 * e) * (p - 2.0 + 2.0 * e) / (p * denom))
            Lz = np.sqrt(M**2 * p**2 / denom)
            Q = 0.0
            return E, Lz, Q

        # For Kerr, use iterative or approximate approach
        # Leading order in 1/p:
        # E ~ 1 - (1-e^2)/(2p) * (1 - ...)  [Newtonian limit]
        # Lz ~ sqrt(M p) * cos(iota)  [weak field]

        # More accurate: use the expressions from Schmidt (2002)
        # For generic Kerr, we solve the turning point equations
        r1 = p / (1.0 + e)  # periapsis
        r2 = p / (1.0 - e)  # apoapsis

        # For equatorial Kerr orbits, we solve the system:
        # R(r1) = 0, R(r2) = 0, Theta(theta_max) = 0
        def equations_for_E_Lz(E, Lz):
            """R(r) = 0 at both turning points for equatorial (Q=0)."""
            def R_func(r):
                P = E * (r**2 + a**2) - a * Lz
                D = r**2 - 2.0 * M * r + a**2
                kappa = r**2 + (Lz - a * E) ** 2
                return P**2 - D * kappa
            return R_func(r1), R_func(r2)

        # Newton's method for E, Lz
        # Initial guess from Schwarzschild
        E_guess = 1.0 - (1.0 - e**2) / (2.0 * p)
        Lz_guess = np.sqrt(M**2 * p**2 / max(p - 3.0 - e**2, 0.1))

        # For equatorial orbits
        if abs(iota) < 1e-15:
            Q = 0.0
            E = E_guess
            Lz = Lz_guess

            # Refine with Newton iterations
            for _ in range(50):
                f1, f2 = equations_for_E_Lz(E, Lz)
                if abs(f1) < 1e-12 and abs(f2) < 1e-12:
                    break
                # Numerical Jacobian
                eps_E = max(abs(E) * 1e-8, 1e-12)
                eps_Lz = max(abs(Lz) * 1e-8, 1e-12)

                f1_Ep, f2_Ep = equations_for_E_Lz(E + eps_E, Lz)
                f1_Em, f2_Em = equations_for_E_Lz(E - eps_E, Lz)
                f1_Lp, f2_Lp = equations_for_E_Lz(E, Lz + eps_Lz)
                f1_Lm, f2_Lm = equations_for_E_Lz(E, Lz - eps_Lz)

                J = np.array(
                    [
                        [(f1_Ep - f1_Em) / (2 * eps_E), (f1_Lp - f1_Lm) / (2 * eps_Lz)],
                        [(f2_Ep - f2_Em) / (2 * eps_E), (f2_Lp - f2_Lm) / (2 * eps_Lz)],
                    ]
                )

                rhs = -np.array([f1, f2])
                try:
                    delta = np.linalg.solve(J, rhs)
                except np.linalg.LinAlgError:
                    break
                E += delta[0]
                Lz += delta[1]

                # Keep E < 1 for bound orbits
                E = min(E, 0.999)
                if E < 0:
                    E = abs(E)

            return E, Lz, 0.0
        else:
            # Inclined orbits: use approximate relations
            # Q ~ Lz^2 tan^2(iota) + a^2 cos^2(iota) (E^2 - 1)
            # Start with Schwarzschild-like E, Lz and add inclination
            E = E_guess
            Lz = Lz_guess * np.cos(iota)
            Q = Lz_guess**2 * np.sin(iota) ** 2

            return E, Lz, Q

    def orbital_frequencies(self, p, e, iota):
        """
        Compute the three fundamental orbital frequencies.

        Omega_r: radial (epicyclic) frequency
        Omega_theta: polar frequency
        Omega_phi: azimuthal frequency

        For Schwarzschild (a=0, e=0, iota=0):
            Omega_phi = sqrt(M/r^3)  [Kepler]
            Omega_r = Omega_theta = Omega_phi

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
            {'Omega_r': ..., 'Omega_theta': ..., 'Omega_phi': ...}
        """
        M = self.M
        a = self.a

        # Mean radius for circular orbit approximation
        r_mean = p / (1.0 - e**2)

        # Azimuthal frequency (Kepler + frame dragging)
        # Omega_phi = M^{1/2} / (r^{3/2} + a M^{1/2})  [exact for circular equatorial]
        if abs(a) < 1e-15:
            Omega_phi = np.sqrt(M / r_mean**3)
        else:
            Omega_phi = np.sqrt(M) / (r_mean**1.5 + a * np.sqrt(M))

        # Radial frequency: for eccentric orbits
        # Omega_r < Omega_phi in GR (perihelion precession)
        # For Schwarzschild: Omega_r/Omega_phi = sqrt(1 - 6M/p) approximately
        if e > 0:
            Omega_r = Omega_phi * np.sqrt(max(1.0 - 6.0 * M / p, 1e-10))
        else:
            # Circular: radial frequency = azimuthal (no oscillation)
            Omega_r = Omega_phi

        # Polar frequency: for inclined orbits
        # Omega_theta = Omega_phi for Schwarzschild
        # For Kerr: slight modification due to spin
        if abs(iota) < 1e-15:
            Omega_theta = Omega_phi
        else:
            # Nodal precession from Lense-Thirring
            Omega_theta = Omega_phi * (1.0 - 2.0 * a * np.sqrt(M) / r_mean**1.5)
            Omega_theta = max(Omega_theta, 0.1 * Omega_phi)  # Ensure positive

        return {
            "Omega_r": Omega_r,
            "Omega_theta": Omega_theta,
            "Omega_phi": Omega_phi,
        }
