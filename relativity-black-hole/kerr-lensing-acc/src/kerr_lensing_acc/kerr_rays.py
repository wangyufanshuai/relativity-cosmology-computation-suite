"""
Null geodesics in Kerr spacetime via Hamilton-Jacobi formalism.

Implements ray tracing for photons in the equatorial plane and off-equatorial
trajectories using Boyer-Lindquist coordinates and Mino time parametrization.

Key equations:
    Sigma^2 (dr/dlambda)^2 = R(r) = [E(r^2+a^2) - a*L_z]^2
                                   - Delta*[r^2 + (L_z - a*E)^2 + Q]
    Sigma^2 (dtheta/dlambda)^2 = Theta(theta) = Q - cos^2(theta)*[a^2*(1-E^2) + L_z^2/sin^2(theta)]
    where E=1 for null geodesics, Sigma = r^2 + a^2*cos^2(theta),
    Delta = r^2 - 2*M*r + a^2.

We use Mino time tau defined by dlambda = dtau / Sigma, which decouples the
radial and polar equations:
    (dr/dtau)^2 = R(r)
    (dtheta/dtau)^2 = Theta(theta)
    dt/dtau = ...
    dphi/dtau = ...
"""

import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def delta(r: float, M: float, a: float) -> float:
    """Kerr metric function Delta = r^2 - 2*M*r + a^2."""
    return r**2 - 2.0 * M * r + a**2


def sigma(r: float, theta: float, a: float) -> float:
    """Kerr metric function Sigma = r^2 + a^2 * cos^2(theta)."""
    return r**2 + a**2 * np.cos(theta) ** 2


def metric_coefficients(r: float, theta: float, M: float, a: float):
    """
    Return the contravariant metric components g^{tt}, g^{tphi}, g^{phiphi},
    and g^{rr}, g^{thetatheta} at a given point in Boyer-Lindquist coords.

    Useful for redshift / transfer-function calculations.
    """
    Sig = sigma(r, theta, a)
    D = delta(r, M, a)
    A = (r**2 + a**2) ** 2 - a**2 * D * np.sin(theta) ** 2

    g_tt = -(A - a**2 * D * np.sin(theta) ** 2) / (Sig * D)  # simplified = -(r^2+a^2)^2-a^2 D sin^2 / (Sig A) in contravariant form
    # Actually let's compute the *covariant* components and then invert.
    # Covariant Kerr metric:
    g_tt_cov = -(1.0 - 2.0 * M * r / Sig)
    g_tphi_cov = -2.0 * M * a * r * np.sin(theta) ** 2 / Sig
    g_phiphi_cov = (A * np.sin(theta) ** 2) / Sig
    g_rr_cov = Sig / D
    g_thth_cov = Sig

    # Build 2x2 (t,phi) block to invert
    det = g_tt_cov * g_phiphi_cov - g_tphi_cov**2
    g_tt_con = g_phiphi_cov / det
    g_tphi_con = -g_tphi_cov / det
    g_phiphi_con = g_tt_cov / det

    g_rr_con = 1.0 / g_rr_cov
    g_thth_con = 1.0 / g_thth_cov

    return {
        "g_tt": g_tt_cov,
        "g_tphi": g_tphi_cov,
        "g_phiphi": g_phiphi_cov,
        "g_rr": g_rr_cov,
        "g_thth": g_thth_cov,
        "g_tt_con": g_tt_con,
        "g_tphi_con": g_tphi_con,
        "g_phiphi_con": g_phiphi_con,
        "g_rr_con": g_rr_con,
        "g_thth_con": g_thth_con,
    }


# ---------------------------------------------------------------------------
# Geodesic equations in Mino time
# ---------------------------------------------------------------------------

def _R_potential(r: float, M: float, a: float, Lz: float, Q: float) -> float:
    """Radial potential R(r) for null geodesics (E=1)."""
    E = 1.0
    R = (E * (r**2 + a**2) - a * Lz) ** 2 - delta(r, M, a) * (
        r**2 + (Lz - a * E) ** 2 + Q
    )
    return R


def _Theta_potential(theta: float, a: float, Lz: float, Q: float) -> float:
    """Polar potential Theta(theta) for null geodesics (E=1)."""
    E = 1.0
    sth = np.sin(theta)
    cth = np.cos(theta)
    # Avoid division by zero near poles
    sin2 = sth**2
    if sin2 < 1e-30:
        sin2 = 1e-30
    Theta = Q - cth**2 * (a**2 * (1.0 - E**2) + Lz**2 / sin2)
    return Theta


def _geodesic_rhs(tau, y, M, a, Lz, Q):
    """
    RHS of the geodesic equations in Mino time.

    State vector y = [r, theta, phi, t, p_r_sign, p_theta_sign]
    where p_r_sign = sign(dr/dtau), p_theta_sign = sign(dtheta/dtau).
    We need to track these signs to handle turning points.
    """
    r, theta, phi, t_coord, pr_sign, pth_sign = y
    E = 1.0

    D = delta(r, M, a)
    Sig = r**2 + a**2 * np.cos(theta) ** 2

    # Radial potential
    R = _R_potential(r, M, a, Lz, Q)
    if R < 0:
        R = 0.0

    # Polar potential
    Th = _Theta_potential(theta, a, Lz, Q)
    if Th < 0:
        Th = 0.0

    dr_dtau = pr_sign * np.sqrt(R)
    dtheta_dtau = pth_sign * np.sqrt(Th)

    # dt/dtau and dphi/dtau from geodesic equations
    # Using the standard Kerr geodesic equations in Mino time:
    # dt/dtau = -a*(a*E*sin^2(theta) - Lz) + (r^2+a^2)/Delta * P(r)
    # dphi/dtau = -(a*E - Lz/sin^2(theta)) + a/Delta * P(r)
    # where P(r) = E*(r^2+a^2) - a*Lz
    sin2 = np.sin(theta) ** 2
    if sin2 < 1e-30:
        sin2 = 1e-30
    P_r = E * (r**2 + a**2) - a * Lz

    dt_dtau = -a * (a * E * sin2 - Lz) + (r**2 + a**2) / D * P_r
    dphi_dtau = -(a * E - Lz / sin2) + a / D * P_r

    # p_r_sign and p_theta_sign stay the same (turning points handled by events)
    return [dr_dtau, dtheta_dtau, dphi_dtau, dt_dtau, 0.0, 0.0]


def trace_ray(
    alpha: float,
    beta: float,
    r_obs: float,
    theta_obs: float,
    M: float = 1.0,
    a: float = 0.0,
    lambda_max: float = 50.0,
    n_steps: int = 5000,
    r_disk_outer: float = 20.0,
):
    """
    Trace a null geodesic from the observer back to the accretion disk.

    Parameters
    ----------
    alpha, beta : float
        Impact parameters (angular position on the observer's sky in units of M).
    r_obs : float
        Observer radial coordinate (Boyer-Lindquist r).
    theta_obs : float
        Observer inclination angle (0 = pole, pi/2 = equatorial).
    M : float
        Black hole mass (default 1, geometric units).
    a : float
        Black hole spin parameter (0 <= a <= M).
    lambda_max : float
        Maximum Mino time for integration.
    n_steps : int
        Number of integration steps.
    r_disk_outer : float
        Outer edge of accretion disk (for detecting disk crossing).

    Returns
    -------
    dict with keys:
        'hit_disk': bool
        'r_disk': float or None  (radius where ray hits disk, theta=pi/2)
        'phi_disk': float or None
        't_coord': float or None
        'r_final': float
        'theta_final': float
        'tau_final': float
        'trajectory': array of shape (N, 6) or None
    """
    E = 1.0  # photon energy at infinity

    # Convert impact parameters to constants of motion (Lz, Q)
    # For an observer at large distance and inclination theta_obs:
    #   Lz = -alpha * sin(theta_obs)  (approximate for distant observer)
    #   Q = beta^2 + alpha^2 * cos^2(theta_obs)
    # More precisely, from Bardeen (1972):
    Lz = -alpha * np.sin(theta_obs)
    Q = beta**2 + alpha**2 * np.cos(theta_obs)

    # Initial signs: dr/dtau is negative (ray goes inward), dtheta/dtau depends on beta
    pr_sign0 = -1.0
    pth_sign0 = 1.0 if beta >= 0 else -1.0

    y0 = [r_obs, theta_obs, 0.0, 0.0, pr_sign0, pth_sign0]

    tau_span = (0, lambda_max)
    tau_eval = np.linspace(0, lambda_max, n_steps)

    # Event: ray crosses equatorial plane (theta = pi/2)
    def equatorial_crossing(tau, y, M, a, Lz, Q):
        return y[1] - np.pi / 2.0

    equatorial_crossing.terminal = True
    equatorial_crossing.direction = -1 if beta >= 0 else 1  # downward/upward crossing

    # Event: ray falls into horizon (r = r_horizon + epsilon)
    r_horizon = M + np.sqrt(M**2 - a**2)
    eps_h = 0.1 * M

    def horizon_crossing(tau, y, M, a, Lz, Q):
        return y[0] - (r_horizon + eps_h)

    horizon_crossing.terminal = True
    horizon_crossing.direction = -1

    # Event: ray escapes to large r
    def escape(tau, y, M, a, Lz, Q):
        return y[0] - 2.0 * r_obs

    escape.terminal = True
    escape.direction = 1

    try:
        sol = solve_ivp(
            _geodesic_rhs,
            tau_span,
            y0,
            args=(M, a, Lz, Q),
            method="RK45",
            t_eval=tau_eval,
            events=[equatorial_crossing, horizon_crossing, escape],
            rtol=1e-8,
            atol=1e-10,
            max_step=lambda_max / n_steps * 5,
        )
    except Exception:
        return {
            "hit_disk": False,
            "r_disk": None,
            "phi_disk": None,
            "t_coord": None,
            "r_final": r_obs,
            "theta_final": theta_obs,
            "tau_final": 0.0,
            "trajectory": None,
        }

    r_final = sol.y[0, -1]
    theta_final = sol.y[1, -1]
    phi_final = sol.y[2, -1]
    t_final = sol.y[3, -1]
    tau_final = sol.t[-1]

    hit_disk = False
    r_disk = None
    phi_disk = None
    t_coord = None

    # Check if equatorial crossing occurred
    if len(sol.t_events[0]) > 0:
        # Ray crossed theta = pi/2
        idx = 0  # first crossing
        r_cross = sol.y_events[0][idx][0]
        phi_cross = sol.y_events[0][idx][2]
        t_cross = sol.y_events[0][idx][3]

        r_isco = compute_isco(M, a)

        if r_isco - 0.5 <= r_cross <= r_disk_outer:
            hit_disk = True
            r_disk = r_cross
            phi_disk = phi_cross
            t_coord = t_cross

    trajectory = np.array([sol.y[0], sol.y[1], sol.y[2], sol.y[3]]).T

    return {
        "hit_disk": hit_disk,
        "r_disk": r_disk,
        "phi_disk": phi_disk,
        "t_coord": t_coord,
        "r_final": r_final,
        "theta_final": theta_final,
        "tau_final": tau_final,
        "trajectory": trajectory,
    }


def trace_ray_equatorial(
    b: float,
    r_start: float,
    M: float = 1.0,
    a: float = 0.0,
    lambda_max: float = 100.0,
    n_steps: int = 10000,
):
    """
    Trace an equatorial (theta=pi/2) null geodesic for deflection calculations.

    Parameters
    ----------
    b : float
        Impact parameter. For equatorial photons, L_z = b, Q = 0.
    r_start : float
        Starting radial coordinate.
    M : float
        Black hole mass.
    a : float
        Black hole spin parameter.
    lambda_max : float
        Maximum Mino time.
    n_steps : int
        Number of steps.

    Returns
    -------
    dict with 'r_final', 'phi_final', 'trajectory', 'captured'
    """
    E = 1.0
    Lz = b
    Q = 0.0  # equatorial

    # Start moving inward
    pr_sign0 = -1.0
    pth_sign0 = 0.0  # stays in equatorial plane

    y0 = [r_start, np.pi / 2.0, 0.0, 0.0, pr_sign0, pth_sign0]

    r_horizon = M + np.sqrt(M**2 - a**2)
    eps_h = 0.05 * M

    def horizon_event(tau, y, M, a, Lz, Q):
        return y[0] - (r_horizon + eps_h)

    horizon_event.terminal = True
    horizon_event.direction = -1

    # Turning point: dr/dtau changes sign (R=0)
    def turning_point(tau, y, M, a, Lz, Q):
        R = _R_potential(y[0], M, a, Lz, Q)
        return R

    turning_point.terminal = True
    turning_point.direction = -1  # R going to zero from positive

    tau_span = (0, lambda_max)
    tau_eval = np.linspace(0, lambda_max, n_steps)

    try:
        sol = solve_ivp(
            _geodesic_rhs,
            tau_span,
            y0,
            args=(M, a, Lz, Q),
            method="RK45",
            t_eval=tau_eval,
            events=[horizon_event, turning_point],
            rtol=1e-9,
            atol=1e-11,
            max_step=lambda_max / n_steps * 5,
        )
    except Exception:
        return {
            "r_final": r_start,
            "phi_final": 0.0,
            "trajectory": None,
            "captured": False,
        }

    captured = len(sol.t_events[0]) > 0
    r_final = sol.y[0, -1]
    phi_final = sol.y[2, -1]

    trajectory = np.array([sol.y[0], sol.y[1], sol.y[2], sol.y[3]]).T

    return {
        "r_final": r_final,
        "phi_final": phi_final,
        "trajectory": trajectory,
        "captured": captured,
    }


# ---------------------------------------------------------------------------
# ISCO and photon sphere
# ---------------------------------------------------------------------------

def compute_isco(M: float = 1.0, a: float = 0.0, prograde: bool = True) -> float:
    """
    Compute the Innermost Stable Circular Orbit (ISCO) radius for the Kerr metric.

    For prograde orbits around a black hole with spin a:
        r_isco is found from the radial effective potential.

    For a=0 (Schwarzschild): r_isco = 6M
    For a=M (extreme Kerr, prograde): r_isco = M
    """
    a_star = a / M  # dimensionless spin
    z1 = 1 + (1 - a_star**2) ** (1 / 3) * (
        (1 + a_star) ** (1 / 3) + (1 - a_star) ** (1 / 3)
    )
    z2 = np.sqrt(3 * a_star**2 + z1**2)

    if prograde:
        r_isco = M * (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2)))
    else:
        r_isco = M * (3 + z2 + np.sqrt((3 - z1) * (3 + z1 + 2 * z2)))

    return r_isco


def photon_sphere_radius(M: float = 1.0, a: float = 0.0) -> float:
    """
    Compute the photon sphere (circular photon orbit) radius for equatorial photons.

    For Schwarzschild (a=0): r_ph = 3M.
    For Kerr, there are prograde and retrograde photon orbits.
    Returns the prograde orbit radius.
    """
    # r_ph for prograde equatorial orbits satisfies:
    # r^2 - 3*M*r + 2*a*sqrt(M*r) = 0
    # Let u = sqrt(r/M), then M*u^2 - 3*M^2*u + 2*a*M^(3/2)*u ...
    # More directly:
    # r_ph = 2M(1 + cos(2/3 * arccos(-a/M)))  [prograde]
    # But simpler: solve r^2 - 3Mr +/- 2a*sqrt(Mr) = 0
    # For a=0: r = 3M exactly.

    if abs(a) < 1e-12:
        return 3.0 * M

    # Prograde: r^2 - 3Mr + 2a*sqrt(Mr) = 0
    # Solve numerically
    from scipy.optimize import brentq

    def eq_pro(r):
        return r**2 - 3 * M * r + 2 * a * np.sqrt(M * r)

    r_horizon = M + np.sqrt(M**2 - a**2)
    r_min = r_horizon + 0.01
    r_max = 6.0 * M

    try:
        r_ph = brentq(eq_pro, r_min, r_max)
    except ValueError:
        r_ph = 3.0 * M  # fallback

    return r_ph


def critical_impact_parameter(M: float = 1.0, a: float = 0.0) -> float:
    """
    Critical impact parameter b_c for photon capture.

    For Schwarzschild: b_c = 3*sqrt(3)*M ~ 5.196M.
    For Kerr equatorial prograde: b_c = L_z/E at the photon sphere.
    """
    r_ph = photon_sphere_radius(M, a)

    # For equatorial prograde photon orbit:
    # L_z = (r^2 + a^2 - 2*M*r) / (a - r^2 * sqrt(M/r) / (a*sqrt(M/r) - M + r^2/r))
    # Simpler: use the standard result
    # b_c = -a*sin(theta) + ... but for equatorial, theta=pi/2:
    # b = L_z/E with E=1
    # At photon sphere: L_z = -(r_ph^2 + a^2)/a + r_ph * sqrt(...)
    # Actually for equatorial:
    # b_c = (r_ph^2 + a^2 - a*sqrt(...))/...
    # Use numerical approach: impact parameter at photon sphere
    if abs(a) < 1e-12:
        return 3.0 * np.sqrt(3.0) * M

    # L_z = (r^2 + a^2 - 2Mr) / (a + r*sqrt(r/M) * sign)
    # For prograde: sign is such that |L_z| is smaller
    # b = L_z (since E=1)
    denom = a + np.sqrt(M * r_ph)
    if abs(denom) < 1e-12:
        return 3.0 * np.sqrt(3.0) * M

    D = delta(r_ph, M, a)
    # L_z for prograde photon orbit:
    # P = E*(r^2 + a^2) - a*L_z
    # At photon orbit R=0 and dR/dr=0
    # L_z = -(r_ph^2 + a^2 - a*r_ph*sqrt(r_ph/M)) / (a - sqrt(r_ph/M))  ...
    # Let's use the known formula:
    # b_c = L_z/E = -a*sin^2(theta) + (r^2+a^2)/sqrt(...) for theta=pi/2
    sqrt_term = np.sqrt(r_ph / M)
    b_c = (r_ph**2 + a**2 - a * r_ph * sqrt_term) / (r_ph * sqrt_term - a)

    return abs(b_c)


def compute_deflection_angle(
    b: float,
    r_obs: float = 1000.0,
    M: float = 1.0,
    a: float = 0.0,
) -> float:
    """
    Compute the gravitational deflection angle for a photon with impact parameter b.

    For weak-field (large b), should give ~4M/b (Einstein result).
    Uses equatorial ray tracing and measures total phi change.

    Parameters
    ----------
    b : float
        Impact parameter (perpendicular distance from BH center).
    r_obs : float
        Distance of observer/source (large for weak-field limit).
    M : float
        Black hole mass.
    a : float
        Spin parameter.

    Returns
    -------
    float
        Deflection angle in radians (positive = toward the BH).
    """
    result = trace_ray_equatorial(b, r_obs, M, a, lambda_max=200.0, n_steps=20000)

    if result["captured"]:
        return np.inf

    # For a ray starting at r_obs going inward and coming back out:
    # phi_final is the total azimuthal angle traversed.
    # Deflection = phi_final - pi (straight line would give pi for passing through)
    # Actually for a ray starting at phi=0, going inward and back out,
    # the deflection angle is |phi_final| - pi
    deflection = abs(result["phi_final"]) - np.pi

    return deflection
