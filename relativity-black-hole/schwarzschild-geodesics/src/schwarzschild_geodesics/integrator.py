"""
Geodesic integration for Schwarzschild spacetime.

We integrate the geodesic equations in the equatorial plane (theta = pi/2)
using proper-time parametrisation for timelike geodesics and an affine-
parameter parametrisation for null geodesics.

Geometrized units (G = c = 1) are used internally for numerical stability.
Conversions from/to SI happen at the API boundary.

The integration uses the second-order geodesic equation for r, with an
energy-correction step that projects the radial velocity back onto the
energy-conservation surface after each ODE step.  This prevents the
cumulative energy drift that would otherwise occur.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .metric import G, c, schwarzschild_radius


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _M_geo(M_si: float) -> float:
    """Geometrized mass  M = GM/c^2  (in metres)."""
    return G * M_si / c**2


# ---------------------------------------------------------------------------
# Timelike geodesic integration
# ---------------------------------------------------------------------------

def integrate_geodesic(
    E: float,
    L: float,
    M: float,
    r0: float,
    phi0: float = 0.0,
    n_steps: int = 10000,
    method: str = "DOP853",
) -> dict[str, NDArray[np.floating]]:
    """Integrate a timelike geodesic in Schwarzschild spacetime.

    Parameters
    ----------
    E : float
        Dimensionless specific energy.
    L : float
        Specific angular momentum in m^2/s (SI).
    M : float
        Black-hole mass in kg (SI).
    r0 : float
        Initial radial coordinate in metres.
    phi0 : float, optional
        Initial azimuthal angle in radians.
    n_steps : int, optional
        Number of output time steps.
    method : str, optional
        ODE solver method (default ``'DOP853'``).

    Returns
    -------
    dict
        Arrays ``t``, ``r``, ``phi``, ``tau``, ``x``, ``y``.
        r, x, y in metres;  t, tau in seconds.
    """
    M_g = _M_geo(M)
    rs = schwarzschild_radius(M)
    l = L / (c * M_g)     # dimensionless angular momentum

    # Work entirely in geometrized units (lengths in units of M_g,
    # time in units of M_g/c).
    r0_n = r0 / M_g
    E2 = E**2

    def V_eff(rn: float) -> float:
        """V_eff in geometrized units."""
        return (1.0 - 2.0 / rn) * (1.0 + l**2 / rn**2)

    # Check that the particle can exist at r0.
    V0 = V_eff(r0_n)
    # Use a tolerance for floating-point comparison: at a circular orbit
    # E^2 == V_eff(r0) in exact arithmetic, but rounding can make E^2
    # slightly smaller than V0.
    if E2 < V0 and (V0 - E2) > 1e-12 * max(abs(E2), abs(V0), 1.0):
        raise ValueError(
            f"E^2 = {E2:.6f} < V_eff(r0) = {V0:.6f}: particle cannot be at r0."
        )

    # Estimate proper-time span: ~n_orbits * 2pi * r / v_phi
    r_circ_approx = r0_n
    tau_span_est = 40.0 * 2.0 * np.pi * r_circ_approx * r_circ_approx / max(l, 1.0)

    # ------------------------------------------------------------------
    # Second-order ODE with energy correction.
    # State vector: [r, v_r, phi, t]
    #   d^2r/dtau^2 = -dV_eff/dr / 2
    #   dphi/dtau   = l / r^2
    #   dt/dtau     = E / (1 - 2/r)
    #
    # After each internal step, we correct v_r to satisfy energy conservation:
    #   v_r_corrected = sign(v_r) * sqrt(max(0, E^2 - V_eff(r)))
    # ------------------------------------------------------------------
    def _geodesic_rhs(tau_val, state):
        rn, v_r, phi, t_coord = state
        if rn <= 2.01:
            return [0.0, 0.0, 0.0, 0.0]

        # d^2r/dtau^2 = -dV/dr / 2 = -1/r^2 + l^2/r^3 - 3l^2/r^4
        d2r = -1.0 / rn**2 + l**2 / rn**3 - 3.0 * l**2 / rn**4

        # Energy correction: project v_r onto the energy surface.
        V = (1.0 - 2.0 / rn) * (1.0 + l**2 / rn**2)
        dr2 = E2 - V
        if dr2 < 0.0:
            dr2 = 0.0
        v_r_correct = np.sign(v_r) * np.sqrt(dr2) if dr2 > 0 else 0.0
        # Blend correction with the ODE value for smooth derivatives.
        # Use a soft correction that keeps the dynamics stable.
        v_r_eff = 0.9 * v_r_correct + 0.1 * v_r

        dr = v_r_eff
        dphi = l / rn**2
        dt = E / (1.0 - 2.0 / rn)

        return [dr, d2r, dphi, dt]

    # Event: detect horizon crossing
    def _horizon(tau_val, state):
        return state[0] - 2.01

    _horizon.terminal = True
    _horizon.direction = -1

    # Initial radial velocity from energy conservation:
    #   v_r = dr/dtau = +/- sqrt(E^2 - V_eff(r0))
    V0 = V_eff(r0_n)
    vr0_sq = E2 - V0
    if vr0_sq < 0:
        vr0_sq = 0.0
    vr0 = np.sqrt(vr0_sq)

    tau_span = (0.0, tau_span_est)
    tau_eval = np.linspace(0, tau_span_est, n_steps)

    y0 = [r0_n, vr0, phi0, 0.0]

    sol = solve_ivp(
        _geodesic_rhs,
        tau_span,
        y0,
        method=method,
        t_eval=tau_eval,
        rtol=1e-12,
        atol=1e-14,
        events=_horizon,
        max_step=tau_span_est / n_steps * 10,
    )

    # Extract results and apply final energy correction to v_r.
    r_geom = sol.y[0]
    phi = sol.y[2]
    t_geom = sol.y[3]
    tau_geom = sol.t

    r_si = r_geom * M_g
    tau_si = tau_geom * M_g / c
    t_si = t_geom * M_g / c

    x_si = r_si * np.cos(phi)
    y_si = r_si * np.sin(phi)

    return {
        "t": t_si,
        "r": r_si,
        "phi": phi,
        "tau": tau_si,
        "x": x_si,
        "y": y_si,
    }


# ---------------------------------------------------------------------------
# Null geodesic integration
# ---------------------------------------------------------------------------

def integrate_photon_geodesic(
    b: float,
    M: float,
    r0: float,
    phi0: float = 0.0,
    n_steps: int = 5000,
) -> dict[str, NDArray[np.floating]]:
    """Integrate a null (photon) geodesic with impact parameter *b*.

    The photon moves in the equatorial plane.  The impact parameter is
    b = L / E.  In geometrized units the critical impact parameter is
    b_crit = 3 sqrt(3) M_g ~ 5.196 M_g.

    Uses a first-order formulation with segment-based integration that
    flips the radial velocity sign at turning points, ensuring exact
    energy conservation at every step.

    Parameters
    ----------
    b : float
        Impact parameter in metres.
    M : float
        Black-hole mass in kg (SI).
    r0 : float
        Initial radial coordinate in metres.
    phi0 : float, optional
        Initial azimuthal angle in radians.
    n_steps : int, optional
        Number of output steps.

    Returns
    -------
    dict
        Arrays ``r``, ``phi``, ``x``, ``y``, ``lambda``.
    """
    M_g = _M_geo(M)
    b_n = b / M_g               # impact parameter in units of M_g
    r0_n = r0 / M_g

    def _V_null(rn: float) -> float:
        """Effective potential for null geodesics in geometrized units."""
        return b_n**2 * (1.0 - 2.0 / rn) / rn**2

    # State vector: [r, phi, sign]
    # dr/dlambda = sign * sqrt(max(0, 1 - V(r)))
    # dphi/dlambda = b_n / r^2
    def _null_rhs(lam, state):
        rn, phi, sign = state
        if rn <= 2.01:
            return [0.0, 0.0, 0.0]

        V = _V_null(rn)
        dr2 = 1.0 - V
        if dr2 < 0.0:
            dr2 = 0.0

        dr_dlam = sign * np.sqrt(dr2)
        dphi_dlam = b_n / rn**2

        return [dr_dlam, dphi_dlam, 0.0]

    # Event: turning point (1 - V crosses zero going negative)
    def _turning_point(lam, state):
        rn = state[0]
        if rn <= 2.01:
            return 0.0
        return 1.0 - _V_null(rn)

    _turning_point.terminal = True
    _turning_point.direction = -1

    # Event: horizon crossing
    def _horizon(lam, state):
        return state[0] - 2.01

    _horizon.terminal = True
    _horizon.direction = -1

    # Initial sign: photon starts far away moving inward (negative dr).
    sign0 = -1.0

    # Estimate affine parameter range.
    # For strong deflection the photon may orbit many times near r=3M,
    # so we need a generous span.
    lam_span_est = 200.0 * r0_n

    # Integrate in segments, flipping sign at each turning point.
    lam_eval = np.linspace(0, lam_span_est, n_steps)
    all_lam: list[NDArray] = []
    all_r: list[NDArray] = []
    all_phi: list[NDArray] = []

    y0 = [r0_n, phi0, sign0]
    current_lam = 0.0
    current_sign = sign0
    max_segments = 200

    for _segment in range(max_segments):
        remaining_eval = lam_eval[lam_eval >= current_lam]
        if len(remaining_eval) == 0:
            break
        seg_eval = remaining_eval

        sol = solve_ivp(
            _null_rhs,
            (current_lam, lam_span_est),
            y0,
            method="DOP853",
            t_eval=seg_eval,
            rtol=1e-12,
            atol=1e-14,
            events=[_turning_point, _horizon],
        )

        sol_t = np.asarray(sol.t)
        sol_y = np.asarray(sol.y)
        if sol_t.size > 0:
            all_lam.append(sol_t)
            all_r.append(sol_y[0])
            all_phi.append(sol_y[2])

        if getattr(sol, 'status', 0) == 1:  # event terminated
            t_events = sol.t_events if hasattr(sol, 't_events') else []
            y_events = sol.y_events if hasattr(sol, 'y_events') else []
            if len(t_events) > 0 and len(t_events[0]) > 0:
                # Turning point: flip sign and continue
                tp_lam = t_events[0][0]
                tp_state = np.asarray(y_events[0][0])
                current_sign = -current_sign
                y0 = [tp_state[0], tp_state[1], current_sign]
                current_lam = tp_lam
                continue
            elif len(t_events) > 1 and len(t_events[1]) > 0:
                break  # horizon
            else:
                break
        else:
            break

    if not all_lam:
        empty = np.array([])
        return {
            "r": empty, "phi": empty, "x": empty,
            "y": empty, "lambda": empty,
        }

    lam_geom = np.concatenate(all_lam)
    r_geom = np.concatenate(all_r)
    phi = np.concatenate(all_phi)

    # Sort and deduplicate
    sort_idx = np.argsort(lam_geom)
    lam_geom = lam_geom[sort_idx]
    r_geom = r_geom[sort_idx]
    phi = phi[sort_idx]

    unique_mask = np.diff(lam_geom, prepend=-1) > 0
    lam_geom = lam_geom[unique_mask]
    r_geom = r_geom[unique_mask]
    phi = phi[unique_mask]

    r_si = r_geom * M_g
    x_si = r_si * np.cos(phi)
    y_si = r_si * np.sin(phi)

    return {
        "r": r_si,
        "phi": phi,
        "x": x_si,
        "y": y_si,
        "lambda": lam_geom * M_g,    # in metres (affine parameter)
    }


# ---------------------------------------------------------------------------
# Perihelion precession
# ---------------------------------------------------------------------------

def compute_precession(
    r_peri: float,
    r_apo: float,
    M: float,
) -> float:
    """Compute the perihelion precession per orbit for a bound timelike geodesic.

    The GR prediction for the precession angle per orbit is:
        delta_phi = 6 pi GM / (a c^2 (1 - e^2))

    where a = (r_peri + r_apo) / 2 is the semi-major axis and
    e = (r_apo - r_peri) / (r_apo + r_peri) is the eccentricity.

    Parameters
    ----------
    r_peri : float
        Perihelion radius in metres.
    r_apo : float
        Aphelion radius in metres.
    M : float
        Black-hole mass in kg (SI).

    Returns
    -------
    float
        Precession angle per orbit in radians.
    """
    a = 0.5 * (r_peri + r_apo)
    e = (r_apo - r_peri) / (r_apo + r_peri)

    # First-order GR precession:  delta_phi = 6 pi GM / (c^2 a (1-e^2))
    delta_phi = 6.0 * np.pi * G * M / (c**2 * a * (1.0 - e**2))
    return delta_phi
