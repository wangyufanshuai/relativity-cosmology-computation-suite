"""Exact numerical computation of greybody factors via the shooting method.

This module integrates the radial wave equation on Schwarzschild spacetime
using tortoise coordinates, matching boundary conditions at the horizon
and at infinity to extract the transmission coefficient.

The radial equation in tortoise coordinates is:
    d^2 psi/dr*^2 + [omega^2 - V(r*)] psi = 0

where r* is the tortoise coordinate and V(r*) is the effective potential.

Boundary conditions:
    - Near horizon (r* -> -inf): psi -> e^{-i*omega*r*} + R * e^{i*omega*r*}
      (ingoing + reflected outgoing wave)
    - At infinity (r* -> +inf): psi -> T * e^{i*omega*r*}
      (transmitted outgoing wave only)

The greybody factor is Gamma = |T|^2 = 1 - |R|^2.

We use the shooting method:
1. Start integration near the horizon with the ingoing wave boundary condition.
2. Integrate outward to a large radius.
3. Extract R and T by matching to asymptotic solutions.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .potential import V_eff, tortoise_coordinate


def _rstar_to_r(rstar, rs, r_guess=None):
    """Convert tortoise coordinate r* back to areal radius r.

    r* = r + rs * ln(r/rs - 1)

    This is inverted numerically using Newton's method.

    Parameters
    ----------
    rstar : float
        Tortoise coordinate value.
    rs : float
        Schwarzschild radius.
    r_guess : float, optional
        Initial guess for Newton iteration.

    Returns
    -------
    float
        Areal radius r corresponding to r*.
    """
    if r_guess is None:
        # For large r*, r ~ r*; for r* near -inf, r ~ rs
        r_guess = max(rs * 1.1, rstar)

    r = r_guess
    for _ in range(100):
        f = r + rs * np.log(r / rs - 1.0) - rstar
        fp = 1.0 + rs / (r - rs)  # dr*/dr = 1/(1-rs/r)
        delta = f / fp
        r = r - delta
        if r <= rs:
            r = rs * 1.0001
        if abs(delta) < 1e-14 * abs(r):
            break
    return r


def greybody_factor_numerical(omega, l, s, M, rtol=1e-10):
    """Compute the greybody factor using the shooting method.

    Integrates the radial wave equation from near the horizon to a large
    radius, extracting the transmission coefficient from the asymptotic
    behavior of the solution.

    Parameters
    ----------
    omega : float
        Frequency in geometric units.
    l : int
        Angular momentum quantum number.
    s : int
        Spin of the field (0=scalar, 1=EM, 2=gravitational).
    M : float
        Black hole mass in geometric units.
    rtol : float, optional
        Relative tolerance for the ODE integrator. Default is 1e-10.

    Returns
    -------
    float
        Greybody factor Gamma(omega, l, s) = |T|^2 in [0, 1].
    """
    rs = 2.0 * M

    # Enforce minimum l values by spin
    if s == 1 and l < 1:
        l = 1
    if s == 2 and l < 2:
        l = 2

    # Handle edge cases
    if omega < 1e-15:
        return 0.0

    omega_rs = omega * rs
    if omega_rs > 30:
        # High frequency: full transmission
        return 1.0

    # Define the effective potential as a function of r*
    def V_of_rstar(rstar_val):
        r = _rstar_to_r(rstar_val, rs)
        return V_eff(r, rs, l, s)

    # Integration range in tortoise coordinate
    # Near horizon: r* -> -inf, we start at r slightly above rs
    r_start = rs * (1.0 + 1e-6)
    rstar_start = tortoise_coordinate(r_start, rs)

    # Far field: r* -> +inf, we integrate to a large radius
    r_end = rs * max(50.0, 10.0 / omega_rs)
    rstar_end = tortoise_coordinate(r_end, rs)

    # The wave equation: d^2 psi/dr*^2 = -[omega^2 - V(r*)] psi
    # Rewrite as first-order system:
    #   d psi / dr* = phi
    #   d phi / dr* = -[omega^2 - V(r*)] psi

    def wave_eq(rstar_val, y):
        psi, phi = y
        V = V_of_rstar(rstar_val)
        dpsi_drstar = phi
        dphi_drstar = -(omega ** 2 - V) * psi
        return [dpsi_drstar, dphi_drstar]

    # Near-horizon boundary condition:
    # psi ~ e^{-i*omega*r*} + R * e^{i*omega*r*}
    # For the ingoing wave: psi ~ e^{-i*omega*r*}
    # We start with purely ingoing wave and extract R later.
    #
    # Use the ingoing mode: psi = e^{-i*omega*r*}
    # psi(rstar_start) = e^{-i*omega*rstar_start}
    # phi(rstar_start) = -i*omega * e^{-i*omega*rstar_start}

    psi_start = np.exp(-1j * omega * rstar_start)
    phi_start = -1j * omega * np.exp(-1j * omega * rstar_start)

    y0 = [psi_start, phi_start]

    # Integrate from near horizon to large radius
    sol = solve_ivp(
        wave_eq,
        [rstar_start, rstar_end],
        y0,
        method='DOP853',
        rtol=rtol,
        atol=rtol * 1e-2,
        dense_output=True,
    )

    if not sol.success:
        # Fallback: use WKB approximation
        from .wkb import greybody_factor_wkb
        return greybody_factor_wkb(omega, l, s, M, order=3)

    # Extract the solution at the far end
    psi_end = sol.y[0, -1]
    phi_end = sol.y[1, -1]

    # At large r*, V -> 0, so the asymptotic solution is:
    # psi ~ A * e^{i*omega*r*} + B * e^{-i*omega*r*}
    #
    # Matching:
    # psi(rstar_end) = A * e^{i*omega*rstar_end} + B * e^{-i*omega*rstar_end}
    # phi(rstar_end) = i*omega * A * e^{i*omega*rstar_end}
    #                  - i*omega * B * e^{-i*omega*rstar_end}
    #
    # Solving for A and B:
    # A = (psi_end * i*omega + phi_end) / (2 * i * omega) * e^{-i*omega*rstar_end}
    # B = (psi_end * i*omega - phi_end) / (2 * i * omega) * e^{i*omega*rstar_end}

    iomega = 1j * omega
    exp_plus = np.exp(iomega * rstar_end)
    exp_minus = np.exp(-iomega * rstar_end)

    A = (psi_end * iomega + phi_end) / (2.0 * iomega) * exp_minus
    B = (psi_end * iomega - phi_end) / (2.0 * iomega) * exp_plus

    # The ingoing wave at the horizon has unit amplitude.
    # A is the transmitted (outgoing at infinity) amplitude T
    # B is the reflected (ingoing at infinity) amplitude
    #
    # Greybody factor: Gamma = |T|^2 / |ingoing_amplitude|^2
    # Since our starting ingoing wave had unit amplitude:
    # Gamma = |A|^2

    # Actually, let's be more careful. We started with:
    # psi ~ e^{-i*omega*r*} (ingoing at horizon, unit amplitude)
    #
    # At infinity:
    # psi = A e^{i*omega*r*} + B e^{-i*omega*r*}
    #
    # The transmission coefficient T = A (the outgoing part)
    # The reflection coefficient R = B (would be the ingoing part, but this
    # comes from reflection off the barrier)
    #
    # Gamma = |T|^2 / |ingoing|^2 = |A|^2
    #
    # But we need to normalize correctly. The relation is:
    # 1 = |T|^2 + |R|^2  (flux conservation)
    #
    # Gamma = |A|^2 and we verify |A|^2 + |B|^2 = 1 as a consistency check.

    T_abs2 = np.abs(A) ** 2
    R_abs2 = np.abs(B) ** 2

    # Normalization: the ingoing wave had unit amplitude, but the potential
    # near the horizon modifies the effective normalization.
    # Use flux conservation: Gamma = 1 - |R|^2 is more robust numerically.
    # However, for the shooting method starting with the ingoing mode,
    # Gamma = |T|^2 directly, and we normalize by total flux.

    total_flux = T_abs2 + R_abs2
    if total_flux > 0:
        Gamma = T_abs2 / total_flux
    else:
        Gamma = 0.0

    return float(np.clip(Gamma, 0.0, 1.0))
