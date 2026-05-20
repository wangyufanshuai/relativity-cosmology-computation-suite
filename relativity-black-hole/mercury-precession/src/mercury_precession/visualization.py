"""Visualization and reporting for Mercury perihelion precession results."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .analytical import total_analytical, precession_vs_eccentricity, precession_vs_semi_major
from .constants import A_MERCURY, E_MERCURY, M_SUN, G, C
from .geodesic import (
    effective_potential,
    integrate_orbit,
    orbital_params_to_conserved,
    schwarzschild_radius,
)


def plot_orbit_3d(result: dict, n_points: int = 5000) -> plt.Figure:
    """Plot Mercury's orbit in Schwarzschild spacetime (projected to x-y plane)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    t = result["t"]
    r = result["r"]
    phi = result["phi"]

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Plot orbit
    ax.plot(x[:n_points], y[:n_points], "b-", linewidth=0.3, alpha=0.7, label="Schwarzschild orbit")

    # Mark perihelion passages
    if len(result["perihelion_times"]) > 0:
        peri_indices = np.searchsorted(t, result["perihelion_times"])
        peri_indices = peri_indices[peri_indices < len(r)]
        ax.plot(
            r[peri_indices] * np.cos(result["perihelion_phis"][:len(peri_indices)]),
            r[peri_indices] * np.sin(result["perihelion_phis"][:len(peri_indices)]),
            "r.",
            markersize=8,
            label="Perihelion passages",
        )

    # Newtonian orbit for comparison
    ax.plot(x[:n_points], y[:n_points], "b-", linewidth=0.3)

    # Central body
    ax.plot(0, 0, "yo", markersize=15, label="Sun")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Mercury Orbit in Schwarzschild Spacetime")
    ax.set_aspect("equal")
    ax.legend()
    return fig


def plot_precession_comparison(a: float = A_MERCURY, e: float = E_MERCURY, M: float = M_SUN) -> plt.Figure:
    """Compare analytical, numerical, and PPN predictions of perihelion precession."""
    from .ppn import precession_ppn

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Panel 1: Effective potential
    ax1 = fig.add_subplot(gs[0, 0])
    rs = schwarzschild_radius(M)
    E_tilde, L_tilde, r_p, r_a = orbital_params_to_conserved(a, e, M)

    r_range = np.linspace(r_p * 0.95, r_a * 1.05, 1000)
    V_eff = effective_potential(r_range, L_tilde, rs)

    # Newtonian effective potential for comparison
    V_newt = 1.0 + L_tilde**2 / r_range**2 - rs / (2.0 * r_range)

    ax1.plot(r_range / 1e10, V_eff, "b-", linewidth=2, label="Schwarzschild V_eff")
    ax1.plot(r_range / 1e10, V_newt, "g--", linewidth=1.5, label="Newtonian V_eff")
    ax1.axhline(y=E_tilde**2, color="r", linestyle=":", label=f"E² = {E_tilde**2:.10f}")
    ax1.axvline(x=r_p / 1e10, color="gray", linestyle=":", alpha=0.5)
    ax1.axvline(x=r_a / 1e10, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("r (×10¹⁰ m)")
    ax1.set_ylabel("V_eff")
    ax1.set_title("Effective Potential")
    ax1.legend(fontsize=8)

    # Panel 2: Precession vs eccentricity
    ax2 = fig.add_subplot(gs[0, 1])
    e_range = np.linspace(0.01, 0.95, 200)
    prec_vs_e = precession_vs_eccentricity(e_range, a, M)
    prec_arcsec = prec_vs_e * 415.0 * (180.0 * 3600.0 / np.pi)  # rough orbits/century

    ax2.plot(e_range, prec_arcsec, "b-", linewidth=2)
    ax2.axvline(x=e, color="r", linestyle="--", alpha=0.7, label=f"Mercury (e={e})")
    ax2.set_xlabel("Eccentricity")
    ax2.set_ylabel("Precession (arcsec/century)")
    ax2.set_title("Precession vs Eccentricity")
    ax2.legend()

    # Panel 3: Precession vs semi-major axis
    ax3 = fig.add_subplot(gs[1, 0])
    a_range = np.linspace(0.1 * A_MERCURY, 5.0 * A_MERCURY, 200)
    prec_vs_a = precession_vs_semi_major(a_range, e, M)
    prec_arcsec_a = prec_vs_a * 415.0 * (180.0 * 3600.0 / np.pi)

    ax3.plot(a_range / A_MERCURY, prec_arcsec_a, "b-", linewidth=2)
    ax3.axvline(x=1.0, color="r", linestyle="--", alpha=0.7, label="Mercury")
    ax3.set_xlabel("Semi-major axis (a/a_Mercury)")
    ax3.set_ylabel("Precession (arcsec/century)")
    ax3.set_title("Precession vs Semi-Major Axis")
    ax3.legend()

    # Panel 4: PPN parameter space
    ax4 = fig.add_subplot(gs[1, 1])
    gamma_range = np.linspace(0.9, 1.1, 200)
    beta_range = np.linspace(0.9, 1.1, 200)
    G_grid, B_grid = np.meshgrid(gamma_range, beta_range)
    prec_grid = np.zeros_like(G_grid)
    for i in range(len(beta_range)):
        for j in range(len(gamma_range)):
            prec_grid[i, j] = precession_ppn(G_grid[i, j], B_grid[i, j], a, e, M)

    prec_total = prec_grid * 415.0 * (180.0 * 3600.0 / np.pi)
    im = ax4.contourf(G_grid, B_grid, prec_total, levels=30, cmap="viridis")
    plt.colorbar(im, ax=ax4, label="arcsec/century")
    ax4.plot(1.0, 1.0, "r*", markersize=15, label="GR (γ=β=1)")
    ax4.set_xlabel("PPN γ")
    ax4.set_ylabel("PPN β")
    ax4.set_title("Precession in PPN Parameter Space")
    ax4.legend()

    plt.tight_layout()
    return fig


def print_report(a: float = A_MERCURY, e: float = E_MERCURY, M: float = M_SUN) -> None:
    """Print a comprehensive report of Mercury's perihelion precession."""
    print("=" * 70)
    print("  MERCURY PERIHELION PRECESSION — COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    print()

    # Physical parameters
    rs = schwarzschild_radius(M)
    print(f"  Solar mass:         M = {M:.4e} kg")
    print(f"  Schwarzschild radius: rs = {rs:.4f} m = {rs/1e3:.6f} km")
    print(f"  Mercury semi-major: a  = {a:.4e} m = {a/1e10:.4f} × 10¹⁰ m")
    print(f"  Mercury eccentricity: e = {e:.6f}")
    print(f"  Perihelion:         r_p = {a*(1-e):.4e} m")
    print(f"  Aphelion:           r_a = {a*(1+e):.4e} m")
    print(f"  a/r_s ratio:        {a/rs:.4e}")
    print()

    # Analytical results
    results = total_analytical(a, e, M)
    print("  ANALYTICAL RESULTS")
    print("  " + "-" * 50)
    print(f"  1PN Schwarzschild:  {results['schwarzschild_1pn_rad']:.6e} rad/orbit")
    print(f"  2PN correction:     {results['schwarzschild_2pn_rad']:.6e} rad/orbit")
    print(f"  J2 quadrupole:      {results['quadrupole_J2_rad']:.6e} rad/orbit")
    print(f"  Total per orbit:    {results['total_per_orbit_rad']:.6e} rad/orbit")
    print(f"  Total (arcsec/cen): {results['total_arcsec_per_century']:.4f}")
    print(f"  Orbits per century: {results['orbits_per_century']:.2f}")
    print()

    # Breakdown
    phi_1pn = results["schwarzschild_1pn_rad"]
    phi_J2 = results["quadrupole_J2_rad"]
    total = results["total_per_orbit_rad"]
    print(f"  GR fraction:        {phi_1pn/total*100:.2f}%")
    print(f"  J2 fraction:        {phi_J2/total*100:.4f}%")
    print()

    # Numerical result (10 orbits for speed)
    print("  NUMERICAL INTEGRATION (10 orbits)")
    print("  " + "-" * 50)
    num_result = integrate_orbit(a, e, M, n_orbits=10)
    print(f"  Numerical:          {num_result['precession_per_orbit_rad']:.6e} rad/orbit")
    print(f"  Numerical (″/cen):  {num_result['precession_arcsec_per_century']:.4f}")
    print(f"  Analytical:         {num_result['analytical_precession_rad']:.6e} rad/orbit")

    ratio = num_result["precession_per_orbit_rad"] / num_result["analytical_precession_rad"]
    print(f"  Numerical/Analytical: {ratio:.10f}")
    print()

    # Observed value
    print("  OBSERVED VALUE (from MESSENGER data)")
    print("  " + "-" * 50)
    observed_total = 574.10  # arcsec/century total observed advance
    venus_pert = 277.86
    earth_pert = 90.04
    mars_pert = 2.54
    jupiter_pert = 153.58
    other_pert = venus_pert + earth_pert + mars_pert + jupiter_pert
    gr_signal = observed_total - other_pert
    print(f"  Total observed:     {observed_total:.2f} ″/century")
    print(f"  Venus perturbation: {venus_pert:.2f} ″/century")
    print(f"  Earth perturbation: {earth_pert:.2f} ″/century")
    print(f"  Mars perturbation:  {mars_pert:.2f} ″/century")
    print(f"  Jupiter perturb:    {jupiter_pert:.2f} ″/century")
    print(f"  Total Newtonian:    {other_pert:.2f} ″/century")
    print(f"  GR signal (resid.): {gr_signal:.2f} ″/century")
    print(f"  GR prediction:      {results['total_arcsec_per_century']:.2f} ″/century")
    print(f"  Agreement:          {gr_signal/results['total_arcsec_per_century']*100:.2f}%")
    print()

    # PPN constraint
    from .ppn import precession_ppn
    print("  PPN ANALYSIS")
    print("  " + "-" * 50)
    for g, b, label in [(1.0, 1.0, "GR"), (0.99, 1.0, "Brans-Dicke (ω=200)"), (1.0, 0.99, "β deviation")]:
        p = precession_ppn(g, b, a, e, M)
        p_asc = p * results["orbits_per_century"] * (180.0 * 3600.0 / np.pi)
        print(f"  {label:30s}: {p_asc:.4f} ″/century")
    print()
    print("=" * 70)
