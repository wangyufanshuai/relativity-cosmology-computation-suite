"""Effective potentials for perturbations on Schwarzschild spacetime.

All functions work in geometric units (G = c = 1) internally.
The Schwarzschild radius rs = 2M where M is the black hole mass.

References:
    Regge & Wheeler, Phys. Rev. 108, 1063 (1957)
    Zerilli, Phys. Rev. D 2, 2141 (1970)
"""

import numpy as np
import matplotlib.pyplot as plt


def tortoise_coordinate(r, rs):
    """Compute the tortoise coordinate r* from the areal radius r.

    The relation is dr*/dr = 1/(1 - rs/r), which integrates to:
        r* = r + rs * ln|r/rs - 1|

    Parameters
    ----------
    r : float or ndarray
        Areal (Schwarzschild) radial coordinate. Must be > rs.
    rs : float
        Schwarzschild radius rs = 2M.

    Returns
    -------
    float or ndarray
        Tortoise coordinate r*.
    """
    return r + rs * np.log(np.abs(r / rs - 1.0))


def V_eff_scalar(r, rs, l):
    """Effective potential for scalar (spin-0) field perturbations.

    V = (1 - rs/r) * [l(l+1)/r^2 + rs/r^3]

    This follows from the Klein-Gordon equation on Schwarzschild background,
    decomposed into spherical harmonics with quantum number l.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate (r > rs).
    rs : float
        Schwarzschild radius.
    l : int
        Angular momentum quantum number (l >= 0).

    Returns
    -------
    float or ndarray
        Effective potential V(r).
    """
    f = 1.0 - rs / r
    return f * (l * (l + 1.0) / r ** 2 + rs / r ** 3)


def V_eff_em(r, rs, l):
    """Effective potential for electromagnetic (spin-1) field perturbations.

    For electromagnetic perturbations on Schwarzschild, the odd-parity (Regge-Wheeler)
    and even-parity (Zerilli) potentials are isospectral. We use the Regge-Wheeler form:

    V = (1 - rs/r) * [l(l+1)/r^2 - rs/r^3 * (1 - s)]

    For s=1 (electromagnetic):
    V = (1 - rs/r) * [l(l+1)/r^2]

    Note: the electromagnetic potential has no 1/r^3 term (the rs/r^3 term
    cancels for spin-1), giving a simpler form compared to the scalar case.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate (r > rs).
    rs : float
        Schwarzschild radius.
    l : int
        Angular momentum quantum number (l >= 1 for EM).

    Returns
    -------
    float or ndarray
        Effective potential V(r).
    """
    f = 1.0 - rs / r
    return f * l * (l + 1.0) / r ** 2


def V_eff_gravitational(r, rs, l):
    """Effective potential for gravitational (spin-2) perturbations.

    Uses the Regge-Wheeler potential for odd-parity gravitational perturbations:

    V = (1 - rs/r) * [l(l+1)/r^2 - 3*rs/r^3]

    For l=2 the peak value is approximately 0.15/rs^2 at r ~ 1.6*rs.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate (r > rs).
    rs : float
        Schwarzschild radius.
    l : int
        Angular momentum quantum number (l >= 2 for gravitational).

    Returns
    -------
    float or ndarray
        Effective potential V(r).
    """
    f = 1.0 - rs / r
    return f * (l * (l + 1.0) / r ** 2 - 3.0 * rs / r ** 3)


def V_eff(r, rs, l, s):
    """Unified effective potential for spin-s perturbations.

    General Regge-Wheeler form:
    V = (1 - rs/r) * [l(l+1)/r^2 + (1 - s^2) * rs/r^3]

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate (r > rs).
    rs : float
        Schwarzschild radius.
    l : int
        Angular momentum quantum number.
    s : int
        Spin of the perturbing field (0=scalar, 1=EM, 2=gravitational).

    Returns
    -------
    float or ndarray
        Effective potential V(r).
    """
    f = 1.0 - rs / r
    return f * (l * (l + 1.0) / r ** 2 + (1.0 - s ** 2) * rs / r ** 3)


def plot_potentials(rs, omega, l_range):
    """Plot effective potentials for spin s=0, 1, 2 for given l values.

    Parameters
    ----------
    rs : float
        Schwarzschild radius.
    omega : float
        Frequency (used to draw the omega^2 line showing the tunneling region).
    l_range : array-like
        List of l values to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with potential plots.
    """
    r_min = rs * 1.01
    r_max = rs * 12.0
    r = np.linspace(r_min, r_max, 1000)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"s=0": "blue", "s=1": "green", "s=2": "red"}
    linestyles = ["-", "--", "-.", ":"]

    for i, l in enumerate(l_range):
        ls = linestyles[i % len(linestyles)]
        ax.plot(r / rs, V_eff_scalar(r, rs, l),
                color=colors["s=0"], linestyle=ls,
                label=f"s=0, l={l}")
        if l >= 1:
            ax.plot(r / rs, V_eff_em(r, rs, l),
                    color=colors["s=1"], linestyle=ls,
                    label=f"s=1, l={l}")
        if l >= 2:
            ax.plot(r / rs, V_eff_gravitational(r, rs, l),
                    color=colors["s=2"], linestyle=ls,
                    label=f"s=2, l={l}")

    # Draw omega^2 line
    ax.axhline(y=omega ** 2, color="black", linestyle=":", alpha=0.5,
               label=r"$\omega^2$")

    ax.set_xlabel(r"$r / r_s$", fontsize=14)
    ax.set_ylabel(r"$V_{\mathrm{eff}}(r)$", fontsize=14)
    ax.set_title("Effective potentials on Schwarzschild spacetime", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(bottom=0)
    ax.set_xlim(r_min / rs, r_max / rs)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
