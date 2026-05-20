"""Analysis tools for quintessence models.

Provides classification (thawing vs freezing), CPL parameter fitting,
and tracker-solution diagnostics.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit


def classify_model(w_array: NDArray, a_array: NDArray) -> str:
    """Classify a quintessence model as 'thawing' or 'freezing'.

    A model is **thawing** if |w| generally increases with time (w moves
    away from -1 toward 0) and **freezing** if |w| generally decreases
    (w approaches -1 from above).

    The classification is determined by the slope of w(a) over the
    latter half of the evolution.  If w is trending upward (dw/da > 0
    on average) the model is thawing; if trending downward it is freezing.

    Parameters
    ----------
    w_array : NDArray
        Equation of state values w(a).
    a_array : NDArray
        Corresponding scale factor values.

    Returns
    -------
    str
        Either "thawing" or "freezing".
    """
    w = np.asarray(w_array, dtype=float)
    a = np.asarray(a_array, dtype=float)

    # Use the latter half where the dark energy dynamics are most relevant
    n = len(w)
    half = n // 2
    w_late = w[half:]
    a_late = a[half:]

    if len(a_late) < 2:
        return "thawing"

    # Linear fit to w(a) over the late-time segment
    coeffs = np.polyfit(a_late, w_late, 1)
    slope = coeffs[0]

    # Positive slope => w increases with a => thawing (moving away from -1)
    # Negative slope => w decreases with a => freezing (approaching -1)
    if slope > 0:
        return "thawing"
    else:
        return "freezing"


def _cpl_form(a: float | NDArray, w0: float, wa: float) -> float | NDArray:
    """CPL parameterisation: w(a) = w0 + wa * (1 - a)."""
    return w0 + wa * (1.0 - a)


def w0_wa_fit(w_array: NDArray, a_array: NDArray) -> tuple[float, float]:
    """Fit the CPL (Chevallier-Polarski-Linder) parameterisation.

    w(a) = w0 + wa * (1 - a)

    Parameters
    ----------
    w_array : NDArray
        Equation of state values.
    a_array : NDArray
        Scale factor values.

    Returns
    -------
    tuple (w0, wa)
        Best-fit CPL parameters.
    """
    w = np.asarray(w_array, dtype=float)
    a = np.asarray(a_array, dtype=float)

    # Filter out any NaN values
    mask = np.isfinite(w) & np.isfinite(a)
    w_clean = w[mask]
    a_clean = a[mask]

    if len(a_clean) < 2:
        return -1.0, 0.0

    try:
        popt, _ = curve_fit(_cpl_form, a_clean, w_clean, p0=[-0.9, 0.1])
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        # Fallback: simple least-squares on the linear form
        # w = w0 + wa - wa*a  =>  w = c0 + c1 * a  with c0 = w0+wa, c1 = -wa
        coeffs = np.polyfit(a_clean, w_clean, 1)
        c1 = coeffs[0]
        c0 = coeffs[1]
        wa = -c1
        w0 = c0 - wa
        return float(w0), float(wa)


def tracker_test(
    V_func: Callable,
    n_points: int = 200,
    phi_range: tuple[float, float] = (0.1, 10.0),
) -> dict:
    """Test whether a potential admits a tracker solution.

    A tracker solution exists when the potential satisfies certain
    conditions on the "gamma" parameter:
        Gamma = V * d2V / (dV)^2
    If Gamma > 1 and slowly varying, the potential admits a tracker
    (attractor) solution that is insensitive to initial conditions.

    Parameters
    ----------
    V_func : callable
        Potential object with V(phi), dV(phi), d2V(phi) methods.
    n_points : int
        Number of phi values to evaluate.
    phi_range : tuple
        Range of phi values to test.

    Returns
    -------
    dict with keys:
        'phi' : NDArray  - field values tested
        'Gamma' : NDArray  - Gamma = V * d2V / (dV)^2
        'is_tracker' : bool  - True if Gamma > 1 and roughly constant
        'Gamma_mean' : float  - mean value of Gamma
        'Gamma_std' : float  - standard deviation of Gamma (constancy check)
    """
    phi = np.linspace(phi_range[0], phi_range[1], n_points)

    V_vals = V_func.V(phi)
    dV_vals = V_func.dV(phi)
    d2V_vals = V_func.d2V(phi)

    # Gamma = V * d2V / (dV)^2
    with np.errstate(divide="ignore", invalid="ignore"):
        Gamma = np.where(
            np.abs(dV_vals) > 1e-30,
            V_vals * d2V_vals / dV_vals**2,
            1.0,
        )

    # Filter out non-finite values
    finite_mask = np.isfinite(Gamma)
    Gamma_finite = Gamma[finite_mask]

    if len(Gamma_finite) == 0:
        return {
            "phi": phi,
            "Gamma": Gamma,
            "is_tracker": False,
            "Gamma_mean": 1.0,
            "Gamma_std": 0.0,
        }

    Gamma_mean = float(np.mean(Gamma_finite))
    Gamma_std = float(np.std(Gamma_finite))

    # Tracker condition: Gamma > 1 and roughly constant (std / mean small)
    is_tracker = bool(Gamma_mean > 1.0 and Gamma_std / max(abs(Gamma_mean), 1e-30) < 0.5)

    return {
        "phi": phi,
        "Gamma": Gamma,
        "is_tracker": is_tracker,
        "Gamma_mean": Gamma_mean,
        "Gamma_std": Gamma_std,
    }
