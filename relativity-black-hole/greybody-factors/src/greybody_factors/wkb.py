"""WKB approximation for computing greybody factors.

The WKB method approximates the greybody factor (transmission coefficient)
by computing the tunneling probability through the effective potential barrier.
This amounts to evaluating the integral of sqrt(V - omega^2) between the
classical turning points where V(r) = omega^2.

Higher-order WKB corrections (beyond the leading-order exponential approximation)
include contributions from the potential derivatives at the turning points,
following the method developed by Iyer and Will (1987) and extended by
Iyer (1987), Konoplya (2003), and others.

References:
    Schutz & Will, ApJ 291, L33 (1985) - 1st order WKB
    Iyer & Will, Phys. Rev. D 35, 3621 (1987) - 3rd order
    Iyer, Phys. Rev. D 35, 3632 (1987) - 3rd order details
    Konoplya, Phys. Rev. D 68, 024018 (2003) - 6th order extension
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .potential import V_eff


def _find_turning_points(V_func, r_min, r_max, omega):
    """Find the two classical turning points where V(r) = omega^2.

    The effective potential for Schwarzschild perturbations rises from 0 at the
    horizon to a maximum and then falls back to 0 at infinity. For omega^2
    below the peak, there are exactly two turning points r1 < r2.

    Parameters
    ----------
    V_func : callable
        The effective potential function V(r).
    r_min : float
        Lower search bound (just outside the horizon).
    r_max : float
        Upper search bound (large radius).
    omega : float
        Frequency.

    Returns
    -------
    tuple of (float, float)
        The inner and outer turning points (r1, r2).

    Raises
    ------
    ValueError
        If turning points cannot be found (omega^2 may exceed peak potential).
    """
    omega2 = omega ** 2

    def diff(r):
        return V_func(r) - omega2

    # Sample to find where V exceeds omega^2
    r_sample = np.linspace(r_min, r_max, 5000)
    v_sample = np.array([V_func(r) for r in r_sample])
    d_sample = v_sample - omega2

    # Find sign changes: d goes from negative to positive to negative
    sign_changes = np.where(np.diff(np.sign(d_sample)))[0]

    if len(sign_changes) < 2:
        raise ValueError(
            f"Cannot find two turning points for omega={omega:.4e}. "
            f"The frequency may exceed the potential barrier peak."
        )

    # First turning point: V crosses omega^2 going up
    r1 = brentq(diff, r_sample[sign_changes[0]], r_sample[sign_changes[0] + 1])
    # Second turning point: V crosses omega^2 going down
    r2 = brentq(diff, r_sample[sign_changes[-1]], r_sample[sign_changes[-1] + 1])

    return r1, r2


def _barrier_integral(V_func, r1, r2, omega):
    """Compute the leading WKB barrier integral.

    K = integral_{r1}^{r2} sqrt(V(r) - omega^2) dr

    Parameters
    ----------
    V_func : callable
        The effective potential function V(r).
    r1 : float
        Inner turning point.
    r2 : float
        Outer turning point.
    omega : float
        Frequency.

    Returns
    -------
    float
        The barrier integral K.
    """
    omega2 = omega ** 2

    def integrand(r):
        val = V_func(r) - omega2
        if val <= 0:
            return 0.0
        return np.sqrt(val)

    result, _ = quad(integrand, r1, r2, limit=200)
    return result


def _potential_derivatives(V_func, r0, dr=1e-6):
    """Compute numerical derivatives of V at a point r0.

    Uses central finite differences for up to 6th derivative.

    Parameters
    ----------
    V_func : callable
        The potential function.
    r0 : float
        Evaluation point.
    dr : float
        Step size for finite differences.

    Returns
    -------
    dict
        Dictionary with keys 'V', 'V1', 'V2', ... 'V6' for V and its
        1st through 6th derivatives evaluated at r0.
    """
    result = {}
    result['V'] = V_func(r0)

    # 1st derivative
    result['V1'] = (V_func(r0 + dr) - V_func(r0 - dr)) / (2 * dr)

    # 2nd derivative
    result['V2'] = (V_func(r0 + dr) - 2 * V_func(r0) + V_func(r0 - dr)) / dr ** 2

    # 3rd derivative
    result['V3'] = (V_func(r0 + 2*dr) - 2*V_func(r0 + dr) +
                     2*V_func(r0 - dr) - V_func(r0 - 2*dr)) / (2 * dr ** 3)

    # 4th derivative
    result['V4'] = (V_func(r0 + 2*dr) - 4*V_func(r0 + dr) + 6*V_func(r0) -
                     4*V_func(r0 - dr) + V_func(r0 - 2*dr)) / dr ** 4

    # 5th derivative
    h = dr
    result['V5'] = (V_func(r0 + 3*h) - 4*V_func(r0 + 2*h) + 5*V_func(r0 + h) -
                     5*V_func(r0 - h) + 4*V_func(r0 - 2*h) - V_func(r0 - 3*h)) / (2 * h ** 5)

    # 6th derivative
    result['V6'] = (V_func(r0 + 3*h) - 6*V_func(r0 + 2*h) + 15*V_func(r0 + h) -
                     20*V_func(r0) + 15*V_func(r0 - h) - 6*V_func(r0 - 2*h) +
                     V_func(r0 - 3*h)) / (2 * h ** 6)

    return result


def _find_potential_peak(V_func, r_min, r_max):
    """Find the radial coordinate where V reaches its maximum.

    Parameters
    ----------
    V_func : callable
        The effective potential function.
    r_min : float
        Lower bound.
    r_max : float
        Upper bound.

    Returns
    -------
    float
        The radius r_peak where V is maximum.
    """
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(lambda r: -V_func(r),
                             bounds=(r_min, r_max), method='bounded')
    return result.x


def wkb_order3(V_func, r_range, omega):
    """Compute greybody factor using 3rd-order WKB approximation.

    The 3rd-order WKB formula (Iyer & Will 1987) for the greybody factor is:

    Gamma = 1 / (1 + exp(2*K_3))

    where K_3 includes the barrier integral K plus 3rd-order correction terms
    involving derivatives of V at the potential peak.

    The 3rd-order WKB expression for K is:

    K_3 = K_0 + correction_3

    with:
    K_0 = integral_{r1}^{r2} sqrt(V - omega^2) dr  (barrier integral)

    correction_3 involves Lambda_i coefficients computed from derivatives
    of the potential at the peak r0.

    Parameters
    ----------
    V_func : callable
        The effective potential V(r).
    r_range : tuple of (float, float)
        The (r_min, r_max) search range for turning points.
    omega : float
        Frequency of the wave.

    Returns
    -------
    float
        Greybody factor Gamma(omega) in [0, 1].
    """
    r_min, r_max = r_range
    omega2 = omega ** 2

    # Check if omega^2 exceeds the peak potential (full transmission)
    r_peak = _find_potential_peak(V_func, r_min, r_max)
    V_peak = V_func(r_peak)

    if omega2 >= V_peak:
        # No barrier: full transmission
        # Use smooth interpolation near the top
        if omega2 > V_peak * 1.01:
            return 1.0

    # Check if omega is too small (effectively zero transmission)
    if omega < 1e-15:
        return 0.0

    try:
        r1, r2 = _find_turning_points(V_func, r_min, r_max, omega)
    except ValueError:
        # No turning points: omega^2 above barrier peak
        return 1.0

    # Leading order barrier integral
    K0 = _barrier_integral(V_func, r1, r2, omega)

    # Compute potential derivatives at the peak
    derivs = _potential_derivatives(V_func, r_peak)
    V0 = derivs['V']
    V2 = derivs['V2']

    if abs(V2) < 1e-30:
        # Nearly flat potential: treat as no barrier
        return 1.0

    Q0 = V0 - omega2
    sqrt_Q0 = np.sqrt(abs(Q0)) if Q0 > 0 else 0.0
    V2_abs = abs(V2)

    # 3rd order WKB correction terms
    # Following Iyer & Will (1987), the correction is:
    # Delta_3 = (Lambda_3 - Lambda_2 + Lambda_1) / (2 * sqrt(2 * |V''|))
    # where Lambda_i involve derivatives of V at the peak

    V4 = derivs['V4']
    V3 = derivs['V3']
    V5 = derivs['V5']
    V6 = derivs['V6']

    # Sigma function: Sigma(r0) = (V''/2) * (r - r0)^2 + ...
    # Lambda coefficients (3rd order, from Iyer & Will)
    if V2_abs > 0:
        eta = V2 ** 2
        # 1st correction
        Lambda_1 = (V4 / (64 * eta)) * (Q0 ** 2 + 18 * Q0 * eta + 65 * eta ** 2 / 4)

        # 2nd correction
        Lambda_2 = (V3 ** 2 / (eta ** 2)) * (
            Q0 ** 2 / 72 + 7 * Q0 * eta / 24 + 115 * eta ** 2 / 64
        )

        # 3rd correction
        Lambda_3 = -(V5 * V3 / (eta ** 2)) * (
            Q0 / 24 + 7 * eta / 16
        ) + (V6 / (1152 * eta)) * (
            Q0 ** 2 + 26 * Q0 * eta + 97 * eta ** 2 / 4
        )

        correction_3 = (Lambda_1 - Lambda_2 + Lambda_3) / np.sqrt(2 * V2_abs)
    else:
        correction_3 = 0.0

    K_total = K0 + correction_3

    if K_total < 0:
        # WKB can give negative K for very high frequencies
        return 1.0

    # Greybody factor from WKB
    Gamma = 1.0 / (1.0 + np.exp(2.0 * K_total))
    return np.clip(Gamma, 0.0, 1.0)


def wkb_order6(V_func, r_range, omega):
    """Compute greybody factor using 6th-order WKB approximation.

    Extends the 3rd-order method with additional correction terms up to
    6th order, following Konoplya (2003). The additional terms involve
    higher derivatives of the potential at the peak.

    For most practical purposes, 3rd order is sufficient. The 6th order
    provides improved accuracy especially near the peak of the potential
    where the WKB expansion converges slowly.

    Parameters
    ----------
    V_func : callable
        The effective potential V(r).
    r_range : tuple of (float, float)
        The (r_min, r_max) search range for turning points.
    omega : float
        Frequency of the wave.

    Returns
    -------
    float
        Greybody factor Gamma(omega) in [0, 1].
    """
    r_min, r_max = r_range
    omega2 = omega ** 2

    r_peak = _find_potential_peak(V_func, r_min, r_max)
    V_peak = V_func(r_peak)

    if omega2 >= V_peak and omega2 > V_peak * 1.01:
        return 1.0

    if omega < 1e-15:
        return 0.0

    try:
        r1, r2 = _find_turning_points(V_func, r_min, r_max, omega)
    except ValueError:
        return 1.0

    K0 = _barrier_integral(V_func, r1, r2, omega)

    derivs = _potential_derivatives(V_func, r_peak)
    V0 = derivs['V']
    V2 = derivs['V2']

    if abs(V2) < 1e-30:
        return 1.0

    Q0 = V0 - omega2
    V2_abs = abs(V2)

    # 3rd order correction (same as wkb_order3)
    V3 = derivs['V3']
    V4 = derivs['V4']
    V5 = derivs['V5']
    V6 = derivs['V6']

    eta = V2 ** 2
    if eta > 0:
        # 3rd order terms
        Lambda_1 = (V4 / (64 * eta)) * (Q0 ** 2 + 18 * Q0 * eta + 65 * eta ** 2 / 4)
        Lambda_2 = (V3 ** 2 / (eta ** 2)) * (
            Q0 ** 2 / 72 + 7 * Q0 * eta / 24 + 115 * eta ** 2 / 64
        )
        Lambda_3 = -(V5 * V3 / (eta ** 2)) * (
            Q0 / 24 + 7 * eta / 16
        ) + (V6 / (1152 * eta)) * (
            Q0 ** 2 + 26 * Q0 * eta + 97 * eta ** 2 / 4
        )
        correction_3 = (Lambda_1 - Lambda_2 + Lambda_3) / np.sqrt(2 * V2_abs)

        # 4th through 6th order corrections
        # These follow the Konoplya (2003) expansion
        # The higher-order terms involve increasingly complex combinations
        # of potential derivatives. We include the dominant contributions.

        # 4th order: proportional to (V3*V4/eta^2, V3^2*V4/eta^3, etc.)
        corr_4_a = -(V3 * V4 / (384 * eta ** 2)) * (Q0 + 11 * eta / 2)
        corr_4_b = (V3 ** 2 * V4 / (576 * eta ** 3)) * (
            Q0 ** 2 / 4 + 9 * Q0 * eta + 21 * eta ** 2
        )
        correction_4 = (corr_4_a + corr_4_b) / np.sqrt(2 * V2_abs)

        # 5th and 6th order: dominant terms only
        corr_5 = (V4 ** 2 / (12288 * eta ** 2)) * (
            3 * Q0 ** 2 + 40 * Q0 * eta + 105 * eta ** 2
        )
        corr_6 = -(V3 * V5 * V4 / (1152 * eta ** 3)) * (
            Q0 + 5 * eta
        )
        correction_56 = (corr_5 + corr_6) / np.sqrt(2 * V2_abs)

        correction_total = correction_3 + correction_4 + correction_56
    else:
        correction_total = 0.0

    K_total = K0 + correction_total

    if K_total < 0:
        return 1.0

    Gamma = 1.0 / (1.0 + np.exp(2.0 * K_total))
    return np.clip(Gamma, 0.0, 1.0)


def greybody_factor_wkb(omega, l, s, M, order=3):
    """Compute the greybody factor Gamma(omega, l, s) using WKB approximation.

    Internally uses geometric units (G = c = 1) where the Schwarzschild
    radius is rs = 2M. The mass M is given in geometric units (M has dimensions
    of length). To convert from SI: M_geo = G * M_SI / c^2.

    Parameters
    ----------
    omega : float
        Frequency in geometric units (omega has dimensions of 1/length).
    l : int
        Angular momentum quantum number.
    s : int
        Spin of the field (0=scalar, 1=EM, 2=gravitational).
    M : float
        Black hole mass in geometric units.
    order : int, optional
        WKB order (3 or 6). Default is 3.

    Returns
    -------
    float
        Greybody factor Gamma(omega, l, s) in [0, 1].
    """
    rs = 2.0 * M

    # For very high frequency, full transmission
    if omega * rs > 50:
        return 1.0

    # For very low frequency with l > 0, essentially no transmission
    # V ~ l(l+1)/r^2 at large r, barrier integral K ~ sqrt(l(l+1)) * ln(V_peak/omega^2)
    # which is enormous for omega << 1
    if omega * rs < 1e-3 and l > 0:
        return 0.0

    # For l=0 and very low omega, there is still some transmission
    # (the potential barrier has finite width for s=0, l=0)
    if omega < 1e-15:
        return 0.0

    # Minimum l values by spin
    if s == 1 and l < 1:
        l = 1
    if s == 2 and l < 2:
        l = 2

    V_func = lambda r: V_eff(r, rs, l, s)
    # For low omega, the outer turning point can be at very large r
    # since V ~ l(l+1)/r^2 at large r, V=omega^2 at r ~ sqrt(l(l+1))/omega
    r_outer = max(rs * 100.0, rs * 10.0 / max(omega * rs, 1e-10))
    r_range = (rs * 1.001, min(r_outer, rs * 1e6))

    if order == 3:
        return wkb_order3(V_func, r_range, omega)
    elif order == 6:
        return wkb_order6(V_func, r_range, omega)
    else:
        raise ValueError(f"WKB order must be 3 or 6, got {order}")
