"""
Non-Gaussian signatures in CMB and Large-Scale Structure (LSS).

Implements observational consequences of primordial non-Gaussianity (PNG):

- Scale-dependent bias in the halo power spectrum
- CMB bispectrum (angular bispectrum via reduced bispectrum formalism)
- Squeezed-limit consistency relation
"""

import numpy as np
from scipy.special import spherical_jn


# ---------------------------------------------------------------------------
# Scale-dependent bias from local PNG
# ---------------------------------------------------------------------------

def scale_dependent_bias(f_nl, k, k_pivot=0.05, b_G=1.0, delta_c=1.686,
                         Omega_m=0.315, z=0.0):
    """Scale-dependent halo bias correction from local PNG.

    In the presence of local-type non-Gaussianity, the halo bias acquires a
    scale-dependent correction (Dalal et al. 2008, Slosar et al. 2008):

        Delta b(k) = f_NL * delta_c * (b_G - 1) * 3 * Omega_m * H0^2
                     / (c^2 * k^2 * T(k) * k_pivot^2)

    In simplified dimensionless form with transfer function T(k) ~ 1
    on large scales:

        Delta b(k) ~ f_NL * (b_G - 1) * const * (k / k_pivot)^{-2}

    Parameters
    ----------
    f_nl : float
        Local non-linearity parameter.
    k : float or ndarray
        Wavenumber in h/Mpc.
    k_pivot : float
        Pivot scale.
    b_G : float
        Gaussian (scale-independent) bias.
    delta_c : float
        Critical overdensity for spherical collapse.
    Omega_m : float
        Matter density fraction.
    z : float
        Redshift.

    Returns
    -------
    float or ndarray
        Total bias b(k) = b_G + Delta b(k).
    """
    # Simplified constant: absorb cosmology into a single factor
    # In full calculation: const = 3 * Omega_m * (H0/c)^2 / k_pivot^2
    # Here we use a normalized version
    const = 2.0  # O(1) normalization matching literature conventions

    delta_b = f_nl * (b_G - 1.0) * const * delta_c * (k / k_pivot) ** (-2)

    # At very high k the correction should not diverge; apply a smooth cutoff
    # (transfer function suppression)
    if np.isscalar(k):
        k_arr = np.array([k])
    else:
        k_arr = np.asarray(k, dtype=float)

    # Suppress at high k (small scales) where T(k) ~ 0
    k_eq = 0.01  # equality scale ~0.01 h/Mpc
    T_k = 1.0 / (1.0 + (k_arr / k_eq) ** 2)  # simplified transfer function

    delta_b_arr = np.asarray(delta_b) * T_k

    if np.isscalar(k):
        return b_G + float(delta_b_arr[0])
    return b_G + delta_b_arr


def bias_correction(f_nl, k, k_pivot=0.05, b_G=1.0):
    """Return only the correction Delta b(k) (without Gaussian part).

    Parameters
    ----------
    f_nl : float
        Local f_NL.
    k : float or ndarray
        Wavenumber.
    k_pivot : float
        Pivot scale.
    b_G : float
        Gaussian bias.

    Returns
    -------
    float or ndarray
    """
    b_total = scale_dependent_bias(f_nl, k, k_pivot, b_G)
    if np.isscalar(b_total):
        return b_total - b_G
    return b_total - b_G


# ---------------------------------------------------------------------------
# CMB angular bispectrum
# ---------------------------------------------------------------------------

def reduced_cmb_bispectrum(l1, l2, l3, B_primordial, r_lss=14.0,
                            k_pivot=0.05, dk=0.01):
    """Compute the CMB reduced angular bispectrum b_{l1 l2 l3} from the
    primordial bispectrum.

    b_{l1 l2 l3} = (2/pi)^3 int dk1 dk2 dk3 k1^2 k2^2 k3^2
                       B(k1, k2, k3) * j_{l1}(k1 r) j_{l2}(k2 r) j_{l3}(k3 r)

    This is a simplified single-evaluation version.  A full computation
    would use a 3-d quadrature; here we evaluate the integrand at a
    representative set of k values to give a qualitative answer.

    Parameters
    ----------
    l1, l2, l3 : int
        Multipole moments.
    B_primordial : callable
        B(k1, k2, k3) -> float, the primordial bispectrum.
    r_lss : float
        Comoving distance to last-scattering surface (Gpc).
    k_pivot : float
        Characteristic scale.
    dk : float
        Step size for numerical integration (simplified).

    Returns
    -------
    float
        Estimated reduced bispectrum b_{l1 l2 l3}.
    """
    # Use a discrete approximation: sample at N points
    N = 20
    k_min = 1e-4
    k_max = 0.3  # / Mpc
    k_vals = np.linspace(k_min, k_max, N)
    dk_step = (k_max - k_min) / N

    result = 0.0
    pre = (2.0 / np.pi) ** 3

    for i in range(N):
        k1 = k_vals[i]
        j1 = spherical_jn(l1, k1 * r_lss)
        for j in range(N):
            k2 = k_vals[j]
            j2 = spherical_jn(l2, k2 * r_lss)
            for m in range(N):
                k3 = k_vals[m]
                j3 = spherical_jn(l3, k3 * r_lss)
                B_val = B_primordial(k1, k2, k3)
                result += (k1**2 * k2**2 * k3**2 * B_val
                           * j1 * j2 * j3 * dk_step**3)

    return pre * result


# ---------------------------------------------------------------------------
# Squeezed-limit relation
# ---------------------------------------------------------------------------

def squeezed_limit_bispectrum(f_nl, P_k, P_ks):
    """Squeezed-limit bispectrum: B(k, k_s, k_s) for k_s << k.

    From the consistency relation:

        B(k, k_s, k_s) -> (12/5) * f_NL * P(k) * P(k_s)

    Parameters
    ----------
    f_nl : float
        Local non-linearity parameter.
    P_k : float
        Power spectrum at the long-wavelength mode P(k).
    P_ks : float
        Power spectrum at the short-wavelength mode P(k_s).

    Returns
    -------
    float
        Squeezed-limit bispectrum amplitude.
    """
    return (12.0 / 5.0) * f_nl * P_k * P_ks


def squeezed_limit_fnl(B_squeezed, P_k, P_ks):
    """Infer f_NL from a squeezed-limit bispectrum measurement.

    f_NL = 5/12 * B(k, k_s, k_s) / [P(k) P(k_s)]

    Parameters
    ----------
    B_squeezed : float
        Measured squeezed-limit bispectrum.
    P_k : float
        Power at long wavelength.
    P_ks : float
        Power at short wavelength.

    Returns
    -------
    float
    """
    denominator = P_k * P_ks
    if abs(denominator) < 1e-30:
        return 0.0
    return (5.0 / 12.0) * B_squeezed / denominator


# ---------------------------------------------------------------------------
# C_l from power spectrum (helper for bispectrum normalization)
# ---------------------------------------------------------------------------

def cmb_cl_from_power_spectrum(l, P_func, r_lss=14.0):
    """Compute C_l from the primordial power spectrum.

    C_l = (2/pi) int dk k^2 P(k) [j_l(k r)]^2

    Parameters
    ----------
    l : int
        Multipole.
    P_func : callable
        P(k) -> float.
    r_lss : float
        Comoving distance to LSS.

    Returns
    -------
    float
    """
    k_min = 1e-4
    k_max = 0.3
    N = 200
    k_vals = np.linspace(k_min, k_max, N)
    dk = (k_max - k_min) / N

    integrand = k_vals**2 * np.array([P_func(k) * spherical_jn(l, k * r_lss)**2
                                       for k in k_vals])
    return (2.0 / np.pi) * np.trapz(integrand, dx=dk)
