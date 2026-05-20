"""
Schwinger-Keldysh (in-in) formalism for computing the 3-point function.

The in-in formalism computes expectation values at time t starting from
the Bunch-Davies vacuum at t -> -infinity:

    <zeta^3(t)> = < [T-bar exp(-i int H_int)] zeta^3(0) [T exp(-i int H_int)] ] >

At first order in H_int this gives:

    <zeta^3> ~ int dtau  G_k(tau) F(k1, k2, k3, tau)

where G_k is a bulk-to-boundary propagator and F encodes the interaction
vertex.
"""

import numpy as np
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Background quantities
# ---------------------------------------------------------------------------

def hubble_parameter(V_star=1e-10, M_pl=1.0):
    """Hubble parameter during inflation.

    H^2 ~ V / (3 M_pl^2).  Working in reduced Planck units M_pl = 1.

    Parameters
    ----------
    V_star : float
        Inflationary potential energy scale.
    M_pl : float
        Reduced Planck mass.

    Returns
    -------
    float
        Hubble parameter.
    """
    return np.sqrt(V_star / (3.0 * M_pl**2))


def slow_roll_epsilon(V, dV, ddV, M_pl=1.0):
    """First slow-roll parameter epsilon_V.

    epsilon_V = (M_pl^2 / 2) * (V' / V)^2
    """
    return (M_pl**2 / 2.0) * (dV / V) ** 2


def slow_roll_eta(V, dV, ddV, M_pl=1.0):
    """Second slow-roll parameter eta_V.

    eta_V = M_pl^2 * V'' / V
    """
    return M_pl**2 * ddV / V


# ---------------------------------------------------------------------------
# Bulk-to-boundary propagator and Green's function
# ---------------------------------------------------------------------------

def bulk_to_boundary_propagator(k, tau, k_t=1.0):
    """Bulk-to-boundary propagator for scalar perturbations in de Sitter.

    G_k(tau) = (1 - i k tau) exp(i k tau)   (for the +i branch)

    In conformal time tau < 0 (running from -inf to 0).

    Parameters
    ----------
    k : float
        Wavenumber.
    tau : float or ndarray
        Conformal time (negative).
    k_t : float
        Normalization scale (default 1).

    Returns
    -------
    complex or ndarray
    """
    x = k * tau
    return (1.0 - 1j * x) * np.exp(1j * x)


def bulk_propagator(k, tau, tau_prime):
    """Bulk-to-bulk propagator (retarded).

    G_k(tau, tau') = (1/k^3) * [ (1 + i k tau') exp(-i k tau')
                                   - (1 + i k tau) exp(-i k tau) ]
                     for tau > tau'

    Parameters
    ----------
    k : float
        Wavenumber.
    tau : float
        Late conformal time.
    tau_prime : float
        Early conformal time.

    Returns
    -------
    complex
    """
    x = k * tau
    xp = k * tau_prime
    return (1.0 / k**3) * ((1.0 + 1j * xp) * np.exp(-1j * xp)
                             - (1.0 + 1j * x) * np.exp(-1j * x))


# ---------------------------------------------------------------------------
# Interaction Hamiltonian kernels
# ---------------------------------------------------------------------------

def cubic_interaction_kernel_local(k1, k2, k3, tau, epsilon=0.01):
    """Cubic interaction kernel for the local-type bispectrum.

    In the effective field theory of inflation the leading cubic interaction
    that produces a local shape comes from the eta-epsilon correction:

    H_int ~ epsilon * eta * int d^3x [ zeta'^3,  zeta^2 zeta', ... ]

    Simplified kernel:
    F(k1, k2, k3, tau) ~ epsilon * sum_perms (k1 k2 / k3) * (k_t tau)^2 * exp(-k_t tau)

    where k_t = k1 + k2 + k3.

    Parameters
    ----------
    k1, k2, k3 : float
        Wavenumbers.
    tau : float
        Conformal time (negative).
    epsilon : float
        Slow-roll parameter.

    Returns
    -------
    complex
    """
    k_t = k1 + k2 + k3
    # Simple model kernel peaked at horizon crossing k*tau ~ -1
    x_t = k_t * tau
    kernel = epsilon * (k1 * k2 / k3 + k1 * k3 / k2 + k2 * k3 / k1)
    kernel *= x_t**2 * np.exp(-x_t)
    return kernel


def cubic_interaction_kernel_equilateral(k1, k2, k3, tau, c_s=1.0, epsilon=0.01):
    """Cubic interaction kernel for equilateral-type bispectrum.

    Arises from higher-derivative operators (e.g. (del zeta)^2 zeta')
    whose interactions are strongest when all three modes cross the
    horizon at the same time.

    F ~ epsilon / c_s^2 * k1^2 k2^2 k3^2 * tau^4 * exp(-k_t tau)

    Parameters
    ----------
    k1, k2, k3 : float
        Wavenumbers.
    tau : float
        Conformal time (negative).
    c_s : float
        Sound speed (c_s < 1 enhances equilateral).
    epsilon : float
        Slow-roll parameter.

    Returns
    -------
    complex
    """
    k_t = k1 + k2 + k3
    x_t = k_t * tau
    kernel = epsilon / c_s**2 * (k1 * k2 * k3)**2
    kernel *= x_t**4 * np.exp(-x_t)
    return kernel


# ---------------------------------------------------------------------------
# In-in integral
# ---------------------------------------------------------------------------

def in_in_integral(k1, k2, k3, interaction_kernel, tau_end=-0.01,
                   tau_start=-1000.0, **kernel_kwargs):
    """Evaluate the first-order in-in integral for the bispectrum.

    B(k1, k2, k3) = -2 * Im  int_{tau_start}^{tau_end} dtau
                         G_{k1}(tau) G_{k2}(tau) G_{k3}(tau)
                         * F(k1, k2, k3, tau)

    The integral runs over conformal time from the far past (tau_start < 0)
    to some late-time cutoff near tau -> 0.

    Parameters
    ----------
    k1, k2, k3 : float
        Wavenumbers forming the triangle.
    interaction_kernel : callable
        F(k1, k2, k3, tau, **kwargs) returning the interaction vertex.
    tau_end : float
        Late-time cutoff (close to 0, but negative).
    tau_start : float
        Early-time start of integration (large negative number).
    **kernel_kwargs
        Additional keyword arguments passed to the interaction kernel.

    Returns
    -------
    float
        Real bispectrum amplitude B(k1, k2, k3).
    """
    def integrand_real(tau):
        G1 = bulk_to_boundary_propagator(k1, tau)
        G2 = bulk_to_boundary_propagator(k2, tau)
        G3 = bulk_to_boundary_propagator(k3, tau)
        F = interaction_kernel(k1, k2, k3, tau, **kernel_kwargs)
        # -2 * Im[ G1 * G2 * G3 * F ]
        product = G1 * G2 * G3 * F
        return -2.0 * np.imag(product)

    def integrand_imag(tau):
        G1 = bulk_to_boundary_propagator(k1, tau)
        G2 = bulk_to_boundary_propagator(k2, tau)
        G3 = bulk_to_boundary_propagator(k3, tau)
        F = interaction_kernel(k1, k2, k3, tau, **kernel_kwargs)
        product = G1 * G2 * G3 * F
        return -2.0 * np.real(product)

    result_real, err_real = quad(integrand_real, tau_start, tau_end,
                                 limit=200, epsabs=1e-12, epsrel=1e-10)
    return result_real


def compute_bispectrum_in_in(k1, k2, k3, shape="local", **kwargs):
    """Compute the bispectrum using the in-in formalism.

    Parameters
    ----------
    k1, k2, k3 : float
        Wavenumbers.
    shape : str
        'local' or 'equilateral'.
    **kwargs
        Passed to the interaction kernel.

    Returns
    -------
    float
        Bispectrum B(k1, k2, k3).
    """
    kernels = {
        "local": cubic_interaction_kernel_local,
        "equilateral": cubic_interaction_kernel_equilateral,
    }
    if shape not in kernels:
        raise ValueError(f"Shape '{shape}' not supported for in-in. "
                         f"Choose from {list(kernels)}")

    # Choose integration range appropriate for the wavenumbers.
    # The integrand is exponentially suppressed for tau << -1/k_t,
    # so we set tau_start accordingly.
    k_t = k1 + k2 + k3
    tau_start = min(-1000.0, -50.0 / k_t)
    tau_end = -0.001 / k_t  # close to zero but not exactly zero

    return in_in_integral(k1, k2, k3, kernels[shape],
                          tau_end=tau_end, tau_start=tau_start, **kwargs)
