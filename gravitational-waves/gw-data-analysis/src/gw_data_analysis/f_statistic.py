"""F-statistic for continuous gravitational wave searches.

Implements the F-statistic used in searches for persistent,
nearly-monochromatic gravitational wave signals from spinning neutron stars.
"""

import numpy as np

from .matched_filter import inner_product


def f_statistic(data, freq, theta, phi, psi, phi0, psd_func, freqs):
    """Compute the F-statistic for a continuous gravitational wave signal.

    The F-statistic is the log-likelihood ratio maximized analytically over
    the four amplitude parameters (h0, cos(iota), phi0, psi). Here we
    compute a simplified version for a monochromatic signal at a known
    frequency with given sky position and orientation parameters.

    Parameters
    ----------
    data : array_like
        Complex frequency-domain data.
    freq : float
        Expected signal frequency in Hz.
    theta : float
        Polar angle of source on the sky (colatitude) in radians.
    phi : float
        Azimuthal angle of source on the sky in radians.
    psi : float
        Polarization angle in radians.
    phi0 : float
        Initial phase in radians.
    psd_func : callable
        Function f(freqs) -> PSD values.
    freqs : array_like
        Frequency array in Hz.

    Returns
    -------
    f_stat : float
        Value of the F-statistic (2x log-likelihood ratio).
    """
    data = np.asarray(data, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)

    # Find the frequency bin closest to the target frequency
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    idx = int(round(freq / df))
    idx = min(max(idx, 0), len(freqs) - 1)

    # Build antenna pattern functions (simplified for single detector)
    # F+ and Fx for a given sky position and polarization
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_2phi = np.cos(2.0 * phi)
    sin_2phi = np.sin(2.0 * phi)
    cos_2psi = np.cos(2.0 * psi)
    sin_2psi = np.sin(2.0 * psi)

    # Simplified antenna patterns for an interferometer at zenith
    # (In practice, these depend on detector orientation and time)
    f_plus = 0.5 * (1.0 + cos_theta**2) * cos_2phi * cos_2psi \
             - cos_theta * sin_2phi * sin_2psi
    f_cross = 0.5 * (1.0 + cos_theta**2) * cos_2phi * sin_2psi \
              + cos_theta * sin_2phi * cos_2psi

    # Monochromatic signal template at the target frequency bin
    # h(t) = F+ * h+(t) + Fx * hx(t), with h+(t) = A cos(2*pi*f*t + phi0)
    # In frequency domain this is concentrated at f = freq
    template = np.zeros_like(data)

    psd_val = psd_func(freqs)
    psd_at_freq = psd_val[idx]

    if not np.isfinite(psd_at_freq) or psd_at_freq <= 0:
        return 0.0

    # The F-statistic in the simplest form is 2 * |data_at_f|^2 / S_n(f)
    # maximized over amplitude and phase
    signal_norm = f_plus**2 + f_cross**2

    if signal_norm <= 0:
        return 0.0

    # Data at the frequency bin
    data_at_f = data[idx]

    # F-statistic: 2 * |X|^2 / S_n where X is the matched filter output
    # For a monochromatic signal, this simplifies to:
    # F = (2 / S_n(f)) * |sum over antenna-pattern-weighted data|^2
    f_stat = (2.0 / psd_at_freq) * abs(data_at_f)**2 * signal_norm

    return float(f_stat)


def cwb_snr(data, freq, psd_func, freqs):
    """Approximate SNR for continuous wave signals.

    Computes an approximate SNR by comparing the data power at the
    target frequency to the expected noise floor.

    Parameters
    ----------
    data : array_like
        Complex frequency-domain data.
    freq : float
        Target frequency in Hz.
    psd_func : callable
        Function f(freqs) -> PSD values.
    freqs : array_like
        Frequency array in Hz.

    Returns
    -------
    snr : float
        Approximate SNR for the continuous wave signal.
    """
    data = np.asarray(data, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)

    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    idx = int(round(freq / df))
    idx = min(max(idx, 0), len(freqs) - 1)

    psd_val = psd_func(freqs)
    psd_at_freq = psd_val[idx]

    if not np.isfinite(psd_at_freq) or psd_at_freq <= 0:
        return 0.0

    # SNR^2 ~ 4 * |data(f)|^2 * df / S_n(f)
    snr_sq = 4.0 * abs(data[idx])**2 * df / psd_at_freq

    return float(np.sqrt(max(0.0, snr_sq)))
