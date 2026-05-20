"""Matched filtering for gravitational wave detection.

Implements noise-weighted inner products and signal-to-noise ratio
calculations following the standard LIGO/Virgo formalism.
"""

import numpy as np


def inner_product(h1, h2, psd_func, freqs):
    """Compute the noise-weighted inner product (h1|h2).

    .. math::
        (h_1|h_2) = 4 \\operatorname{Re} \\int_0^\\infty
                     \\frac{\\tilde{h}_1^*(f) \\tilde{h}_2(f)}{S_n(f)} \\, df

    The integral is approximated as a Riemann sum over the given
    frequency array.

    Parameters
    ----------
    h1, h2 : array_like
        Complex frequency-domain representations of the signals.
        Must have the same length as `freqs`.
    psd_func : callable
        Function f(freqs) -> PSD values.
    freqs : array_like
        Frequency array in Hz. Must be monotonically increasing.

    Returns
    -------
    ip : float
        The noise-weighted inner product (h1|h2), which is real-valued.
    """
    h1 = np.asarray(h1, dtype=np.complex128)
    h2 = np.asarray(h2, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)

    psd = psd_func(freqs)

    # df from frequency spacing
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # Avoid division by zero / inf: mask out pathological bins
    valid = np.isfinite(psd) & (psd > 0)

    integrand = np.zeros(len(freqs), dtype=np.complex128)
    integrand[valid] = (np.conj(h1[valid]) * h2[valid]) / psd[valid]

    ip = 4.0 * np.real(np.sum(integrand) * df)
    return float(ip)


def matched_filter_snr(template, data, psd_func, freqs):
    """Compute the matched filter signal-to-noise ratio.

    SNR = (template|data) / sqrt((template|template))

    This is the SNR obtained by filtering the data with the template.

    Parameters
    ----------
    template : array_like
        Complex frequency-domain template waveform.
    data : array_like
        Complex frequency-domain data.
    psd_func : callable
        Function f(freqs) -> PSD values.
    freqs : array_like
        Frequency array in Hz.

    Returns
    -------
    snr : float
        The matched filter SNR.
    """
    cross = inner_product(template, data, psd_func, freqs)
    auto = inner_product(template, template, psd_func, freqs)

    if auto <= 0:
        return 0.0

    return abs(cross) / np.sqrt(auto)


def optimal_snr(h, psd_func, freqs):
    """Compute the optimal signal-to-noise ratio for a known signal.

    .. math::
        \\rho_{\\mathrm{opt}} = \\sqrt{(h|h)}

    Parameters
    ----------
    h : array_like
        Complex frequency-domain signal waveform.
    psd_func : callable
        Function f(freqs) -> PSD values.
    freqs : array_like
        Frequency array in Hz.

    Returns
    -------
    snr : float
        The optimal SNR.
    """
    auto = inner_product(h, h, psd_func, freqs)
    return np.sqrt(max(0.0, auto))
