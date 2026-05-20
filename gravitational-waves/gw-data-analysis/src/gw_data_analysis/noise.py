"""Noise models for gravitational wave detectors.

Provides analytic PSD models for Advanced LIGO and utilities for generating
colored Gaussian noise time series.
"""

import numpy as np


def advanced_ligo_psd(f):
    """Advanced LIGO design power spectral density (analytic fit).

    Uses the simplified analytic model from LIGO-T0900288:
        S_n(f) = S_0 * [(f_0/f)^4 + 2*(1 + (f/f_0)^2)] / (1 + (f/f_0)^2)

    Below the seismic wall (f < f_seismic), the PSD rises steeply.
    A minimum frequency floor is applied to avoid singularity at f=0.

    Parameters
    ----------
    f : array_like
        Frequencies in Hz. Must be non-negative.

    Returns
    -------
    psd : ndarray
        One-sided power spectral density in Hz^-1.
    """
    f = np.asarray(f, dtype=np.float64)

    # aLIGO design parameters
    f_0 = 215.0        # reference frequency [Hz]
    S_0 = 1.0e-49      # PSD scale [Hz^-1] (roughly ~1e-49 at ~100 Hz)
    f_seismic = 10.0    # seismic wall frequency [Hz]

    psd = np.zeros_like(f)

    # Only compute for positive frequencies
    mask = f > 0
    f_pos = f[mask]

    # Analytic fit for f > 0
    x = f_pos / f_0
    seismic_wall = (f_seismic / f_pos) ** 8  # steep seismic rise
    body = S_0 * ((1.0 / x) ** 4 + 2.0 * (1.0 + x**2)) / (1.0 + x**2)

    psd[mask] = body + S_0 * seismic_wall

    # f = 0: set to a very large but finite value (DC does not carry GW info)
    psd[~mask] = np.inf

    return psd


def white_noise_psd(f, sigma):
    """White noise power spectral density.

    For white noise with standard deviation sigma sampled at rate fs,
    the one-sided PSD is constant: S_n = 2 * sigma^2 * dt.
    Here we return S_n = 2 * sigma^2 for simplicity (independent of fs
    since the user should match the normalization).

    Parameters
    ----------
    f : array_like
        Frequencies in Hz (not used, included for interface consistency).
    sigma : float
        Standard deviation of the white noise.

    Returns
    -------
    psd : ndarray
        One-sided PSD, constant at 2 * sigma^2.
    """
    f = np.asarray(f, dtype=np.float64)
    return np.full_like(f, 2.0 * sigma**2)


def generate_noise(psd_func, duration, dt):
    """Generate a colored Gaussian noise time series.

    Generates white noise in the frequency domain, colors it by the PSD,
    and transforms back to the time domain.

    Parameters
    ----------
    psd_func : callable
        Function f(freqs) -> PSD values.
    duration : float
        Duration of the time series in seconds.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    noise : ndarray
        Noise time series of length int(duration / dt).
    """
    n_samples = int(round(duration / dt))
    fs = 1.0 / dt

    # Frequency array (one-sided: 0, df, 2*df, ..., fs/2)
    freqs = np.fft.rfftfreq(n_samples, d=dt)

    # Get PSD at these frequencies
    psd = psd_func(freqs)

    # Generate white complex Gaussian noise in frequency domain
    # The one-sided FFT has n_freqs = n_samples // 2 + 1 bins
    n_freqs = len(freqs)

    # Random complex coefficients: real and imaginary parts independent N(0,1)
    # Scale by sqrt(PSD * df / 2) for proper noise coloring
    # The factor 1/sqrt(2) accounts for the variance split between real/imag
    df = fs / n_samples

    # Handle inf/nan in PSD (e.g., at f=0)
    psd_safe = np.where(np.isfinite(psd), psd, 0.0)

    amplitude = np.sqrt(psd_safe * df) / np.sqrt(2.0)
    # For f=0 and Nyquist, the factor is sqrt(2) different (real-only)
    amplitude[0] = np.sqrt(psd_safe[0] * df)
    if n_samples % 2 == 0:
        amplitude[-1] = np.sqrt(psd_safe[-1] * df)

    noise_freq = amplitude * (
        np.random.standard_normal(n_freqs) + 1j * np.random.standard_normal(n_freqs)
    )
    # f=0 and Nyquist are real
    noise_freq[0] = amplitude[0] * np.random.standard_normal()
    if n_samples % 2 == 0:
        noise_freq[-1] = amplitude[-1] * np.random.standard_normal()

    # Inverse FFT to get time-domain noise
    noise = np.fft.irfft(noise_freq, n=n_samples)

    return noise
