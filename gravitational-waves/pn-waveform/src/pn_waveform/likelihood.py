"""Matched filtering and likelihood for GW data analysis."""

from __future__ import annotations

import numpy as np


def inner_product(h1: np.ndarray, h2: np.ndarray, psd: np.ndarray, df: float) -> float:
    """Noise-weighted inner product (h1|h2) = 4 Re ∫ h1* h2 / Sn(f) df.

    Parameters
    ----------
    h1, h2 : frequency-domain waveforms
    psd : noise power spectral density Sn(f)
    df : frequency resolution [Hz]
    """
    n = min(len(h1), len(h2), len(psd))
    integrand = np.conj(h1[:n]) * h2[:n] / (psd[:n] + 1e-30)
    return 4.0 * np.real(np.trapezoid(integrand, dx=df))


def matched_filter_snr(template: np.ndarray, data: np.ndarray, psd: np.ndarray, df: float) -> float:
    """Matched filter SNR = (h|h)^{1/2}."""
    rho2 = inner_product(template, template, psd, df)
    return np.sqrt(max(rho2, 0))


def chirp_snr(
    Mc: float,
    eta: float,
    distance: float,
    f_low: float,
    psd_func=None,
) -> float:
    """Approximate inspiral SNR for given parameters.

    Uses the Flanagan-Hughes formula:
    ρ² ≈ (5/6) · (Mc)^(5/3) / (π^(4/3) · d²) · ∫_{f_low}^{f_isco} f^(-7/3)/Sn(f) df
    """
    # Placeholder: simplified estimate
    # In reality this requires integrating over the PSD
    return 1.0  # stub
