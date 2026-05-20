from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FREF_HZ = 1.0 / (365.25 * 24.0 * 3600.0)


@dataclass(frozen=True)
class PTAData:
    frequency_hz: np.ndarray
    strain: np.ndarray
    sigma: np.ndarray


def power_law_strain(frequency_hz: np.ndarray, amplitude: float, gamma: float) -> np.ndarray:
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    return amplitude * (frequency_hz / FREF_HZ) ** ((3.0 - gamma) / 2.0)


def gaussian_loglike(data: PTAData, amplitude: float, gamma: float) -> float:
    model = power_law_strain(data.frequency_hz, amplitude, gamma)
    residual = (data.strain - model) / data.sigma
    return float(-0.5 * np.sum(residual * residual))


def spectral_slope_label(gamma: float) -> str:
    if abs(gamma - 13.0 / 3.0) < 0.15:
        return "smbh-binary-like"
    if gamma > 5.0:
        return "red-early-universe-like"
    return "blue-or-flat-source-like"
