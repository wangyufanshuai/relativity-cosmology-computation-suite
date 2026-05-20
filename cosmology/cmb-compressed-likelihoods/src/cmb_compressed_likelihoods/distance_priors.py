from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistancePrior:
    labels: tuple[str, ...]
    mean: np.ndarray
    covariance: np.ndarray

    def chi2(self, prediction: np.ndarray) -> float:
        return gaussian_chi2(np.asarray(prediction, dtype=float), self.mean, self.covariance)


def gaussian_chi2(prediction: np.ndarray, observed: np.ndarray, covariance: np.ndarray) -> float:
    residual = np.asarray(prediction, dtype=float) - np.asarray(observed, dtype=float)
    inv_cov = np.linalg.inv(np.asarray(covariance, dtype=float))
    return float(residual.T @ inv_cov @ residual)


def planck_like_distance_prior() -> DistancePrior:
    return DistancePrior(
        labels=("shift_parameter_R", "acoustic_scale_lA", "omega_b_h2"),
        mean=np.array([1.749, 301.47, 0.02237]),
        covariance=np.diag([0.0049**2, 0.09**2, 0.00015**2]),
    )
