from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .models import Cosmology


Prediction = Callable[[Cosmology], np.ndarray]


@dataclass(frozen=True)
class GaussianBlock:
    name: str
    observed: np.ndarray
    covariance: np.ndarray
    predict: Prediction

    def chi2(self, cosmo: Cosmology) -> float:
        residual = np.asarray(self.observed, dtype=float) - np.asarray(self.predict(cosmo), dtype=float)
        inv_cov = np.linalg.inv(np.asarray(self.covariance, dtype=float))
        return float(residual.T @ inv_cov @ residual)


@dataclass(frozen=True)
class JointLikelihood:
    blocks: tuple[GaussianBlock, ...]

    def chi2(self, cosmo: Cosmology) -> float:
        return float(sum(block.chi2(cosmo) for block in self.blocks))

    def loglike(self, cosmo: Cosmology) -> float:
        return -0.5 * self.chi2(cosmo)


def grid_search(likelihood: JointLikelihood, candidates: Iterable[Cosmology]) -> tuple[Cosmology, float]:
    scored = [(candidate, likelihood.chi2(candidate)) for candidate in candidates]
    if not scored:
        raise ValueError("grid_search requires at least one candidate")
    return min(scored, key=lambda item: item[1])
