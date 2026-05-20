from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class Constraint:
    label: str
    mean: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


def tension_sigma(a: Constraint, b: Constraint) -> float:
    return abs(a.mean - b.mean) / sqrt(a.sigma * a.sigma + b.sigma * b.sigma)


def combined_constraint(label: str, constraints: list[Constraint]) -> Constraint:
    if not constraints:
        raise ValueError("at least one constraint is required")
    weights = np.array([1.0 / c.sigma**2 for c in constraints], dtype=float)
    means = np.array([c.mean for c in constraints], dtype=float)
    mean = float(np.sum(weights * means) / np.sum(weights))
    sigma = float(1.0 / sqrt(np.sum(weights)))
    return Constraint(label, mean, sigma)


def tension_summary(local: Constraint, early: Constraint) -> dict[str, float | str]:
    return {
        "local": local.label,
        "early": early.label,
        "delta_h0": local.mean - early.mean,
        "sigma": tension_sigma(local, early),
    }
