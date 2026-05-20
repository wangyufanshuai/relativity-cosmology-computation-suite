from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

C_KM_S = 299792.458


@dataclass(frozen=True)
class SirenEvent:
    name: str
    redshift: float
    luminosity_distance_mpc: float
    distance_sigma_mpc: float

    def __post_init__(self) -> None:
        if self.redshift <= 0 or self.luminosity_distance_mpc <= 0 or self.distance_sigma_mpc <= 0:
            raise ValueError("event redshift and distance quantities must be positive")


def h0_from_event(event: SirenEvent) -> tuple[float, float]:
    h0 = C_KM_S * event.redshift / event.luminosity_distance_mpc
    sigma = h0 * event.distance_sigma_mpc / event.luminosity_distance_mpc
    return float(h0), float(sigma)


def estimate_h0(events: list[SirenEvent]) -> tuple[float, float]:
    if not events:
        raise ValueError("at least one siren event is required")
    estimates = np.array([h0_from_event(event)[0] for event in events])
    sigmas = np.array([h0_from_event(event)[1] for event in events])
    weights = 1.0 / sigmas**2
    mean = float(np.sum(weights * estimates) / np.sum(weights))
    sigma = float(1.0 / sqrt(np.sum(weights)))
    return mean, sigma
