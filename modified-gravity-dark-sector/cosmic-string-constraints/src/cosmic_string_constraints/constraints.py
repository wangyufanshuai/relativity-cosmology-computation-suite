from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class StringNetwork:
    gmu: float
    loop_size: float = 0.1
    reconnection_probability: float = 1.0

    def __post_init__(self) -> None:
        if self.gmu <= 0 or self.loop_size <= 0 or self.reconnection_probability <= 0:
            raise ValueError("network parameters must be positive")


def amplitude_proxy(network: StringNetwork, normalization: float = 1.0e-15) -> float:
    """Approximate PTA-band strain scaling for ranking model points."""
    return normalization * sqrt(network.gmu / 1.0e-11) * sqrt(0.1 / network.loop_size) / sqrt(
        network.reconnection_probability
    )


def excluded_by_pta_limit(network: StringNetwork, strain_limit: float) -> bool:
    return amplitude_proxy(network) > strain_limit
