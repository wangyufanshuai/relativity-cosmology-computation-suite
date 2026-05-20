from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BAOMeasurement:
    redshift: float
    dv_over_rd: float
    sigma: float


@dataclass(frozen=True)
class SupernovaMeasurement:
    redshift: float
    distance_modulus: float
    sigma: float


def _read_rows(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return list(payload.get("measurements", []))
        return list(payload)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_bao_measurements(path: str | Path) -> list[BAOMeasurement]:
    rows = _read_rows(path)
    return [
        BAOMeasurement(float(row["redshift"]), float(row["dv_over_rd"]), float(row["sigma"]))
        for row in rows
    ]


def load_supernovae(path: str | Path) -> list[SupernovaMeasurement]:
    rows = _read_rows(path)
    return [
        SupernovaMeasurement(float(row["redshift"]), float(row["distance_modulus"]), float(row["sigma"]))
        for row in rows
    ]


def diagonal_covariance(sigmas: list[float] | np.ndarray) -> np.ndarray:
    sigmas = np.asarray(sigmas, dtype=float)
    if np.any(sigmas <= 0):
        raise ValueError("all uncertainties must be positive")
    return np.diag(sigmas * sigmas)
