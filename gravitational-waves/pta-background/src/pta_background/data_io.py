from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import PTAData


def load_binned_spectrum(path: str | Path) -> PTAData:
    payload = json.loads(Path(path).read_text())
    return PTAData(
        np.asarray(payload["frequency_hz"], dtype=float),
        np.asarray(payload["strain"], dtype=float),
        np.asarray(payload["sigma"], dtype=float),
    )


def load_phase_transition_spectrum(path: str | Path) -> dict[str, np.ndarray | str | float]:
    payload = json.loads(Path(path).read_text())
    return {
        "model": payload.get("model", "unknown"),
        "frequency_hz": np.asarray(payload["frequency_hz"], dtype=float),
        "omega_gw": np.asarray(payload["omega_gw"], dtype=float),
        "alpha": float(payload.get("alpha", payload.get("parameters", {}).get("alpha", 0.0))),
        "beta_over_h": float(payload.get("beta_over_h", payload.get("parameters", {}).get("beta_over_h", 0.0))),
        "temperature_gev": float(payload.get("temperature_gev", payload.get("parameters", {}).get("temperature_gev", 0.0))),
    }
