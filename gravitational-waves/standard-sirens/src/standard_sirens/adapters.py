from __future__ import annotations

import csv
import json
from pathlib import Path

from .inference import SirenEvent


def load_posterior_summary(path: str | Path) -> list[SirenEvent]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        rows = payload["events"] if isinstance(payload, dict) else payload
    else:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    return [
        SirenEvent(
            str(row["name"]),
            float(row["redshift"]),
            float(row["luminosity_distance_mpc"]),
            float(row["distance_sigma_mpc"]),
        )
        for row in rows
    ]
