from __future__ import annotations

import json
from pathlib import Path

from .metrics import Constraint, tension_summary


def load_constraints(path: str | Path) -> list[Constraint]:
    payload = json.loads(Path(path).read_text())
    rows = payload["constraints"] if isinstance(payload, dict) else payload
    return [Constraint(str(row["label"]), float(row["mean"]), float(row["sigma"])) for row in rows]


def constraint_by_label(constraints: list[Constraint], label: str) -> Constraint:
    for constraint in constraints:
        if constraint.label == label:
            return constraint
    raise KeyError(label)


def grouped_tension_report(path: str | Path) -> dict[str, object]:
    constraints = load_constraints(path)
    local = constraint_by_label(constraints, "local-distance-ladder")
    early = constraint_by_label(constraints, "early-universe")
    standard_siren = constraint_by_label(constraints, "standard-siren")
    return {
        "inputs": [constraint.__dict__ for constraint in constraints],
        "local_vs_early": tension_summary(local, early),
        "siren_vs_early": tension_summary(standard_siren, early),
        "siren_vs_local": tension_summary(standard_siren, local),
    }
