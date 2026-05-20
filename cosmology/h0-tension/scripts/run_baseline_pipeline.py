from __future__ import annotations

import json
from pathlib import Path

from h0_tension import grouped_tension_report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = grouped_tension_report(root / "data" / "toy_constraints.json")
    output = {"run": "baseline-v1", **report}
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "baseline_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
