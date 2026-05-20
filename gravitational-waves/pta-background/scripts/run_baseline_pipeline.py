from __future__ import annotations

import json
from pathlib import Path

from pta_background import compare_power_law_sources, load_binned_spectrum


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_binned_spectrum(root / "data" / "toy_binned_spectrum.json")
    candidates = [
        ("smbh-binary", 2.0e-15, 13.0 / 3.0),
        ("cosmic-string-proxy", 1.3e-15, 5.2),
        ("phase-transition-proxy", 8.0e-16, 3.0),
    ]
    output = {"run": "baseline-v1", "models": compare_power_law_sources(data, candidates)}
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "baseline_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
