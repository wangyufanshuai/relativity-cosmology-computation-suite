from __future__ import annotations

import json
from pathlib import Path

from standard_sirens import SirenEvent, estimate_h0


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    events = [
        SirenEvent("toy-a", 0.01, 42.827494, 4.0),
        SirenEvent("toy-b", 0.02, 85.654988, 4.0),
    ]
    mean, sigma = estimate_h0(events)
    output = {
        "run": "smoke",
        "events": [event.__dict__ for event in events],
        "h0_mean": mean,
        "h0_sigma": sigma,
    }
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "smoke_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
