from __future__ import annotations

import json
from pathlib import Path

from standard_sirens import estimate_h0, load_posterior_summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    events = load_posterior_summary(root / "data" / "toy_posterior_summary.json")
    mean, sigma = estimate_h0(events)
    output = {
        "run": "baseline-v1",
        "source": "toy_posterior_summary.json",
        "n_events": len(events),
        "h0_mean": mean,
        "h0_sigma": sigma,
    }
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "baseline_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
