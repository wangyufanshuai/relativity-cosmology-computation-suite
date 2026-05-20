from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pta_background import PTAData, gaussian_loglike, power_law_strain, spectral_slope_label


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    frequencies = np.array([1.0e-8, 2.0e-8, 4.0e-8])
    truth_amp = 2.0e-15
    truth_gamma = 13.0 / 3.0
    data = PTAData(frequencies, power_law_strain(frequencies, truth_amp, truth_gamma), np.ones(3) * 1.0e-16)
    candidates = [(1.5e-15, 13.0 / 3.0), (2.0e-15, 13.0 / 3.0), (2.0e-15, 5.5)]
    scored = [
        {"amplitude": amp, "gamma": gamma, "loglike": gaussian_loglike(data, amp, gamma), "class": spectral_slope_label(gamma)}
        for amp, gamma in candidates
    ]
    best = max(scored, key=lambda row: row["loglike"])
    output = {"run": "smoke", "best_fit": best, "candidates": scored}
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "smoke_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
