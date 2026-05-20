from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from joint_dark_energy_inference import (
    GaussianBlock,
    JointLikelihood,
    compare_model_grid,
    cosmology_grid,
    diagonal_covariance,
    load_bao_measurements,
    load_supernovae,
)
from joint_dark_energy_inference.models import bao_dv_over_rd, distance_modulus


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    bao = load_bao_measurements(root / "data" / "toy_bao.json")
    sne = load_supernovae(root / "data" / "toy_supernovae.csv")
    blocks = (
        GaussianBlock(
            "bao",
            observed=np.array([row.dv_over_rd for row in bao]),
            covariance=diagonal_covariance([row.sigma for row in bao]),
            predict=lambda c: np.array([bao_dv_over_rd(row.redshift, c) for row in bao]),
        ),
        GaussianBlock(
            "supernovae",
            observed=np.array([row.distance_modulus for row in sne]),
            covariance=diagonal_covariance([row.sigma for row in sne]),
            predict=lambda c: np.array([distance_modulus(row.redshift, c) for row in sne]),
        ),
    )
    likelihood = JointLikelihood(blocks)
    lcdm = compare_model_grid(
        "lcdm",
        likelihood,
        cosmology_grid([67.0, 70.0, 73.0], [0.27, 0.3, 0.33]),
        n_parameters=2,
        n_data=5,
    )
    wcdm = compare_model_grid(
        "wcdm",
        likelihood,
        cosmology_grid([67.0, 70.0, 73.0], [0.27, 0.3, 0.33], [-1.1, -1.0, -0.9]),
        n_parameters=3,
        n_data=5,
    )
    cpl = compare_model_grid(
        "cpl",
        likelihood,
        cosmology_grid([67.0, 70.0, 73.0], [0.27, 0.3, 0.33], [-1.1, -1.0, -0.9], [-0.2, 0.0, 0.2]),
        n_parameters=4,
        n_data=5,
    )
    output = {"run": "baseline-v1", "data": ["toy_bao.json", "toy_supernovae.csv"], "models": [lcdm, wcdm, cpl]}
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "baseline_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
