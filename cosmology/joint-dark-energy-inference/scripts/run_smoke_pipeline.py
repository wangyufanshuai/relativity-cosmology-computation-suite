from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from joint_dark_energy_inference import Cosmology, GaussianBlock, JointLikelihood, grid_search
from joint_dark_energy_inference.models import bao_dv_over_rd, distance_modulus


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    truth = Cosmology(h0=70.0, omega_m=0.3)
    bao_z = np.array([0.5, 1.0])
    sn_z = np.array([0.05, 0.2, 0.8])
    blocks = (
        GaussianBlock(
            "toy-bao",
            observed=np.array([bao_dv_over_rd(float(z), truth) for z in bao_z]),
            covariance=np.eye(len(bao_z)) * 0.01,
            predict=lambda c: np.array([bao_dv_over_rd(float(z), c) for z in bao_z]),
        ),
        GaussianBlock(
            "toy-supernovae",
            observed=np.array([distance_modulus(float(z), truth) for z in sn_z]),
            covariance=np.eye(len(sn_z)) * 0.04,
            predict=lambda c: np.array([distance_modulus(float(z), c) for z in sn_z]),
        ),
    )
    like = JointLikelihood(blocks)
    candidates = [
        Cosmology(h0=h0, omega_m=om, w0=w0)
        for h0 in (67.0, 70.0, 73.0)
        for om in (0.27, 0.3, 0.33)
        for w0 in (-1.1, -1.0, -0.9)
    ]
    best, chi2 = grid_search(like, candidates)
    output = {
        "run": "smoke",
        "model": "flat-wcdm-grid",
        "best_fit": best.__dict__,
        "chi2": chi2,
        "blocks": [block.name for block in blocks],
    }
    results = root / "results"
    results.mkdir(exist_ok=True)
    (results / "smoke_summary.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
