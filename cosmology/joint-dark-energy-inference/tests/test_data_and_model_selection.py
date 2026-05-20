from pathlib import Path

import numpy as np

from joint_dark_energy_inference import (
    Cosmology,
    GaussianBlock,
    JointLikelihood,
    aic,
    bic,
    compare_model_grid,
    cosmology_grid,
    diagonal_covariance,
    load_bao_measurements,
)


def test_bao_loader_reads_json_fixture():
    path = Path(__file__).resolve().parents[1] / "data" / "toy_bao.json"
    rows = load_bao_measurements(path)
    assert len(rows) == 2
    assert rows[0].redshift == 0.5


def test_model_selection_reports_information_criteria():
    block = GaussianBlock(
        "h0",
        observed=np.array([70.0]),
        covariance=diagonal_covariance([1.0]),
        predict=lambda c: np.array([c.h0]),
    )
    report = compare_model_grid(
        "toy",
        JointLikelihood((block,)),
        cosmology_grid([69.0, 70.0], [0.3]),
        n_parameters=2,
        n_data=2,
    )
    assert report["best_fit"] == Cosmology(h0=70.0, omega_m=0.3).__dict__
    assert report["aic"] == aic(0.0, 2)
    assert report["bic"] == bic(0.0, 2, 2)
