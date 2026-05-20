from pathlib import Path

from h0_tension import constraint_by_label, grouped_tension_report, load_constraints


def test_load_constraints_and_grouped_report():
    path = Path(__file__).resolve().parents[1] / "data" / "toy_constraints.json"
    constraints = load_constraints(path)
    assert constraint_by_label(constraints, "early-universe").mean == 67.4
    report = grouped_tension_report(path)
    assert report["local_vs_early"]["sigma"] > report["siren_vs_early"]["sigma"]
