from pathlib import Path

from standard_sirens import estimate_h0, load_posterior_summary


def test_load_posterior_summary_json():
    path = Path(__file__).resolve().parents[1] / "data" / "toy_posterior_summary.json"
    events = load_posterior_summary(path)
    mean, _ = estimate_h0(events)
    assert len(events) == 2
    assert round(mean, 1) == 70.0
