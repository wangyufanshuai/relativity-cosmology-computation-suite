from pathlib import Path

from pta_background import compare_power_law_sources, load_binned_spectrum


def test_load_binned_spectrum_and_rank_sources():
    path = Path(__file__).resolve().parents[1] / "data" / "toy_binned_spectrum.json"
    data = load_binned_spectrum(path)
    rows = compare_power_law_sources(data, [("a", 2e-15, 13 / 3), ("b", 1e-15, 3.0)])
    assert len(rows) == 2
    assert rows[0]["loglike"] >= rows[1]["loglike"]
