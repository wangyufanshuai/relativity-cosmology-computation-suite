from __future__ import annotations

from .models import PTAData, gaussian_loglike, spectral_slope_label


def compare_power_law_sources(data: PTAData, candidates: list[tuple[str, float, float]]) -> list[dict[str, float | str]]:
    rows = []
    for name, amplitude, gamma in candidates:
        rows.append(
            {
                "source": name,
                "amplitude": amplitude,
                "gamma": gamma,
                "class": spectral_slope_label(gamma),
                "loglike": gaussian_loglike(data, amplitude, gamma),
            }
        )
    return sorted(rows, key=lambda row: float(row["loglike"]), reverse=True)
