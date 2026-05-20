from __future__ import annotations

from dataclasses import asdict

from .likelihood import JointLikelihood, grid_search
from .models import Cosmology


def cosmology_grid(
    h0_values: list[float],
    omega_m_values: list[float],
    w0_values: list[float] | None = None,
    wa_values: list[float] | None = None,
) -> list[Cosmology]:
    w0_values = [-1.0] if w0_values is None else w0_values
    wa_values = [0.0] if wa_values is None else wa_values
    return [
        Cosmology(h0=h0, omega_m=omega_m, w0=w0, wa=wa)
        for h0 in h0_values
        for omega_m in omega_m_values
        for w0 in w0_values
        for wa in wa_values
    ]


def aic(chi2: float, n_parameters: int) -> float:
    return float(chi2 + 2 * n_parameters)


def bic(chi2: float, n_parameters: int, n_data: int) -> float:
    if n_data <= 1:
        raise ValueError("n_data must be greater than 1")
    from math import log

    return float(chi2 + n_parameters * log(n_data))


def compare_model_grid(
    name: str,
    likelihood: JointLikelihood,
    candidates: list[Cosmology],
    n_parameters: int,
    n_data: int,
) -> dict[str, object]:
    best, chi2 = grid_search(likelihood, candidates)
    return {
        "model": name,
        "best_fit": asdict(best),
        "chi2": chi2,
        "n_parameters": n_parameters,
        "n_data": n_data,
        "aic": aic(chi2, n_parameters),
        "bic": bic(chi2, n_parameters, n_data),
    }
