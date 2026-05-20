import numpy as np

from pta_background import PTAData, gaussian_loglike, power_law_strain, spectral_slope_label


def test_power_law_exact_at_reference_frequency():
    assert power_law_strain(np.array([1.0 / (365.25 * 24 * 3600)]), 2e-15, 13 / 3)[0] == 2e-15


def test_loglike_prefers_truth():
    f = np.array([1e-8, 2e-8, 4e-8])
    y = power_law_strain(f, 2e-15, 13 / 3)
    data = PTAData(f, y, np.ones_like(y) * 1e-16)
    assert gaussian_loglike(data, 2e-15, 13 / 3) > gaussian_loglike(data, 1e-15, 13 / 3)
    assert spectral_slope_label(13 / 3) == "smbh-binary-like"
