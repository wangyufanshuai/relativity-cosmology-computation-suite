"""Parameter space scan for superradiant instabilities.

Scans the μM (scalar field mass × BH mass) parameter space to map
instability regions and growth rates as a function of the dimensionless
spin parameter a/M and the dimensionless product μM.
"""

import numpy as np

from .instability import instability_growth_rate
from .superradiance import (
    horizon_angular_velocity,
    superradiance_condition,
)


def scan_mu_M(
    a_over_M: float,
    l_max: int = 3,
    m_max: int = 3,
    mu_range: tuple[float, float] = (0.01, 0.5),
    n_points: int = 100,
) -> dict:
    """Scan μM parameter space for superradiant instability regions.

    For each (l, m) mode with 1 <= m <= min(l, m_max), scans μM values
    and computes the instability growth rate.

    Parameters
    ----------
    a_over_M : float
        Dimensionless spin parameter a/M (0 < a/M < 1).
    l_max : int
        Maximum l quantum number to scan.
    m_max : int
        Maximum m quantum number to scan.
    mu_range : tuple
        (mu_min * M, mu_max * M) range of dimensionless μM to scan.
    n_points : int
        Number of points in μM to evaluate.

    Returns
    -------
    dict
        Results dictionary with keys:
        - 'muM_array': array of μM values
        - 'modes': dict keyed by (l, m) with sub-dict containing:
            - 'growth_rates': array of growth rates
            - 'superradiant': boolean array (True where superradiance occurs)
            - 'peak_muM': μM value at maximum growth rate
            - 'peak_rate': maximum growth rate
        - 'total_growth': sum of growth rates over all modes
    """
    M = 1.0  # Work in units of M=1
    a = a_over_M * M

    muM_array = np.linspace(mu_range[0], mu_range[1], n_points)

    results = {
        'muM_array': muM_array,
        'a_over_M': a_over_M,
        'modes': {},
        'total_growth': np.zeros(n_points),
    }

    for l in range(1, l_max + 1):
        for m in range(1, min(l, m_max) + 1):
            growth_rates = np.zeros(n_points)
            is_superradiant = np.zeros(n_points, dtype=bool)

            for i, muM in enumerate(muM_array):
                mu = muM / M

                # Check superradiance: bound state frequency ω ≈ μ
                # Superradiance requires μ < m * Ω_H
                Omega_H = horizon_angular_velocity(a, M)
                if mu < m * Omega_H:
                    is_superradiant[i] = True

                growth_rates[i] = instability_growth_rate(mu, a, l, m, M)

            results['modes'][(l, m)] = {
                'growth_rates': growth_rates,
                'superradiant': is_superradiant,
            }

            # Find peak
            if np.max(growth_rates) > 0:
                idx_peak = np.argmax(growth_rates)
                results['modes'][(l, m)]['peak_muM'] = muM_array[idx_peak]
                results['modes'][(l, m)]['peak_rate'] = growth_rates[idx_peak]
            else:
                results['modes'][(l, m)]['peak_muM'] = 0.0
                results['modes'][(l, m)]['peak_rate'] = 0.0

            results['total_growth'] += growth_rates

    return results
