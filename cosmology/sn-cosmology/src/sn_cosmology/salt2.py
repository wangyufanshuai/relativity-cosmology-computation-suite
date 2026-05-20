"""SALT2 Type Ia supernova light curve model."""

import numpy as np


def salt2_light_curve(t, t0=0.0, x0=1.0, x1=0.0, c=0.0,
                      alpha=0.14, beta=3.1, M_B=-19.36):
    """Simplified SALT2 light curve model.

    Returns apparent B-band magnitude at each time.
    """
    t = np.asarray(t, dtype=float)
    phase = t - t0
    flux = x0 * np.exp(-0.5 * (phase / (1.0 + 0.1 * x1))**2)
    flux = np.where(flux > 0, flux, 1e-30)
    m_B = -2.5 * np.log10(flux) + M_B - alpha * x1 + beta * c
    return m_B
