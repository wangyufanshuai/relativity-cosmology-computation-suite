"""Kerr QNM database and convenience functions."""

from __future__ import annotations

import numpy as np
from .leaver import find_qnm


# Known Schwarzschild QNM frequencies (in units of c³/GM)
# From Nollert (1993) and Berti, Cardoso, Will (2006)
_SCHWARZSCHILD_QNM = {
    # (l, n): (omega_R, omega_I)  with s=-2
    (2, 0): (0.3736716844, -0.0889623156),
    (2, 1): (0.346711, -0.273915),
    (2, 2): (0.301053, -0.478276),
    (3, 0): (0.599443, -0.092703),
    (3, 1): (0.582689, -0.281293),
    (4, 0): (0.809178, -0.094169),
}


def schwartzschild_qnm(l: int = 2, n: int = 0) -> complex:
    """Return known Schwarzschild QNM frequency for s=-2, m=l.

    Parameters
    ----------
    l : angular quantum number
    n : overtone number

    Returns
    -------
    omega : complex frequency in units of c³/(GM)
    """
    key = (l, n)
    if key in _SCHWARZSCHILD_QNM:
        wr, wi = _SCHWARZSCHILD_QNM[key]
        return complex(wr, wi)
    # Approximate formula for higher overtones
    wr_approx = (l + 0.5) - 0.5 * n
    wi_approx = -0.5 - 0.5 * n
    return complex(wr_approx, wi_approx)


class QNMDatabase:
    """Cache of computed Kerr QNM frequencies."""

    def __init__(self):
        self._cache: dict[tuple, complex] = {}

    def get_qnm(
        self,
        a_star: float,
        l: int = 2,
        m: int = 2,
        n: int = 0,
        s: int = -2,
    ) -> complex:
        """Get QNM frequency, computing if not cached."""
        key = (a_star, l, m, n, s)
        if key not in self._cache:
            if a_star == 0.0 and s == -2 and m == l:
                self._cache[key] = schwartzschild_qnm(l, n)
            else:
                # Use Leaver method
                omega_guess = schwartzschild_qnm(l, n) if (l, n) in _SCHWARZSCHILD_QNM else complex(l + 0.5, -0.5 - n)
                try:
                    self._cache[key] = find_qnm(a_star, l, m, s, omega_guess, n)
                except Exception:
                    self._cache[key] = omega_guess  # fallback
        return self._cache[key]

    def spectrum_table(
        self,
        a_star: float = 0.0,
        l_max: int = 4,
        n_max: int = 5,
        s: int = -2,
    ) -> list[dict]:
        """Generate table of QNM frequencies."""
        rows = []
        for l in range(abs(s), l_max + 1):
            for m in range(-l, l + 1):
                for n in range(n_max + 1):
                    omega = self.get_qnm(a_star, l, m, n, s)
                    Q = qnm_to_quality_factor(omega)
                    rows.append({
                        "l": l, "m": m, "n": n, "s": s,
                        "omega_R": omega.real,
                        "omega_I": omega.imag,
                        "f_Hz": omega.real / (2 * np.pi),  # in units of c³/(GM)
                        "tau": -1.0 / omega.imag if omega.imag < 0 else np.inf,
                        "Q": Q,
                    })
        return rows


def qnm_to_quality_factor(omega: complex) -> float:
    """Quality factor Q = -omega_R / (2*omega_I).

    Higher Q means longer-ringing oscillation.
    """
    if omega.imag >= 0:
        return np.inf
    return -omega.real / (2.0 * omega.imag)
