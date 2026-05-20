"""Experimental constraints on primordial gravitational waves.

Provides upper-limit curves and sensitivity estimates from current and
planned gravitational-wave observatories, CMB polarization experiments,
big bang nucleosynthesis, and pulsar timing arrays.

References
----------
- Planck 2018: Planck Collaboration, A&A 641, A10 (2020), arXiv:1807.06211
- LIGO O3: Abbott et al., PRD 104 (2021) 022004, arXiv:2101.12130
- LISA: Caprini et al., JCAP 04 (2016) 001, arXiv:1512.06239
- NANOGrav 15-year: Agazie et al., ApJL 951 (2023) L8, arXiv:2306.16213
- BBN constraint: Allen & Shellard, PRL 64 (1990) 119; Smith et al.
"""

import numpy as np
from typing import Dict


# ---------------------------------------------------------------------------
# CMB / Planck constraints
# ---------------------------------------------------------------------------

def planck_constraint_r() -> float:
    """Planck 2018 upper limit on tensor-to-scalar ratio.

    Returns
    -------
    float
        r < 0.036 at 95% CL from Planck 2018 (BKP + lensing + BAO).
        This is evaluated at the pivot scale k_* = 0.05 Mpc^{-1}.

    References
    ----------
    Planck Collaboration, A&A 641, A10 (2020), Table 2.
    """
    return 0.036


def cmb_b_mode_constraint(
    ell: np.ndarray,
    r: float = 0.01,
) -> np.ndarray:
    """CMB B-mode power spectrum from primordial tensor perturbations.

    C_ell^{BB,tensor} proportional to r, with the characteristic
    recombination bump at ell ~ 80 and lensing bump at ell ~ 1000.

    Parameters
    ----------
    ell : np.ndarray
        Multipole moment array (must be >= 2 for B-modes).
    r : float
        Tensor-to-scalar ratio.

    Returns
    -------
    np.ndarray
        D_ell^{BB} = ell (ell + 1) C_ell^{BB} / (2 pi) in muK^2.

    Notes
    -----
    The tensor B-mode spectrum has:
    - A broad peak ("recombination bump") at ell ~ 80, amplitude ~ 0.024 r muK^2
    - A second peak ("reionization bump") at ell < 10
    - A minimum near ell ~ 100 before the lensing B-modes dominate

    The model used here captures the main features with a smooth analytic
    approximation calibrated against full Boltzmann code (CLASS/CAMB) output.
    """
    ell = np.asarray(ell, dtype=float)

    # Prevent ell < 2 (no B-modes below quadrupole)
    ell_safe = np.maximum(ell, 2.0)

    # Recombination bump: log-normal centered at ell ~ 80
    sigma1 = 0.6  # width of the main peak in log-ell
    ell_peak1 = 80.0
    bump1 = np.exp(-0.5 * (np.log(ell_safe / ell_peak1) / sigma1) ** 2)

    # Reionization bump: broad feature at low ell
    sigma2 = 1.0
    ell_peak2 = 8.0
    bump2 = np.exp(-0.5 * (np.log(ell_safe / ell_peak2) / sigma2) ** 2)

    # High-ell tail: power-law decay
    tail = (ell_safe / 80.0) ** (-2.5)
    tail = np.where(ell_safe > 150, tail, 0.0)

    # Combine: amplitude calibrated to ~ 0.024 r muK^2 at ell = 80
    # for r = 0.01, D_ell ~ 2.4e-4 muK^2 at peak
    # D_ell^{BB,tensor}(ell=80, r=0.01) ~ 2.4e-4 muK^2  (approx)
    amp_recomb = 0.024 * r  # muK^2 at peak for given r
    amp_reion = 0.008 * r   # reionization bump is ~1/3 of recombination

    D_ell = amp_recomb * bump1 + amp_reion * bump2 + amp_recomb * 0.01 * tail

    return D_ell


# ---------------------------------------------------------------------------
# LIGO constraint
# ---------------------------------------------------------------------------

def ligo_constraint(f: np.ndarray) -> np.ndarray:
    """LIGO O3 constraint curve: upper limit on Omega_GW(f).

    Parameters
    ----------
    f : np.ndarray
        Frequency array in Hz. LIGO O3 sensitivity band: ~20-100 Hz.

    Returns
    -------
    np.ndarray
        Upper limit on Omega_GW(f). Returns inf outside the LIGO band.

    Notes
    -----
    LIGO O3 result: Omega_GW < 6e-9 at 25 Hz (95% CL), roughly flat
    across the band with improved sensitivity near ~25-30 Hz.

    Reference: Abbott et al., PRD 104 (2021) 022004.
    """
    f = np.asarray(f, dtype=float)

    # LIGO O3 power-law integrated (PLI) curve approximation
    # Omega_GW(f) < 6e-9 in [20, 100] Hz band, with shape
    result = np.full_like(f, np.inf)

    # Within LIGO band
    mask = (f >= 20.0) & (f <= 100.0)
    # Slight frequency dependence: best at ~25 Hz, degrading at edges
    f_center = 25.0
    Omega_min = 6.0e-9
    # Approximate the O3 PLI curve shape
    shape = 1.0 + 0.3 * ((f - f_center) / 40.0) ** 2
    result = np.where(mask, Omega_min * shape, result)

    # Extended range with degraded sensitivity
    mask_ext = (f >= 10.0) & (f < 20.0)
    result = np.where(
        mask_ext,
        Omega_min * 10.0 ** (1.0 - (f - 10.0) / 10.0),
        result,
    )
    mask_ext2 = (f > 100.0) & (f <= 500.0)
    result = np.where(
        mask_ext2,
        Omega_min * (f / 100.0) ** 3,
        result,
    )

    return result


# ---------------------------------------------------------------------------
# LISA projected sensitivity
# ---------------------------------------------------------------------------

def lisa_constraint(f: np.ndarray) -> np.ndarray:
    """LISA projected sensitivity curve for Omega_GW(f).

    Parameters
    ----------
    f : np.ndarray
        Frequency array in Hz. LISA band: ~0.1 mHz to 100 mHz.

    Returns
    -------
    np.ndarray
        Projected sensitivity (upper limit) on Omega_GW(f).
        Returns inf outside the LISA band.

    Notes
    -----
    LISA is a space-based laser interferometer (ESA/NASA, launch ~2035).
    Expected peak sensitivity ~ 10^{-12} near ~3 mHz.

    The sensitivity curve is approximated from the LISA Science
    Requirement Document and Caprini et al. (2016).
    """
    f = np.asarray(f, dtype=float)

    result = np.full_like(f, np.inf)

    # LISA sensitivity band: ~10^{-5} to 1 Hz
    # Peak sensitivity at f_peak ~ 3 mHz = 3e-3 Hz
    f_peak = 3.0e-3
    Omega_peak = 1.0e-12  # Best sensitivity

    # Approximate LISA sensitivity curve (broken power law)
    mask = (f >= 1e-5) & (f <= 1.0)

    # Shape: Omega ~ Omega_peak * (f/f_peak)^alpha + noise floor
    # Low-f slope: f^2 (acceleration noise)
    # High-f slope: f^{-4} (shot noise)
    low_f = Omega_peak * (f_peak / np.maximum(f, 1e-10)) ** 2.0
    high_f = Omega_peak * (f / f_peak) ** 4.0
    shape = np.sqrt(low_f**2 + high_f**2)
    # Smooth minimum
    shape = np.maximum(shape, Omega_peak)

    result = np.where(mask, shape, result)

    return result


# ---------------------------------------------------------------------------
# BBN constraint
# ---------------------------------------------------------------------------

def bbn_constraint() -> float:
    """BBN constraint on integrated gravitational-wave energy density.

    Returns
    -------
    float
        Upper limit on the log-integrated energy density:
            integral d(ln f) Omega_GW(f) < 1.7e-5

    This comes from the constraint on extra effective neutrino species
    at BBN: N_eff < 3.4 (95% CL), which limits the total energy in
    gravitational waves that were present at T ~ 1 MeV.

    References
    ----------
    - Allen & Shellard, PRL 64 (1990) 119
    - Smith, Caldwell, No, arXiv:1905.09557
    - Planck 2018: N_eff = 2.99 +/- 0.17
    """
    return 1.7e-5


# ---------------------------------------------------------------------------
# Pulsar timing array constraints
# ---------------------------------------------------------------------------

def pulsar_timing_constraint(f: np.ndarray) -> np.ndarray:
    """Pulsar timing array constraint from NANOGrav/EPTA/PPTA.

    Parameters
    ----------
    f : np.ndarray
        Frequency array in Hz. PTA sensitivity: ~10^{-9} to 10^{-7} Hz.

    Returns
    -------
    np.ndarray
        Upper limit on Omega_GW(f). Returns inf outside the PTA band.

    Notes
    -----
    NANOGrav 15-year dataset (2023) reported evidence for a stochastic
    GW background with characteristic strain:
        h_c(f) = A_GWB (f / yr^{-1})^{-2/3}
    with A_GWB ~ 2.4e-15, consistent with a supermassive black hole
    binary astrophysical background.

    The 95% upper limit on a primordial (spectral index = 0) background
    is Omega_GW < 2e-9 at f ~ 3e-8 Hz (nHz).

    Reference: Agazie et al., ApJL 951 (2023) L8.
    """
    f = np.asarray(f, dtype=float)

    result = np.full_like(f, np.inf)

    # PTA band: ~10^{-9} to 10^{-6} Hz
    mask = (f >= 1e-9) & (f <= 1e-6)

    # NANOGrav 15-yr: Omega_GW(f) ~ 2e-9 * (f / f_yr)^{gamma - 5}
    # For a flat spectrum (gamma = 5, i.e., spectral index = 0):
    f_yr = 1.0 / (365.25 * 24.0 * 3600.0)  # 1 yr^{-1} in Hz ~ 3.17e-8 Hz
    Omega_ref = 2.0e-9  # at f_yr

    # PTA sensitivity scales as f^2 at low f and f^{-2/3} at high f
    # for the primordial GW search. Approximate shape:
    shape = Omega_ref * (f / f_yr) ** (-2.0 / 3.0)
    # Floor at the reference level
    shape = np.maximum(shape, Omega_ref * 0.5)

    result = np.where(mask, shape, result)

    return result


# ---------------------------------------------------------------------------
# Combined constraints
# ---------------------------------------------------------------------------

def combined_constraints(f: np.ndarray) -> Dict[str, np.ndarray]:
    """Return all applicable experimental constraints for given frequencies.

    Parameters
    ----------
    f : np.ndarray
        Frequency array in Hz.

    Returns
    -------
    dict
        Dictionary mapping constraint name to Omega_GW upper limit array.
        Keys:
        - 'planck_r': scalar r < 0.036 (single value, repeated)
        - 'cmb_b_mode': B-mode constraint (at CMB frequencies)
        - 'ligo': LIGO O3 upper limit
        - 'lisa': LISA projected sensitivity
        - 'pulsar_timing': NANOGrav/EPTA constraint
        - 'bbn_integrated': BBN integrated constraint (single value)
        - 'bbn': BBN per-log-frequency constraint

    Notes
    -----
    Each entry is an array the same length as f. Values are np.inf where
    a given experiment has no sensitivity at that frequency.
    """
    return {
        "planck_r": np.full_like(f, planck_constraint_r(), dtype=float),
        "cmb_b_mode": cmb_b_mode_constraint(np.maximum(f, 2.0)),
        "ligo": ligo_constraint(f),
        "lisa": lisa_constraint(f),
        "pulsar_timing": pulsar_timing_constraint(f),
        "bbn_integrated": np.full_like(f, bbn_constraint(), dtype=float),
        # BBN per-log-frequency: divide integrated constraint by
        # the ~30 e-folds of observable frequency range
        "bbn": np.full_like(
            f, bbn_constraint() / 30.0, dtype=float
        ),
    }
