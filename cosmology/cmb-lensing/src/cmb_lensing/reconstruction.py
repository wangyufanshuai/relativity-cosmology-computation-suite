"""CMB lensing potential reconstruction via quadratic estimators."""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Helper: simplified CMB TT power spectrum (model)
# ---------------------------------------------------------------------------

def _simplified_ctt(ell: np.ndarray) -> np.ndarray:
    """Return a simplified model of the unlensed CMB TT power spectrum.

    This captures the main acoustic peaks and the damping tail, useful as a
    stand-in when a full Boltzmann solver is not available.

    D_ell = ell*(ell+1)/(2*pi) * C_ell   [uK^2]
    """
    ell = np.asarray(ell, dtype=np.float64)
    # avoid ell = 0
    ell_safe = np.where(ell < 2, 2.0, ell)

    # rough model: envelope * acoustic oscillations
    x = ell_safe / 300.0
    envelope = 6000.0 * (ell_safe / 1000.0) ** 2 * np.exp(-0.5 * ((ell_safe - 220) / 350) ** 2)
    # damping tail
    envelope *= np.exp(-(ell_safe / 1500) ** 1.5)
    acoustic = 1.0 + 0.6 * np.cos(2.0 * np.pi * ell_safe / 280)
    D_ell = envelope * acoustic
    # convert D_ell -> C_ell = 2*pi * D_ell / (ell*(ell+1))
    C_ell = 2.0 * np.pi * D_ell / (ell_safe * (ell_safe + 1))
    return C_ell


def _simplified_ctt_with_noise(
    ell: np.ndarray,
    noise_uK_arcmin: float = 1.0,
    beam_fwhm_arcmin: float = 1.0,
) -> np.ndarray:
    """C_ell^{TT,tot} = C_ell^{TT,signal} + N_ell."""
    C_signal = _simplified_ctt(ell)
    sigma_b = beam_fwhm_arcmin / np.sqrt(8.0 * np.log(2.0))  # arcmin
    theta_rad = sigma_b * np.arctan(1.0) / (60.0 * 45.0 / np.pi)
    beam2 = np.exp(-ell * (ell + 1) * theta_rad**2)
    N_ell = (noise_uK_arcmin * np.pi / 180.0 / 60.0) ** 2 / beam2
    return C_signal + N_ell


# ---------------------------------------------------------------------------
# Main estimator functions
# ---------------------------------------------------------------------------

def quadratic_estimator_phi(
    T_map: np.ndarray,
    pixel_scale: float = 1.0,
    l_max: int = 3000,
) -> np.ndarray:
    """Hu-Okamoto quadratic estimator for the lensing potential.

    Implements the temperature-only (TT) quadratic estimator:

        phi_hat(L) = N_L^{-1} integral d^2 ell  f(ell, L) T(ell) T(L-ell)

    where the weight function is

        f(ell, L) = (ell . L) * (C_{|L-ell|}^{TT,tot})^{-1}
                    + ((L - ell) . L) * (C_ell^{TT,tot})^{-1}

    In practice the estimator is computed via an inverse-variance filtered map
    and FFT-based convolutions.

    Parameters
    ----------
    T_map : np.ndarray
        2D lensed CMB temperature map.
    pixel_scale : float
        Arcmin per pixel.
    l_max : int
        Maximum multipole to include.

    Returns
    -------
    np.ndarray
        Reconstructed lensing potential map (same shape as input).
    """
    T_map = np.asarray(T_map, dtype=np.float64)
    ny, nx = T_map.shape
    arcmin_to_rad = np.pi / (180.0 * 60.0)
    dx_rad = pixel_scale * arcmin_to_rad

    # 2D Fourier coordinates
    kx = np.fft.fftfreq(nx, d=dx_rad) * 2.0 * np.pi  # rad^{-1}
    ky = np.fft.fftfreq(ny, d=dx_rad) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    Lx, Ly = KX, KY
    L2 = Lx**2 + Ly**2
    L_mag = np.sqrt(L2)

    # Temperature FFT
    T_fft = np.fft.fft2(T_map)

    # multipole bin for C_ell lookup
    ell_1d = np.abs(np.fft.fftfreq(nx, d=1.0 / nx))
    ell_2d = L_mag.copy()
    ell_2d_int = np.round(ell_2d).astype(np.intp)

    # simplified C_ell^{TT,tot}
    ell_for_cl = np.arange(0, l_max + 1, dtype=np.float64)
    C_tot = _simplified_ctt(ell_for_cl)
    # avoid division by zero
    C_tot[0] = 1.0
    C_tot[1] = 1.0

    # build inverse-variance filter  1/C_ell^{tot}
    C_inv = np.zeros_like(ell_2d)
    mask = (ell_2d_int <= l_max) & (ell_2d_int >= 2)
    C_inv[mask] = 1.0 / C_tot[ell_2d_int[mask]]
    C_inv[~mask] = 0.0

    # inverse-variance filtered map in Fourier space
    X_ell = T_fft * C_inv

    # --- Build the two terms of the quadratic estimator ---
    # term A:  F^{-1}[ ell * C_inv(|ell|) * T(ell) ]  * T(x)
    # weighting factor  g(ell) = ell_x (or ell_y) * C_inv  for the gradient
    # We use the scalar weight  (L . ell) in Fourier space

    # Real-space inverse-variance filtered map
    x_filtered = np.fft.ifft2(X_ell).real

    # Gradient of the inverse-variance filtered map (real space)
    #   d/dx component:  F^{-1}[ i*kx * X_ell ]
    grad_x = np.fft.ifft2(1j * KX * X_ell).real
    grad_y = np.fft.ifft2(1j * KY * X_ell).real

    # Quadratic combination in real space
    # phi_hat component from  x_filtered * grad  - grad * x_filtered  (simplified)
    # Full form: phi_hat ~ x_filtered * nabla(T) - T * nabla(x_filtered)
    T_grad_x = np.fft.ifft2(1j * KX * T_fft).real
    T_grad_y = np.fft.ifft2(1j * KY * T_fft).real

    # estimator source in real space
    S_x = x_filtered * T_grad_x - np.fft.ifft2(T_fft).real * grad_x
    S_y = x_filtered * T_grad_y - np.fft.ifft2(T_fft).real * grad_y

    # The lensing potential estimator:  phi_hat(L) ~ (L_x * S_x_fft + L_y * S_y_fft) / L^2
    S_x_fft = np.fft.fft2(S_x)
    S_y_fft = np.fft.fft2(S_y)

    phi_hat_fft = np.zeros_like(T_fft)
    nonzero = L2 > 0
    phi_hat_fft[nonzero] = (
        Lx[nonzero] * S_x_fft[nonzero] + Ly[nonzero] * S_y_fft[nonzero]
    ) / L2[nonzero]

    # Normalization (simplified — use the N_L^{(0)} from the dedicated function)
    N_L = estimator_normalization(L_mag, l_max=l_max)
    # N_L from the 1-D helper is for a 1-D L array; broadcast to 2-D
    phi_hat_fft *= N_L

    # zero mean
    phi_hat_fft[0, 0] = 0.0

    phi_hat = np.fft.ifft2(phi_hat_fft).real
    return phi_hat


def estimator_normalization(
    L: np.ndarray,
    noise_uK_arcmin: float = 1.0,
    beam_fwhm_arcmin: float = 1.0,
    l_max: int = 3000,
) -> np.ndarray:
    """Compute the N_L^{(0)} normalization for the quadratic estimator.

    N_L^{(0)} = [integral d^2 ell  F(ell, L)^2 C_ell^{TT} C_{|L-ell|}^{TT} ]^{-1}

    Uses a simplified analytical integration following the Hu-Okamoto
    approximation for the TT estimator.

    Parameters
    ----------
    L : np.ndarray
        2-D array of multipole magnitudes (or 1-D).
    noise_uK_arcmin : float
        White-noise level in uK-arcmin.
    beam_fwhm_arcmin : float
        Gaussian beam FWHM in arcmin.
    l_max : int
        Maximum multipole.

    Returns
    -------
    np.ndarray
        Normalization array N_L^{(0)}, same shape as *L*.
    """
    L = np.asarray(L, dtype=np.float64)

    # approximate analytic normalization following Hu & Okamoto (2002) scaling
    # N_L ~ L^2 / (C_L^{phi, fiducial} * A) where A depends on noise
    # We use a simplified model calibrated to give reasonable amplitudes:
    #
    #   N_L^{(0)} ~ (L^2 + L_star^2) / (ell_star^2 * f_sky * n_eff)
    #
    # Here we compute a simpler version:  N_L = 1 / (R * L^2 * C_L^{phi})
    # where R ~ integral of (ell^2 C_ell^{TT}/C_ell^{tot})^2

    ell = np.arange(2, l_max + 1, dtype=np.float64)
    C_sig = _simplified_ctt(ell)
    C_tot = _simplified_ctt_with_noise(ell, noise_uK_arcmin, beam_fwhm_arcmin)

    # Fisher response  R ~ sum_ell ell^2 * (C_ell^{sig}/C_ell^{tot})^2
    weight = (C_sig / C_tot) ** 2
    R = np.sum(ell**2 * weight)
    if R == 0:
        R = 1.0

    # Normalization  N_L^{(0)} ≈ L^2 / R   (simplified Hu-Okamoto)
    L_safe = np.where(L < 2, 2.0, L)
    N_L = L_safe**2 / R

    # damp at high L (beyond the information content)
    damping = np.exp(-((L_safe / (0.75 * l_max)) ** 4))
    N_L *= damping

    # zero out L = 0
    N_L = np.where(L < 1, 0.0, N_L)

    return N_L


def mean_field(
    n_simulations: int = 100,
    map_size: int = 256,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Compute the mean field from Monte-Carlo simulations.

    The mean field is the average reconstructed phi from *unlensed* Gaussian
    simulations.  It captures survey-mask and anisotropic-noise biases so that
    they can be subtracted from real-data reconstructions.

    Parameters
    ----------
    n_simulations : int
        Number of random realisations to average.
    map_size : int
        Side length in pixels (square patch).
    pixel_scale : float
        Arcmin per pixel.

    Returns
    -------
    np.ndarray
        Mean-field map of shape (map_size, map_size).
    """
    arcmin_to_rad = np.pi / (180.0 * 60.0)
    dx_rad = pixel_scale * arcmin_to_rad

    # 1-D power spectrum for random field generation
    kx = np.fft.fftfreq(map_size, d=dx_rad) * 2.0 * np.pi
    ky = np.fft.fftfreq(map_size, d=dx_rad) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    ell_2d = np.sqrt(K2)

    ell_for_cl = np.arange(0, 5000, dtype=np.float64)
    C_ell = _simplified_ctt(ell_for_cl)
    C_ell[0] = 0.0
    C_ell[1] = 0.0

    # 2-D power spectrum (isotropic interpolation)
    ell_idx = np.clip(np.round(ell_2d).astype(np.intp), 0, len(C_ell) - 1)
    Pk_2d = C_ell[ell_idx]
    # single-valued random phases => real map
    amplitude = np.sqrt(Pk_2d / (2.0 * (map_size * dx_rad) ** 2))

    accumulator = np.zeros((map_size, map_size), dtype=np.float64)
    rng = np.random.default_rng()

    for _ in range(n_simulations):
        phases = rng.standard_normal((map_size, map_size)) + 1j * rng.standard_normal(
            (map_size, map_size)
        )
        phases /= np.abs(phases)

        T_fft = amplitude * phases
        T_map = np.fft.ifft2(T_fft).real

        phi_rec = quadratic_estimator_phi(T_map, pixel_scale=pixel_scale, l_max=3000)
        accumulator += phi_rec

    mean_field_map = accumulator / n_simulations
    return mean_field_map


def lensing_power_spectrum_reconstructed(
    phi_reconstructed: np.ndarray,
    N_L: Optional[np.ndarray] = None,
    pixel_scale: float = 1.0,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate C_L^{phi phi} from a reconstructed lensing-potential map.

    The power is measured as the azimuthally averaged 2-D power spectrum of
    *phi_reconstructed*, with an optional noise-bias subtraction from *N_L*.

    Parameters
    ----------
    phi_reconstructed : np.ndarray
        2-D reconstructed lensing potential.
    N_L : np.ndarray or None
        1-D noise power spectrum for bias subtraction.  If ``None``, no
        subtraction is performed.
    pixel_scale : float
        Arcmin per pixel.
    n_bins : int
        Number of logarithmic L bins.

    Returns
    -------
    L_centers, C_L_phiphi : np.ndarray
        Binned multipole centres and the estimated lensing-potential power.
    """
    phi = np.asarray(phi_reconstructed, dtype=np.float64)
    ny, nx = phi.shape
    arcmin_to_rad = np.pi / (180.0 * 60.0)
    dx_rad = pixel_scale * arcmin_to_rad

    kx = np.fft.fftfreq(nx, d=dx_rad) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dx_rad) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    L2 = KX**2 + KY**2
    L_mag = np.sqrt(L2)

    # 2-D power spectrum
    phi_fft = np.fft.fft2(phi)
    P2d = (np.abs(phi_fft) ** 2) / (nx * ny) ** 2  # normalised

    # DFT estimator:  C_L = P2d * (map area in rad^2)
    map_area_sr = (nx * dx_rad) * (ny * dx_rad)
    P2d *= map_area_sr

    # azimuthal binning
    L_max = np.min([nx, ny]) * np.pi / (nx * dx_rad) * 0.5
    L_min = 2.0 * np.pi / (np.max([nx, ny]) * dx_rad)
    L_edges = np.logspace(np.log10(max(L_min, 2.0)), np.log10(L_max), n_bins + 1)

    L_centers = np.zeros(n_bins)
    C_L = np.zeros(n_bins)

    L_flat = L_mag.ravel()
    P_flat = P2d.ravel()

    for i in range(n_bins):
        mask = (L_flat >= L_edges[i]) & (L_flat < L_edges[i + 1])
        if mask.any():
            L_centers[i] = np.mean(L_flat[mask])
            C_L[i] = np.mean(P_flat[mask])

    # noise-bias subtraction
    if N_L is not None:
        N_interp = np.interp(
            L_centers,
            np.arange(len(N_L), dtype=np.float64),
            N_L,
            left=0.0,
            right=0.0,
        )
        C_L -= N_interp

    # enforce non-negative
    C_L = np.maximum(C_L, 0.0)

    return L_centers, C_L


def delensing_efficiency(
    phi_reconstructed: np.ndarray,
    phi_true: Optional[np.ndarray] = None,
    l_max: int = 100,
) -> float:
    """Compute delensing efficiency  eta = 1 - residual / total.

    When *phi_true* is provided the residual is computed directly.
    Otherwise a theoretical estimate is returned based on the variance
    of the reconstructed map.

    Parameters
    ----------
    phi_reconstructed : np.ndarray
        Reconstructed lensing potential.
    phi_true : np.ndarray or None
        True input lensing potential (for validation).
    l_max : int
        Maximum multipole used for the efficiency calculation.

    Returns
    -------
    float
        Delensing efficiency in [0, 1].
    """
    phi_rec = np.asarray(phi_reconstructed, dtype=np.float64)

    if phi_true is not None:
        phi_tr = np.asarray(phi_true, dtype=np.float64)
        # compute in harmonic space up to l_max
        ny, nx = phi_rec.shape
        phi_rec_fft = np.fft.fft2(phi_rec) / (nx * ny)
        phi_tr_fft = np.fft.fft2(phi_tr) / (nx * ny)

        # multipole mask (simplified)
        kx = np.fft.fftfreq(nx) * nx
        ky = np.fft.fftfreq(ny) * ny
        KX, KY = np.meshgrid(kx, ky)
        ell_2d = np.sqrt(KX**2 + KY**2)
        mask = ell_2d <= l_max

        var_total = np.sum(np.abs(phi_tr_fft[mask]) ** 2).real
        residual = phi_rec_fft - phi_tr_fft
        var_residual = np.sum(np.abs(residual[mask]) ** 2).real

        if var_total > 0:
            eta = 1.0 - var_residual / var_total
        else:
            eta = 0.0
    else:
        # theoretical estimate: assume reconstruction noise ~ 50% of signal
        # This is a rough approximation; real analyses use full covariance.
        var_rec = np.var(phi_rec)
        # model signal variance (heuristic)
        var_signal = var_rec / 0.5  # assume 50% signal fraction
        if var_signal > 0:
            var_noise = max(var_signal - var_rec, 0.0)
            eta = var_rec / (var_rec + var_noise) if (var_rec + var_noise) > 0 else 0.0
        else:
            eta = 0.0

    return float(np.clip(eta, 0.0, 1.0))
