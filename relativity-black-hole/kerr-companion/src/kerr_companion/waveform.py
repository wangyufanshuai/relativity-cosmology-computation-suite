"""
EMRI waveform generation in the adiabatic approximation.

Produces gravitational waveforms h+ and hx for extreme mass ratio inspirals
using a sum over harmonics of the orbital frequencies, with slowly evolving
amplitude and phase driven by radiation reaction.

References:
    - Barack & Cutler (2004) "LISA capture sources: Approximate waveforms..."
    - Gair et al. (2017) "Fast and accurate EMRI waveforms"
    - Chua, Gair & Cutler (2021) "Generalized inspiral-merger-ringdown waveforms"
"""

import numpy as np
from .orbital_elements import OrbitalElements
from .radiation_reaction import RadiationReaction


class EMRIWaveform:
    """
    EMRI waveform generator in the adiabatic approximation.

    The waveform is a sum over harmonics of the three fundamental frequencies
    (Omega_r, Omega_theta, Omega_phi):

        h(t) = Sum_{k,l,m} A_{klm}(t) * cos(Phi_{klm}(t))

    where the amplitudes A_{klm} and phases Phi_{klm} evolve slowly due to
    radiation reaction.

    Parameters
    ----------
    M : float
        Primary black hole mass.
    a : float
        Primary spin parameter (|a| <= M).
    mu : float
        Secondary mass (mu << M).
    distance : float
        Distance to the source (in geometric units).
    """

    def __init__(self, M=1.0, a=0.0, mu=1e-4, distance=1.0):
        self.M = M
        self.a = a
        self.mu = mu
        self.distance = distance
        self.oe = OrbitalElements(M, a)
        self.rr = RadiationReaction(M, a, mu)

    def harmonic_amplitudes(self, p, e, iota, n_harmonics=3):
        """
        Compute harmonic amplitudes A_{klm} for the waveform.

        Uses a simplified model based on the Hansen-Malik prescription.
        Dominant mode is (2, 2) for nearly circular equatorial orbits.

        Parameters
        ----------
        p : float
            Semi-latus rectum.
        e : float
            Eccentricity.
        iota : float
            Inclination.
        n_harmonics : int
            Number of harmonics to include (n = 1, 2, ..., n_harmonics).

        Returns
        -------
        dict
            Dictionary with keys (k, l, m) -> amplitude.
        """
        M = self.M
        mu = self.mu
        dist = self.distance

        r = p / (1.0 - e**2)
        v = np.sqrt(M / r)  # characteristic velocity

        # Leading-order amplitude (quadrupole formula)
        A0 = 4.0 * mu * M / (dist * r) * v**2

        amplitudes = {}

        for k in range(1, n_harmonics + 1):
            for l in range(k, k + 2):
                for m in range(-l, l + 1):
                    # Amplitude scaling
                    # Dominant: l=2, m=2 (and m=-2)
                    # Eccentricity excites higher harmonics: k * Omega_r
                    # Inclination excites m != l modes

                    if l == 2 and abs(m) == 2:
                        # Dominant quadrupole mode
                        a_klm = A0 * 1.0
                    elif l == 2 and abs(m) == 1:
                        # Sub-dominant, excited by spin
                        a_klm = A0 * abs(self.a / self.M) * 0.5
                    elif l == 2 and m == 0:
                        # Breathing mode, excited by eccentricity
                        a_klm = A0 * e * 0.3
                    else:
                        # Higher harmonics, suppressed
                        a_klm = A0 * (e * 0.5) ** (k - 1) * 0.1

                    # Inclination modulation
                    if abs(m) < l:
                        a_klm *= np.sin(iota)

                    # Only include if amplitude is significant
                    if a_klm > A0 * 1e-6:
                        amplitudes[(k, l, m)] = a_klm

        # Ensure at least the dominant mode is present
        if not amplitudes:
            amplitudes[(1, 2, 2)] = A0
            amplitudes[(1, 2, -2)] = A0

        return amplitudes

    def generate_waveform(self, p0, e0, iota0, t_span, n_points=4096):
        """
        Generate the EMRI waveform h+(t) and hx(t).

        Parameters
        ----------
        p0 : float
            Initial semi-latus rectum.
        e0 : float
            Initial eccentricity.
        iota0 : float
            Initial inclination.
        t_span : tuple
            (t_start, t_end) coordinate time span.
        n_points : int
            Number of time samples.

        Returns
        -------
        dict
            {'t': time array, 'h_plus': h+(t), 'h_cross': hx(t),
             'p': p(t), 'e': e(t), 'iota': iota(t),
             'phases': dict of (k,l,m) -> phase array}
        """
        t = np.linspace(t_span[0], t_span[1], n_points)
        dt = t[1] - t[0]

        # Evolve the orbital elements
        n_evolution = min(n_points, 500)
        evolution = self.rr.evolve_orbit(p0, e0, iota0, t_span, n_steps=n_evolution)

        # Interpolate orbital elements onto the waveform time grid
        p_t = np.interp(t, evolution["t"], evolution["p"])
        e_t = np.interp(t, evolution["t"], evolution["e"])
        iota_t = np.interp(t, evolution["t"], evolution["iota"])

        # Initialize waveforms
        h_plus = np.zeros(n_points)
        h_cross = np.zeros(n_points)

        # Compute phases for each harmonic
        phases = {}
        n_harmonics = 3

        # Initial phases
        phi_r = 0.0
        phi_theta = 0.0
        phi_phi = 0.0

        # Integrate phases
        phi_r_arr = np.zeros(n_points)
        phi_theta_arr = np.zeros(n_points)
        phi_phi_arr = np.zeros(n_points)

        for i in range(n_points):
            freqs = self.oe.orbital_frequencies(p_t[i], e_t[i], iota_t[i])
            if i == 0:
                phi_r_arr[0] = 0.0
                phi_theta_arr[0] = 0.0
                phi_phi_arr[0] = 0.0
            else:
                phi_r_arr[i] = phi_r_arr[i - 1] + freqs["Omega_r"] * dt
                phi_theta_arr[i] = phi_theta_arr[i - 1] + freqs["Omega_theta"] * dt
                phi_phi_arr[i] = phi_phi_arr[i - 1] + freqs["Omega_phi"] * dt

        # Sum over harmonics
        for i in range(n_points):
            amps = self.harmonic_amplitudes(p_t[i], e_t[i], iota_t[i], n_harmonics=n_harmonics)

            for (k, l, m), amp in amps.items():
                # Phase: Phi_{klm} = k * phi_r + (l - |m|) * phi_theta + m * phi_phi
                phase = k * phi_r_arr[i] + (l - abs(m)) * phi_theta_arr[i] + m * phi_phi_arr[i]

                # Polarization decomposition
                # h+ = sum A * cos(Phi)
                # hx = sum A * sin(Phi) * sign(m)
                h_plus[i] += amp * np.cos(phase)
                h_cross[i] += amp * np.sin(phase) * np.sign(m) if m != 0 else 0.0

        return {
            "t": t,
            "h_plus": h_plus,
            "h_cross": h_cross,
            "p": p_t,
            "e": e_t,
            "iota": iota_t,
            "phi_r": phi_r_arr,
            "phi_theta": phi_theta_arr,
            "phi_phi": phi_phi_arr,
        }

    def stationary_phase_waveform(self, p0, e0, iota0, t_span, n_points=4096):
        """
        Generate frequency-domain waveform using the stationary phase approximation.

        h(f) ~ A(f) * exp(i * Psi(f))

        where Psi(f) is the stationary phase:
        Psi(f) = 2*pi*f*t_c - phi_c - pi/4 + (3/128)(pi M f)^{-5/3} / eta

        Parameters
        ----------
        p0 : float
            Initial semi-latus rectum.
        e0 : float
            Initial eccentricity.
        iota0 : float
            Initial inclination.
        t_span : tuple
            Time span.
        n_points : int
            Number of frequency points.

        Returns
        -------
        dict
            {'f': frequency array, 'h_f': complex strain, 'phase': Psi(f)}
        """
        M = self.M
        mu = self.mu
        eta = mu / M  # symmetric mass ratio

        # Get time-domain waveform
        td = self.generate_waveform(p0, e0, iota0, t_span, n_points)

        # FFT-based frequency domain
        h = td["h_plus"] + 1j * td["h_cross"]
        dt = td["t"][1] - td["t"][0]

        # Window to reduce spectral leakage
        window = np.hanning(len(h))
        h_windowed = h * window

        # FFT
        h_f = np.fft.rfft(h_windowed) * dt
        freqs = np.fft.rfftfreq(len(h), dt)

        # Stationary phase approximation for comparison
        # f * h_tilde(f) ~ A * (M f)^{-7/6} * exp(i Psi(f))
        # Only valid where signal has power
        f_peak = 1.0 / (2.0 * np.pi) * np.sqrt(M / p0**3)
        f_arr = freqs[1:]  # skip DC

        # SPA phase (leading order)
        Mf = np.pi * M * f_arr
        spa_phase = np.zeros_like(f_arr)
        valid = Mf > 0
        spa_phase[valid] = 3.0 / 128.0 / (Mf[valid] ** (5.0 / 3.0) * eta)

        # SPA amplitude
        spa_amp = np.zeros_like(f_arr)
        spa_amp[valid] = (Mf[valid]) ** (-7.0 / 6.0)

        return {
            "f": freqs,
            "h_f": h_f,
            "f_spa": f_arr,
            "phase_spa": spa_phase,
            "amplitude_spa": spa_amp,
        }

    def lisa_snr(self, p0, e0, iota0, t_span, n_points=4096):
        """
        Estimate the signal-to-noise ratio for LISA detection.

        SNR = sqrt((h|h)) where the inner product uses the LISA noise curve.

        Parameters
        ----------
        p0 : float
            Initial semi-latus rectum.
        e0 : float
            Initial eccentricity.
        iota0 : float
            Initial inclination.
        t_span : tuple
            Observation time span.
        n_points : int
            Number of points.

        Returns
        -------
        float
            Estimated SNR.
        """
        # Get frequency-domain waveform
        fd = self.stationary_phase_waveform(p0, e0, iota0, t_span, n_points)

        freqs = fd["f"]
        h_f = fd["h_f"]

        # LISA-like noise curve (simplified)
        # S_n(f) = S_0 * (f/f_ref)^{-4} * (1 + (f/f_ref)^2)
        # where f_ref ~ 1 mHz (scaled to geometric units)
        f_ref = 1e-3  # reference frequency (approximate)

        # Noise power spectral density
        S_n = np.ones_like(freqs) * 1e-40  # floor
        valid = freqs > 0
        f = freqs[valid]
        S_n[valid] = 1e-42 * (f / f_ref) ** (-4) * (1.0 + (f / f_ref) ** 2)

        # SNR^2 = 4 * integral |h(f)|^2 / S_n(f) df
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        integrand = np.abs(h_f) ** 2 / S_n
        snr_sq = 4.0 * np.sum(integrand) * df

        return np.sqrt(max(snr_sq, 0.0))

    def estimate_snr_simple(self, p0, e0, iota0, t_obs):
        """
        Simple SNR estimate based on characteristic strain.

        SNR ~ h_c * sqrt(T_obs) / sqrt(S_n(f_peak))

        Parameters
        ----------
        p0 : float
            Semi-latus rectum.
        e0 : float
            Eccentricity.
        iota0 : float
            Inclination.
        t_obs : float
            Observation time.

        Returns
        -------
        float
            Rough SNR estimate.
        """
        M = self.M
        mu = self.mu
        dist = self.distance

        r = p0 / (1.0 - e0**2)
        v = np.sqrt(M / r)

        # Characteristic strain
        h_c = 4.0 * mu * v**2 / dist * np.sqrt(M / r)

        # Peak frequency
        f_peak = v**3 / (2.0 * np.pi * M)

        # Simple noise level (order of magnitude for LISA)
        S_n = 1e-42 * (f_peak / 1e-3) ** (-4) * (1.0 + (f_peak / 1e-3) ** 2)

        # Number of cycles
        n_cycles = f_peak * t_obs

        # SNR
        snr = h_c * np.sqrt(n_cycles) / np.sqrt(S_n)

        return snr
