"""Tests for kerr-companion: geodesics, orbital elements, radiation reaction, waveforms."""

import numpy as np
import pytest

from kerr_companion.kerr_geodesics import KerrGeodesics
from kerr_companion.orbital_elements import OrbitalElements
from kerr_companion.radiation_reaction import RadiationReaction


class TestKerrGeodesics:
    """Test Kerr geodesic equations."""

    def test_schwarzschild_delta(self):
        """For a=0, Delta = r^2 - 2Mr."""
        kg = KerrGeodesics(M=1.0, a=0.0)
        r = 6.0
        assert kg.delta(r) == pytest.approx(r**2 - 2.0 * r)

    def test_sigma_schwarzschild(self):
        """For a=0, Sigma = r^2."""
        kg = KerrGeodesics(M=1.0, a=0.0)
        assert kg.sigma(5.0, np.pi / 2) == pytest.approx(25.0)

    def test_invalid_spin_raises(self):
        """Spin |a| > M should raise ValueError."""
        with pytest.raises(ValueError):
            KerrGeodesics(M=1.0, a=1.5)

    def test_radial_potential_circular(self):
        """For circular equatorial orbit, R(r) should be ~0."""
        oe = OrbitalElements(M=1.0, a=0.0)
        E, Lz, Q = oe.circular_orbit_constants(r_circ=10.0)
        kg = KerrGeodesics(M=1.0, a=0.0)
        R_val = kg.R(10.0, E, Lz, Q)
        assert abs(R_val) < 1e-8


class TestOrbitalElements:
    """Test orbital element computations."""

    def test_isco_schwarzschild(self):
        """ISCO for Schwarzschild (a=0) should be 6M."""
        oe = OrbitalElements(M=1.0, a=0.0)
        r_isco = oe.isco_radius()
        assert r_isco == pytest.approx(6.0, rel=1e-4)

    def test_isco_extremal_kerr(self):
        """ISCO for near-extremal Kerr should be close to M."""
        oe = OrbitalElements(M=1.0, a=0.999)
        r_isco = oe.isco_radius()
        assert r_isco < 2.0

    def test_isco_retrograde_larger(self):
        """Retrograde ISCO should be larger than prograde."""
        oe = OrbitalElements(M=1.0, a=0.9)
        assert oe.isco_radius_retrograde() > oe.isco_radius()

    def test_circular_orbit_constants_physical(self):
        """E should be < 1 and Lz should be positive for bound orbits."""
        oe = OrbitalElements(M=1.0, a=0.0)
        E, Lz, Q = oe.circular_orbit_constants(r_circ=10.0)
        assert 0 < E < 1
        assert Lz > 0
        assert Q == 0.0


class TestRadiationReaction:
    """Test gravitational radiation reaction."""

    def test_energy_flux_negative(self):
        """Energy flux should be negative (energy loss)."""
        rr = RadiationReaction(M=1.0, a=0.0, mu=1e-4)
        dEdt = rr.peters_mathews_energy_flux(p=10.0, e=0.1)
        assert dEdt < 0

    def test_angular_momentum_flux_negative(self):
        """Angular momentum flux should be negative (loss)."""
        rr = RadiationReaction(M=1.0, a=0.0, mu=1e-4)
        dLzdt = rr.peters_mathews_angular_momentum_flux(p=10.0, e=0.1)
        assert dLzdt < 0

    def test_inspiral_time_positive(self):
        """Inspiral time should be positive."""
        rr = RadiationReaction(M=1.0, a=0.0, mu=1e-4)
        t_insp = rr.inspiral_time(p=20.0, e=0.0)
        assert t_insp > 0

    def test_invalid_spin_raises(self):
        """Invalid spin should raise ValueError."""
        with pytest.raises(ValueError):
            RadiationReaction(M=1.0, a=2.0, mu=1e-4)
