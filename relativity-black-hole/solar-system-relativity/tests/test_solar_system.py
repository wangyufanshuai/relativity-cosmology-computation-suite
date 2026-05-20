"""Tests for solar system relativistic N-body simulation."""

import numpy as np
import pytest
from solar_system_relativity.constants import BODIES, G, C
from solar_system_relativity.analyze import orbital_elements


class TestOrbitalElements:
    def test_circular_orbit(self):
        """Circular orbit should have e ≈ 0."""
        M = BODIES["Sun"]["mass"]
        a = BODIES["Earth"]["a"]
        v_circ = np.sqrt(G * M / a)
        pos = np.array([a, 0.0, 0.0])
        vel = np.array([0.0, v_circ, 0.0])
        elements = orbital_elements(pos, vel, M)
        assert abs(elements["e"]) < 1e-6
        assert abs(elements["a"] - a) / a < 1e-6

    def test_elliptical_orbit(self):
        """Known elliptical orbit should recover correct a and e."""
        M = BODIES["Sun"]["mass"]
        a = BODIES["Mercury"]["a"]
        e = BODIES["Mercury"]["e"]
        r_peri = a * (1.0 - e)
        v_peri = np.sqrt(G * M * (2.0 / r_peri - 1.0 / a))
        pos = np.array([r_peri, 0.0, 0.0])
        vel = np.array([0.0, v_peri, 0.0])
        elements = orbital_elements(pos, vel, M)
        assert abs(elements["e"] - e) / e < 1e-4
        assert abs(elements["a"] - a) / a < 1e-4

    def test_inclined_orbit(self):
        """Orbit with known inclination should recover it."""
        M = BODIES["Sun"]["mass"]
        a = 1.0e11
        v = np.sqrt(G * M / a)
        incl = 30.0 * np.pi / 180.0
        pos = np.array([a, 0.0, 0.0])
        vel = np.array([0.0, v * np.cos(incl), v * np.sin(incl)])
        elements = orbital_elements(pos, vel, M)
        assert abs(elements["incl"] - incl) < 0.01


class TestConstants:
    def test_bodies_defined(self):
        assert "Sun" in BODIES
        assert "Mercury" in BODIES
        assert len(BODIES) >= 7

    def test_masses_positive(self):
        for name, body in BODIES.items():
            assert body["mass"] > 0

    def test_mercury_eccentricity(self):
        assert abs(BODIES["Mercury"]["e"] - 0.205630) < 1e-4
