"""Tests for Kerr QNM database."""

import numpy as np
import pytest
from kerr_qnm.database import (
    schwartzschild_qnm,
    QNMDatabase,
    qnm_to_quality_factor,
)


class TestSchwarzschildQNM:
    def test_fundamental_l2(self):
        """Fundamental l=2, n=0 mode: ω ≈ 0.3737 - 0.0890i."""
        omega = schwartzschild_qnm(2, 0)
        assert abs(omega.real - 0.3737) < 0.01
        assert abs(omega.imag - (-0.0890)) < 0.01

    def test_l3_fundamental(self):
        """l=3, n=0 mode."""
        omega = schwartzschild_qnm(3, 0)
        assert omega.real > 0.5
        assert omega.imag < 0

    def test_imaginary_negative(self):
        """All QNM frequencies should have negative imaginary part (damped)."""
        for l in range(2, 5):
            for n in range(3):
                omega = schwartzschild_qnm(l, n)
                assert omega.imag < 0, f"l={l}, n={n}: omega_I should be negative"

    def test_real_positive(self):
        """Fundamental modes should have positive real part (oscillatory)."""
        for l in range(2, 5):
            omega = schwartzschild_qnm(l, 0)
            assert omega.real > 0


class TestQualityFactor:
    def test_positive(self):
        """Quality factor should be positive for physical modes."""
        omega = schwartzschild_qnm(2, 0)
        Q = qnm_to_quality_factor(omega)
        assert Q > 0

    def test_fundamental_value(self):
        """l=2, n=0 has Q ≈ 2."""
        omega = schwartzschild_qnm(2, 0)
        Q = qnm_to_quality_factor(omega)
        assert 1.5 < Q < 3.0

    def test_higher_overtone_lower_Q(self):
        """Higher overtones should have lower quality factor."""
        Q0 = qnm_to_quality_factor(schwartzschild_qnm(2, 0))
        Q1 = qnm_to_quality_factor(schwartzschild_qnm(2, 1))
        assert Q0 > Q1


class TestDatabase:
    def test_schwarzschild_cached(self):
        """a=0 should return known Schwarzschild values."""
        db = QNMDatabase()
        omega = db.get_qnm(0.0, l=2, m=2, n=0)
        assert abs(omega.real - 0.3737) < 0.01

    def test_spectrum_table(self):
        """Should generate a non-empty table."""
        db = QNMDatabase()
        table = db.spectrum_table(a_star=0.0, l_max=3, n_max=2)
        assert len(table) > 0
        assert all("omega_R" in row for row in table)
        assert all("omega_I" in row for row in table)

    def test_caching(self):
        """Second call should return same value."""
        db = QNMDatabase()
        w1 = db.get_qnm(0.0, l=2, m=2, n=0)
        w2 = db.get_qnm(0.0, l=2, m=2, n=0)
        assert w1 == w2
