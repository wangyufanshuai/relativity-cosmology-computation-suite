"""Tests for primordial-gravitational-waves."""

import numpy as np
import pytest


class TestPrimordialGW:
    """Test primordial gravitational wave module."""

    def test_import(self):
        """Module should be importable."""
        import primordial_gravitational_waves
        assert primordial_gravitational_waves is not None

    def test_version(self):
        """Module should have a version string."""
        import primordial_gravitational_waves
        assert hasattr(primordial_gravitational_waves, '__version__')

    def test_docstring(self):
        """Module should have a descriptive docstring."""
        import primordial_gravitational_waves
        assert primordial_gravitational_waves.__doc__ is not None
        assert "gravitational wave" in primordial_gravitational_waves.__doc__.lower()
