"""Tests for killing-fields: Killing vector field and symmetry analysis."""

import numpy as np
import pytest


class TestKillingFields:
    """Test Killing vector field functionality from __init__."""

    def test_import(self):
        """Module should be importable."""
        import killing_fields
        assert killing_fields is not None

    def test_module_has_docstring(self):
        """Module should have a docstring."""
        import killing_fields
        assert killing_fields.__doc__ is not None
        assert "Killing" in killing_fields.__doc__
