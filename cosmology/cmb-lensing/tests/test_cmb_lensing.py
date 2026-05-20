"""Tests for CMB lensing reconstruction.

NOTE: The cmb_lensing package is currently a stub with no source modules.
Tests verify the package is importable and define the expected API contract
for CMB lensing reconstruction, potential estimation, and Hu-Okamoto estimators.
"""

import pytest

import cmb_lensing


class TestPackageImport:
    """Verify the package can be imported."""

    def test_import_package(self):
        """cmb_lensing should be importable."""
        assert cmb_lensing is not None

    def test_package_docstring(self):
        """Package should have a docstring."""
        assert cmb_lensing.__doc__ is not None


class TestLensingPotential:
    """Tests for CMB lensing potential reconstruction.

    The lensing potential phi is related to the projected mass distribution
    along the line of sight and should be a smooth field on the sky.
    """

    @pytest.mark.skip(reason="Module not yet implemented")
    def test_lensing_potential_smooth(self):
        """The lensing potential should be a smooth field (no sharp features)."""
        pass

    @pytest.mark.skip(reason="Module not yet implemented")
    def test_potential_power_spectrum_positive(self):
        """The lensing potential power spectrum C_l^phi should be positive."""
        pass


class TestHuOkamotoEstimator:
    """Tests for the Hu-Okamoto quadratic estimator for lensing.

    The estimator uses the off-diagonal correlations induced by lensing
    in the CMB to reconstruct the lensing potential.
    """

    @pytest.mark.skip(reason="Module not yet implemented")
    def test_reconstruction_unbiased(self):
        """For a known input lensing potential, the reconstruction should be unbiased."""
        pass

    @pytest.mark.skip(reason="Module not yet implemented")
    def test_estimator_response_positive(self):
        """The estimator response function should be positive."""
        pass


class TestLensingDeflection:
    """Tests for CMB lensing deflection field.

    The deflection angle alpha = nabla(phi) where phi is the lensing potential.
    """

    @pytest.mark.skip(reason="Module not yet implemented")
    def test_deflection_from_potential(self):
        """Deflection should be the gradient of the lensing potential."""
        pass

    @pytest.mark.skip(reason="Module not yet implemented")
    def test_deflection_field_magnitude(self):
        """Typical CMB lensing deflections should be ~arcminute scale."""
        pass
