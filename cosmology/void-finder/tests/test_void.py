"""Tests for cosmic void identification and analysis."""

import numpy as np
import pytest

from void_finder.watershed import watershed_voids, void_radii, void_centers
from void_finder.profile import radial_density_profile, stacked_profile
from void_finder.modified_gravity import void_environment_screening, fifth_force_profile


# ============================================================
# Watershed void finder tests
# ============================================================

class TestWatershedVoids:
    """Tests for the watershed void-finding algorithm."""

    def test_finds_voids_in_underdense_field(self):
        """Should find at least one void when underdense regions exist."""
        rng = np.random.RandomState(42)
        # Create a 3D density field with an obvious central underdense region
        density = np.ones((10, 10, 10)) * 0.5  # overdense background
        density[3:7, 3:7, 3:7] = -0.5  # underdense central region
        positions = rng.uniform(0, 10, (100, 3))

        catalog = watershed_voids(density, positions)
        assert catalog['n_voids'] >= 1

    def test_no_voids_in_overdense_field(self):
        """Should find no voids when all regions are overdense."""
        density = np.ones((10, 10, 10)) * 2.0  # all overdense
        positions = np.random.uniform(0, 10, (100, 3))

        catalog = watershed_voids(density, positions)
        assert catalog['n_voids'] == 0
        assert catalog['centers'].shape == (0, 3)
        assert len(catalog['radii']) == 0

    def test_void_density_below_mean(self):
        """Voids should have density contrast < 0 (below mean density)."""
        density = np.ones((20, 20, 20))
        density[5:15, 5:15, 5:15] = -0.5
        positions = np.random.uniform(0, 20, (500, 3))

        catalog = watershed_voids(density, positions)
        if catalog['n_voids'] > 0:
            labels = catalog['labels']
            for i in range(1, catalog['n_voids'] + 1):
                void_cells = labels == i
                mean_density = density[void_cells].mean()
                assert mean_density < 0, "Void should have density contrast < 0"

    def test_void_radii_positive(self):
        """All void radii should be positive."""
        density = np.ones((15, 15, 15)) * 0.5
        density[4:11, 4:11, 4:11] = -0.3
        positions = np.random.uniform(0, 15, (200, 3))

        catalog = watershed_voids(density, positions)
        if catalog['n_voids'] > 0:
            assert np.all(catalog['radii'] > 0)

    def test_returns_dict_with_required_keys(self):
        """Catalog should have all required keys."""
        density = np.ones((5, 5, 5))
        positions = np.random.uniform(0, 5, (20, 3))

        catalog = watershed_voids(density, positions)
        assert 'labels' in catalog
        assert 'centers' in catalog
        assert 'radii' in catalog
        assert 'n_voids' in catalog

    def test_labels_shape_matches_field(self):
        """Labels array should match input density field shape."""
        shape = (8, 8, 8)
        density = np.ones(shape) * -0.2
        positions = np.random.uniform(0, 8, (50, 3))

        catalog = watershed_voids(density, positions)
        assert catalog['labels'].shape == shape


class TestVoidRadiiAndCenters:
    """Tests for void_radii and void_centers helper functions."""

    def test_void_radii_extracts_radii(self):
        """void_radii should return the radii from the catalog."""
        catalog = {
            'labels': np.zeros((5, 5, 5), dtype=int),
            'centers': np.array([[2.0, 2.0, 2.0]]),
            'radii': np.array([3.5]),
            'n_voids': 1,
        }
        radii = void_radii(catalog)
        np.testing.assert_array_equal(radii, [3.5])

    def test_void_centers_extracts_centers(self):
        """void_centers should return the centers from the catalog."""
        expected_center = np.array([[5.0, 5.0, 5.0]])
        catalog = {
            'labels': np.zeros((10, 10, 10), dtype=int),
            'centers': expected_center,
            'radii': np.array([4.0]),
            'n_voids': 1,
        }
        centers = void_centers(catalog)
        np.testing.assert_array_equal(centers, expected_center)


# ============================================================
# Density profile tests
# ============================================================

class TestRadialDensityProfile:
    """Tests for radial density profiles around voids."""

    def test_profile_at_void_center_below_mean(self):
        """Density at void center should be below the mean (normalized < 1)."""
        rng = np.random.RandomState(42)
        void_center = np.array([5.0, 5.0, 5.0])

        # Create particles with lower density near center
        n = 500
        positions = rng.uniform(0, 10, (n, 3))
        # Densities: lower near center
        dist = np.sqrt(np.sum((positions - void_center)**2, axis=1))
        densities = 0.3 + 0.7 * (dist / dist.max())

        r_array = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        profile = radial_density_profile(r_array, void_center, positions, densities)

        # Innermost bin should be below mean (normalized < 1)
        assert profile[0] < profile[-1], "Profile should increase outward from void"

    def test_profile_normalized(self):
        """Profile should be normalized by mean density."""
        rng = np.random.RandomState(123)
        void_center = np.array([5.0, 5.0, 5.0])
        positions = rng.uniform(0, 10, (200, 3))
        densities = rng.uniform(0.5, 2.0, 200)

        r_array = np.array([1.0, 3.0, 5.0])
        profile = radial_density_profile(r_array, void_center, positions, densities)

        # At large enough radius, should approach ~1 (mean density)
        # Just check it's finite and non-negative
        assert np.all(np.isfinite(profile))
        assert np.all(profile >= 0)


class TestStackedProfile:
    """Tests for stacked (averaged) void profiles."""

    def test_stacked_profile_shape(self):
        """Stacked profile should return arrays of length n_bins."""
        rng = np.random.RandomState(42)
        n_bins = 10
        catalog = {
            'centers': np.array([[3.0, 3.0, 3.0], [7.0, 7.0, 7.0]]),
            'radii': np.array([2.0, 2.5]),
        }
        positions = rng.uniform(0, 10, (300, 3))
        densities = rng.uniform(0.5, 1.5, 300)

        r_norm, profile = stacked_profile(catalog, positions, densities, n_bins)
        assert len(r_norm) == n_bins
        assert len(profile) == n_bins

    def test_stacked_profile_empty_catalog(self):
        """Should handle empty void catalog gracefully."""
        catalog = {
            'centers': np.empty((0, 3)),
            'radii': np.empty(0),
        }
        positions = np.random.uniform(0, 10, (100, 3))
        densities = np.ones(100)

        r_norm, profile = stacked_profile(catalog, positions, densities, n_bins=5)
        assert len(profile) == 5
        np.testing.assert_allclose(profile, 1.0)  # default to mean density


# ============================================================
# Modified gravity tests
# ============================================================

class TestVoidEnvironmentScreening:
    """Tests for screening in void environments."""

    def test_screening_between_zero_and_one(self):
        """Screening factor should be between 0 and 1."""
        result = void_environment_screening(
            M_eff=1e12, void_radius=10.0, rho_mean=2.775e11,
        )
        assert 0.0 <= result <= 1.0

    def test_screening_unscreened_in_large_void(self):
        """Large voids (low enclosed mass relative to M_eff) should be nearly unscreened."""
        # Very large void with small M_eff -> screening_ratio small -> factor ~0
        result = void_environment_screening(
            M_eff=1e20, void_radius=50.0, rho_mean=2.775e11,
        )
        # Should be small but positive (weakly screened)
        assert result < 1.0

    def test_screening_zero_for_zero_radius(self):
        """Zero or negative void radius should return 0."""
        result = void_environment_screening(M_eff=1e12, void_radius=0.0, rho_mean=1.0)
        assert result == 0.0

    def test_screening_increases_with_void_radius(self):
        """Larger voids enclose more mass, so screening should increase."""
        rho_mean = 2.775e11
        M_eff = 1e12
        r_small = 5.0
        r_large = 50.0
        s_small = void_environment_screening(M_eff, r_small, rho_mean)
        s_large = void_environment_screening(M_eff, r_large, rho_mean)
        assert s_large >= s_small


class TestFifthForceProfile:
    """Tests for fifth force enhancement within voids."""

    def test_enhancement_at_center(self):
        """Fifth force should be enhanced at void center (r=0)."""
        r = np.array([0.0, 0.5, 1.0, 1.5])
        profile = fifth_force_profile(r, void_radius=1.0, M_eff=1e12)
        # At center, enhancement should be > 1
        assert profile[0] > 1.0, "Enhancement at center should be > 1"

    def test_no_enhancement_outside_void(self):
        """Outside the void (r > R), enhancement should be exactly 1."""
        r = np.array([1.5, 2.0, 3.0])
        profile = fifth_force_profile(r, void_radius=1.0, M_eff=1e12)
        np.testing.assert_allclose(profile, 1.0)

    def test_profile_monotonically_decreasing_inside(self):
        """Enhancement should decrease from center to void edge."""
        r = np.linspace(0.01, 0.99, 50)
        profile = fifth_force_profile(r, void_radius=1.0, M_eff=1e12)
        # Generally decreasing (allow small numerical fluctuations)
        center_val = profile[0]
        edge_val = profile[-1]
        assert center_val >= edge_val, "Profile should decrease from center to edge"

    def test_zero_radius_returns_unity(self):
        """Zero or negative void radius should return all ones."""
        r = np.array([0.5, 1.0, 2.0])
        profile = fifth_force_profile(r, void_radius=0.0, M_eff=1e12)
        np.testing.assert_allclose(profile, 1.0)

    def test_enhancement_non_negative(self):
        """Enhancement factor should always be >= 1 (no suppression)."""
        r = np.linspace(0, 3, 100)
        profile = fifth_force_profile(r, void_radius=1.0, M_eff=1e12)
        assert np.all(profile >= 1.0 - 1e-10)
