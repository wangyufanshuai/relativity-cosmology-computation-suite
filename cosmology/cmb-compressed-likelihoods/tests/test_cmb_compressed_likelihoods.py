import numpy as np

from cmb_compressed_likelihoods import gaussian_chi2, planck_like_distance_prior


def test_distance_prior_zero_at_mean():
    prior = planck_like_distance_prior()
    assert prior.chi2(prior.mean) == 0.0


def test_gaussian_chi2_positive_for_offset():
    assert gaussian_chi2(np.array([1.0]), np.array([0.0]), np.array([[4.0]])) == 0.25
