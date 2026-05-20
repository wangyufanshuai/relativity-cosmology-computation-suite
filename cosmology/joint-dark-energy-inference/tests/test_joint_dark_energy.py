import numpy as np

from joint_dark_energy_inference import Cosmology, GaussianBlock, JointLikelihood, distance_modulus, grid_search
from joint_dark_energy_inference.models import bao_dv_over_rd


def test_lcdm_distances_are_monotonic():
    cosmo = Cosmology()
    values = [distance_modulus(z, cosmo) for z in (0.05, 0.2, 0.8)]
    assert values == sorted(values)


def test_joint_likelihood_prefers_matching_candidate():
    truth = Cosmology(h0=70.0, omega_m=0.3)
    bao_z = np.array([0.5, 1.0])
    observed = np.array([bao_dv_over_rd(float(z), truth) for z in bao_z])
    block = GaussianBlock(
        "toy-bao",
        observed=observed,
        covariance=np.eye(2) * 0.01,
        predict=lambda c: np.array([bao_dv_over_rd(float(z), c) for z in bao_z]),
    )
    like = JointLikelihood((block,))
    best, chi2 = grid_search(like, [Cosmology(h0=67.0, omega_m=0.35), truth])
    assert best == truth
    assert chi2 < 1e-9
