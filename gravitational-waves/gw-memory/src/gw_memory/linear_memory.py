"""Linear (ordinary) gravitational wave memory.

Linear memory arises from the permanent change in the mass quadrupole moment
of the source system. For a binary system, this comes from unbound components
(e.g., hyperbolic encounters, debris ejected during merger) that produce a
net change in the quadrupole.

References:
    Braginskii & Grishchuk (1985), Thorne (1992)
"""

import numpy as np

from .constants import G, C, MPC


def linear_memory_delta_h(
    masses,
    velocities_before,
    velocities_after,
    theta,
    phi,
    distance,
):
    """Compute linear memory from permanent change in mass quadrupole.

    Calculates the permanent displacement Delta h in the GW strain caused by
    a change in the mass quadrupole moment of the system. The linear memory
    is given by:

        Delta h = (2 / r) * (Delta I_ij * e^ij) / c^4

    where Delta I_ij is the change in the reduced quadrupole moment tensor
    and e^ij is the transverse-traceless projection tensor for the observer
    at angles (theta, phi).

    Parameters
    ----------
    masses : array_like, shape (N,)
        Masses of the N particles/components [kg].
    velocities_before : array_like, shape (N, 3)
        Velocities of each component before the event [m/s].
        Each row is [vx, vy, vz].
    velocities_after : array_like, shape (N, 3)
        Velocities of each component after the event [m/s].
        Each row is [vx, vy, vz].
    theta : float
        Polar angle of observer (measured from z-axis) [radians].
    phi : float
        Azimuthal angle of observer in the x-y plane [radians].
    distance : float
        Luminosity distance to source [m].

    Returns
    -------
    float
        The linear memory Delta h (dimensionless strain). Always >= 0 by
        construction of the plus polarization.
    """
    masses = np.asarray(masses, dtype=float)
    velocities_before = np.asarray(velocities_before, dtype=float)
    velocities_after = np.asarray(velocities_after, dtype=float)

    # Compute the mass-weighted second moment tensor Q_ij = sum m * v_i * v_j / c^2
    # The quadrupole moment I_ij ~ sum m * x_i * x_j, so its second time derivative
    # is I_ddot_ij = sum m * (v_i v_j + v_j v_i) = 2 * sum m * v_i * v_j
    # The change Delta I_ddot_ij = 2 * sum m * (v_after_i * v_after_j - v_before_i * v_before_j)
    # Memory ~ (1/r) * integral of I_ddot, which for impulsive changes gives:
    # Delta h ~ (G/c^4) * (2/r) * Delta I_ij * e^ij

    # Compute quadrupole change (mass-weighted velocity outer product difference)
    delta_Q = np.zeros((3, 3))
    for i in range(len(masses)):
        v_after = velocities_after[i]
        v_before = velocities_before[i]
        delta_Q += masses[i] * (np.outer(v_after, v_after) - np.outer(v_before, v_before))

    # TT projection tensor for plus polarization
    # e^ij_+ for observer direction (theta, phi)
    # Unit vector from source to observer
    n_hat = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    # Projection tensor P_ij = delta_ij - n_i * n_j
    P = np.eye(3) - np.outer(n_hat, n_hat)

    # TT projection of delta_Q
    PdQ = P @ delta_Q @ P

    # Plus polarization tensor: e_+ = (1/2) * (e_theta^i e_theta^j - e_phi^i e_phi^j)
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])

    # The TT-projected quadrupole contracted with the plus-polarization basis
    # h_+ ~ (PdQ_ij * e_theta^i * e_theta^j - PdQ_ij * e_phi^i * e_phi^j)
    h_plus_raw = np.einsum('ij,i,j->', PdQ, e_theta, e_theta) - np.einsum('ij,i,j->', PdQ, e_phi, e_phi)

    # Memory amplitude: Delta h = (G / (r * c^4)) * h_plus_raw
    # Factor of 2 from the TT projection convention
    delta_h = (2.0 * G / (distance * C**4)) * h_plus_raw

    # Linear memory magnitude (return absolute value as the sign depends on
    # observer orientation; the physical observable is |Delta h|)
    return abs(delta_h)


def linear_memory_from_burst(E_radiated, distance, theta=None, phi=None):
    """Simplified burst formula for linear memory order-of-magnitude estimate.

    For an isotropic burst of gravitational radiation carrying energy E_radiated,
    the linear memory is approximately:

        Delta h ~ (G * E_radiated) / (4 * pi * r * c^4)

    This follows from the quadrupole formula: the energy flux in GWs produces
    a permanent strain change proportional to the energy emitted, with the
    standard G/c^4 prefactor. This is an order-of-magnitude estimate.

    Parameters
    ----------
    E_radiated : float
        Total energy radiated in gravitational waves [J].
    distance : float
        Luminosity distance to source [m].
    theta : float, optional
        Polar angle of observer [radians]. Not used in this simplified formula
        but kept for API consistency.
    phi : float, optional
        Azimuthal angle of observer [radians]. Not used in this simplified formula
        but kept for API consistency.

    Returns
    -------
    float
        Order-of-magnitude linear memory Delta h (dimensionless strain).
    """
    return G * E_radiated / (4.0 * np.pi * distance * C**4)
