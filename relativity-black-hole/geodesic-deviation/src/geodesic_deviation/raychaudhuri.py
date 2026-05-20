"""Raychaudhuri equation for geodesic congruences."""

from scipy.integrate import solve_ivp


def raychaudhuri_expansion(dtheta_dt, theta, sigma_sq=0.0, omega_sq=0.0,
                           R_munu_un_un=0.0):
    """Raychaudhuri equation: dtheta/dtau = -1/3 theta^2 - sigma^2 + omega^2 - R_{mu nu} u^mu u^nu."""
    return -theta**2 / 3.0 - sigma_sq + omega_sq - R_munu_un_un


def raychaudhuri_geodesic_congruence(theta0=1.0, sigma_sq=0.0, omega_sq=0.0,
                                     R_munu=0.0, tau_span=(0, 10)):
    """Solve Raychaudhuri equation for a geodesic congruence."""
    def rhs(tau, y):
        theta = y[0]
        return [raychaudhuri_expansion(0, theta, sigma_sq, omega_sq, R_munu)]

    sol = solve_ivp(rhs, tau_span, [theta0], dense_output=True, max_step=0.1)
    return sol
