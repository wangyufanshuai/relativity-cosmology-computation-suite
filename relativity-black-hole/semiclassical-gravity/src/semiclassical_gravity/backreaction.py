"""
Backreaction: Coupled Semiclassical Einstein Equations.

Implements the coupled evolution of metric (Einstein equations) and quantum
field (Klein-Gordon) with the expectation value of the stress-energy tensor
<T_mu_nu> as the source term.

The semiclassical Einstein equations:
    G_mu_nu + Lambda * g_mu_nu = 8*pi*G * <T_mu_nu>

In 1+1D, this reduces to a much simpler system where the backreaction
is driven by the conformal anomaly.

Key equations (CGHS semiclassical):
    partial_+ partial_- rho = kappa * quantum_source
    <T_{++}> = -(kappa/2) * [d^2 rho/dx^+^2 - (drho/dx^+)^2]

where kappa = N*hbar/(24*pi) controls the strength of quantum effects.
"""

import numpy as np
from .constants import HBAR, PI
from .cghs import cghs_quantum_correction
from .stress_energy import vacuum_stress_1d


def semiclassical_einstein_1d(rho, f, stress_tensor, kappa):
    """
    Compute the right-hand side of the semiclassical Einstein equations in 1+1D.

    The semiclassical Einstein equation in conformal gauge reads:
        partial_+ partial_- rho = -kappa * (source from <T_mu_nu>)

    This function evaluates the source term given the current state.

    Parameters
    ----------
    rho : array_like
        Current conformal factor on the spatial grid.
    f : array_like
        Current matter field on the spatial grid.
    stress_tensor : dict
        Dictionary with quantum stress tensor components:
        - 'T_plus_plus': <T_{++}> array
        - 'T_minus_minus': <T_{--}> array
    kappa : float
        Backreaction coupling constant: kappa = N*hbar/(24*pi).

    Returns
    -------
    dict
        Dictionary with:
        - 'drho_dt': time derivative of conformal factor
        - 'df_dt': time derivative of matter field
        - 'source': semiclassical source term
        - 'quantum_source': quantum contribution to source
        - 'classical_source': classical contribution to source
    """
    rho = np.asarray(rho, dtype=float)
    f = np.asarray(f, dtype=float)
    n_pts = len(rho)

    T_pp = stress_tensor['T_plus_plus']
    T_mm = stress_tensor['T_minus_minus']

    # Classical source: matter stress-energy
    # For a free scalar field: T^{classical}_{++} = (partial_+ f)^2 / 2
    df = np.gradient(f)
    classical_source_pp = 0.5 * df**2
    classical_source_mm = 0.5 * df**2

    # Total source = classical + quantum
    quantum_source = kappa * (T_pp + T_mm)
    classical_source = classical_source_pp + classical_source_mm
    total_source = classical_source + quantum_source

    # Time derivatives
    # drho/dt from the semiclassical equations
    drho_dt = np.zeros_like(rho)
    if n_pts >= 3:
        # Spatial second derivative of rho
        d2rho_dx2 = np.zeros_like(rho)
        for i in range(1, n_pts - 1):
            d2rho_dx2[i] = rho[i+1] - 2*rho[i] + rho[i-1]

        # d^2 rho / dt^2 = d^2 rho / dx^2 + source
        # For time derivative: drho/dt ~ integral of source
        drho_dt = np.gradient(rho) + dt_factor * total_source

    # Matter field derivative: free wave equation + metric coupling
    df_dt = np.zeros_like(f)
    if n_pts >= 3:
        df_dt = np.gradient(f)

    return {
        'drho_dt': drho_dt,
        'df_dt': df_dt,
        'source': total_source,
        'quantum_source': quantum_source,
        'classical_source': classical_source,
    }


# Helper to avoid reference errors in semiclassical_einstein_1d
dt_factor = 1.0


def run_semiclassical_simulation(rho_init, f_init, n_steps, dt, kappa):
    """
    Run a full semiclassical gravity simulation with backreaction.

    Evolves the coupled Einstein-quantum field system:
    1. Compute <T_mu_nu> from the current metric (rho)
    2. Update metric using semiclassical Einstein equations
    3. Update matter field on the new background
    4. Repeat

    Uses a leapfrog (Stoermer-Verlet) integrator for stability.

    Parameters
    ----------
    rho_init : array_like
        Initial conformal factor profile (spatial grid).
    f_init : array_like
        Initial matter field profile.
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.
    kappa : float
        Backreaction coupling: kappa = N*hbar/(24*pi).

    Returns
    -------
    dict
        Simulation results:
        - 'rho': final conformal factor
        - 'f': final matter field
        - 'time': array of time values
        - 'rho_history': conformal factor at each time step
        - 'f_history': matter field at each time step
        - 'stress_history': quantum stress tensor at each time step
        - 'n_steps': number of steps completed
    """
    rho = np.array(rho_init, dtype=float)
    f = np.array(f_init, dtype=float)
    n_pts = len(rho)

    time = np.arange(n_steps + 1) * dt

    rho_history = np.zeros((n_steps + 1, n_pts))
    f_history = np.zeros((n_steps + 1, n_pts))
    stress_history = []

    rho_history[0] = rho.copy()
    f_history[0] = f.copy()

    # Initial stress tensor
    T_q = cghs_quantum_correction(rho, f)
    stress_history.append({
        'T_plus_plus': T_q['T_plus_plus'].copy(),
        'T_minus_minus': T_q['T_minus_minus'].copy(),
    })

    # We need two previous states for leapfrog; initialize with Euler step
    if n_pts < 3:
        # Not enough grid points for finite differences
        for step in range(n_steps):
            rho_history[step + 1] = rho.copy()
            f_history[step + 1] = f.copy()
            T_q = cghs_quantum_correction(rho, f)
            stress_history.append({
                'T_plus_plus': T_q['T_plus_plus'].copy(),
                'T_minus_minus': T_q['T_minus_minus'].copy(),
            })
        return {
            'rho': rho,
            'f': f,
            'time': time,
            'rho_history': rho_history,
            'f_history': f_history,
            'stress_history': stress_history,
            'n_steps': n_steps,
        }

    # Compute initial velocity (drho/dt)
    drho_dt = np.gradient(rho)
    df_dt = np.gradient(f)

    # Compute initial acceleration from semiclassical source
    T_q = cghs_quantum_correction(rho, f)
    quantum_source = kappa * (T_q['T_plus_plus'] + T_q['T_minus_minus'])

    # Laplacian of rho
    d2rho_dx2 = np.zeros(n_pts)
    for i in range(1, n_pts - 1):
        d2rho_dx2[i] = rho[i+1] - 2*rho[i] + rho[i-1]

    # Acceleration: d2rho/dt2 = d2rho/dx2 + quantum_source
    d2rho_dt2 = d2rho_dx2 + quantum_source

    # First half-step for velocity
    drho_dt_half = drho_dt + 0.5 * dt * d2rho_dt2

    # Laplacian of f
    d2f_dx2 = np.zeros(n_pts)
    for i in range(1, n_pts - 1):
        d2f_dx2[i] = f[i+1] - 2*f[i] + f[i-1]

    df_dt_half = df_dt + 0.5 * dt * d2f_dx2

    for step in range(n_steps):
        # Full position update
        rho_new = rho + dt * drho_dt_half
        f_new = f + dt * df_dt_half

        # Boundary conditions: fixed endpoints
        rho_new[0] = rho[0]
        rho_new[-1] = rho[-1]
        f_new[0] = f[0]
        f_new[-1] = f[-1]

        # Compute new quantum stress tensor on updated metric
        T_q = cghs_quantum_correction(rho_new, f_new)
        quantum_source = kappa * (T_q['T_plus_plus'] + T_q['T_minus_minus'])

        # Laplacian of new rho
        d2rho_dx2 = np.zeros(n_pts)
        for i in range(1, n_pts - 1):
            d2rho_dx2[i] = rho_new[i+1] - 2*rho_new[i] + rho_new[i-1]

        # New acceleration
        d2rho_dt2_new = d2rho_dx2 + quantum_source

        # Update velocity
        drho_dt_half = drho_dt_half + dt * d2rho_dt2_new

        # Similarly for f
        d2f_dx2 = np.zeros(n_pts)
        for i in range(1, n_pts - 1):
            d2f_dx2[i] = f_new[i+1] - 2*f_new[i] + f_new[i-1]

        df_dt_half = df_dt_half + dt * d2f_dx2

        rho = rho_new
        f = f_new

        rho_history[step + 1] = rho.copy()
        f_history[step + 1] = f.copy()
        stress_history.append({
            'T_plus_plus': T_q['T_plus_plus'].copy(),
            'T_minus_minus': T_q['T_minus_minus'].copy(),
        })

    return {
        'rho': rho,
        'f': f,
        'time': time,
        'rho_history': rho_history,
        'f_history': f_history,
        'stress_history': stress_history,
        'n_steps': n_steps,
    }
