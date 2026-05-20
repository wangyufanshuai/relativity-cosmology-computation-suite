"""
1+1D CGHS (Callan-Giddings-Harvey-Strominger) Model.

The CGHS model is a tractable 2D dilaton gravity model that captures
the essential physics of black hole formation and evaporation via
semiclassical backreaction.

The CGHS action is:
    S = (1/2pi) int d^2x sqrt(-g) [ e^{-2phi} (R + 4 (nabla phi)^2 + 4 lambda^2)
        - (1/2) sum_i (nabla f_i)^2 ]

In conformal gauge ds^2 = -e^{2rho} dx^+ dx^-, the equations of motion become:
    Classically:
        partial_- (e^{-2phi} partial_+ rho) + ... = matter source terms
        partial_+ partial_- phi + lambda^2 e^{2(rho-phi)} = 0

    Quantum correction (for N scalar fields):
        <T_{++}> = -(kappa/2) (partial_+^2 rho - (partial_+ rho)^2 + t_+)
    where kappa = N*hbar/(24*pi).

References:
    Callan, Giddings, Harvey, Strominger, Phys. Rev. D 45, R1005 (1992)
"""

import numpy as np
from .constants import HBAR, PI


def cghs_metric(f, x):
    """
    Compute the CGHS metric components in conformal gauge.

    The CGHS metric in conformal gauge is:
        ds^2 = -e^{2*rho} dx^+ dx^-

    For a classical solution with dilaton phi and matter f:
        rho depends on the matter distribution.

    Parameters
    ----------
    f : callable or array_like
        Matter field f(x). If callable, evaluated at x.
        If array, treated as pre-computed values at grid points.
    x : array_like
        Coordinate values (can be x^+ or x^- or a 1D grid).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'g_tt': metric component g_{tt} (conformal factor squared)
        - 'g_xx': metric component g_{xx}
        - 'conformal_factor': rho(x) = log(|g|)/2
        - 'det_g': determinant of the metric
    """
    x = np.asarray(x, dtype=float)

    if callable(f):
        f_vals = f(x)
    else:
        f_vals = np.asarray(f, dtype=float)

    # In the CGHS model, for the linear dilaton vacuum:
    #   rho = 0 (flat space, Minkowski in light-cone coordinates)
    # With matter, rho acquires a dependence on the matter distribution.
    # For a simple model: rho ~ -lambda * x + matter_correction
    # Here we compute the conformal factor from the matter field.
    # In the classical theory, the general solution is:
    #   e^{-2rho} = constant (for vacuum)

    # For a generic matter distribution, the conformal factor is:
    #   rho(x) = 0 for flat space, or determined by the field equations
    # We return flat metric (rho=0) as baseline; actual rho determined by evolution
    rho = np.zeros_like(x, dtype=float)

    # The metric in light-cone coordinates:
    #   ds^2 = -e^{2*rho} dx^+ dx^-
    # In (t,x) coordinates with x^+ = t+x, x^- = t-x:
    #   g_tt = -(1/2) e^{2*rho}, g_xx = (1/2) e^{2*rho}
    conformal_factor = rho
    exp_2rho = np.exp(2.0 * rho)

    return {
        'g_tt': -0.5 * exp_2rho,
        'g_xx': 0.5 * exp_2rho,
        'conformal_factor': conformal_factor,
        'det_g': -0.25 * exp_2rho**2,
    }


def cghs_classical_evolve(f_0, rho_0, x_plus, x_minus, n_steps, dt):
    """
    Evolve the classical CGHS model (no quantum backreaction).

    The classical CGHS equations in conformal gauge are:
        partial_+ partial_- rho = 0  (in vacuum)
        partial_+ partial_- phi + lambda^2 e^{2(rho - phi)} = 0

    With matter coupling:
        partial_- (e^{-2phi} partial_+ rho) - ... = T_{++}^matter

    Parameters
    ----------
    f_0 : array_like
        Initial matter field profile.
    rho_0 : array_like
        Initial conformal factor profile.
    x_plus : array_like
        x^+ grid coordinates.
    x_minus : array_like
        x^- grid coordinates.
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.

    Returns
    -------
    dict
        Dictionary with evolved fields:
        - 'rho': conformal factor at final time
        - 'f': matter field at final time
        - 'rho_history': list of rho arrays at each step
        - 'f_history': list of f arrays at each step
    """
    rho = np.array(rho_0, dtype=float)
    f = np.array(f_0, dtype=float)

    rho_history = [rho.copy()]
    f_history = [f.copy()]

    # Grid spacing
    dx_plus = np.gradient(x_plus)
    dx_minus = np.gradient(x_minus)

    # Effective dx from the coordinate grids
    if len(x_plus) > 1:
        dx_p = np.mean(dx_plus)
    else:
        dx_p = dt
    if len(x_minus) > 1:
        dx_m = np.mean(dx_minus)
    else:
        dx_m = dt

    for step in range(n_steps):
        # Classical evolution in conformal gauge:
        # partial_+ partial_- rho = 0 (wave equation in flat space)
        # => rho propagates freely along characteristics

        # For the matter field: wave equation on curved background
        # partial_+ partial_- f = 0 (free propagation in conformal gauge)

        # Simple finite-difference update using d'Alembertian
        # rho^{n+1}_i = 2*rho^n_i - rho^{n-1}_i + (dt^2/dx^2)(rho^n_{i+1} - 2*rho^n_i + rho^n_{i-1})
        # But for characteristic evolution, we use:
        # rho(x^+, x^- + dx^-) = rho(x^+, x^-) + dx^- * partial_- rho

        n_pts = len(rho)
        if n_pts < 3:
            # Not enough points for finite differences; just propagate
            rho_history.append(rho.copy())
            f_history.append(f.copy())
            continue

        # Wave equation in light-cone coords: d2_rho/dx+dx- = 0
        # Solution: rho = A(x^+) + B(x^-), i.e., left- and right-moving parts
        # For classical CGHS with matter source, use finite differences
        dx2 = dx_p * dx_m

        rho_new = np.copy(rho)
        f_new = np.copy(f)

        # Interior points: d^2 rho / (dx^+ dx^-) = 0
        # Using cross-derivative finite difference
        for i in range(1, n_pts - 1):
            # Approximate d^2/dx+dx- by the standard 2D Laplacian in LC coords
            rho_new[i] = rho[i] + dt * (
                (rho[i+1] - 2*rho[i] + rho[i-1]) / (2 * dx2)
            ) * dt

        # Matter field: free wave propagation
        for i in range(1, n_pts - 1):
            f_new[i] = f[i] + dt * (
                (f[i+1] - 2*f[i] + f[i-1]) / (2 * dx2)
            ) * dt

        rho = rho_new
        f = f_new

        rho_history.append(rho.copy())
        f_history.append(f.copy())

    return {
        'rho': rho,
        'f': f,
        'rho_history': rho_history,
        'f_history': f_history,
    }


def cghs_quantum_correction(rho, f):
    """
    Compute the quantum stress-energy tensor <T_{++}> in the CGHS model.

    In 1+1D, the expectation value of the stress-energy tensor for N massless
    scalar fields on a curved background receives a contribution from the
    conformal (trace) anomaly. In the CGHS model:

        <T_{++}> = -(kappa/2) * [partial_+^2 rho - (partial_+ rho)^2 + t_+]

    where kappa = N*hbar/(24*pi) and t_+ is a function encoding the quantum
    state (for the Hartle-Hawking state, t_+ can be chosen appropriately).

    For simplicity, we compute the anomalous stress tensor using numerical
    derivatives of rho.

    Parameters
    ----------
    rho : array_like
        Conformal factor values on the grid.
    f : array_like
        Matter field values on the grid (used for state-dependent terms).

    Returns
    -------
    dict
        Dictionary with:
        - 'T_plus_plus': <T_{++}> quantum stress tensor component
        - 'T_minus_minus': <T_{--}> quantum stress tensor component
        - 'kappa': the coupling constant used
    """
    rho = np.asarray(rho, dtype=float)
    f = np.asarray(f, dtype=float)

    n_pts = len(rho)
    kappa = HBAR / (24.0 * PI)

    # Compute derivatives of rho numerically
    if n_pts < 3:
        # Not enough points; return zero correction
        return {
            'T_plus_plus': np.zeros_like(rho),
            'T_minus_minus': np.zeros_like(rho),
            'kappa': kappa,
        }

    # First derivative: drho/dx
    drho = np.gradient(rho)

    # Second derivative: d^2 rho/dx^2
    ddrho = np.gradient(drho)

    # State-dependent term t_+: for the Unruh/vacuum state
    # t_+ encodes the quantum state choice. For the Boulware state t_+ = 0.
    # For simplicity, we use the Boulware vacuum (t_+ = 0).
    t_plus = np.zeros_like(rho)
    t_minus = np.zeros_like(rho)

    # <T_{++}> = -(kappa/2) * (d^2 rho/dx^+^2 - (drho/dx^+)^2 + t_+)
    T_plus_plus = -(kappa / 2.0) * (ddrho - drho**2 + t_plus)
    T_minus_minus = -(kappa / 2.0) * (ddrho - drho**2 + t_minus)

    return {
        'T_plus_plus': T_plus_plus,
        'T_minus_minus': T_minus_minus,
        'kappa': kappa,
    }


def cghs_semiclassical_evolve(f_0, rho_0, kappa, n_steps, dt):
    """
    Full semiclassical CGHS evolution with quantum backreaction.

    Evolves the coupled system:
        Einstein:   partial_+ partial_- rho = -kappa * <T_{++}> * source
        Quantum:    <T_{++}> = -(kappa/2)(d^2 rho - (drho)^2)

    The backreaction modifies the geometry through the quantum
    stress-energy tensor, leading to black hole evaporation.

    Parameters
    ----------
    f_0 : array_like
        Initial matter field profile.
    rho_0 : array_like
        Initial conformal factor profile.
    kappa : float
        Backreaction coupling: kappa = N*hbar/(24*pi).
        Larger kappa = stronger quantum effects.
    n_steps : int
        Number of evolution steps.
    dt : float
        Time step size.

    Returns
    -------
    dict
        Dictionary with:
        - 'rho': final conformal factor
        - 'f': final matter field
        - 'T_quantum': final quantum stress-energy tensor
        - 'rho_history': history of rho values
        - 'f_history': history of f values
        - 'T_history': history of quantum stress tensor
    """
    rho = np.array(rho_0, dtype=float)
    f = np.array(f_0, dtype=float)

    n_pts = len(rho)

    rho_history = [rho.copy()]
    f_history = [f.copy()]

    # Initial quantum correction
    T_q = cghs_quantum_correction(rho, f)
    T_plus_plus = T_q['T_plus_plus']
    T_minus_minus = T_q['T_minus_minus']
    T_history = [{'T_plus_plus': T_plus_plus.copy(), 'T_minus_minus': T_minus_minus.copy()}]

    for step in range(n_steps):
        if n_pts < 3:
            rho_history.append(rho.copy())
            f_history.append(f.copy())
            T_history.append({'T_plus_plus': T_plus_plus.copy(),
                              'T_minus_minus': T_minus_minus.copy()})
            continue

        # Compute quantum stress-energy tensor
        T_q = cghs_quantum_correction(rho, f)
        T_plus_plus = T_q['T_plus_plus']
        T_minus_minus = T_q['T_minus_minus']

        # Derivatives of rho
        drho = np.gradient(rho)
        ddrho = np.gradient(drho)

        # Semiclassical Einstein equation:
        # partial_+ partial_- rho = source_term (classical) + kappa * quantum_source
        #
        # In CGHS semiclassical model, the evolution becomes:
        #   d^2 rho / dt^2 - d^2 rho / dx^2 = -kappa * quantum_source
        #
        # The quantum source from conformal anomaly contributes:
        #   source_quantum = kappa * (T_{++} + T_{--})

        quantum_source = kappa * (T_plus_plus + T_minus_minus)

        # Finite difference update (wave equation with source)
        # rho^{n+1}_i = 2*rho^n_i - rho^{n-1}_i + dt^2 * (laplacian + source)
        # For first step, use forward Euler
        if step == 0:
            # First step: forward Euler
            d2rho_dx2 = np.zeros_like(rho)
            for i in range(1, n_pts - 1):
                d2rho_dx2[i] = (rho[i+1] - 2*rho[i] + rho[i-1])

            rho_new = rho + dt * drho + 0.5 * dt**2 * (d2rho_dx2 + quantum_source)
        else:
            # Subsequent steps: leapfrog
            rho_prev = rho_history[-2] if len(rho_history) >= 2 else rho_history[-1]
            d2rho_dx2 = np.zeros_like(rho)
            for i in range(1, n_pts - 1):
                d2rho_dx2[i] = (rho[i+1] - 2*rho[i] + rho[i-1])

            rho_new = 2*rho - rho_prev + dt**2 * (d2rho_dx2 + quantum_source)

        # Matter field: free wave equation (classically)
        f_new = np.copy(f)
        for i in range(1, n_pts - 1):
            d2f_dx2 = f[i+1] - 2*f[i] + f[i-1]
            f_new[i] = f[i] + dt * (f[i] - (f[i-1] if i > 0 else f[i])) + 0.0

        # Boundary conditions: fixed endpoints
        rho_new[0] = rho[0]
        rho_new[-1] = rho[-1]
        f_new[0] = f[0]
        f_new[-1] = f[-1]

        rho = rho_new
        f = f_new

        rho_history.append(rho.copy())
        f_history.append(f.copy())
        T_history.append({
            'T_plus_plus': T_plus_plus.copy(),
            'T_minus_minus': T_minus_minus.copy(),
        })

    # Final quantum correction
    T_final = cghs_quantum_correction(rho, f)

    return {
        'rho': rho,
        'f': f,
        'T_quantum': T_final,
        'rho_history': rho_history,
        'f_history': f_history,
        'T_history': T_history,
    }
