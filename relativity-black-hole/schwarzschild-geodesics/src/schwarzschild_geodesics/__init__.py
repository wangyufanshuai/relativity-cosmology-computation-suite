"""
schwarzschild_geodesics — Schwarzschild geodesic panorama analyzer.

Provides tools for analyzing all orbit types in Schwarzschild spacetime:
effective potentials, ISCO, photon sphere, orbit integration,
perihelion precession, light deflection, and Poincare sections.
"""

from .metric import (
    G,
    c,
    M_SUN,
    schwarzschild_radius,
    photon_sphere,
    isco_radius,
    marginally_bound_orbit,
    metric_components,
    christoffel_schwarzschild,
    kretschner_scalar,
)

from .effective_potential import (
    V_eff_timelike,
    V_eff_null,
    circular_orbit_params,
    isco_energy_angular,
    find_unstable_circular,
    classify_orbit,
)

from .integrator import (
    integrate_geodesic,
    integrate_photon_geodesic,
    compute_precession,
)

from .poincare import poincare_section

__all__ = [
    # Constants
    "G",
    "c",
    "M_SUN",
    # Metric
    "schwarzschild_radius",
    "photon_sphere",
    "isco_radius",
    "marginally_bound_orbit",
    "metric_components",
    "christoffel_schwarzschild",
    "kretschner_scalar",
    # Effective potential
    "V_eff_timelike",
    "V_eff_null",
    "circular_orbit_params",
    "isco_energy_angular",
    "find_unstable_circular",
    "classify_orbit",
    # Integrator
    "integrate_geodesic",
    "integrate_photon_geodesic",
    "compute_precession",
    # Poincare
    "poincare_section",
]
