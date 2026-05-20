"""Bianchi classification of 3D Lie algebras and isometry groups."""

import numpy as np
from typing import Tuple, Optional, Dict, Any

# Bianchi classification: Class A (a=0) and Class B (a!=0)
# The classification uses the Ellis-MacCallum parameterisation with
# a diagonal tensor n^{ab} = diag(n1, n2, n3) and a vector a^a = (a, 0, 0).
BIANCHI_TYPES: Dict[str, Dict[str, Any]] = {
    "I":     {"a": 0, "n1": 0, "n2": 0, "n3": 0,  "class": "A"},
    "II":    {"a": 0, "n1": 1, "n2": 0, "n3": 0,  "class": "A"},
    "VII_0": {"a": 0, "n1": 0, "n2": 1, "n3": 1,  "class": "A"},
    "VI_0":  {"a": 0, "n1": 0, "n2": 1, "n3": -1, "class": "A"},
    "IX":    {"a": 0, "n1": 1, "n2": 1, "n3": 1,  "class": "A"},
    "VIII":  {"a": 0, "n1": 1, "n2": 1, "n3": -1, "class": "A"},
    "V":     {"a": 1, "n1": 0, "n2": 0, "n3": 0,  "class": "B"},
    "IV":    {"a": 1, "n1": 0, "n2": 0, "n3": 1,  "class": "B"},
    "VII_h": {"a": 1, "n1": 0, "n2": 1, "n3": 1,  "class": "B"},
    "III":   {"a": 1, "n1": 0, "n2": 1, "n3": -1, "class": "B"},
    "VI_h":  {"a": 1, "n1": 0, "n2": 1, "n3": -1, "class": "B"},
}


def bianchi_structure_constants(bianchi_type: str, h: float = 1.0) -> np.ndarray:
    """Return structure constants C^k_{ij} for the given Bianchi type.

    Uses the Ellis-MacCallum decomposition::

        C^k_{ij} = epsilon_{ijl} n^{lk} + delta^k_i a_j - delta^k_j a_i

    where n^{lk} is diagonal ``(n1, n2, n3)`` and ``a = (a, 0, 0)``.

    Parameters
    ----------
    bianchi_type : str
        One of the keys in :data:`BIANCHI_TYPES`.
    h : float
        Free parameter for types that support it (VII_h, VI_h).
        Ignored for other types.

    Returns
    -------
    np.ndarray of shape (3, 3, 3)
        Structure constants C^k_{ij} with upper index first.
    """
    key = bianchi_type
    if key not in BIANCHI_TYPES:
        raise ValueError(
            f"Unknown Bianchi type '{bianchi_type}'. "
            f"Valid types: {list(BIANCHI_TYPES.keys())}"
        )

    params = BIANCHI_TYPES[key]
    a_val = params["a"]
    n_diag = np.array([params["n1"], params["n2"], params["n3"]], dtype=float)

    # For parameterised types, allow h to modify n values
    if key == "VII_h" and h != 1.0:
        n_diag = np.array([0.0, h, 1.0])
    elif key == "VI_h" and h != 1.0:
        n_diag = np.array([0.0, h, -1.0])

    # Levi-Civita symbol in 3D
    def levi_civita_3d() -> np.ndarray:
        e = np.zeros((3, 3, 3))
        e[0, 1, 2] = 1.0
        e[1, 2, 0] = 1.0
        e[2, 0, 1] = 1.0
        e[0, 2, 1] = -1.0
        e[2, 1, 0] = -1.0
        e[1, 0, 2] = -1.0
        return e

    eps = levi_civita_3d()

    # n^{lk} is diagonal
    n_tensor = np.diag(n_diag)  # shape (3, 3)

    # a_j vector
    a_vec = np.array([a_val, 0.0, 0.0])

    # Kronecker delta
    delta = np.eye(3)

    # Build C^k_{ij} = eps_{ijl} n^{lk} + delta^k_i a_j - delta^k_j a_i
    C = np.zeros((3, 3, 3))
    for k in range(3):
        for i in range(3):
            for j in range(3):
                term1 = 0.0
                for l in range(3):
                    term1 += eps[i, j, l] * n_tensor[l, k]
                term2 = delta[k, i] * a_vec[j]
                term3 = delta[k, j] * a_vec[i]
                C[k, i, j] = term1 + term2 - term3

    return C


def isometry_group_dimension(bianchi_type: str) -> int:
    """Return the dimension of the isometry group.

    All Bianchi types correspond to 3-dimensional Lie algebras, so the
    isometry group is always 3-dimensional.  Some types (e.g. IX = SO(3),
    I = R^3) have well-known group names, but the algebra dimension is
    always 3.

    Parameters
    ----------
    bianchi_type : str
        One of the keys in :data:`BIANCHI_TYPES`.

    Returns
    -------
    int
        Dimension of the isometry group (always 3 for standard Bianchi types).
    """
    if bianchi_type not in BIANCHI_TYPES:
        raise ValueError(
            f"Unknown Bianchi type '{bianchi_type}'. "
            f"Valid types: {list(BIANCHI_TYPES.keys())}"
        )
    return 3


def classify_metric_algebra(
    structure_constants: np.ndarray,
    tol: float = 1e-10,
) -> Optional[str]:
    """Classify the Bianchi type from structure constants C^k_{ij}.

    Extracts the Ellis-MacCallum parameters ``(a, n1, n2, n3)`` from the
    given structure constants and matches them against the known types.

    Parameters
    ----------
    structure_constants : np.ndarray of shape (3, 3, 3)
        Structure constants C^k_{ij} with upper index first.
    tol : float
        Tolerance for numerical comparison.

    Returns
    -------
    str or None
        The matched Bianchi type name, or ``None`` if no match is found.
    """
    C = structure_constants
    dim = 3

    # Extract a_j from the antisymmetric part involving Kronecker deltas:
    # C^k_{ij} contains delta^k_i a_j - delta^k_j a_i.
    # For k=i != j:  C^i_{ij} = a_j  (no epsilon contribution when i=j)
    # Actually: when k=i, C^i_{ij} = eps_{ijl} n^{li} + a_j - 0
    #   and for diagonal n, eps_{ijl} n^{li} = 0 when i=j (eps is zero).
    # Wait, i != j for epsilon to be nonzero.
    #
    # Let us use: C^k_{kj} (trace over upper and first lower index):
    #   C^k_{kj} = eps_{kjl} n^{lk} + delta^k_k a_j - delta^k_j a_k
    #            = eps_{kjl} n^{lk} + 3 a_j - a_j
    #            = eps_{kjl} n^{lk} + 2 a_j
    #
    # For j=0: C^k_{k0} = eps_{k0l} n^{lk} + 2 a_0
    #   eps_{k0l} is nonzero only for (k,l) = (1,2) or (2,1):
    #   eps_{102} = -1, eps_{201} = 1
    #   So eps_{k0l} n^{lk} = eps_{102} n^{21} + eps_{201} n^{12}
    #                        = -n^{21} + n^{12} = 0  (n is diagonal)
    #   Thus C^k_{k0} = 2 a_0 = 2a

    # Extract a: sum over k of C^k_{k,0}
    a_val = 0.0
    for k in range(dim):
        a_val += C[k, k, 0]
    a_val /= 2.0

    # Extract n^{lk} from: C^k_{ij} - (delta^k_i a_j - delta^k_j a_i) = eps_{ijl} n^{lk}
    # Contract with eps_{ijm} (using eps_{ijm} eps_{ijl} = 2 delta_{ml}):
    # eps_{ijm} (C^k_{ij} - delta^k_i a_j + delta^k_j a_i) = 2 n^{mk}
    delta = np.eye(dim)

    # Compute eps_{ijm} * C^k_{ij} for each m, k
    def levi_civita_3d() -> np.ndarray:
        e = np.zeros((3, 3, 3))
        e[0, 1, 2] = 1.0
        e[1, 2, 0] = 1.0
        e[2, 0, 1] = 1.0
        e[0, 2, 1] = -1.0
        e[2, 1, 0] = -1.0
        e[1, 0, 2] = -1.0
        return e

    eps = levi_civita_3d()

    # n^{mk} = (1/2) eps_{ijm} (C^k_{ij} - delta^k_i a_j + delta^k_j a_i)
    n_tensor = np.zeros((dim, dim))
    for m in range(dim):
        for k in range(dim):
            s = 0.0
            for i in range(dim):
                for j in range(dim):
                    correction = delta[k, i] * (a_val if j == 0 else 0.0) \
                                 - delta[k, j] * (a_val if i == 0 else 0.0)
                    s += eps[i, j, m] * (C[k, i, j] - correction)
            n_tensor[m, k] = 0.5 * s

    # n should be diagonal; extract diagonal elements
    n1 = n_tensor[0, 0]
    n2 = n_tensor[1, 1]
    n3 = n_tensor[2, 2]

    # Round to suppress floating-point noise
    a_r = round(a_val, 8)
    n1_r = round(n1, 8)
    n2_r = round(n2, 8)
    n3_r = round(n3, 8)

    # Match against known types
    best_match = None
    best_err = float("inf")
    for name, params in BIANCHI_TYPES.items():
        err = (
            (a_r - params["a"]) ** 2
            + (n1_r - params["n1"]) ** 2
            + (n2_r - params["n2"]) ** 2
            + (n3_r - params["n3"]) ** 2
        )
        if err < best_err:
            best_err = err
            best_match = name

    if best_err < tol ** 2:
        return best_match
    return None
