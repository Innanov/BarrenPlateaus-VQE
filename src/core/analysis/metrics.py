"""VQE metrics: solution quality and gradient diagnostics.

Solution quality: fidelity is the state overlap with the exact ground state, and
energy error is the absolute difference from the FCI energy. Gradient diagnostics:
the variance and L2 norm of a gradient, used by the barren-plateau analysis.
"""

from __future__ import annotations

import numpy as np

from ...utils.helpers import statevector
from ..ansatze import Ansatz
from ..backend import MolecularSystem


def energy_error(final_energy: float, system: MolecularSystem) -> float:
    """Return the absolute energy error against the cached FCI energy.

    Args:
        final_energy: The optimized VQE energy.
        system: The molecular system (provides fci_energy).

    Returns:
        |final_energy - fci_energy|.
    """
    return abs(float(final_energy) - system.fci_energy)


def fidelity(ansatz: Ansatz, params, system: MolecularSystem) -> float:
    """Return the real state fidelity with the exact ground state.

    Computes F = |<psi_exact | psi(theta)>|^2 using the exact ground state from
    MolecularSystem.ground_state.

    Args:
        ansatz: The ansatz.
        params: Optimized parameter vector.
        system: The molecular system (provides the exact ground state).

    Returns:
        A fidelity in [0, 1].
    """
    psi = statevector(ansatz, system.n_qubits, params=params)
    exact = system.ground_state
    overlap = np.vdot(exact, psi)
    return float(np.abs(overlap) ** 2)


def gradient_variance(gradients: np.ndarray) -> float:
    """Return the variance of gradient components (the barren-plateau metric).

    Args:
        gradients: An array of gradient samples (e.g. one partial derivative
            gathered over many random initializations).

    Returns:
        The sample variance.
    """
    return float(np.var(np.asarray(gradients)))


def gradient_norm(gradient: np.ndarray) -> float:
    """Return the Euclidean norm of a gradient vector.

    Args:
        gradient: A gradient vector.

    Returns:
        ||gradient||_2.
    """
    return float(np.linalg.norm(np.asarray(gradient)))
