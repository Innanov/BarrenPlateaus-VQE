"""Local-Global warm-start VQE and the local-cost observable it uses.

Generalizes qubap's VQE_shift [1]: a two-stage warm start that trains on a local
cost, then refines on the full Hamiltonian. Two deliberate improvements over
qubap. First, the local cost is the ground-state local observable
(local_cost_observable below), built from the target ground state so its minimum
IS that state, rather than qubap's global2local Pauli-splitting whose minimum is
generally orthogonal to the target. Both realize the local-cost idea of [2].
Second, the method is optimizer-agnostic (any PennyLane optimizer), where qubap is
SPSA-only.

References:
    [1] qubap, https://github.com/jgidi/quantum-barren-plateaus
    [2] Cerezo, Sone, Volkoff, Cincio, Coles, "Cost function dependent barren
        plateaus in shallow parametrized quantum circuits", Nat. Commun. 12(1),
        1791 (2021).
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ...utils.helpers import single_qubit_marginals
from ..ansatze import EfficientSU2
from ..backend import MolecularSystem, build_optimizer
from .base import MethodConfig, MethodResult, _fixed_hamiltonian, _make_cost, _optimize

_PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_PAULI_OP = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def local_global(system: MolecularSystem, config: MethodConfig) -> MethodResult:
    """Local-Global VQE: warm-start on a local cost, then refine on energy.

    The local stage minimizes a LOCAL cost (local_cost_observable, the local-cost
    idea of [2] in the module docstring) built from the exact ground state, which
    has polynomially vanishing gradients and is minimized at the true ground
    state. The global stage then refines against the actual molecular energy.

    Of the max_iters total iterations, warm_iters go to the local warm-up and the
    rest to the global stage, so the total matches every other method for a fair
    comparison. One optimizer spans both stages (reset=False on the global stage),
    so its state carries across the handoff.
    """
    n = system.n_qubits
    ansatz = EfficientSU2(n, d=config.depth)
    rng = np.random.default_rng(config.seed)
    params = ansatz.random_params(rng)

    h_local = local_cost_observable(system.ground_state, n)
    cost_local = _make_cost(ansatz, h_local, n)
    cost_global = _make_cost(ansatz, system.hamiltonian, n)

    warm_iters = min(config.warm_iters, config.max_iters)
    main_iters = config.max_iters - warm_iters

    opt = build_optimizer(config.optimizer, seed=config.seed, **config.optimizer_kwargs)
    params, hist_local, phist_local = _optimize(cost_local, params, opt, warm_iters)
    boundary = len(hist_local)

    params, hist_global, phist_global = _optimize(cost_global, params, opt, main_iters, reset=False)

    history = hist_local + hist_global
    full_params = phist_local + phist_global
    history_global = [float(cost_global(pnp.array(p, requires_grad=False))) for p in full_params]
    return MethodResult(
        method="local_global",
        energy_history=history,
        final_energy=float(cost_global(params)),
        params=params,
        ansatz=ansatz,
        n_params=ansatz.n_params,
        stage_boundaries=[boundary],
        param_history=phist_local + phist_global,
        energy_history_global=history_global,
    )


def local_cost_observable(ground_state, n_qubits: int) -> qml.Hamiltonian:
    """Build a local cost observable whose minimum is the given ground state.

    Following the cost-function-dependent barren plateaus of [2] (module
    docstring), this is a LOCAL observable (a sum of single-qubit terms), so it
    has polynomially (not exponentially) vanishing gradients. It is built from the
    target ground state |psi_gs> so its minimum is that state:

        O_L = I - (1/n) sum_j (sigma_j^target on qubit j),

    where sigma_j^target is qubit j's reduced density matrix of |psi_gs>.
    Minimizing <psi|O_L|psi> drives each qubit's marginal toward the ground
    state's. Only single-qubit (2x2) marginals are needed, never a dense 2**n
    operator.

    Args:
        ground_state: The exact ground-state statevector (length 2**n).
        n_qubits: Number of qubits.

    Returns:
        A qml.Hamiltonian for O_L.
    """
    marginals = single_qubit_marginals(ground_state, n_qubits)
    coeffs = [1.0]  # the identity term
    obs = [qml.Identity(0)]
    inv_n = 1.0 / n_qubits
    for j, sigma in enumerate(marginals):
        # Decompose the 2x2 Hermitian marginal into the Pauli basis:
        #   sigma = (1/2) sum_P Tr[P sigma] P,  P in {I, X, Y, Z}.
        for label, mat in _PAULI_MATS.items():
            c = 0.5 * np.trace(mat @ sigma).real
            if abs(c) < 1e-12:
                continue
            # subtract (1/n) * sigma_j -> coefficient -inv_n * c
            if label == "I":
                coeffs.append(-inv_n * c)
                obs.append(qml.Identity(0))
            else:
                coeffs.append(-inv_n * c)
                obs.append(_PAULI_OP[label](j))
    return _fixed_hamiltonian(coeffs, obs)
