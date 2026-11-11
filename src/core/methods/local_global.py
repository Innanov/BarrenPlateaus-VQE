"""Local-Global warm-start VQE and the local-cost observable it uses.

The method runs VQE in two stages. First it trains on a LOCAL cost (a sum of
single-qubit terms) that has milder barren plateaus, to move the parameters into a
good region cheaply. Then it switches to the full molecular Hamiltonian and refines
to the energy. The same ansatz and optimizer carry across the switch, so the second
stage picks up where the first left off.

The local cost is anchored on the Hartree-Fock reference state (see
local_cost_observable), which uses only information known before solving. The anchor
must be a product state, which the HF state is.

This generalizes qubap's VQE_shift [1], replacing qubap's global2local
Pauli-splitting (whose minimum is generally not the target state) with the HF local
cost. The local-cost idea is from [2].

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
    """Run Local-Global VQE: warm-start on a local cost, then refine on energy.

    Stage one runs config.warm_iters steps on the Hartree-Fock local cost. Stage two
    runs config.max_iters steps on the full molecular Hamiltonian, continuing from
    the same parameters and optimizer state (no reset). warm_iters is extra
    preparation on top of max_iters, so the main (energy) stage is max_iters long
    for every method, which keeps the methods comparable.

    Args:
        system: The molecule to solve (provides the Hamiltonian, qubit count, and
            the Hartree-Fock reference state used for the local cost).
        config: Run settings (ansatz depth, optimizer, seed, warm_iters, max_iters).

    Returns:
        A MethodResult. energy_history is both stages concatenated;
        energy_history_global re-scores every parameter set against the full
        Hamiltonian (so it is comparable across the switch); stage_boundaries marks
        the handoff index; final_energy is the full-Hamiltonian energy at the end.
    """
    n = system.n_qubits
    ansatz = EfficientSU2(n, d=config.depth)
    rng = np.random.default_rng(config.seed)
    params = ansatz.random_params(rng)

    h_local = local_cost_observable(system.hf_state, n)
    cost_local = _make_cost(ansatz, h_local, n)
    cost_global = _make_cost(ansatz, system.hamiltonian, n)

    opt = build_optimizer(config.optimizer, seed=config.seed, **config.optimizer_kwargs)
    params, hist_local, phist_local = _optimize(cost_local, params, opt, config.warm_iters)
    boundary = len(hist_local)

    params, hist_global, phist_global = _optimize(
        cost_global, params, opt, config.max_iters, reset=False
    )

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


def local_cost_observable(reference_state, n_qubits: int) -> qml.Hamiltonian:
    """Build a local (single-qubit-sum) cost observable minimized at the reference.

    Returns the observable

        O_L = I - (1/n) sum_j (sigma_j^ref on qubit j),

    where sigma_j^ref is qubit j's reduced density matrix of the reference state.
    Minimizing <psi|O_L|psi> pushes each qubit's marginal toward the reference's.
    Being a sum of single-qubit terms, its gradients vanish only polynomially, not
    exponentially, which is what makes it a useful warm-start cost (the local-cost
    idea of [2] in the module docstring). Only the single-qubit (2x2) marginals are
    used, so this is cheap even for many qubits.

    Pass a product state as the reference (e.g. the Hartree-Fock state): O_L is a
    sum of independent per-qubit terms (see the formula above), so it can only
    specify a target qubit by qubit, which pins down a product state but not an
    entangled one.

    Args:
        reference_state: A product-state reference statevector (length 2**n), the
            Hartree-Fock state here.
        n_qubits: Number of qubits.

    Returns:
        A qml.Hamiltonian for O_L.
    """
    marginals = single_qubit_marginals(reference_state, n_qubits)
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
