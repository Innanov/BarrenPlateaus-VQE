"""Adiabatic staged-anneal VQE.

Generalizes qubap's VQE_adiabatic [1], which anneals the cost from a local to the
global Hamiltonian, H(s) = (1-s) H_0 + s H_global. qubap ramps the mixing
coefficient s continuously, rebuilding H every cost evaluation. We instead hold H
fixed for a block of iterations across adiabatic_steps stages, which is a superset
of that schedule: setting adiabatic_steps = max_iters gives one stage per
iteration and recovers qubap's per-iteration ramp as a limit. The variational
adiabatic schedule follows [2].

The anneal starts from a Hartree-Fock reference Hamiltonian H_0 whose ground state
is the HF occupation as a computational-basis state (e.g. |1100>). That state is
easy to prepare (one layer of X gates) and known before solving, so it is a sound
starting point. H_0 is built from the reference occupation alone
(hf_reference_hamiltonian below), never the exact ground state, and the schedule
interpolates from it to the true molecular Hamiltonian.

References:
    [1] qubap, https://github.com/jgidi/quantum-barren-plateaus
    [2] Harwood, Trenev, Stober, Barkoutsos, Gujarati, Mostame, Greenberg,
        "Improving the Variational Quantum Eigensolver Using Variational Adiabatic
        Quantum Computing", ACM Trans. Quantum Comput. 3(1), 1-20 (2022),
        doi:10.1145/3479197.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ..ansatze import EfficientSU2
from ..backend import MolecularSystem, build_optimizer
from .base import MethodConfig, MethodResult, _fixed_hamiltonian, _make_cost, _optimize


def adiabatic(system: MolecularSystem, config: MethodConfig) -> MethodResult:
    """Adiabatic VQE: a staged anneal from H_0 to H_global, warm-started.

    Follows the reference (qubap AdiabaticVQE.run_vqe): the schedule is split
    into adiabatic_steps stages (default 10). Within each stage the interpolated
    Hamiltonian is held FIXED while the optimizer runs its share of the iterations,
    each stage warm-started from the previous. The max_iters total is spread evenly
    across the stages (so the total matches every other method):

        s_i = i / (n_stages - 1),   H(s_i) = (1 - s_i) H_0 + s_i H_global

    The optimizer is reused across stages without a reset, so its state carries
    through the handoffs and the trace stays smooth.

    The anneal starts from the Hartree-Fock reference Hamiltonian H_0, whose ground
    state is the HF occupation basis state, and ends at the true
    molecular Hamiltonian. The max_iters iterations are spread across the stages so they sum
    to exactly max_iters (the first max_iters % n_stages stages get one extra),
    keeping the budget equal to every other method.
    """
    n = system.n_qubits
    ansatz = EfficientSU2(n, d=config.depth)
    rng = np.random.default_rng(config.seed)
    params = ansatz.random_params(rng)

    h_local = hf_reference_hamiltonian(system.hf_occupation)
    h_global = system.hamiltonian
    cost_global = _make_cost(ansatz, h_global, n)

    iters = max(config.max_iters, 1)
    n_stages = max(int(getattr(config, "adiabatic_steps", 10) or 10), 1)
    n_stages = min(n_stages, iters)  # never more stages than iters
    base, extra = divmod(iters, n_stages)
    stage_iters = [base + (1 if i < extra else 0) for i in range(n_stages)]

    optimizer = build_optimizer(config.optimizer, seed=config.seed, **config.optimizer_kwargs)
    params = optimizer.reset(params)

    history = []
    param_history = []
    for i in range(n_stages):
        s = i / (n_stages - 1) if n_stages > 1 else 1.0
        # Hamiltonian held fixed for the whole stage.
        h_s = _interpolate(h_local, h_global, s)
        cost_s = _make_cost(ansatz, h_s, n)
        # reset=False: keep optimizer state across stages so the anneal is smooth.
        params, hist, phist = _optimize(cost_s, params, optimizer, stage_iters[i], reset=False)
        history.extend(hist)
        param_history.extend(phist)

    # Energy vs the FULL Hamiltonian along the whole path -> continuous curve.
    history_global = [float(cost_global(pnp.array(p, requires_grad=False))) for p in param_history]

    return MethodResult(
        method="adiabatic",
        energy_history=history,
        final_energy=float(cost_global(params)),
        params=params,
        ansatz=ansatz,
        n_params=ansatz.n_params,
        # No boundary: one parameter space, plotted vs the global H throughout.
        stage_boundaries=[],
        param_history=param_history,
        energy_history_global=history_global,
    )


def hf_reference_hamiltonian(occupation) -> qml.Hamiltonian:
    """Hartree-Fock reference Hamiltonian whose ground state is |occupation>.

    A sum of single-qubit Z terms, H_0 = sum_j sign_j Z_j, with sign_j = +1 on the
    occupied qubits and -1 on the empty ones. Because Z|1> = -|1> and Z|0> = +|0>,
    the sign_j Z_j term is lowest when qubit j is in its HF state (|1> if occupied,
    |0> if empty). The basis state |occupation> puts every qubit in that state at
    once, so it is the exact, non-degenerate ground state of H_0. This is the
    easy-to-prepare anchor of the adiabatic schedule and depends only on the
    reference occupation, not on the solution.

    Args:
        occupation: Length-n array of 0/1 HF occupations (qubit j occupied iff 1).

    Returns:
        A qml.Hamiltonian for H_0 on n qubits.
    """
    coeffs = [1.0 if int(o) == 1 else -1.0 for o in occupation]
    obs = [qml.PauliZ(j) for j in range(len(occupation))]
    return _fixed_hamiltonian(coeffs, obs)


def _interpolate(h_local: qml.Hamiltonian, h_global: qml.Hamiltonian, s: float) -> qml.Hamiltonian:
    """Return (1 - s) * H_local + s * H_global.

    Args:
        h_local: The local Hamiltonian.
        h_global: The global Hamiltonian.
        s: Interpolation parameter in [0, 1].

    Returns:
        The interpolated Hamiltonian, simplified.
    """
    coeffs = list((1.0 - s) * pnp.array(h_local.coeffs)) + list(s * pnp.array(h_global.coeffs))
    obs = list(h_local.ops) + list(h_global.ops)
    return _fixed_hamiltonian(coeffs, obs)
