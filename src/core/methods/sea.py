"""State Efficient Ansatz VQE.

Standard single-stage VQE with the State Efficient Ansatz [2]. qubap [1] ships the
SEA as an ansatz constructor only (state_efficient_ansatz), with no dedicated SEA
VQE, so it is meant to be plugged into the ordinary VQE loop. We do exactly that:
sea is the standard method run with the SEA ansatz, no extra behavior. The ansatz
itself (ported in src/core/ansatze) is qubap's Schmidt-coefficient-layer
construction.

References:
    [1] qubap, https://github.com/jgidi/quantum-barren-plateaus
    [2] Liu, Liu, Zhang, Huang, Wang, "Mitigating barren plateaus of variational
        quantum eigensolvers", IEEE Trans. Quantum Eng. 5, 1-19 (2024).
"""

from __future__ import annotations

import numpy as np

from ..ansatze import SEA
from ..backend import MolecularSystem, build_optimizer
from .base import MethodConfig, MethodResult, _make_cost, _optimize


def sea(system: MolecularSystem, config: MethodConfig) -> MethodResult:
    """SEA VQE: one stage with the State Efficient Ansatz.

    Draws seeded initial parameters, then optimizes the SEA ansatz directly
    against the full molecular Hamiltonian for config.max_iters iterations. The
    SEA requires an even qubit count. Same single-stage shape as standard, only
    the ansatz differs.
    """
    n = system.n_qubits
    ansatz = SEA(n, d=config.depth)
    rng = np.random.default_rng(config.seed)
    params = ansatz.random_params(rng)
    cost = _make_cost(ansatz, system.hamiltonian, n)
    opt = build_optimizer(config.optimizer, seed=config.seed, **config.optimizer_kwargs)
    params, history, phist = _optimize(cost, params, opt, config.max_iters)
    return MethodResult(
        method="sea",
        energy_history=history,
        final_energy=float(cost(params)),
        params=params,
        ansatz=ansatz,
        n_params=ansatz.n_params,
        param_history=phist,
    )
