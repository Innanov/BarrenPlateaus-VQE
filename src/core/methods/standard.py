"""Standard VQE: the unmitigated baseline.

Plain VQE with no barren-plateau mitigation: a single stage that optimizes the
EfficientSU2 ansatz directly against the full molecular Hamiltonian. This is the
reference the four mitigation methods (local_global, adiabatic, sea, pretrained)
are compared against, at the same depth and iteration budget.
"""

from __future__ import annotations

import numpy as np

from ..ansatze import EfficientSU2
from ..backend import MolecularSystem, build_optimizer
from .base import MethodConfig, MethodResult, _make_cost, _optimize


def standard(system: MolecularSystem, config: MethodConfig) -> MethodResult:
    """Standard VQE: one stage, no mitigation.

    Draws seeded initial parameters, then optimizes the EfficientSU2 ansatz
    directly against the full molecular Hamiltonian for config.max_iters
    iterations. No warm-up or annealing, so this is the baseline the mitigation
    methods are compared against.
    """
    n = system.n_qubits
    ansatz = EfficientSU2(n, d=config.depth)
    rng = np.random.default_rng(config.seed)
    params = ansatz.random_params(rng)
    cost = _make_cost(ansatz, system.hamiltonian, n)
    opt = build_optimizer(config.optimizer, seed=config.seed, **config.optimizer_kwargs)
    params, history, phist = _optimize(cost, params, opt, config.max_iters)
    return MethodResult(
        method="standard",
        energy_history=history,
        final_energy=float(cost(params)),
        params=params,
        ansatz=ansatz,
        n_params=ansatz.n_params,
        param_history=phist,
    )
