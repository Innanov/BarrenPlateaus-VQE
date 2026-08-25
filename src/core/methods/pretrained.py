"""Pretrained (MPS warm-start) VQE.

Generalizes qubap's VQE_pretrained [1]: train a diagonal (MPS, classically
tractable) ansatz, then transfer its parameters into the full ansatz by prefix
zero-pad. The diagonal parameters occupy the same leading indices in both
circuits, and a zero off-diagonal block is the identity, so the full stage starts
exactly at the pretrained state. This is the MPS pre-training method of [2]. We
transfer the final pretrained iterate, where qubap averages the last ten. The
method is also optimizer-agnostic, where qubap is SPSA-only.

References:
    [1] qubap, https://github.com/jgidi/quantum-barren-plateaus
    [2] Dborin, Barratt, Wimalaweera, Wright, Green, "Matrix product state
        pre-training for quantum machine learning", Quantum Sci. Technol. 7(3),
        035014 (2022).
"""

from __future__ import annotations

import numpy as np
from pennylane import numpy as pnp

from ..ansatze import MPS
from ..backend import MolecularSystem, build_optimizer
from .base import MethodConfig, MethodResult, _make_cost, _optimize


def pretrained(system: MolecularSystem, config: MethodConfig) -> MethodResult:
    """Pretrained VQE: train the diagonal MPS stage, then the full MPS stage.

    Stage 1 optimizes the diagonal (classically tractable) MPS ansatz. Its
    parameters warm-start stage 2, the full MPS ansatz that adds off-diagonal
    blocks. The two circuits differ, so their energies are not on one continuous
    curve, and only the full stage is reported as the convergence.

    The MPS pre-training runs warm_iters as extra preparation, then the full stage
    runs the full max_iters, so every method's main stage has the same length (the
    warm-up is on top, not carved out of max_iters). The full stage refines with a
    reduced stepsize (a fifth of the base), because the transfer already lands a
    near-optimal state that a full-stepsize first step would overshoot and disrupt.
    This mirrors qubap continuing its decayed optimizer schedule into the full
    stage rather than restarting at full step size.
    """
    n = system.n_qubits
    rng = np.random.default_rng(config.seed)

    # Stage 1: MPS pretraining (diagonal-only staircase).
    mps = MPS(n, d=config.depth, diagonal=True)
    mps_params = mps.random_params(rng)
    cost_mps = _make_cost(mps, system.hamiltonian, n)
    opt = build_optimizer(config.optimizer, seed=config.seed, **config.optimizer_kwargs)
    mps_params, hist_mps, _ = _optimize(cost_mps, mps_params, opt, config.warm_iters)
    boundary = len(hist_mps)

    # Stage 2: full ansatz, warm-started by the prefix zero-pad transfer.
    full = MPS(n, d=config.depth, diagonal=False)
    init = np.zeros(full.n_params)
    take = min(len(mps_params), full.n_params)
    init[:take] = np.asarray(mps_params)[:take]
    full_params = pnp.array(init, requires_grad=True)
    cost_full = _make_cost(full, system.hamiltonian, n)
    stage2_kwargs = dict(config.optimizer_kwargs)
    stage2_kwargs["stepsize"] = 0.2 * config.optimizer_kwargs.get("stepsize", 0.1)
    opt2 = build_optimizer(config.optimizer, seed=config.seed + 1, **stage2_kwargs)
    full_params, hist_full, phist_full = _optimize(cost_full, full_params, opt2, config.max_iters)

    combined = hist_mps + hist_full  # full record (MPS pre-stage + quantum stage)
    return MethodResult(
        method="pretrained",
        energy_history=combined,
        final_energy=float(cost_full(full_params)),
        params=full_params,
        ansatz=full,
        n_params=full.n_params,
        stage_boundaries=[boundary],
        param_history=phist_full,
        # Reported convergence is the full (quantum) stage only, from the warm-start.
        energy_history_global=hist_full,
    )
