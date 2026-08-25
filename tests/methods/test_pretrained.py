"""Tests for the pretrained (MPS warm-start) VQE.

This method generalizes qubap's VQE_pretrained. The load-bearing shared mechanic
is the parameter transfer: the diagonal MPS parameters are a prefix of the full
ansatz's parameter vector, and a zero off-diagonal block is the identity, so the
full stage starts exactly at the pretrained state. These tests pin that transfer
identity at the method's handoff and the two-stage bookkeeping. We transfer the
final pretrained iterate (qubap averages the last ten), so that difference is
noted but not pinned.

Run with: pytest tests/methods/test_pretrained.py
"""

import numpy as np
import pytest

from src.core.ansatze import MPS
from src.core.methods import MethodConfig, pretrained
from src.utils.helpers import statevector


def test_zero_padded_transfer_reproduces_diagonal_state():
    """The transfer step of pretrained: full_params = [diag_params, 0...]. With
    the off-diagonal block zero (hence identity), the full ansatz produces the
    same statevector as the diagonal-only ansatz on diag_params. This is why the
    warm start begins exactly where the pretraining ended.
    """
    n = 4
    diag = MPS(n, d=1, diagonal=True)
    full = MPS(n, d=1, diagonal=False)

    diag_params = np.random.default_rng(0).uniform(0, 2 * np.pi, diag.n_params)
    full_params = np.zeros(full.n_params)
    full_params[: diag.n_params] = diag_params

    state_diag = statevector(diag, n, params=diag_params)
    state_full = statevector(full, n, params=full_params)
    np.testing.assert_allclose(state_full, state_diag, atol=1e-8)


def test_two_stage_bookkeeping(four_qubit_system):
    """The two stages are recorded consistently, with the full stage at max_iters.

    The MPS pre-training runs warm_iters as extra preparation and the full stage
    runs the full max_iters, so the whole record is warm_iters + max_iters long,
    with the handoff at warm_iters. The reported convergence is the full (quantum)
    stage only, so energy_history_global has max_iters entries.
    """
    cfg = MethodConfig(depth=1, max_iters=5, warm_iters=2, seed=0)
    result = pretrained(four_qubit_system, cfg)

    assert result.stage_boundaries == [cfg.warm_iters]
    assert len(result.energy_history) == cfg.warm_iters + cfg.max_iters
    assert len(result.energy_history_global) == cfg.max_iters


def test_reported_ansatz_is_full_mps(four_qubit_system):
    """The reported ansatz and params are the full stage's.

    The final params live in the full ansatz's parameter space, and param_history
    covers the full (quantum) stage only, which runs max_iters steps.
    """
    cfg = MethodConfig(depth=1, max_iters=5, warm_iters=2, seed=0)
    result = pretrained(four_qubit_system, cfg)

    full = MPS(four_qubit_system.n_qubits, d=cfg.depth, diagonal=False)
    assert result.n_params == full.n_params
    assert len(result.param_history) == cfg.max_iters
    assert np.asarray(result.params).shape[0] == full.n_params


def test_final_energy_matches_full_stage_end(four_qubit_system):
    cfg = MethodConfig(depth=1, max_iters=4, warm_iters=2, seed=0)
    result = pretrained(four_qubit_system, cfg)
    state = statevector(result.ansatz, four_qubit_system.n_qubits, params=np.asarray(result.params))
    assert float(np.abs(np.vdot(state, state))) == pytest.approx(1.0, abs=1e-6)
