"""Tests for the standard (unmitigated baseline) VQE.

The baseline has no mitigation: a single stage of EfficientSU2 optimized directly
against the full Hamiltonian. These tests pin that single-stage structure (no warm
up, no stage boundary, no separate global curve), determinism under a fixed seed,
and that the recorded final energy matches the reported parameters.

Run with: pytest tests/methods/test_standard.py
"""

import numpy as np
import pytest

from src.core.ansatze import EfficientSU2
from src.core.backend import build_optimizer
from src.core.methods import MethodConfig, standard
from src.core.methods.base import _make_cost, _optimize


def test_standard_is_single_stage(two_qubit_system):
    """Standard runs as a single stage, so the two-stage bookkeeping stays empty.

    Because there is no warm-up handoff, stage_boundaries and the separate
    global-energy curve (energy_history_global, used only by the two-stage
    methods) are both empty, and the energy and parameter histories have exactly
    one entry per iteration (max_iters total).
    """
    cfg = MethodConfig(depth=1, max_iters=4, seed=0)
    result = standard(two_qubit_system, cfg)
    assert result.method == "standard"
    assert result.stage_boundaries == []
    assert result.energy_history_global == []
    assert len(result.energy_history) == cfg.max_iters
    assert len(result.param_history) == cfg.max_iters


def test_standard_uses_efficientsu2(two_qubit_system):
    cfg = MethodConfig(depth=1, max_iters=3, seed=0)
    result = standard(two_qubit_system, cfg)
    assert isinstance(result.ansatz, EfficientSU2)
    assert result.n_params == result.ansatz.n_params


def test_standard_matches_direct_pipeline(two_qubit_system):
    """standard is exactly a single-stage optimize of EfficientSU2 against the
    full H: reconstruct that pipeline by hand with the same seed and confirm an
    identical energy history (the baseline adds no hidden behavior).
    """
    cfg = MethodConfig(depth=1, max_iters=5, seed=0, optimizer="adam")
    result = standard(two_qubit_system, cfg)

    n = two_qubit_system.n_qubits
    ansatz = EfficientSU2(n, d=cfg.depth)
    params = ansatz.random_params(np.random.default_rng(cfg.seed))
    cost = _make_cost(ansatz, two_qubit_system.hamiltonian, n)
    opt = build_optimizer(cfg.optimizer, seed=cfg.seed, **cfg.optimizer_kwargs)
    _, expected_history, _ = _optimize(cost, params, opt, cfg.max_iters)

    np.testing.assert_allclose(result.energy_history, expected_history, atol=1e-12)


def test_standard_is_deterministic(two_qubit_system):
    """Same seed gives the same run."""
    cfg = MethodConfig(depth=1, max_iters=4, seed=7)
    r1 = standard(two_qubit_system, cfg)
    r2 = standard(two_qubit_system, cfg)
    np.testing.assert_allclose(r1.energy_history, r2.energy_history, atol=1e-12)
    assert r1.final_energy == pytest.approx(r2.final_energy, abs=1e-12)


def test_final_energy_matches_last_param(two_qubit_system):
    """final_energy is the cost at the returned params, so re-scoring the final
    param_history entry reproduces it.
    """
    cfg = MethodConfig(depth=1, max_iters=4, seed=0)
    result = standard(two_qubit_system, cfg)
    n = two_qubit_system.n_qubits
    cost = _make_cost(EfficientSU2(n, d=cfg.depth), two_qubit_system.hamiltonian, n)
    assert float(cost(result.params)) == pytest.approx(result.final_energy, abs=1e-9)
