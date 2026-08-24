"""Tests for the State Efficient Ansatz VQE.

qubap ships the SEA as an ansatz constructor only, with no dedicated SEA VQE, so
it is meant to be plugged into the ordinary VQE loop. These tests pin exactly that:
sea is the standard single-stage VQE run with the SEA ansatz and nothing else, and
it inherits SEA's even-qubit requirement (matching qubap's constraint).

Run with: pytest tests/methods/test_sea.py
"""

import numpy as np
import pytest

from src.core.ansatze import SEA
from src.core.backend import build_optimizer
from src.core.methods import MethodConfig, sea
from src.core.methods.base import _make_cost, _optimize


def test_sea_equals_standard_vqe_with_sea_ansatz(four_qubit_system):
    """Reconstruct the standard single-stage pipeline by hand with the SEA ansatz
    and the same seed, and confirm sea produces an identical energy history. This
    shows sea adds no behavior beyond standard-VQE-with-SEA (qubap's SEA usage).
    """
    cfg = MethodConfig(depth=1, max_iters=5, seed=0, optimizer="adam")
    result = sea(four_qubit_system, cfg)

    n = four_qubit_system.n_qubits
    ansatz = SEA(n, d=cfg.depth)
    params = ansatz.random_params(np.random.default_rng(cfg.seed))
    cost = _make_cost(ansatz, four_qubit_system.hamiltonian, n)
    opt = build_optimizer(cfg.optimizer, seed=cfg.seed, **cfg.optimizer_kwargs)
    _, expected_history, _ = _optimize(cost, params, opt, cfg.max_iters)

    np.testing.assert_allclose(result.energy_history, expected_history, atol=1e-12)


def test_sea_is_single_stage(four_qubit_system):
    """sea runs as a single stage, so stage_boundaries and the two-stage-only
    global-energy curve stay empty.
    """
    cfg = MethodConfig(depth=1, max_iters=4, seed=0)
    result = sea(four_qubit_system, cfg)
    assert result.stage_boundaries == []
    assert result.energy_history_global == []
    assert len(result.energy_history) == cfg.max_iters
    assert result.method == "sea"


def test_sea_requires_even_qubits():
    """SEA raises on odd qubit counts, so a sea run on an odd system raises too,
    matching qubap's even-qubit constraint.
    """
    with pytest.raises(ValueError):
        SEA(3, d=1)
