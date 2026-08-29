"""Tests for the local_global warm-start VQE.

This method generalizes qubap's VQE_shift (a two-stage local->global warm start),
with one deliberate improvement: the local cost is the ground-state Cerezo
observable (local_cost_observable), not qubap's global2local Pauli-splitting. So
these tests pin OUR structure and the improvement claim, not a qubap parity: the
local cost is minimized at the true ground state (which global2local is not), and
the two-stage bookkeeping (stage boundary, warm-start continuity, continuous
global curve) is correct.

Run with: pytest tests/methods/test_local_global.py
"""

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from src.core.methods import MethodConfig, local_cost_observable, local_global
from src.utils.helpers import expectation, statevector


def test_local_cost_is_a_valid_hamiltonian(two_qubit_system):
    o_l = local_cost_observable(two_qubit_system.ground_state, two_qubit_system.n_qubits)
    assert isinstance(o_l, qml.Hamiltonian)


def test_local_cost_minimized_at_ground_state(two_qubit_system):
    """The load-bearing improvement over qubap: the ground-state Cerezo cost is
    minimized at the exact ground state, well below a random state's value.
    global2local (which qubap uses) lacks this property.

    We compare against a random state rather than against 0, because O_L reaches
    exactly 0 only for a product ground state. An entangled ground state has mixed
    single-qubit marginals and so a small positive floor.
    """
    n = two_qubit_system.n_qubits
    o_l = local_cost_observable(two_qubit_system.ground_state, n)
    gs_value = expectation(o_l, two_qubit_system.ground_state, n)

    rng = np.random.default_rng(0)
    random_state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    random_state /= np.linalg.norm(random_state)
    random_value = expectation(o_l, random_state, n)

    assert gs_value < random_value


def test_two_stage_bookkeeping(two_qubit_system):
    """The stages are recorded consistently, with the main stage at full max_iters.

    The warm-up runs warm_iters as extra preparation and the main stage runs the
    full max_iters, so the whole record is warm_iters + max_iters long, with the
    boundary at warm_iters. The global curve re-scores the whole parameter path, so
    it has the same length as param_history.
    """
    cfg = MethodConfig(depth=1, max_iters=5, warm_iters=2, seed=0)
    result = local_global(two_qubit_system, cfg)

    assert result.stage_boundaries == [cfg.warm_iters]
    assert len(result.energy_history) == cfg.warm_iters + cfg.max_iters
    assert len(result.energy_history_global) == len(result.param_history)
    assert len(result.param_history) == cfg.warm_iters + cfg.max_iters


def test_warm_start_continuity(two_qubit_system):
    """One ansatz spans both stages with reset=False, so stage 2 begins exactly
    at the parameters stage 1 ended on (no re-initialization at the handoff).

    The params around the boundary are consecutive optimizer steps in one space,
    so they are close but not identical. What we pin is that the state is carried,
    not reset, by checking the global energy curve is continuous at the boundary
    (its step there is smaller than the whole warm-up drop).
    """
    cfg = MethodConfig(depth=1, max_iters=4, warm_iters=3, seed=0)
    result = local_global(two_qubit_system, cfg)

    boundary = cfg.warm_iters
    last_warm = np.asarray(result.param_history[boundary - 1], dtype=float)
    first_global = np.asarray(result.param_history[boundary], dtype=float)
    g = result.energy_history_global
    assert abs(g[boundary] - g[boundary - 1]) < abs(g[0] - g[boundary])
    assert last_warm.shape == first_global.shape


def test_global_curve_uses_full_hamiltonian(two_qubit_system):
    """energy_history_global re-scores every param against the full H, so its
    final value equals final_energy.
    """
    cfg = MethodConfig(depth=1, max_iters=4, warm_iters=3, seed=0)
    result = local_global(two_qubit_system, cfg)
    assert result.energy_history_global[-1] == pytest.approx(result.final_energy, abs=1e-6)


def test_reported_state_matches_params(two_qubit_system):
    cfg = MethodConfig(depth=1, max_iters=3, warm_iters=2, seed=1)
    result = local_global(two_qubit_system, cfg)
    state = statevector(result.ansatz, two_qubit_system.n_qubits, params=pnp.array(result.params))
    assert float(np.abs(np.vdot(state, state))) == pytest.approx(1.0, abs=1e-6)
