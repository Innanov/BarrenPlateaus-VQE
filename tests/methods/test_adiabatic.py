"""Tests for the adiabatic staged-anneal VQE.

This method generalizes qubap's VQE_adiabatic. qubap ramps the mixing coefficient
s continuously per iteration, while we hold H fixed across adiabatic_steps discrete
stages, which is a superset: setting adiabatic_steps = max_iters gives one stage
per iteration and recovers qubap's per-iteration granularity as a limit. These
tests pin the schedule endpoints (s=0 pure local, s=1 pure global, exactly qubap's
ramp bounds), the stage-count clamping, and that superset limit.

Run with: pytest tests/methods/test_adiabatic.py
"""

import numpy as np
import pennylane as qml
import pytest

from src.core.methods import MethodConfig, adiabatic
from src.core.methods.adiabatic import _interpolate


def _matrix(hamiltonian, n):
    return qml.matrix(hamiltonian, wire_order=range(n))


def test_interpolate_endpoints_match_qubap_ramp():
    """H(s) = (1-s) H_local + s H_global. At s=0 it is pure local, at s=1 pure
    global. These are exactly the bounds qubap's continuous ramp moves between.
    """
    n = 2
    h_local = qml.Hamiltonian([1.0, 0.5], [qml.PauliZ(0), qml.PauliX(1)])
    h_global = qml.Hamiltonian([1.0, 1.0], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0)])

    at0 = _matrix(_interpolate(h_local, h_global, 0.0), n)
    at1 = _matrix(_interpolate(h_local, h_global, 1.0), n)
    np.testing.assert_allclose(at0, _matrix(h_local, n), atol=1e-10)
    np.testing.assert_allclose(at1, _matrix(h_global, n), atol=1e-10)


def test_interpolate_midpoint_is_average():
    n = 2
    h_local = qml.Hamiltonian([1.0], [qml.PauliZ(0)])
    h_global = qml.Hamiltonian([1.0], [qml.PauliX(1)])
    mid = _matrix(_interpolate(h_local, h_global, 0.5), n)
    expected = 0.5 * _matrix(h_local, n) + 0.5 * _matrix(h_global, n)
    np.testing.assert_allclose(mid, expected, atol=1e-10)


def test_stage_count_never_exceeds_iters(two_qubit_system):
    """n_stages is clamped to at most max_iters (never more stages than iters),
    so energy_history has one entry per iteration actually run.

    Here max_iters=5 with adiabatic_steps=10 clamps to 5 stages of 1 iteration.
    """
    cfg = MethodConfig(depth=1, max_iters=5, adiabatic_steps=10, seed=0)
    result = adiabatic(two_qubit_system, cfg)
    assert len(result.energy_history) == 5


def test_total_iters_exact_when_steps_do_not_divide(two_qubit_system):
    """The max_iters total is spread across the stages so it sums to exactly
    max_iters, even when adiabatic_steps does not divide it (7 into 20 would be 2
    per stage = 14 with naive integer division, undershooting by 6). This keeps
    the budget equal to every other method.
    """
    for max_iters, steps in [(20, 7), (20, 3), (13, 5), (7, 4)]:
        cfg = MethodConfig(depth=1, max_iters=max_iters, adiabatic_steps=steps, seed=0)
        result = adiabatic(two_qubit_system, cfg)
        assert len(result.energy_history) == max_iters


def test_single_stage_is_pure_global(two_qubit_system):
    """adiabatic_steps=1 means s=1 for the whole run: a pure-global optimization
    (the s ramp collapses to the global endpoint).

    With one stage the whole run optimizes the global H. energy_history records
    the energy before each step, while energy_history_global re-scores the param
    after each step, so the two are the same single-H trajectory offset by one:
    the before-step energy of iteration k+1 equals the after-step energy of k.
    """
    cfg = MethodConfig(depth=1, max_iters=4, adiabatic_steps=1, seed=0)
    result = adiabatic(two_qubit_system, cfg)
    assert len(result.energy_history) == 4
    np.testing.assert_allclose(
        result.energy_history[1:], result.energy_history_global[:-1], atol=1e-9
    )


def test_superset_limit_one_stage_per_iteration(two_qubit_system):
    """adiabatic_steps == max_iters gives per_stage=1: one stage per iteration,
    reaching the per-iteration granularity of qubap's continuous ramp.
    """
    cfg = MethodConfig(depth=1, max_iters=6, adiabatic_steps=6, seed=0)
    result = adiabatic(two_qubit_system, cfg)
    assert len(result.energy_history) == cfg.max_iters


def test_global_curve_length_matches_param_path(two_qubit_system):
    cfg = MethodConfig(depth=1, max_iters=6, adiabatic_steps=3, seed=0)
    result = adiabatic(two_qubit_system, cfg)
    assert len(result.energy_history_global) == len(result.param_history)
    assert result.stage_boundaries == []
    assert result.energy_history_global[-1] == pytest.approx(result.final_energy, abs=1e-6)
