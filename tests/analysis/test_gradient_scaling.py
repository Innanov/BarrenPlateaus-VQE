"""Tests for the gradient-variance measurements (random and warm-start samples).

The warm-start tests pin the property the barren-plateau comparison relies on:
the point warm_start_gradient_scaling differentiates is exactly the handoff
point of the method's own run, the parameters its energy stage starts from.

Run with: pytest tests/analysis/test_gradient_scaling.py
"""

import numpy as np
import pytest

from src.core.analysis import (
    COSTS,
    cost_observable,
    gradient_scaling,
    warm_start_gradient_scaling,
)
from src.core.methods import MethodConfig, local_global, pretrained
from src.utils.helpers import expectation, ising_system


@pytest.fixture
def system():
    """A 4-qubit Ising MolecularSystem (even n, so SEA and MPS are valid)."""
    return ising_system(4)


def test_energy_cost_is_the_molecular_hamiltonian(system):
    assert cost_observable(system, "energy") is system.hamiltonian


def test_first_stage_costs_are_minimized_at_hf(system):
    """At the HF state local_cost is 0 and hf_reference is -n, its minimum."""
    n = system.n_qubits
    local = cost_observable(system, "local_cost")
    reference = cost_observable(system, "hf_reference")
    assert np.isclose(expectation(local, system.hf_state, n), 0.0, atol=1e-12)
    assert np.isclose(expectation(reference, system.hf_state, n), -n, atol=1e-12)


def test_unknown_cost_raises(system):
    with pytest.raises(ValueError):
        cost_observable(system, "global2local")


@pytest.mark.parametrize("cost", COSTS)
def test_random_sweep_records_cost_and_start(system, cost):
    points = gradient_scaling(system, depths=[1, 2], n_samples=4, cost=cost)
    assert [p.n_layers for p in points] == [1, 2]
    for p in points:
        assert (p.cost, p.start, p.n_samples) == (cost, "random", 4)
        assert np.isfinite(p.mean_grad_var) and p.mean_grad_var >= 0
        assert np.isnan(p.mean_energy_error)


def test_default_sweep_matches_explicit_energy_cost(system):
    """Adding the cost option leaves the default measurement unchanged."""
    default = gradient_scaling(system, depths=[2], n_samples=3, seed=5)[0]
    explicit = gradient_scaling(
        system, depths=[2], n_samples=3, seed=5, cost="energy"
    )[0]
    assert default.mean_grad_var == explicit.mean_grad_var
    assert default.mean_grad_norm == explicit.mean_grad_norm


def test_local_global_warm_start_is_its_handoff_point(system):
    """The warm-start energy is local_global's energy at its stage boundary."""
    warm_iters = 5
    run = local_global(
        system, MethodConfig(depth=1, max_iters=2, warm_iters=warm_iters, seed=3)
    )
    handoff_energy = run.energy_history_global[warm_iters - 1]
    point = warm_start_gradient_scaling(
        system,
        "local_global",
        depths=[1],
        n_samples=1,
        seed=3,
        warm_iters=warm_iters,
    )[0]
    assert (point.start, point.cost, point.n_samples) == (
        "warm_start",
        "energy",
        1,
    )
    assert np.isclose(
        point.mean_energy_error, handoff_energy - system.fci_energy, atol=1e-10
    )


def test_pretrained_warm_start_is_its_transfer_point(system):
    """The warm-start energy equals pretrained's first full-stage energy."""
    run = pretrained(
        system, MethodConfig(depth=1, max_iters=2, warm_iters=5, seed=2)
    )
    point = warm_start_gradient_scaling(
        system, "pretrained", depths=[1], n_samples=1, seed=2, warm_iters=5
    )[0]
    assert point.n_parameters == run.n_params
    assert np.isclose(
        point.mean_energy_error,
        run.energy_history_global[0] - system.fci_energy,
        atol=1e-10,
    )


def test_warm_start_samples_use_consecutive_seeds(system):
    """Two samples from seed 0 match the mean of seeds 0 and 1 run one by one."""
    both = warm_start_gradient_scaling(
        system, "local_global", [1], n_samples=2, warm_iters=3
    )[0]
    first = warm_start_gradient_scaling(
        system, "local_global", [1], 1, seed=0, warm_iters=3
    )[0]
    second = warm_start_gradient_scaling(
        system, "local_global", [1], 1, seed=1, warm_iters=3
    )[0]
    expected = (first.mean_energy_error + second.mean_energy_error) / 2
    assert np.isclose(both.mean_energy_error, expected, atol=1e-12)


def test_warm_start_rejects_single_stage_methods(system):
    with pytest.raises(ValueError):
        warm_start_gradient_scaling(system, "standard", depths=[1])
