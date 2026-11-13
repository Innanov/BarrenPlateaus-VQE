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


def test_point_reports_the_mean_variance_over_parameters(system):
    """mean_grad_var averages every parameter, grad_var_param0 keeps the first."""
    from src.core.analysis.gradient_scaling import _point

    grads = np.array([[1.0, 0.0, 2.0], [3.0, 4.0, 2.0]])
    p = _point(system, "efficient_su2", 1, 3, grads, 0.0, "energy", "random")
    per_param = np.var(grads, axis=0)
    assert np.isclose(p.mean_grad_var, per_param.mean())
    assert np.isclose(p.std_grad_var, per_param.std())
    assert np.isclose(p.grad_var_param0, per_param[0])


def test_old_files_get_the_mean_from_the_norms(tmp_path):
    """A file without grad_var_param0 is read back as the mean over parameters."""
    from src.utils.io import load_gradient_points

    path = tmp_path / "scaling_analysis_data.csv"
    path.write_text(
        "n_layers,n_parameters,mean_grad_var,std_grad_var,mean_grad_norm,"
        "std_grad_norm\n"
        "2,4,0.5,0.1,3.0,1.0\n"
        "1,4,0.5,0.1,2.0,0.0\n",
        encoding="utf-8",
    )
    points = load_gradient_points(str(path))
    assert [p.n_layers for p in points] == [1, 2]
    assert np.isclose(points[0].mean_grad_var, 2.0**2 / 4)
    assert np.isclose(points[1].mean_grad_var, (3.0**2 + 1.0**2) / 4)
    assert load_gradient_points(str(path), max_depth=1)[0].n_layers == 1


def test_same_seed_gives_identical_gradients(system):
    """A sweep rerun with the same seed reproduces every parameter's variance."""
    first = gradient_scaling(system, depths=[1, 2], n_samples=3, seed=7)
    again = gradient_scaling(system, depths=[1, 2], n_samples=3, seed=7)
    for a, b in zip(first, again, strict=False):
        assert np.array_equal(a.grad_var_per_parameter, b.grad_var_per_parameter)
        assert a.mean_grad_var == b.mean_grad_var


def test_saved_per_parameter_variances_match_the_summary(system, tmp_path):
    """grad_var_per_parameter.csv holds the values behind mean and std."""
    import csv
    import importlib.util
    import os

    script = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "scripts",
        "compute_gradient_scaling.py",
    )
    spec = importlib.util.spec_from_file_location("compute_gs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    points = gradient_scaling(system, depths=[1, 2], n_samples=3, seed=7)
    module.write_points(str(tmp_path), points)
    with open(tmp_path / "grad_var_per_parameter.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(tmp_path / "scaling_analysis_data.csv", encoding="utf-8") as fh:
        summary = list(csv.DictReader(fh))
    assert "grad_var_per_parameter" not in summary[0]
    for p, row in zip(points, summary, strict=False):
        saved = np.array(
            [
                float(r["grad_var"])
                for r in rows
                if int(r["n_layers"]) == p.n_layers
            ]
        )
        assert len(saved) == p.n_parameters
        assert float(row["mean_grad_var"]) == saved.mean()
        assert float(row["std_grad_var"]) == saved.std()
        assert float(row["grad_var_param0"]) == saved[0]


def test_point_standard_error_of_the_mean_variance(system):
    """sem_grad_var is the standard error of the per-sample mean deviation."""
    from src.core.analysis.gradient_scaling import _point

    rng = np.random.default_rng(3)
    grads = rng.normal(size=(50, 4))
    p = _point(system, "efficient_su2", 1, 4, grads, 0.0, "energy", "random")
    per_sample = np.mean((grads - grads.mean(axis=0)) ** 2, axis=1)
    assert np.isclose(per_sample.mean(), p.mean_grad_var)
    assert np.isclose(p.sem_grad_var, per_sample.std(ddof=1) / np.sqrt(50))


def test_loader_reads_the_standard_error(tmp_path):
    """The band comes from sem_grad_var, and is NaN for files without it."""
    from src.utils.io import load_gradient_points

    header = (
        "n_layers,n_parameters,mean_grad_var,std_grad_var,mean_grad_norm,"
        "std_grad_norm,grad_var_param0"
    )
    path = tmp_path / "scaling_analysis_data.csv"
    newline = chr(10)
    path.write_text(
        header + newline + "1,3,2.0,1.0,1.0,0.0,1.0" + newline, encoding="utf-8"
    )
    assert np.isnan(load_gradient_points(str(path))[0].sem_grad_var)
    path.write_text(
        header
        + ",sem_grad_var"
        + newline
        + "1,3,2.0,1.0,1.0,0.0,1.0,0.1"
        + newline,
        encoding="utf-8",
    )
    assert load_gradient_points(str(path))[0].sem_grad_var == 0.1


def test_loader_reduces_over_seeds(tmp_path):
    """Several seeds give their mean and a band of one std over seeds."""
    from src.utils.io import load_gradient_points

    newline = chr(10)
    rows = [
        "seed,n_layers,n_parameters,mean_grad_var,std_grad_var,"
        "mean_grad_norm,std_grad_norm,grad_var_param0,sem_grad_var"
    ]
    values = {0: 1.0, 1: 2.0, 2: 4.0}
    rows += [f"{s},1,3,{v},0.5,1.0,0.0,1.0,0.1" for s, v in values.items()]
    path = tmp_path / "scaling_analysis_data.csv"
    path.write_text(newline.join(rows) + newline, encoding="utf-8")
    (point,) = load_gradient_points(str(path))
    assert point.n_seeds == 3
    assert point.mean_grad_var == pytest.approx(np.mean(list(values.values())))
    assert point.seed_std_grad_var == pytest.approx(
        np.std(list(values.values()), ddof=1)
    )


def test_sweep_records_its_seed_and_seeds_differ(system):
    """Each point carries its seed, and another seed draws other samples."""
    a = gradient_scaling(system, depths=[1], n_samples=3, seed=0)[0]
    b = gradient_scaling(system, depths=[1], n_samples=3, seed=1)[0]
    assert (a.seed, b.seed) == (0, 1)
    assert a.mean_grad_var != b.mean_grad_var
