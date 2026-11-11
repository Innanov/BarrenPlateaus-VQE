"""Tests for the mean and std over seeded VQE runs.

Pins that the per-method statistics match numpy's mean and sample std (ddof=1),
that method order is kept, and that missing runs, unequal curve lengths and a
single seed are refused rather than silently averaged.

Run with: pytest tests/analysis/test_seeds.py
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.core.analysis import aggregate_convergence, aggregate_performance


def _table(energies):
    """A performance table with one row per method, final energies as given."""
    return pd.DataFrame(
        [
            dict(
                optimizer="adam",
                method=m,
                final_energy=e,
                energy_error=e + 1.0,
                fidelity=0.5,
                n_params=8,
                runtime_s=1.0,
            )
            for m, e in energies.items()
        ]
    )


def _curves(histories):
    """A results_by_opt mapping for one seed, as io.load_convergence returns."""
    return {"adam": [SimpleNamespace(method=m, energy_history=h) for m, h in histories.items()]}


def test_performance_mean_and_sample_std():
    """Each method's mean and std match numpy with ddof=1, in first-seen order."""
    tables = {
        0: _table({"standard": -1.0, "sea": -2.0}),
        1: _table({"standard": -1.2, "sea": -2.0}),
        2: _table({"standard": -1.1, "sea": -2.0}),
    }
    out = aggregate_performance(tables)
    assert list(out["method"]) == ["standard", "sea"]
    std = out.set_index("method")
    assert std.loc["standard", "final_energy_mean"] == pytest.approx(-1.1)
    assert std.loc["standard", "final_energy_std"] == pytest.approx(
        np.std([-1.0, -1.2, -1.1], ddof=1)
    )
    assert std.loc["sea", "final_energy_std"] == pytest.approx(0.0)
    assert (out["n_seeds"] == 3).all()
    assert (out["n_params"] == 8).all()


def test_performance_refuses_missing_run_and_single_seed():
    """A method absent from one seed, or a lone seed, raises instead of averaging."""
    with pytest.raises(ValueError):
        aggregate_performance(
            {0: _table({"standard": -1.0, "sea": -2.0}), 1: _table({"standard": -1.0})}
        )
    with pytest.raises(ValueError):
        aggregate_performance({0: _table({"standard": -1.0})})


def test_convergence_mean_and_std_per_iteration():
    """Per-iteration mean and std match numpy across seeds."""
    curves = {0: _curves({"standard": [0.0, -1.0]}), 1: _curves({"standard": [-2.0, -1.0]})}
    out = aggregate_convergence(curves)
    assert list(out["iteration"]) == [0, 1]
    assert list(out["energy_mean"]) == pytest.approx([-1.0, -1.0])
    assert list(out["energy_std"]) == pytest.approx([np.std([0.0, -2.0], ddof=1), 0.0])
    assert (out["n_seeds"] == 2).all()


def test_convergence_refuses_unequal_lengths():
    """Curves of different length for the same run mean different settings."""
    curves = {0: _curves({"standard": [0.0, -1.0]}), 1: _curves({"standard": [0.0]})}
    with pytest.raises(ValueError):
        aggregate_convergence(curves)
