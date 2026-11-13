"""Mean and standard deviation over seeded VQE runs.

Each seed draws different initial parameters (MethodConfig.seed), so repeating a
run over several seeds shows how much a method's result depends on where it
starts. These helpers reduce the per-seed outputs of compute_vqe_analysis.py to
a mean and a sample standard deviation (pandas std, ddof=1) per method.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Performance-table columns that vary from seed to seed.
STAT_COLUMNS = ("final_energy", "energy_error", "fidelity", "runtime_s")


def aggregate_performance(tables: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Reduce per-seed performance tables to mean and std per method.

    Args:
        tables: {seed: performance table}, each with the columns of
            performance_table_all_methods.csv (optimizer, method, final_energy,
            energy_error, fidelity, n_params, runtime_s).

    Returns:
        One row per (optimizer, method) in first-seen order, with n_seeds,
        n_params, and <column>_mean and <column>_std for each column in
        STAT_COLUMNS.

    Raises:
        ValueError: If fewer than two seeds are given, or a (optimizer, method)
            pair is missing from some seed.
    """
    if len(tables) < 2:
        raise ValueError(f"need at least 2 seeds for a std, got {len(tables)}")
    frames = [df.assign(seed=seed) for seed, df in tables.items()]
    data = pd.concat(frames, ignore_index=True)
    keys = ["optimizer", "method"]
    grouped = data.groupby(keys, sort=False)
    counts = grouped["seed"].nunique()
    incomplete = counts[counts != len(tables)]
    if not incomplete.empty:
        raise ValueError(
            f"not every seed has these runs: {list(incomplete.index)}"
        )

    stats = grouped[list(STAT_COLUMNS)].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    out = pd.concat(
        [counts.rename("n_seeds"), grouped["n_params"].first(), stats], axis=1
    ).reset_index()
    return out


def aggregate_convergence(curves: dict[int, dict]) -> pd.DataFrame:
    """Reduce per-seed convergence curves to statistics per iteration.

    energy_p16 and energy_p84 are the 16th and 84th percentiles over seeds
    (numpy's linear interpolation), the band the convergence figures shade. They
    hold the middle 68% of the seeds whatever the shape of the spread, where
    mean +- std does so only for normally distributed values, and stuck seeds
    make the spread far from normal.

    Args:
        curves: {seed: results_by_opt}, where results_by_opt is what
            io.load_convergence returns ({optimizer: [result with method and
            energy_history, ...]}).

    Returns:
        A long table with columns optimizer, method, iteration, energy_mean,
        energy_std, energy_p16, energy_p84, n_seeds.

    Raises:
        ValueError: If fewer than two seeds are given, a run is missing from
            some seed, or the seeds' curves for one run differ in length (the
            methods have no early stopping, so equal settings give equal
            lengths).
    """
    if len(curves) < 2:
        raise ValueError(f"need at least 2 seeds for a std, got {len(curves)}")
    # {(optimizer, method): {seed: energy_history}}, in first-seen order.
    runs: dict[tuple[str, str], dict[int, list]] = {}
    for seed, results_by_opt in curves.items():
        for opt, results in results_by_opt.items():
            for r in results:
                runs.setdefault((opt, r.method), {})[seed] = r.energy_history

    rows = []
    for (opt, method), by_seed in runs.items():
        if len(by_seed) != len(curves):
            missing = sorted(set(curves) - set(by_seed))
            raise ValueError(f"{opt}/{method} is missing for seeds {missing}")
        lengths = {seed: len(h) for seed, h in by_seed.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"{opt}/{method} curve lengths differ across seeds: {lengths}"
            )
        energies = np.array(
            list(by_seed.values()), dtype=float
        )  # (n_seeds, n_iters)
        mean = energies.mean(axis=0)
        std = energies.std(axis=0, ddof=1)
        p16, p84 = np.percentile(energies, [16, 84], axis=0)
        for it in range(energies.shape[1]):
            rows.append(
                dict(
                    optimizer=opt,
                    method=method,
                    iteration=it,
                    energy_mean=mean[it],
                    energy_std=std[it],
                    energy_p16=p16[it],
                    energy_p84=p84[it],
                    n_seeds=energies.shape[0],
                )
            )
    return pd.DataFrame(rows)


def replace_methods(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Replace the (optimizer, method) runs of old with the ones in new.

    Used to rerun some methods of a seed without recomputing the others. Works on
    the performance table (one row per run) and the convergence history (one row
    per iteration) alike. Runs keep their place in old, runs that only new has go
    at the end, and the result keeps old's column order.

    Args:
        old: The seed's existing table.
        new: The table for the rerun methods, with the same columns.

    Returns:
        The merged table.

    Raises:
        ValueError: If the two tables have different columns.
    """
    if set(old.columns) != set(new.columns):
        raise ValueError(
            f"columns differ: {sorted(old.columns)} vs {sorted(new.columns)}"
        )
    keys = ["optimizer", "method"]
    old_runs = list(dict.fromkeys(old[keys].itertuples(index=False, name=None)))
    new_runs = list(dict.fromkeys(new[keys].itertuples(index=False, name=None)))
    parts = []
    for run in old_runs + [r for r in new_runs if r not in old_runs]:
        src = new if run in new_runs else old
        parts.append(
            src[(src["optimizer"] == run[0]) & (src["method"] == run[1])]
        )
    return pd.concat(parts, ignore_index=True)[list(old.columns)]
