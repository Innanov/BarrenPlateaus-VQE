#!/usr/bin/env python3
"""
Aggregate step for seeded VQE runs (no computation, no plotting).

Reads every seed_<seed>/ folder of one setting, written by
compute_vqe_analysis.py, and reduces them to the mean and sample standard
deviation over seeds (pandas std, ddof=1). Needs at least two seeds, and refuses
seeds whose settings differ in anything but the seed.

Output in results/<molecule>/vqe_seeds/<run_name>/:
    performance_table_mean_std.csv   <column>_mean and <column>_std per
                                     method+optimizer
    convergence_mean_std.csv         energy_mean and energy_std per iteration
                                     (long format)
    run_parameters.json              the shared settings plus the list of seeds

Plot with plot_vqe_seeds.py.

Usage:
    python scripts/aggregate_vqe_seeds.py --dir results/LiH/vqe_seeds/<run_name>
    python scripts/aggregate_vqe_seeds.py -m LiH --run-name adam_depth4_iters1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.analysis import (  # noqa: E402
    aggregate_convergence,
    aggregate_performance,
)
from src.utils import io  # noqa: E402

# run_parameters.json keys that may differ between seeds of one setting.
_PER_SEED_KEYS = {"seed", "_provenance", "device"}


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dir", help="The results/<molecule>/vqe_seeds/<run_name>/ folder."
    )
    p.add_argument(
        "--molecule",
        "-m",
        help="Molecule, used with --run-name instead of --dir.",
    )
    p.add_argument(
        "--run-name",
        help=(
            "Run name under results/<molecule>/vqe_seeds/. "
            "Optional when there is only one."
        ),
    )
    return p.parse_args(argv)


def shared_parameters(seed_dirs: dict[int, str]) -> dict:
    """Return the settings shared by all seeds, refusing seeds that disagree.

    Args:
        seed_dirs: {seed: seed folder}.

    Returns:
        The common run_parameters.json content without the per-seed keys, plus
        "seeds" (the list of seeds) and "devices" (the devices used).
    """
    params = {}
    for seed, d in seed_dirs.items():
        with open(os.path.join(d, "run_parameters.json"), encoding="utf-8") as fh:
            params[seed] = json.load(fh)
    common = {
        k: v
        for k, v in next(iter(params.values())).items()
        if k not in _PER_SEED_KEYS
    }
    for seed, p in params.items():
        this = {k: v for k, v in p.items() if k not in _PER_SEED_KEYS}
        if this != common:
            diff = sorted(
                k for k in set(this) | set(common) if this.get(k) != common.get(k)
            )
            raise SystemExit(
                f"seed_{seed} has different settings from the others in: {diff}"
            )
    devices = sorted({p.get("device") for p in params.values()} - {None})
    return dict(common, seeds=list(params), devices=devices)


def main(argv=None) -> int:
    """Reduce the seed folders of one setting to mean and std tables."""
    args = parse_args(argv)
    seeds_dir = io.resolve_seeds_dir(args.dir, args.molecule, args.run_name)
    seed_dirs = io.find_seed_dirs(seeds_dir)
    if len(seed_dirs) < 2:
        raise SystemExit(
            f"need at least 2 seed folders in {seeds_dir}, "
            f"found {list(seed_dirs)}"
        )
    params = shared_parameters(seed_dirs)
    if len(params["devices"]) > 1:
        print(
            f"note: seeds ran on different devices {params['devices']}, "
            "so runtime_s mixes them"
        )

    tables = {
        seed: pd.read_csv(os.path.join(d, "performance_table_all_methods.csv"))
        for seed, d in seed_dirs.items()
    }
    curves = {
        seed: io.load_convergence(os.path.join(d, "convergence_history.csv"))
        for seed, d in seed_dirs.items()
    }
    try:
        performance = aggregate_performance(tables)
        convergence = aggregate_convergence(curves)
    except ValueError as err:
        raise SystemExit(str(err)) from err

    perf_path = os.path.join(seeds_dir, "performance_table_mean_std.csv")
    conv_path = os.path.join(seeds_dir, "convergence_mean_std.csv")
    performance.to_csv(perf_path, index=False)
    convergence.to_csv(conv_path, index=False)
    io.write_json(os.path.join(seeds_dir, "run_parameters.json"), params)

    print(f"seeds {list(seed_dirs)} -> {seeds_dir}")
    for row in performance.itertuples(index=False):
        print(
            f"  {row.optimizer:5s} {row.method:14s} "
            f"E = {row.final_energy_mean:.6f} +- {row.final_energy_std:.2e}  "
            f"err = {row.energy_error_mean:.2e} +- {row.energy_error_std:.2e}  "
            f"fid = {row.fidelity_mean:.4f} +- {row.fidelity_std:.4f}"
        )
    print(f"wrote {perf_path}\nwrote {conv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
