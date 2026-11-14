#!/usr/bin/env python3
"""
Analysis step for the VQE runs (no computation): seed tables and figures.

Reads the runs saved by compute_vqe_analysis.py and, chosen by the options:

    -m LiH --run-name R    aggregate the seeds of results/LiH/vqe_seeds/R, then
                           draw its convergence figure
    --dir FOLDER           the same for a seeds folder, or only the figure for
                           one seed or one unseeded run (convergence_history.csv)
    -m LiH --d-sat [N]     aggregate the runs at d_sat and 2 x d_sat, then draw
                           them side by side

Aggregation reads every seed_<seed>/ folder of one setting and reduces them to
the mean and sample standard deviation over seeds (pandas std, ddof=1), plus the
16th and 84th percentile per iteration. It needs at least two seeds and refuses
seeds whose settings differ in anything but the seed. --no-aggregate draws from
the tables already saved.

Every figure with one optimizer has a zoomed inset of iterations 20% to 80%
(--window), and --no-inset leaves it out. How the inset picks its range and
place is in src/utils/plotting.py. The side-by-side figure shares the energy axis
and one inset range across both panels, so the depths compare directly. d_sat is
standard's saturation depth from results/d_sat.csv (written by
gradient_variance.py --d-sat), or the number given after --d-sat.

Output in results/<molecule>/vqe_seeds/<run_name>/:
    performance_table_mean_std.csv   <column>_mean and <column>_std per
                                     method and optimizer
    convergence_mean_std.csv         energy_mean, energy_std, energy_p16 and
                                     energy_p84 per iteration (long format)
    run_parameters.json              the shared settings plus the list of seeds
    energy_convergence_all_methods.pdf
    energy_convergence_overlay.pdf   only when several optimizers were run
and results/<molecule>/vqe_seeds/energy_convergence_d_sat.pdf for --d-sat.

Usage:
    python scripts/vqe_convergence.py -m LiH --run-name adam_depth4_iters1000
    python scripts/vqe_convergence.py --dir results/LiH/vqe_seeds/<run_name>
    python scripts/vqe_convergence.py --dir results/LiH/vqe_seeds/<run>/seed_0
    python scripts/vqe_convergence.py -m LiH --d-sat
    python scripts/vqe_convergence.py -m LiH --d-sat 7 --no-inset
"""

from __future__ import annotations

import argparse
import csv
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
from src.utils import plotting as plots  # noqa: E402

# run_parameters.json keys that may differ between seeds of one setting.
_PER_SEED_KEYS = {"seed", "_provenance", "device"}


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dir", help="A seeds folder, one seed's folder, or an unseeded run."
    )
    p.add_argument("--molecule", "-m", help="Molecule name.")
    p.add_argument(
        "--run-name",
        help=(
            "Run name under results/<molecule>/vqe_seeds/. "
            "Optional when there is only one."
        ),
    )
    p.add_argument(
        "--d-sat",
        nargs="?",
        type=int,
        const=0,
        default=None,
        metavar="N",
        help=(
            "Draw d_sat and 2 x d_sat side by side (N, or standard's d_sat from "
            "results/d_sat.csv)."
        ),
    )
    p.add_argument(
        "--run-template",
        default="adam_depth{depth}_iters1000",
        help="Run folder name for --d-sat, with {depth} for the depth.",
    )
    p.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Draw from the saved tables without aggregating the seeds again.",
    )
    p.add_argument(
        "--no-inset", action="store_true", help="Leave out the zoomed inset."
    )
    p.add_argument(
        "--window",
        type=float,
        nargs=2,
        default=plots.INSET_WINDOW,
        metavar=("START", "STOP"),
        help="Inset iterations as fractions of the iteration budget.",
    )
    p.add_argument(
        "--stuck",
        type=float,
        default=plots.INSET_STUCK,
        help="Median above FCI (Ha) over which a method is left out of the inset "
        "range.",
    )
    args = p.parse_args(argv)
    if args.d_sat is not None and not args.molecule:
        p.error("--d-sat needs --molecule")
    if not (args.dir or args.molecule):
        p.error("give --dir or --molecule")
    return args


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


def aggregate(seeds_dir: str) -> None:
    """Reduce the seed folders of one setting to mean and std tables."""
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


def load_results(folder: str) -> dict:
    """Return {optimizer: [result, ...]} of a seeds folder or a single run."""
    mean_std = os.path.join(folder, "convergence_mean_std.csv")
    history = os.path.join(folder, "convergence_history.csv")
    if os.path.isfile(mean_std):
        return io.load_convergence_mean_std(mean_std)
    if os.path.isfile(history):
        return io.load_convergence(history)
    raise SystemExit(
        f"no convergence_mean_std.csv or convergence_history.csv in {folder}"
    )


def run_figure(folder: str, args) -> None:
    """Aggregate a seeds folder if asked, then draw its convergence figure."""
    if io.find_seed_dirs(folder) and not args.no_aggregate:
        aggregate(folder)
    fci, label = io.system_info(folder)
    paths = plots.write_convergence_figures(
        load_results(folder),
        folder,
        fci_energy=fci,
        label=label,
        max_iters=io.main_iters(folder),
        inset=not args.no_inset,
        window=tuple(args.window),
        stuck=args.stuck,
    )
    for path in paths:
        print(f"wrote {path}")


def standard_d_sat(molecule: str) -> int:
    """Return standard's d_sat of a molecule from results/d_sat.csv."""
    path = os.path.join(io.RESULTS_ROOT, "d_sat.csv")
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["molecule"] == molecule and row["method"] == "standard":
                if not row["d_sat"]:
                    raise SystemExit(
                        f"{molecule}: standard has no d_sat in {path}"
                    )
                return int(float(row["d_sat"]))
    raise SystemExit(f"{molecule}: no standard row in {path}")


def d_sat_figure(args) -> None:
    """Draw one molecule's convergence at d_sat and 2 x d_sat side by side."""
    d_sat = args.d_sat or standard_d_sat(args.molecule)
    seeds_root = os.path.join(io.RESULTS_ROOT, args.molecule, io.SEEDS_ANALYSIS)
    panels = []
    label = None
    for depth, symbol in [
        (d_sat, r"$d_\mathrm{sat}$"),
        (2 * d_sat, r"$2\,d_\mathrm{sat}$"),
    ]:
        folder = os.path.join(seeds_root, args.run_template.format(depth=depth))
        if not os.path.isdir(folder):
            raise SystemExit(f"no run folder {folder}")
        if not args.no_aggregate:
            aggregate(folder)
        fci, label = io.system_info(folder)
        results = next(iter(load_results(folder).values()))
        n_iters = io.main_iters(folder) or max(
            len(r.energy_history) for r in results
        )
        panels.append((f"depth {depth} = {symbol}", results, fci, n_iters))
    out = os.path.join(seeds_root, "energy_convergence_d_sat.pdf")
    plots.plot_convergence_panels(
        panels,
        out,
        title=label,
        inset=not args.no_inset,
        window=tuple(args.window),
        stuck=args.stuck,
    )
    print(f"wrote {out}")


def main(argv=None) -> int:
    """Run the action the options select."""
    args = parse_args(argv)
    if args.d_sat is not None:
        d_sat_figure(args)
    else:
        run_figure(
            io.resolve_seeds_dir(args.dir, args.molecule, args.run_name), args
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
