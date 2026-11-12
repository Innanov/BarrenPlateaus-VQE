#!/usr/bin/env python3
"""
Compute step for the VQE analysis, one seed per call (no plotting).

Runs the five VQE methods under each optimizer from the initial parameters drawn
with --seed, and saves the data needed to rebuild the figures. Run it once per
seed with the same other settings, then reduce the seeds with
aggregate_vqe_seeds.py and plot with plot_vqe_seeds.py. plot_vqe_analysis.py
plots a single seed.

Output in results/<molecule>/vqe_seeds/<run_name>/seed_<seed>/:
    performance_table_all_methods.csv   final energy/error/fidelity per method+optimizer
    convergence_history.csv             per-iteration energy (long format)
    run_parameters.json                 provenance (device, optimizers, seed, run_name)

run_name defaults to the settings the seeds must share, for example
adam_depth4_iters1000, so every seed of one setting lands in the same folder.

Running only some --methods on a seed that already exists replaces just those
methods and keeps the others (see save_seed), so one method can be rerun after a
fix without recomputing the rest.

Usage:
    python scripts/compute_vqe_analysis.py --molecule LiH --optimizers adam --seed 0
    python scripts/compute_vqe_analysis.py --molecule LiH --optimizers adam --seed 1
    python scripts/compute_vqe_analysis.py --molecule LiH --optimizers adam --seed 0 --methods sea
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.analysis import metrics, replace_methods  # noqa: E402
from src.core.backend import active_backend  # noqa: E402
from src.core.backend import load_hamiltonian, verify_against_fci  # noqa: E402
from src.core.methods import METHODS, MethodConfig, run_method  # noqa: E402
from src.utils import io  # noqa: E402
from src.utils.progress import Progress  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--molecule", "-m", default="H2", help="Molecule name.")
    p.add_argument(
        "--geometry", "-g", default="equilibrium", help="Geometry label."
    )
    p.add_argument(
        "--depth",
        "-d",
        type=int,
        default=4,
        help="Ansatz depth (same for all methods).",
    )
    p.add_argument(
        "--optimizers",
        "-o",
        nargs="+",
        default=["adam", "qng", "adagrad"],
        choices=["adam", "qng", "adagrad"],
        help="Optimizers to run.",
    )
    p.add_argument(
        "--iters", type=int, default=1000, help="Main-stage iterations."
    )
    p.add_argument(
        "--warm-iters",
        type=int,
        default=150,
        help="Warm-start/refinement iterations.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
        help="Subset of methods to run.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="RNG seed for the initial parameters."
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Folder shared by the seeds of one setting. Defaults to "
        "<optimizers>_depth<d>_iters<n>, with the geometry in front when it is "
        "not equilibrium and --tag at the end.",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Optional label appended to the default run name.",
    )
    return p.parse_args(argv)


def default_run_name(args) -> str:
    """Name the folder that all seeds of this setting share."""
    parts = []
    if args.geometry != "equilibrium":
        parts.append(args.geometry)
    parts += [
        "-".join(args.optimizers),
        f"depth{args.depth}",
        f"iters{args.iters}",
    ]
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def main(argv=None) -> int:
    """Run the VQE analysis for one seed and save the tables (no figures).

    Runs every (optimizer, method) combination, then saves the performance table
    and the per-iteration convergence histories in the seed's folder.

    The progress clock's work-units are one per (optimizer x method) VQE run, plus
    two for the table and history writes.
    """
    args = parse_args(argv)
    args.run_name = args.run_name or default_run_name(args)
    system = load_hamiltonian(args.molecule, args.geometry)
    if not verify_against_fci(system):
        raise RuntimeError(
            f"{args.molecule} ({args.geometry}) Hamiltonian does not match its FCI energy; refusing to run."
        )

    out_dir = io.seed_run_dir(args.molecule, args.run_name, args.seed)

    prog = Progress(total=len(args.optimizers) * len(args.methods) + 2)
    prog.note(
        f"[{args.molecule}] seed={args.seed}  n_qubits={system.n_qubits}  "
        f"device={active_backend()}  optimizers={args.optimizers}  methods={args.methods}  "
        f"-> {out_dir}"
    )
    if is_partial_rerun(out_dir, args):
        check_same_settings(out_dir, args)
        prog.note(
            f"seed_{args.seed} exists in {args.run_name}, replacing only {args.methods}."
        )
    elif os.path.isfile(
        os.path.join(out_dir, "performance_table_all_methods.csv")
    ):
        prog.note(
            f"seed_{args.seed} already exists in {args.run_name}, overwriting it."
        )

    results_by_opt = {}
    rows = []
    for opt in args.optimizers:
        config = MethodConfig(
            depth=args.depth,
            optimizer=opt,
            max_iters=args.iters,
            warm_iters=args.warm_iters,
            seed=args.seed,
        )
        results = []
        for name in args.methods:
            r = run_method(name, system, config)
            results.append(r)
            rows.append(
                dict(
                    optimizer=opt,
                    method=r.method,
                    final_energy=r.final_energy,
                    energy_error=metrics.energy_error(r.final_energy, system),
                    fidelity=metrics.fidelity(r.ansatz, r.params, system),
                    n_params=r.n_params,
                    runtime_s=r.runtime_s,
                )
            )
            prog.step(
                f"{opt:5s} {name:14s} E={r.final_energy:.6f}  "
                f"err={rows[-1]['energy_error']:.2e}  fid={rows[-1]['fidelity']:.4f}  "
                f"t={r.runtime_s:.1f}s"
            )
        results_by_opt[opt] = results

    save_seed(out_dir, args, rows, results_by_opt)
    prog.step("saved performance_table_all_methods.csv + run_parameters.json")
    prog.step("saved convergence_history.csv")

    prog.note(
        f"[{args.molecule}] seed {args.seed} done. Plot this seed with: "
        f'python scripts/plot_vqe_analysis.py --dir "{out_dir}"'
    )
    return 0


# run_parameters.json keys that may differ between a seed's first run and a rerun.
_RERUN_KEYS = {"methods", "device", "_provenance"}


def is_partial_rerun(out_dir: str, args) -> bool:
    """True when the seed already has results and this call runs only some methods."""
    done = os.path.isfile(
        os.path.join(out_dir, "performance_table_all_methods.csv")
    )
    return done and set(args.methods) != set(METHODS)


def check_same_settings(out_dir: str, args) -> None:
    """Refuse a partial rerun whose settings differ from the seed's existing run.

    The run name covers the optimizers, depth, iterations, geometry and tag but
    not, for example, warm_iters, so the full parameter sets are compared here.
    """
    with open(
        os.path.join(out_dir, "run_parameters.json"), encoding="utf-8"
    ) as fh:
        old = json.load(fh)
    new = dict(vars(args))
    diff = sorted(
        k for k in (set(old) | set(new)) - _RERUN_KEYS if old.get(k) != new.get(k)
    )
    if diff:
        raise SystemExit(f"{out_dir} was run with different settings in: {diff}")


def _read_exact(path: str) -> pd.DataFrame:
    """Read a CSV so its floats round-trip exactly.

    pandas' default parser can change the last bit of a float, which would alter
    the methods a partial rerun is meant to keep.
    """
    return pd.read_csv(path, float_precision="round_trip")


def save_seed(out_dir: str, args, rows, results_by_opt) -> None:
    """Write the seed's performance table, convergence history and run parameters.

    A partial rerun (is_partial_rerun) replaces only the rerun methods' rows in
    the existing tables and keeps the other methods. Its run_parameters.json then
    lists every method the folder holds, in the usual order, and its _provenance
    adds the rerun methods to rerun_methods, so the history stays visible without
    changing the settings that aggregate_vqe_seeds.py compares across seeds.
    Otherwise the tables are written from scratch.
    """
    perf_path = os.path.join(out_dir, "performance_table_all_methods.csv")
    conv_path = os.path.join(out_dir, "convergence_history.csv")
    params = dict(vars(args), device=active_backend())
    provenance = {}

    if is_partial_rerun(out_dir, args):
        new_conv_path = conv_path + ".rerun"
        io.write_convergence_csv(new_conv_path, results_by_opt)
        new_conv = _read_exact(new_conv_path)
        os.remove(new_conv_path)
        perf = replace_methods(_read_exact(perf_path), pd.DataFrame(rows))
        conv = replace_methods(_read_exact(conv_path), new_conv)

        with open(
            os.path.join(out_dir, "run_parameters.json"), encoding="utf-8"
        ) as fh:
            old = json.load(fh)
        held = set(old["methods"]) | set(args.methods)
        params["methods"] = [m for m in METHODS if m in held]
        reruns = old.get("_provenance", {}).get("rerun_methods", [])
        provenance["rerun_methods"] = reruns + [list(args.methods)]

        perf.to_csv(perf_path, index=False)
        conv.to_csv(conv_path, index=False)
    else:
        io.write_csv(perf_path, rows)
        io.write_convergence_csv(conv_path, results_by_opt)

    io.write_run_parameters(
        out_dir, params, data_dir="data", provenance=provenance
    )


if __name__ == "__main__":
    raise SystemExit(main())
