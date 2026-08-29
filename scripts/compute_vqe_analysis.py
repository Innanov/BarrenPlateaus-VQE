#!/usr/bin/env python3
"""
Compute step for the VQE analysis (no plotting).

Runs the five VQE methods under each optimizer and saves all data needed to
rebuild every figure. Render with plot_vqe_analysis.py.

Output in results/<molecule>/vqe_analysis/<timestamp>/:
    performance_table_all_methods.csv   final energy/error/fidelity per method+optimizer
    convergence_history.csv             per-iteration energy (long format)
    landscape_<method>.npz              grid + optimizer trajectories per method
    run_parameters.json                 provenance (device, optimizers, seed, git commit)

Usage:
    python scripts/compute_vqe_analysis.py --molecule H2 --iters 1000
    python scripts/compute_vqe_analysis.py --molecule LiH --iters 1000 --no-landscape
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.analysis import landscape, metrics  # noqa: E402
from src.core.backend import active_backend  # noqa: E402
from src.core.backend import load_hamiltonian, verify_against_fci  # noqa: E402
from src.core.methods import METHODS, MethodConfig, run_method  # noqa: E402
from src.utils import io  # noqa: E402
from src.utils.helpers import timestamp  # noqa: E402
from src.utils.progress import Progress  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--molecule", "-m", default="H2", help="Molecule name.")
    p.add_argument("--geometry", "-g", default="equilibrium", help="Geometry label.")
    p.add_argument(
        "--depth", "-d", type=int, default=4, help="Ansatz depth (same for all methods)."
    )
    p.add_argument(
        "--optimizers",
        "-o",
        nargs="+",
        default=["adam", "qng", "adagrad"],
        choices=["adam", "qng", "adagrad"],
        help="Optimizers to run.",
    )
    p.add_argument("--iters", type=int, default=1000, help="Main-stage iterations.")
    p.add_argument("--warm-iters", type=int, default=150, help="Warm-start/refinement iterations.")
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        choices=list(METHODS),
        help="Subset of methods to run.",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument("--landscape-res", type=int, default=25, help="Landscape grid resolution.")
    p.add_argument("--no-landscape", action="store_true", help="Skip landscape data.")
    p.add_argument(
        "--tag",
        default=None,
        help="Optional label appended to the analysis-type folder, so "
        "runs that differ only by that label land in distinct "
        "directories.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run the VQE analysis and save all plot-source data (no figures).

    Runs every (optimizer, method) combination, then saves the performance table,
    the per-iteration convergence histories, and (unless --no-landscape) the
    optimizer paths and sampled loss landscapes per method, all under a fresh
    timestamped results directory.

    The progress clock's work-units are one per (optimizer x method) VQE run, plus
    two for the table and history writes, plus one per-method landscape build.
    """
    args = parse_args(argv)
    system = load_hamiltonian(args.molecule, args.geometry)
    if not verify_against_fci(system):
        raise RuntimeError(
            f"{args.molecule} ({args.geometry}) Hamiltonian does not match its FCI energy; refusing to run."
        )

    analysis_type = "vqe_analysis" if not args.tag else f"vqe_analysis_{args.tag}"
    out_dir = io.make_output_dir(args.molecule, analysis_type, timestamp())

    n_vqe = len(args.optimizers) * len(args.methods)
    n_land = 0 if args.no_landscape else len(args.methods)
    prog = Progress(total=n_vqe + 2 + n_land)
    prog.note(
        f"[{args.molecule}] n_qubits={system.n_qubits}  device={active_backend()}  "
        f"optimizers={args.optimizers}  methods={args.methods}  -> {out_dir}"
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

    # Save tables + provenance
    io.write_csv(os.path.join(out_dir, "performance_table_all_methods.csv"), rows)
    io.write_run_parameters(out_dir, dict(vars(args), device=active_backend()), data_dir="data")
    prog.step("saved performance_table_all_methods.csv + run_parameters.json")

    # Save per-iteration histories (for the convergence curves)
    io.write_convergence_csv(os.path.join(out_dir, "convergence_history.csv"), results_by_opt)
    prog.step("saved convergence_history.csv")

    # Save optimizer paths + landscapes per method
    if not args.no_landscape:
        land_opt = args.optimizers[-1]
        by_method = {opt: {r.method: r for r in rs} for opt, rs in results_by_opt.items()}
        for r in results_by_opt[land_opt]:
            trajectories = {
                opt: by_method[opt][r.method].param_history
                for opt in args.optimizers
                if r.method in by_method[opt]
            }
            # Persist the raw optimizer paths so a landscape can be rebuilt later
            # without re-running the VQE.
            io.save_optimization_npz(
                os.path.join(out_dir, f"opt_{r.method}.npz"), r.method, r.params, trajectories
            )
            # Also sample the landscape now (so plotting is immediate).
            ls = landscape.compute_landscape(
                r.ansatz,
                system.hamiltonian,
                system.n_qubits,
                r.params,
                resolution=args.landscape_res,
                seed=args.seed,
                trajectories=trajectories,
            )
            io.save_landscape_npz(os.path.join(out_dir, f"landscape_{r.method}.npz"), ls)
            prog.step(f"saved opt_{r.method}.npz + landscape_{r.method}.npz")

    prog.note(
        f"[{args.molecule}] compute done. Render with: "
        f'python scripts/plot_vqe_analysis.py --dir "{out_dir}"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
