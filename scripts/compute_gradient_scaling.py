#!/usr/bin/env python3
"""
Compute step for gradient-variance scaling (no plotting).

Sweeps circuit depth and measures the barren-plateau gradient variance. Saves
data only. Render one run with plot_gradient_scaling.py, or compare the methods
of one molecule with plot_gradient_comparison.py.

Two kinds of sample (see src/core/analysis/gradient_scaling.py):
    random parameters      --ansatz and --cost pick the circuit and the cost
    warm start (--warm-start local_global | pretrained)
                           the method's warm-up from seeds 0..samples-1, then the
                           energy gradient at its handoff point

Output in results/<molecule>/gradient_scaling/<timestamp>_<label>/, where label
is <ansatz>_<cost> or warm_start_<method>:
    scaling_analysis_data.csv     one row per depth (grad var/norm, params, runtime)
    barren_plateau_summary.json   min variance + variance-by-depth
    run_parameters.json           provenance

Usage:
    python scripts/compute_gradient_scaling.py --molecules H2 LiH BeH2 --max-layers 50 --samples 100
    python scripts/compute_gradient_scaling.py -m LiH --ansatz sea --max-layers 20
    python scripts/compute_gradient_scaling.py -m LiH --cost local_cost --max-layers 20
    python scripts/compute_gradient_scaling.py -m LiH --warm-start pretrained --depths 4 8 16 --samples 20
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.analysis import (  # noqa: E402
    COSTS,
    WARM_START_METHODS,
    gradient_scaling,
    warm_start_gradient_scaling,
)
from src.core.backend import active_backend  # noqa: E402
from src.core.backend import load_hamiltonian, verify_against_fci  # noqa: E402
from src.utils import io  # noqa: E402
from src.utils.helpers import timestamp  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--molecules", "-m", nargs="+", default=["H2"], help="Molecule names.")
    p.add_argument("--geometry", "-g", default="equilibrium", help="Geometry label.")
    p.add_argument(
        "--ansatz",
        "-a",
        default="efficient_su2",
        choices=["efficient_su2", "mps", "sea"],
        help="Ansatz family for random samples.",
    )
    p.add_argument(
        "--cost",
        default="energy",
        choices=list(COSTS),
        help="Cost differentiated at random samples.",
    )
    p.add_argument(
        "--warm-start",
        default=None,
        choices=list(WARM_START_METHODS),
        help="Measure at this method's handoff point instead of random parameters.",
    )
    p.add_argument("--max-layers", type=int, default=50, help="Sweep depths 1..max-layers.")
    p.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=None,
        help="Explicit depths instead.",
    )
    p.add_argument("--samples", type=int, default=100, help="Samples per depth.")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed (first seed for warm starts).",
    )
    p.add_argument(
        "--warm-iters",
        type=int,
        default=150,
        help="Warm-up iterations (warm starts).",
    )
    p.add_argument("--optimizer", default="adam", help="Warm-up optimizer (warm starts).")
    args = p.parse_args(argv)
    if args.warm_start and (args.ansatz != "efficient_su2" or args.cost != "energy"):
        p.error(
            "--warm-start uses the method's own ansatz and the energy, drop --ansatz and --cost"
        )
    return args


def run_label(args) -> str:
    """Name the kind of measurement, appended to the run folder's timestamp."""
    if args.warm_start:
        return f"warm_start_{args.warm_start}"
    return f"{args.ansatz}_{args.cost}"


def main(argv=None) -> int:
    """Run the gradient-scaling sweep and save the data (no figure)."""
    args = parse_args(argv)
    ts = f"{timestamp()}_{run_label(args)}"
    depths = args.depths or list(range(1, args.max_layers + 1))

    for mol in args.molecules:
        system = load_hamiltonian(mol, args.geometry)
        if not verify_against_fci(system):
            raise RuntimeError(
                f"{mol} ({args.geometry}) Hamiltonian does not match its FCI energy, refusing to run."
            )
        out_dir = io.make_output_dir(mol, "gradient_scaling", ts)
        print(
            f"[{mol}] n_qubits={system.n_qubits}  device={active_backend()}  "
            f"{run_label(args)}  depths={depths}  -> {out_dir}",
            flush=True,
        )

        if args.warm_start:
            points = warm_start_gradient_scaling(
                system,
                args.warm_start,
                depths=depths,
                n_samples=args.samples,
                seed=args.seed,
                warm_iters=args.warm_iters,
                optimizer=args.optimizer,
                verbose=True,
            )
        else:
            points = gradient_scaling(
                system,
                method=args.ansatz,
                depths=depths,
                n_samples=args.samples,
                seed=args.seed,
                verbose=True,
                cost=args.cost,
            )

        io.write_csv(os.path.join(out_dir, "scaling_analysis_data.csv"), points)
        summary = {
            "molecule": mol,
            "geometry": args.geometry,
            "label": run_label(args),
            "ansatz": args.ansatz,
            "cost": args.cost,
            "warm_start": args.warm_start,
            "n_qubits": system.n_qubits,
            "depths": depths,
            "grad_var_by_depth": {p.n_layers: p.mean_grad_var for p in points},
            "min_grad_var": min(p.mean_grad_var for p in points),
        }
        io.write_json(os.path.join(out_dir, "barren_plateau_summary.json"), summary)
        io.write_run_parameters(out_dir, dict(vars(args), device=active_backend()), data_dir="data")
        print(
            f"[{mol}] compute done. Render with: "
            f'python scripts/plot_gradient_scaling.py --dir "{out_dir}"',
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
