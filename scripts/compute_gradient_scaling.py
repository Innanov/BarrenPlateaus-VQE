#!/usr/bin/env python3
"""
Compute step for gradient-variance scaling (no plotting).

Sweeps circuit depth and measures the barren-plateau gradient variance over
random initializations. Saves data only. Render with plot_gradient_scaling.py.

Output in results/<molecule>/gradient_scaling/<timestamp>/:
    scaling_analysis_data.csv     one row per depth (grad var/norm, params, runtime)
    barren_plateau_summary.json   min variance + variance-by-depth
    run_parameters.json           provenance

Usage:
    python scripts/compute_gradient_scaling.py --molecules H2 LiH BeH2 --max-layers 50 --samples 100
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.analysis import gradient_scaling as gs  # noqa: E402
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
        help="Ansatz family.",
    )
    p.add_argument("--max-layers", type=int, default=50, help="Maximum depth to sweep.")
    p.add_argument("--samples", type=int, default=100, help="Random initializations per depth.")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Run the gradient-scaling sweep and save the data (no figure)."""
    args = parse_args(argv)
    ts = timestamp()
    depths = list(range(1, args.max_layers + 1))

    for mol in args.molecules:
        system = load_hamiltonian(mol, args.geometry)
        if not verify_against_fci(system):
            raise RuntimeError(
                f"{mol} ({args.geometry}) Hamiltonian does not match its FCI energy; refusing to run."
            )
        out_dir = io.make_output_dir(mol, "gradient_scaling", ts)
        print(
            f"[{mol}] n_qubits={system.n_qubits}  device={active_backend()}  "
            f"depths={depths[0]}..{depths[-1]}  -> {out_dir}",
            flush=True,
        )

        points = gs.gradient_scaling(
            system,
            method=args.ansatz,
            depths=depths,
            n_samples=args.samples,
            seed=args.seed,
            verbose=True,
        )

        io.write_csv(os.path.join(out_dir, "scaling_analysis_data.csv"), points)
        summary = {
            "molecule": mol,
            "geometry": args.geometry,
            "ansatz": args.ansatz,
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
