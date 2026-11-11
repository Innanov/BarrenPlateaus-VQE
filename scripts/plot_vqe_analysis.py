#!/usr/bin/env python3
"""
Plot step for one VQE run or one seed (no computation).

Rebuilds the energy-convergence figures from the convergence_history.csv saved
by compute_vqe_analysis.py. Never recomputes physics, so re-run any time to
restyle figures. For the mean +- std over seeds use plot_vqe_seeds.py.

Usage:
    python scripts/plot_vqe_analysis.py --dir results/LiH/vqe_seeds/adam_depth4_iters1000/seed_0
    python scripts/plot_vqe_analysis.py --dir results/H2/vqe_analysis/<timestamp>
    python scripts/plot_vqe_analysis.py --molecule H2        # newest results/H2/vqe_analysis run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import io  # noqa: E402
from src.utils import plotting as plots  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dir", help="Run or seed directory holding convergence_history.csv.")
    p.add_argument(
        "--molecule",
        "-m",
        help="Render the newest results/<molecule>/vqe_analysis run (runs made before seeding).",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="With --molecule, look in 'vqe_analysis_<tag>' instead.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Render the convergence figures from a saved run directory.

    The convergence x-axis is capped at the run's main iteration budget (read from
    run_parameters.json), so a two-stage method's warm-up tail does not stretch the
    axis past the others.
    """
    args = parse_args(argv)
    atype = "vqe_analysis" if not args.tag else f"vqe_analysis_{args.tag}"
    run_dir = io.resolve_run_dir(args.dir, args.molecule, atype)
    conv_csv = os.path.join(run_dir, "convergence_history.csv")
    if not os.path.isfile(conv_csv):
        raise SystemExit(f"no convergence_history.csv in {run_dir}")

    fci, label = io.system_info(run_dir)  # label = "H2 (4 qubits)"
    max_iters = io.main_iters(run_dir)
    paths = plots.write_convergence_figures(
        io.load_convergence(conv_csv), run_dir, fci_energy=fci, label=label, max_iters=max_iters
    )
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
