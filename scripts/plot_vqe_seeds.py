#!/usr/bin/env python3
"""
Plot step for seeded VQE runs (no computation).

Draws the same energy-convergence figures as plot_vqe_analysis.py, but each
method's curve is the mean over seeds with a shaded band from the 16th to the
84th percentile over seeds (the middle 68%). Reads convergence_mean_std.csv,
written by aggregate_vqe_seeds.py.

Output in results/<molecule>/vqe_seeds/<run_name>/:
    energy_convergence_all_methods.pdf
    energy_convergence_overlay.pdf      only when several optimizers were run

Usage:
    python scripts/plot_vqe_seeds.py --dir results/LiH/vqe_seeds/<run_name>
    python scripts/plot_vqe_seeds.py -m LiH --run-name adam_depth4_iters1000
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


def main(argv=None) -> int:
    """Render the seed-averaged convergence figures of one setting."""
    args = parse_args(argv)
    seeds_dir = io.resolve_seeds_dir(args.dir, args.molecule, args.run_name)
    conv_csv = os.path.join(seeds_dir, "convergence_mean_std.csv")
    if not os.path.isfile(conv_csv):
        raise SystemExit(
            f"no convergence_mean_std.csv in {seeds_dir}, "
            "run aggregate_vqe_seeds.py"
        )

    fci, label = io.system_info(seeds_dir)
    max_iters = io.main_iters(seeds_dir)
    paths = plots.write_convergence_figures(
        io.load_convergence_mean_std(conv_csv),
        seeds_dir,
        fci_energy=fci,
        label=label,
        max_iters=max_iters,
    )
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
