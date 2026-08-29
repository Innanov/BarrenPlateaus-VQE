#!/usr/bin/env python3
"""
Plot step for gradient-variance scaling (no computation).

Rebuilds the gradient-variance-scaling figure from scaling_analysis_data.csv
saved by compute_gradient_scaling.py. Re-run any time to restyle.

Usage:
    python scripts/plot_gradient_scaling.py --dir results/H2/gradient_scaling/<timestamp>
    python scripts/plot_gradient_scaling.py --molecule H2   # newest H2 gradient run
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import io  # noqa: E402
from src.utils import plotting as plots  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dir", help="Run directory (results/<mol>/gradient_scaling/<ts>).")
    p.add_argument(
        "--molecule", "-m", help="Render the newest gradient_scaling run for this molecule."
    )
    return p.parse_args(argv)


def _load_points(path):
    """Load scaling_analysis_data.csv into point objects for plotting.

    Returns:
        A list of SimpleNamespace with n_layers, mean_grad_var and std_grad_var
        (what the plot function reads).
    """
    points = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            points.append(
                SimpleNamespace(
                    n_layers=int(row["n_layers"]),
                    mean_grad_var=float(row["mean_grad_var"]),
                    std_grad_var=float(row["std_grad_var"]),
                )
            )
    return points


def main(argv=None) -> int:
    """Render the gradient-scaling figure from saved data."""
    args = parse_args(argv)
    run_dir = io.resolve_run_dir(args.dir, args.molecule, "gradient_scaling")
    csv_path = os.path.join(run_dir, "scaling_analysis_data.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"No scaling_analysis_data.csv in {run_dir}")
    points = _load_points(csv_path)
    # Title from the CSV's molecule/n_qubits columns (first row).
    title = None
    with open(csv_path, newline="", encoding="utf-8") as fh:
        first = next(csv.DictReader(fh), None)
    if first and first.get("molecule") and first.get("n_qubits"):
        title = f"{first['molecule']} ({first['n_qubits']} qubits): gradient-variance scaling"
    out = os.path.join(run_dir, "gradient_scaling.pdf")
    plots.plot_gradient_scaling(points, out, title=title)
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
