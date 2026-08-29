#!/usr/bin/env python3
"""
Plot step for the VQE analysis (no computation).

Rebuilds every VQE figure from data saved by compute_vqe_analysis.py (reads
convergence_history.csv + landscape_<method>.npz, writes PDFs). Never recomputes
physics, so re-run any time to restyle figures.

Usage:
    python scripts/plot_vqe_analysis.py --dir results/H2/vqe_analysis/<timestamp>
    python scripts/plot_vqe_analysis.py --molecule H2               # newest H2 vqe run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import io  # noqa: E402
from src.utils import plotting as plots  # noqa: E402
from src.utils.progress import Progress  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dir", help="Run directory to render (results/<mol>/vqe_analysis/<ts>).")
    p.add_argument("--molecule", "-m", help="Render the newest vqe_analysis run for this molecule.")
    p.add_argument(
        "--tag",
        default=None,
        help="Match the compute step's --tag, so the correct "
        "'vqe_analysis_<tag>' folder is resolved for --molecule.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    """Render all VQE figures from a saved run directory.

    Draws the convergence figures, then one contour + surface landscape per
    method plus an overview grid, from the CSV and NPZ files in the run directory.

    The progress clock's work-units are one for the convergence figures, one per
    landscape method, one for the overview grid, and one final "done". The
    convergence x-axis is capped at the run's main iteration budget (read from
    run_parameters.json), so a two-stage method's warm-up tail does not stretch the
    axis past the others.
    """
    args = parse_args(argv)
    atype = "vqe_analysis" if not args.tag else f"vqe_analysis_{args.tag}"
    run_dir = io.resolve_run_dir(args.dir, args.molecule, atype)
    fci, label = io.system_info(run_dir)  # label = "H2 (4 qubits)"

    n_land = len(glob.glob(os.path.join(run_dir, "landscape_*.npz")))
    prog = Progress(total=1 + n_land + 1 + 1)
    prog.note(f"rendering figures from {run_dir}")

    max_iters = None
    try:
        max_iters = int(json.load(open(os.path.join(run_dir, "run_parameters.json")))["iters"])
    except Exception:
        pass

    # Convergence
    conv_csv = os.path.join(run_dir, "convergence_history.csv")
    if os.path.isfile(conv_csv):
        results_by_opt = io.load_convergence(conv_csv)
        conv_path = os.path.join(run_dir, "energy_convergence_all_methods.pdf")
        opt_list = " vs ".join(o.upper() for o in results_by_opt)
        title = f"{label}: {opt_list}" if label else None
        if len(results_by_opt) > 1:
            plots.plot_convergence_grid(
                results_by_opt, conv_path, fci_energy=fci, title=title, max_iters=max_iters
            )
            # Single-axes overlay too, for a direct per-method comparison.
            plots.plot_convergence_optimizers(
                results_by_opt,
                os.path.join(run_dir, "energy_convergence_overlay.pdf"),
                fci_energy=fci,
                title=title,
                max_iters=max_iters,
            )
        else:
            plots.plot_convergence(
                next(iter(results_by_opt.values())),
                conv_path,
                fci_energy=fci,
                title=(label or None),
                max_iters=max_iters,
            )
        prog.step(f"wrote {os.path.basename(conv_path)}")
    else:
        prog.step("(no convergence_history.csv found)")

    # Landscapes
    # Preferred method column order for the overview grid.
    order = {"standard": 0, "local_global": 1, "adiabatic": 2, "sea": 3, "pretrained": 4}
    grid_items = []
    for npz in sorted(glob.glob(os.path.join(run_dir, "landscape_*.npz"))):
        method = os.path.basename(npz)[len("landscape_") : -len(".npz")]
        ls = io.load_landscape_npz(npz)
        base = f"{label} - {method}" if label else method
        plots.plot_landscape_contour(
            ls,
            os.path.join(run_dir, f"landscape_{method}_contour.pdf"),
            title=f"{base} convergence landscape (contour)",
        )
        plots.plot_landscape_surface(
            ls,
            os.path.join(run_dir, f"landscape_{method}_surface.pdf"),
            title=f"{base} convergence landscape (surface)",
            fci_energy=fci,
        )
        grid_items.append((method, ls))
        prog.step(f"wrote landscape_{method}_(contour|surface).pdf")

    # Overview grid: methods in columns, surface (row 1) + contour (row 2).
    if grid_items:
        grid_items.sort(key=lambda kv: order.get(kv[0], 99))
        grid_title = (
            f"{label}: optimizer convergence landscapes (2-parameter slices)"
            if label
            else "Optimizer convergence landscapes (2-parameter slices)"
        )
        plots.plot_landscape_grid(
            grid_items,
            os.path.join(run_dir, "landscapes_grid.pdf"),
            title=grid_title,
            fci_energy=fci,
        )
        prog.step("wrote landscapes_grid.pdf")

    prog.step("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
