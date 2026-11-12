#!/usr/bin/env python3
"""
Plot step: compare the gradient variance of the VQE methods for one molecule.

Reads every run in results/<molecule>/gradient_scaling/, names each by what it
measured (run_parameters.json), and keeps the newest run of each kind. Each kind
stands for the point where one VQE method starts training:

    efficient_su2, energy        standard
    sea, energy                  sea
    efficient_su2, local_cost    local_global, its first-stage cost
    efficient_su2, hf_reference  adiabatic, its first-stage Hamiltonian
    warm start local_global      local_global, energy after its warm-up
    warm start pretrained        pretrained, energy after pre-training

Output: results/<molecule>/gradient_scaling/gradient_comparison.pdf

Usage:
    python scripts/plot_gradient_comparison.py --molecule LiH
    python scripts/plot_gradient_comparison.py --molecule LiH --max-depth 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import io  # noqa: E402
from src.utils import plotting as plots  # noqa: E402

# (ansatz, cost, warm_start) -> (legend label, VQE method it stands for, start).
KINDS = {
    ("efficient_su2", "energy", None): ("standard", "standard", "random"),
    ("sea", "energy", None): ("sea", "sea", "random"),
    ("efficient_su2", "local_cost", None): (
        "local_global, local cost",
        "local_global",
        "random",
    ),
    ("efficient_su2", "hf_reference", None): (
        "adiabatic, first stage",
        "adiabatic",
        "random",
    ),
    ("efficient_su2", "energy", "local_global"): (
        "local_global, after warm-up",
        "local_global",
        "warm_start",
    ),
    ("efficient_su2", "energy", "pretrained"): (
        "pretrained, after pre-training",
        "pretrained",
        "warm_start",
    ),
}


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--molecule", "-m", required=True, help="Molecule name.")
    p.add_argument(
        "--max-depth", type=int, default=None, help="Drop depths above this."
    )
    return p.parse_args(argv)


def newest_runs(gs_dir: str) -> dict:
    """Return {kind: run_dir} with the newest run of each kind in gs_dir.

    Runs from before the --cost and --warm-start options have neither key, so they
    count as energy at random parameters.
    """
    found = {}
    for name in sorted(os.listdir(gs_dir)):
        run_dir = os.path.join(gs_dir, name)
        params_path = os.path.join(run_dir, "run_parameters.json")
        if not os.path.isfile(os.path.join(run_dir, "scaling_analysis_data.csv")):
            continue
        with open(params_path, encoding="utf-8") as fh:
            params = json.load(fh)
        warm = params.get("warm_start")
        ansatz = (
            "efficient_su2" if warm else params.get("ansatz", "efficient_su2")
        )
        kind = (ansatz, params.get("cost", "energy"), warm)
        if kind in KINDS:
            found[kind] = run_dir
    return found


def load_points(run_dir: str, max_depth: int | None):
    """Load a run's per-depth points (n_layers, mean_grad_var, energy error)."""
    points = []
    with open(
        os.path.join(run_dir, "scaling_analysis_data.csv"),
        newline="",
        encoding="utf-8",
    ) as fh:
        for row in csv.DictReader(fh):
            depth = int(row["n_layers"])
            if max_depth is None or depth <= max_depth:
                points.append(
                    SimpleNamespace(
                        n_layers=depth,
                        mean_grad_var=float(row["mean_grad_var"]),
                        mean_energy_error=float(
                            row.get("mean_energy_error") or "nan"
                        ),
                    )
                )
    return points


def main(argv=None) -> int:
    """Render the per-method gradient-variance comparison of one molecule."""
    args = parse_args(argv)
    gs_dir = os.path.join(io.RESULTS_ROOT, args.molecule, "gradient_scaling")
    runs = newest_runs(gs_dir)
    if not runs:
        raise SystemExit(f"No gradient_scaling runs in {gs_dir}")
    curves = []
    for kind in KINDS:
        if kind in runs:
            label, method, start = KINDS[kind]
            points = load_points(runs[kind], args.max_depth)
            curves.append((label, method, start, points))
            print(f"{label:32s} {os.path.basename(runs[kind])}")
            if start == "warm_start":
                for p in points:
                    print(
                        f"    depth {p.n_layers:>2}: E - E_FCI at the handoff "
                        f"{p.mean_energy_error:.2e} Ha"
                    )
    with open(
        os.path.join(next(iter(runs.values())), "scaling_analysis_data.csv"),
        encoding="utf-8",
    ) as fh:
        first = next(csv.DictReader(fh))
    title = (
        f"{args.molecule} ({first['n_qubits']} qubits): "
        "gradient variance by method"
    )
    out = os.path.join(gs_dir, "gradient_comparison.pdf")
    plots.plot_gradient_comparison(curves, out, title=title)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
