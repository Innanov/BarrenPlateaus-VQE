#!/usr/bin/env python3
"""
Plot step: compare the gradient variance of the VQE methods for one molecule.

Reads every run in results/<molecule>/gradient_scaling/, names each by what it
measured (run_parameters.json), and keeps the newest run of each kind. Each
curve is labelled by its VQE method and shows the cost that method starts
training on, at random parameters:

    efficient_su2, energy        standard
    sea, energy                  sea
    efficient_su2, local_cost    local_global, its first-stage cost
    efficient_su2, hf_reference  adiabatic, its first-stage Hamiltonian
    mps, energy                  pretrained, its diagonal MPS pre-training stage

Warm starts (--warm-start runs) are left out. local_global's warm-up stalls on
the local cost's own plateau in deep circuits, and pretrained hands off in a
different circuit (MPS, many more parameters per layer), so neither handoff point
compares with the random-parameter curves at the same depth.

Each curve is divided by its cost's Pauli weight (the sum of its squared
non-identity Pauli coefficients, src/core/analysis/normalization.py), so the
energy, local_global's local cost and adiabatic's first-stage Hamiltonian share
one scale. --raw plots the variances in each cost's own units instead.

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

from src.core.analysis.normalization import cost_pauli_weight  # noqa: E402
from src.core.analysis.saturation import saturation_depth  # noqa: E402
from src.utils import io  # noqa: E402
from src.utils import plotting as plots  # noqa: E402

# (ansatz, cost, warm_start) -> (legend label, VQE method it stands for, start).
KINDS = {
    ("efficient_su2", "energy", None): ("standard", "standard", "random"),
    ("sea", "energy", None): ("sea", "sea", "random"),
    ("efficient_su2", "local_cost", None): (
        "local_global",
        "local_global",
        "random",
    ),
    ("efficient_su2", "hf_reference", None): ("adiabatic", "adiabatic", "random"),
    ("mps", "energy", None): ("pretrained", "pretrained", "random"),
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
    p.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw variances instead of dividing by the Pauli weight.",
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


def standard_d_sat(curves) -> int | None:
    """Return standard's saturation depth, the dashed line, or None.

    Uses saturation_depth with its defaults, the same rule as
    find_saturation_depths.py, so the figure and results/d_sat.csv agree. None
    when there is no standard curve, it stops before the flat window, or the
    depth is not reached.
    """
    for _, method, _, points in curves:
        if method == "standard":
            try:
                return saturation_depth(
                    [p.n_layers for p in points],
                    [p.mean_grad_var for p in points],
                )[1]
            except ValueError:
                return None
    return None


def normalized(points, weight: float) -> list:
    """Return the points with the variance and its seed band divided by weight."""
    return [
        SimpleNamespace(
            **{
                **vars(p),
                "mean_grad_var": p.mean_grad_var / weight,
                "seed_std_grad_var": p.seed_std_grad_var / weight,
            }
        )
        for p in points
    ]


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
            points = io.load_gradient_points(
                os.path.join(runs[kind], "scaling_analysis_data.csv"),
                args.max_depth,
            )
            with open(
                os.path.join(runs[kind], "run_parameters.json"), encoding="utf-8"
            ) as fh:
                geometry = json.load(fh).get("geometry", "equilibrium")
            weight = cost_pauli_weight(args.molecule, kind[1], geometry)
            if not args.raw:
                points = normalized(points, weight)
            curves.append((label, method, start, points))
            print(
                f"{label:32s} {os.path.basename(runs[kind])}  "
                f"Pauli weight {weight:.4g}"
            )
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
    ylabel = (
        r"Var[$\partial_\theta C$]"
        if args.raw
        else r"Var[$\partial_\theta C$] / $\sum_P c_P^2$"
    )
    plots.plot_gradient_comparison(
        curves, out, title=title, ylabel=ylabel, d_sat=standard_d_sat(curves)
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
