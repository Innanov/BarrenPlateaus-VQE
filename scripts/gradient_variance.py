#!/usr/bin/env python3
"""
Analysis step for the gradient-variance sweeps (no computation).

Reads the runs saved by compute_gradient_scaling.py and does one of three things,
chosen by the options:

    -m LiH [H2 ...]   the comparison figure of each molecule (default action)
    --dir RUN         the figure of one sweep folder
    --d-sat           the saturation depths, written to results/d_sat.csv

The comparison figure reads every run in results/<molecule>/gradient_scaling/,
names each by what it measured (run_parameters.json), and keeps the newest run
of each kind. Each curve is labelled by its VQE method and shows the cost that
method starts training on, at random parameters:

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
one scale. --raw plots the variances in each cost's own units instead. The
dotted line is standard's flat level (its mean over depths 15 to 20), the gray
band is that level +- the tolerance (15%), and the dashed line is standard's
d_sat, the first depth from which the standard curve stays inside the band.

--d-sat applies the rule in src/core/analysis/saturation.py to the standard
curve of each molecule, one row per molecule. Standard is the base level: its
d_sat sets the VQE depths of every method (d_sat and 2 * d_sat), so the other
methods get no d_sat of their own. Their curves are compared with standard's in
the figure instead. Only runs saved with grad_var_per_parameter.csv (the seeded
runs) count. Without -m, every molecule under results/ with such runs is
included, so rerunning never drops a molecule from the table. The flat level is
saved both raw and divided by the Pauli weight (normalized_flat_level), which
compares across molecules.

Output:
    results/<molecule>/gradient_scaling/gradient_comparison.pdf   (-m)
    <RUN>/gradient_scaling.pdf                                     (--dir)
    results/d_sat.csv                                              (--d-sat)

Usage:
    python scripts/gradient_variance.py -m LiH
    python scripts/gradient_variance.py -m LiH BeH2 --max-depth 20 --raw
    python scripts/gradient_variance.py --dir results/H2/gradient_scaling/<ts>
    python scripts/gradient_variance.py --d-sat
    python scripts/gradient_variance.py --d-sat -m LiH BeH2 --tolerance 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.analysis.normalization import cost_pauli_weight  # noqa: E402
from src.core.analysis.saturation import TOLERANCE, saturation_depth  # noqa: E402
from src.core.backend.hamiltonians import json_path  # noqa: E402
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
    p.add_argument("--molecules", "-m", nargs="+", help="Molecule names.")
    p.add_argument("--dir", help="One sweep folder to draw on its own.")
    p.add_argument(
        "--d-sat",
        action="store_true",
        help="Write the saturation depths to results/d_sat.csv instead.",
    )
    p.add_argument(
        "--max-depth", type=int, default=None, help="Drop depths above this."
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw variances instead of dividing by the Pauli weight.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help="Relative distance from the flat level that counts as flat.",
    )
    p.add_argument(
        "--flat-from",
        type=int,
        default=15,
        help="First depth of the window whose mean is the flat level.",
    )
    args = p.parse_args(argv)
    if not (args.molecules or args.dir or args.d_sat):
        p.error("give --molecules, --dir or --d-sat")
    return args


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


def run_geometry(run_dir: str) -> str:
    """Return the geometry label a sweep was run at."""
    with open(
        os.path.join(run_dir, "run_parameters.json"), encoding="utf-8"
    ) as fh:
        return json.load(fh).get("geometry", "equilibrium")


def standard_saturation(curves) -> tuple[float | None, int | None]:
    """Return standard's flat level and saturation depth, for the figure.

    Uses saturation_depth with its defaults, the same rule as --d-sat, so the
    figure and results/d_sat.csv agree. The flat level is drawn with its
    tolerance band and d_sat as a dashed line, which shows why d_sat is where it
    is: the first depth from which the curve stays inside the band. Both are None
    when there is no standard curve or it stops before the flat window, and d_sat
    alone is None when it is not reached.
    """
    for _, method, _, points in curves:
        if method == "standard":
            try:
                return saturation_depth(
                    [p.n_layers for p in points],
                    [p.mean_grad_var for p in points],
                )
            except ValueError:
                return None, None
    return None, None


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


def comparison_figure(molecule: str, max_depth: int | None, raw: bool) -> str:
    """Draw the per-method comparison figure of one molecule."""
    gs_dir = os.path.join(io.RESULTS_ROOT, molecule, "gradient_scaling")
    runs = newest_runs(gs_dir) if os.path.isdir(gs_dir) else {}
    if not runs:
        raise SystemExit(f"No gradient_scaling runs in {gs_dir}")
    curves = []
    for kind in KINDS:
        if kind not in runs:
            continue
        label, method, start = KINDS[kind]
        points = io.load_gradient_points(
            os.path.join(runs[kind], "scaling_analysis_data.csv"), max_depth
        )
        weight = cost_pauli_weight(molecule, kind[1], run_geometry(runs[kind]))
        if not raw:
            points = normalized(points, weight)
        curves.append((label, method, start, points))
        print(
            f"{label:32s} {os.path.basename(runs[kind])}  "
            f"Pauli weight {weight:.4g}"
        )
    with open(
        os.path.join(next(iter(runs.values())), "scaling_analysis_data.csv"),
        encoding="utf-8",
    ) as fh:
        first = next(csv.DictReader(fh))
    title = (
        f"{molecule} ({first['n_qubits']} qubits): gradient variance by method"
    )
    out = os.path.join(gs_dir, "gradient_comparison.pdf")
    ylabel = (
        r"Var[$\partial_\theta C$]"
        if raw
        else r"Var[$\partial_\theta C$] / $\sum_P c_P^2$"
    )
    flat_level, d_sat = standard_saturation(curves)
    plots.plot_gradient_comparison(
        curves,
        out,
        title=title,
        ylabel=ylabel,
        d_sat=d_sat,
        flat_level=flat_level,
    )
    return out


def sweep_figure(run_dir: str) -> str:
    """Draw the figure of one sweep folder from its scaling_analysis_data.csv."""
    csv_path = os.path.join(run_dir, "scaling_analysis_data.csv")
    if not os.path.isfile(csv_path):
        raise SystemExit(f"No scaling_analysis_data.csv in {run_dir}")
    points = io.load_gradient_points(csv_path)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        first = next(csv.DictReader(fh), None)
    title = None
    if first and first.get("molecule") and first.get("n_qubits"):
        title = (
            f"{first['molecule']} ({first['n_qubits']} qubits): "
            "gradient-variance scaling"
        )
    out = os.path.join(run_dir, "gradient_scaling.pdf")
    plots.plot_gradient_scaling(points, out, title=title)
    return out


def molecules_with_sweeps() -> list[str]:
    """Return every molecule under results/ with sweeps, fewest qubits first."""
    names = [
        name
        for name in os.listdir(io.RESULTS_ROOT)
        if os.path.isdir(os.path.join(io.RESULTS_ROOT, name, "gradient_scaling"))
    ]

    def qubits(name: str) -> int:
        with open(json_path(name), encoding="utf-8") as fh:
            return int(json.load(fh)["n_qubits"])

    return sorted(names, key=lambda name: (qubits(name), name))


STANDARD = ("efficient_su2", "energy", None)
"""The KINDS key of standard's sweep, the base level that sets d_sat."""


def d_sat_row(molecule: str, tolerance: float, flat_from: int) -> dict | None:
    """Return standard's d_sat row of a molecule, or None without a seeded run."""
    gs_dir = os.path.join(io.RESULTS_ROOT, molecule, "gradient_scaling")
    run_dir = newest_runs(gs_dir).get(STANDARD) if os.path.isdir(gs_dir) else None
    if run_dir is None or not os.path.isfile(
        os.path.join(run_dir, "grad_var_per_parameter.csv")
    ):
        return None
    points = io.load_gradient_points(
        os.path.join(run_dir, "scaling_analysis_data.csv")
    )
    flat, d_sat = saturation_depth(
        [p.n_layers for p in points],
        [p.mean_grad_var for p in points],
        tolerance=tolerance,
        flat_from=flat_from,
    )
    weight = cost_pauli_weight(molecule, STANDARD[1], run_geometry(run_dir))
    return dict(
        molecule=molecule,
        n_seeds=points[0].n_seeds,
        max_depth=points[-1].n_layers,
        flat_level=flat,
        pauli_weight=weight,
        normalized_flat_level=flat / weight,
        d_sat=d_sat,
        run=os.path.basename(run_dir),
        tolerance=tolerance,
        flat_from=flat_from,
    )


def write_d_sat(molecules: list[str], tolerance: float, flat_from: int) -> int:
    """Compute standard's d_sat of each molecule and save the table."""
    rows = []
    for molecule in molecules:
        row = d_sat_row(molecule, tolerance, flat_from)
        if row is None:
            print(f"[{molecule}] no finished seeded standard run, skipped")
        else:
            rows.append(row)
    if not rows:
        raise SystemExit("no finished seeded standard runs found")
    out = os.path.join(io.RESULTS_ROOT, "d_sat.csv")
    io.write_csv(out, rows)
    for r in rows:
        depths = (
            "not reached"
            if r["d_sat"] is None
            else f"{r['d_sat']}  VQE depths {r['d_sat']} and {2 * r['d_sat']}"
        )
        print(
            f"{r['molecule']:13s} flat level {r['flat_level']:.2e}  "
            f"normalized {r['normalized_flat_level']:.2e}  d_sat {depths}"
        )
    print(f"wrote {out}")
    return 0 if all(not math.isnan(r["flat_level"]) for r in rows) else 1


def main(argv=None) -> int:
    """Run the action the options select."""
    args = parse_args(argv)
    if args.d_sat:
        molecules = args.molecules or molecules_with_sweeps()
        return write_d_sat(molecules, args.tolerance, args.flat_from)
    if args.dir:
        print(f"wrote {sweep_figure(args.dir)}")
    for molecule in args.molecules or []:
        print(f"wrote {comparison_figure(molecule, args.max_depth, args.raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
