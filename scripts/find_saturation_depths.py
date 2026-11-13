#!/usr/bin/env python3
"""
Find the saturation depth d_sat of every molecule's gradient-variance curves.

Reads the newest gradient_scaling run of each method (the curves of
plot_gradient_comparison.py) and applies the rule in
src/core/analysis/saturation.py. Standard's d_sat is the molecule's reference,
which sets the VQE depths (d_sat and 2 * d_sat). Every method is also reported
relative to it, delta_d = d_sat(method) - d_sat(standard). Only runs saved with
grad_var_per_parameter.csv (the seeded runs) are used, so a molecule whose new
runs are not finished yet is skipped. Reads saved data only, so it can be rerun
any time.

Output: results/d_sat.csv, one row per molecule and method, with the flat level
both raw and divided by the cost's Pauli weight (normalized_flat_level, see
src/core/analysis/normalization.py), which compares across methods and
molecules.

Usage:
    python scripts/find_saturation_depths.py
    python scripts/find_saturation_depths.py -m LiH BeH2 --tolerance 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.plot_gradient_comparison import KINDS, newest_runs  # noqa: E402
from src.core.analysis.normalization import cost_pauli_weight  # noqa: E402
from src.core.analysis.saturation import saturation_depth  # noqa: E402
from src.utils import io  # noqa: E402

MOLECULES = ["H2", "H4", "LiH", "BeH2", "NH3"]


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--molecules", "-m", nargs="+", default=MOLECULES, help="Molecules."
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.2,
        help="Relative distance from the flat level that counts as flat.",
    )
    p.add_argument(
        "--flat-from",
        type=int,
        default=15,
        help="First depth of the window whose mean is the flat level.",
    )
    return p.parse_args(argv)


def molecule_rows(molecule: str, tolerance: float, flat_from: int) -> list[dict]:
    """Return one row per method with a finished seeded run, standard first."""
    gs_dir = os.path.join(io.RESULTS_ROOT, molecule, "gradient_scaling")
    if not os.path.isdir(gs_dir):
        return []
    rows = []
    for kind, run_dir in newest_runs(gs_dir).items():
        if not os.path.isfile(
            os.path.join(run_dir, "grad_var_per_parameter.csv")
        ):
            continue
        points = io.load_gradient_points(
            os.path.join(run_dir, "scaling_analysis_data.csv")
        )
        flat, d_sat = saturation_depth(
            [p.n_layers for p in points],
            [p.mean_grad_var for p in points],
            tolerance=tolerance,
            flat_from=flat_from,
        )
        with open(
            os.path.join(run_dir, "run_parameters.json"), encoding="utf-8"
        ) as fh:
            geometry = json.load(fh).get("geometry", "equilibrium")
        weight = cost_pauli_weight(molecule, kind[1], geometry)
        rows.append(
            dict(
                molecule=molecule,
                method=KINDS[kind][1],
                n_seeds=points[0].n_seeds,
                max_depth=points[-1].n_layers,
                flat_level=flat,
                pauli_weight=weight,
                normalized_flat_level=flat / weight,
                d_sat=d_sat,
                run=os.path.basename(run_dir),
            )
        )
    standard = next((r["d_sat"] for r in rows if r["method"] == "standard"), None)
    for r in rows:
        both = r["d_sat"] is not None and standard is not None
        r["delta_d"] = r["d_sat"] - standard if both else None
        r["tolerance"] = tolerance
        r["flat_from"] = flat_from
    return sorted(rows, key=lambda r: r["method"] != "standard")


def main(argv=None) -> int:
    """Compute d_sat for every molecule with finished runs and save the table."""
    args = parse_args(argv)
    rows = []
    for molecule in args.molecules:
        found = molecule_rows(molecule, args.tolerance, args.flat_from)
        if not found:
            print(f"[{molecule}] no finished seeded runs, skipped")
        rows += found
    if not rows:
        raise SystemExit("no finished seeded gradient runs found")
    out = os.path.join(io.RESULTS_ROOT, "d_sat.csv")
    io.write_csv(out, rows)
    for r in rows:
        d_sat = "not reached" if r["d_sat"] is None else r["d_sat"]
        delta = "" if r["delta_d"] is None else f"  delta_d {r['delta_d']:+d}"
        mark = "  <- VQE depths" if r["method"] == "standard" else ""
        print(
            f"{r['molecule']:5s} {r['method']:13s} flat level "
            f"{r['flat_level']:.2e}  "
            f"normalized {r['normalized_flat_level']:.2e}  "
            f"d_sat {d_sat}{delta}{mark}"
        )
        if r["method"] == "standard" and r["d_sat"] is not None:
            print(f"      VQE depths: {r['d_sat']} and {2 * r['d_sat']}")
    print(f"wrote {out}")
    return 0 if all(not math.isnan(r["flat_level"]) for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
