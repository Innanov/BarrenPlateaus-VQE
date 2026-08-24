#!/usr/bin/env python3
"""
Rebuild loss landscapes from saved optimizer paths (no VQE re-run).

Re-samples the landscapes for a run using the optimizer paths saved by
compute_vqe_analysis.py (opt_<method>.npz), regenerating landscape_<method>.npz
cheaply after a landscape-method change. Then re-plot.

Usage:
    python scripts/rebuild_landscapes.py --dir results/H2/vqe_analysis/<ts>
    python scripts/rebuild_landscapes.py --molecule H2      # newest H2 run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.analysis import landscape  # noqa: E402
from src.core.ansatze import SEA, EfficientSU2  # noqa: E402
from src.core.backend import load_hamiltonian  # noqa: E402
from src.utils import io  # noqa: E402


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dir", help="Run directory (results/<mol>/vqe_analysis/<ts>).")
    p.add_argument(
        "--molecule", "-m", help="Rebuild the newest vqe_analysis run for this molecule."
    )
    p.add_argument("--resolution", type=int, help="Landscape grid resolution (else use run's).")
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Only rebuild these methods (default rebuilds all).",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip methods whose landscape_<method>.npz already exists.",
    )
    return p.parse_args(argv)


def _ansatz_for(method, n_qubits, depth):
    """Reconstruct the ansatz a method uses (for landscape sampling)."""
    if method == "sea":
        return SEA(n_qubits, d=depth)
    return EfficientSU2(n_qubits, d=depth)  # standard/local_global/adiabatic/pretrained


def main(argv=None) -> int:
    """Rebuild all landscape NPZs in a run from saved optimizer paths."""
    args = parse_args(argv)
    run_dir = io.resolve_run_dir(args.dir, args.molecule, "vqe_analysis")
    params = json.load(open(os.path.join(run_dir, "run_parameters.json")))
    system = load_hamiltonian(params["molecule"], params.get("geometry", "equilibrium"))
    depth = int(params.get("depth", 4))
    seed = int(params.get("seed", 0))
    res = args.resolution or int(params.get("landscape_res", 25))

    opt_files = sorted(glob.glob(os.path.join(run_dir, "opt_*.npz")))
    if not opt_files:
        raise SystemExit(
            f"No opt_*.npz in {run_dir} (re-run compute_vqe_analysis.py to create them)."
        )

    # Optional filters: a specific method subset, and/or only the missing ones.
    want = set(args.methods) if args.methods else None
    selected = []
    for f in opt_files:
        method = os.path.basename(f)[len("opt_") : -len(".npz")]
        if want is not None and method not in want:
            continue
        land_path = os.path.join(run_dir, f"landscape_{method}.npz")
        if args.only_missing and os.path.exists(land_path):
            print(f"  skip {method} (landscape already present)", flush=True)
            continue
        selected.append((f, method, land_path))

    if not selected:
        print("Nothing to rebuild (all requested landscapes already present).", flush=True)
        return 0

    print(
        f"[{params['molecule']}] rebuilding {len(selected)} landscape(s) (res={res}) "
        f"into {run_dir}",
        flush=True,
    )
    for f, method, land_path in selected:
        _, final_params, histories = io.load_optimization_npz(f)
        ansatz = _ansatz_for(method, system.n_qubits, depth)
        ls = landscape.compute_landscape(
            ansatz,
            system.hamiltonian,
            system.n_qubits,
            final_params,
            resolution=res,
            seed=seed,
            trajectories=histories,
        )
        io.save_landscape_npz(land_path, ls)  # same run folder
        print(f"  rebuilt landscape_{method}.npz", flush=True)

    print(
        f"[{params['molecule']}] done. Re-render: "
        f'python scripts/plot_vqe_analysis.py --dir "{run_dir}"',
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
