"""Output directory management and CSV/JSON writers.

Results are written under results/<molecule>/<analysis_type>/<timestamp>/ (CSVs,
JSON summaries, run parameters, plus PDFs written by plots). Seeded VQE runs go
under results/<molecule>/vqe_seeds/<run_name>/seed_<seed>/ instead. Every
run_parameters.json records the data-source path for provenance.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
SEEDS_ANALYSIS = "vqe_seeds"


def make_output_dir(molecule: str, analysis_type: str, timestamp: str) -> str:
    """Create and return results/<molecule>/<analysis_type>/<timestamp>/.

    Args:
        molecule: Molecule name.
        analysis_type: e.g. 'gradient_scaling' or 'vqe_analysis'.
        timestamp: A caller-supplied timestamp string (the caller controls the
            format).

    Returns:
        The absolute path of the created directory.
    """
    path = os.path.join(RESULTS_ROOT, molecule, analysis_type, timestamp)
    os.makedirs(path, exist_ok=True)
    return path


def resolve_run_dir(
    run_dir: str | None, molecule: str | None, analysis_type: str
) -> str:
    """Resolve a run directory from an explicit path or the newest matching run.

    If run_dir is given it is returned as-is. Otherwise the newest run under
    results/<molecule>/<analysis_type>/ is returned (directories sort
    lexicographically, and the timestamps sort chronologically).

    Args:
        run_dir: An explicit run directory, or None.
        molecule: The molecule whose newest run to find, or None.
        analysis_type: The analysis subfolder, e.g. 'gradient_scaling',
            'vqe_analysis', or 'vqe_analysis_<tag>'.

    Returns:
        The resolved run directory path.

    Raises:
        SystemExit: If neither run_dir nor molecule is given, or no run is found.
    """
    if run_dir:
        return run_dir
    if not molecule:
        raise SystemExit("Provide --dir or --molecule.")
    pattern = os.path.join(RESULTS_ROOT, molecule, analysis_type, "*")
    dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"No {analysis_type} runs found for {molecule}.")
    return dirs[-1]


def load_convergence(path: str) -> dict:
    """Load convergence_history.csv into {optimizer: [plot-ready result, ...]}.

    Uses the energy_global column (energy vs the full Hamiltonian). For the
    two-stage methods, only the main stage is kept (the rows from the stage
    boundary on), so the plotted curve is the refinement against the full H, like
    qubap, without the warm-up stage or the jump at the handoff. A row with a
    blank energy_global is not part of the reported curve and is skipped.

    Args:
        path: Path to convergence_history.csv.

    Returns:
        A mapping {optimizer: [SimpleNamespace(method, energy_history,
        stage_boundaries), ...]} matching what the plot functions read.
    """
    acc = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            opt, m = row["optimizer"], row["method"]
            it = int(row["iteration"])
            acc.setdefault(opt, {}).setdefault(m, {"e": {}, "b": set()})
            g = row.get("energy_global", "")
            if g == "" or g is None:
                continue
            acc[opt][m]["e"][it] = float(g)
            if int(row["is_stage_boundary"]):
                acc[opt][m]["b"].add(it)
    results_by_opt = {}
    for opt, methods in acc.items():
        results = []
        for m, d in methods.items():
            start = min(d["b"]) if d["b"] else 0
            xs = [i for i in sorted(d["e"]) if i >= start]
            hist = [d["e"][i] for i in xs]
            results.append(
                SimpleNamespace(
                    method=m, energy_history=hist, stage_boundaries=[]
                )
            )
        results_by_opt[opt] = results
    return results_by_opt


def load_gradient_points(path: str, max_depth: int | None = None) -> list:
    """Load scaling_analysis_data.csv into plot-ready points.

    mean_grad_var is the mean over parameters of each partial derivative's
    variance (see src/core/analysis/gradient_scaling.py). Files written before the
    grad_var_param0 column existed hold parameter 0's variance in mean_grad_var
    instead. For their random-parameter rows the mean is recovered from the saved
    gradient norms as (mean_grad_norm**2 + std_grad_norm**2) / n_parameters, the
    mean over parameters of each derivative's second moment. At uniformly random
    angles every derivative averages to zero, so this estimates the same mean
    variance. On LiH it agrees with the directly computed mean to within 2%, well
    inside the sampling error of 100 samples (about 14%). Old warm-start rows keep
    parameter 0's variance, since their derivatives do not average to zero.

    A run made with several seeds (a seed column) has one row per seed and
    depth. Each depth is then reduced over seeds: mean_grad_var is the mean of
    the seeds' values and seed_std_grad_var their standard deviation (ddof=1),
    the shaded band. With one seed, seed_std_grad_var is NaN (no band).
    std_grad_var and sem_grad_var are averaged over seeds.

    Args:
        path: Path to scaling_analysis_data.csv.
        max_depth: Drop depths above this, if given.

    Returns:
        A list of SimpleNamespace with n_layers, n_seeds, mean_grad_var,
        seed_std_grad_var, std_grad_var, sem_grad_var and mean_energy_error,
        sorted by depth.
    """
    rows_by_depth = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            depth = int(row["n_layers"])
            if max_depth is not None and depth > max_depth:
                continue
            var = float(row["mean_grad_var"])
            old = "grad_var_param0" not in row
            if old and row.get("start", "random") == "random":
                norm, spread = float(row["mean_grad_norm"]), float(
                    row["std_grad_norm"]
                )
                var = (norm**2 + spread**2) / int(row["n_parameters"])
            rows_by_depth.setdefault(depth, []).append(
                (
                    var,
                    float(row["std_grad_var"]),
                    float(row.get("sem_grad_var") or "nan"),
                    float(row.get("mean_energy_error") or "nan"),
                )
            )
    points = []
    for depth, rows in sorted(rows_by_depth.items()):
        var, std, sem, err = (np.array(c) for c in zip(*rows, strict=True))
        points.append(
            SimpleNamespace(
                n_layers=depth,
                n_seeds=len(rows),
                mean_grad_var=float(var.mean()),
                seed_std_grad_var=(
                    float(var.std(ddof=1)) if len(rows) > 1 else float("nan")
                ),
                std_grad_var=float(std.mean()),
                sem_grad_var=float(sem.mean()),
                mean_energy_error=float(err.mean()),
            )
        )
    return points


def system_info(run_dir: str):
    """Return a run's molecular FCI energy and a title label.

    Reads the molecule from run_parameters.json and loads its cached system.

    Args:
        run_dir: The run directory holding run_parameters.json.

    Returns:
        A (fci_energy, label) tuple, e.g. (..., "H2 (4 qubits)"). Both are None if
        unavailable.
    """
    # Imported here to avoid a core<-utils import cycle at module load.
    from src.core.backend import load_hamiltonian

    try:
        params = json.load(open(os.path.join(run_dir, "run_parameters.json")))
        system = load_hamiltonian(
            params["molecule"], params.get("geometry", "equilibrium")
        )
        label = f"{system.molecule} ({system.n_qubits} qubits)"
        return system.fci_energy, label
    except Exception:
        return None, None


def main_iters(run_dir: str) -> int | None:
    """Return the run's main-stage iteration budget (--iters), or None if unknown.

    Args:
        run_dir: The run directory holding run_parameters.json.
    """
    try:
        with open(
            os.path.join(run_dir, "run_parameters.json"), encoding="utf-8"
        ) as fh:
            return int(json.load(fh)["iters"])
    except (OSError, KeyError, ValueError):
        return None


def _ensure_parent(path: str) -> None:
    """Create the parent directory of path if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _rowdicts(rows):
    """Coerce dataclass rows to plain dicts."""
    out = []
    for r in rows:
        out.append(asdict(r) if is_dataclass(r) else dict(r))
    return out


def write_csv(path: str, rows, fieldnames=None) -> str:
    """Write rows to a CSV file.

    Args:
        path: Destination file path.
        rows: Iterable of dataclass instances or dicts.
        fieldnames: Explicit column order, inferred from the first row if None.

    Returns:
        The written path.
    """
    dicts = _rowdicts(rows)
    if fieldnames is None:
        fieldnames = list(dicts[0].keys()) if dicts else []
    _ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dicts)
    return path


def write_convergence_csv(path: str, results_by_opt) -> str:
    """Save per-iteration energy histories as a tidy (long-format) CSV.

    Each row is one iteration of one method under one optimizer, so the
    convergence curves can be redrawn later without re-running the VQE. Also
    records the stage boundary (where a two-stage method hands off) for plotting.

    Columns:
        energy: Raw per-stage energy (may switch observable at a handoff).
        energy_global: Energy vs the full Hamiltonian for the whole trajectory
            (continuous). Identical to energy for single-stage methods. When it is
            shorter than energy (pretrained reports only the quantum stage), it is
            aligned to the end, and the earlier rows get a blank energy_global.

    Args:
        path: Destination CSV path.
        results_by_opt: Mapping {optimizer: [MethodResult, ...]}.

    Returns:
        The written path.
    """
    fields = [
        "optimizer",
        "method",
        "iteration",
        "energy",
        "energy_global",
        "is_stage_boundary",
    ]
    _ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for opt, results in results_by_opt.items():
            for r in results:
                boundaries = set(getattr(r, "stage_boundaries", []) or [])
                raw = r.energy_history
                gh = getattr(r, "energy_history_global", None) or raw
                # Right-align energy_global (see Columns in the docstring).
                offset = len(raw) - len(gh)
                for it, e in enumerate(raw):
                    gi = it - offset
                    g = gh[gi] if 0 <= gi < len(gh) else ""
                    writer.writerow(
                        {
                            "optimizer": opt,
                            "method": r.method,
                            "iteration": it,
                            "energy": float(e),
                            "energy_global": (float(g) if g != "" else ""),
                            "is_stage_boundary": int(it in boundaries),
                        }
                    )
    return path


def seed_run_dir(molecule: str, run_name: str, seed: int) -> str:
    """Create and return results/<molecule>/vqe_seeds/<run_name>/seed_<seed>/.

    All seeds of one setting share the run_name folder, so the aggregate step can
    find them together.

    Args:
        molecule: Molecule name.
        run_name: Name of the setting shared by the seeds (see
            scripts/compute_vqe_analysis.py).
        seed: The run's seed.

    Returns:
        The absolute path of the created directory.
    """
    return make_output_dir(
        molecule, os.path.join(SEEDS_ANALYSIS, run_name), f"seed_{seed}"
    )


def resolve_seeds_dir(
    run_dir: str | None, molecule: str | None, run_name: str | None
) -> str:
    """Resolve a results/<molecule>/vqe_seeds/<run_name>/ folder.

    Args:
        run_dir: An explicit folder, returned as-is when given.
        molecule: The molecule to look under, when run_dir is None.
        run_name: The run name. May be None when the molecule has only one.

    Returns:
        The folder path.

    Raises:
        SystemExit: If the folder cannot be determined.
    """
    if run_dir:
        return run_dir
    if not molecule:
        raise SystemExit("Provide --dir, or --molecule (and --run-name).")
    root = os.path.join(RESULTS_ROOT, molecule, SEEDS_ANALYSIS)
    if run_name:
        return os.path.join(root, run_name)
    names = sorted(
        d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)
    )
    if len(names) != 1:
        raise SystemExit(
            f"Pick one with --run-name: {[os.path.basename(n) for n in names]}"
        )
    return names[0]


def find_seed_dirs(run_dir: str) -> dict[int, str]:
    """Return {seed: seed_<seed> folder} for the seed runs under run_dir, by seed.

    Args:
        run_dir: A results/<molecule>/vqe_seeds/<run_name>/ folder.

    Returns:
        The seed folders that hold a performance table, sorted by seed.
    """
    found = {}
    for d in glob.glob(os.path.join(run_dir, "seed_*")):
        tail = os.path.basename(d)[len("seed_") :]
        if tail.isdigit() and os.path.isfile(
            os.path.join(d, "performance_table_all_methods.csv")
        ):
            found[int(tail)] = d
    return dict(sorted(found.items()))


def _band(values: list):
    """Return values, or None when the table had no band columns."""
    return None if any(v is None for v in values) else values


def load_convergence_mean_std(path: str) -> dict:
    """Load convergence_mean_std.csv into {optimizer: [plot-ready result, ...]}.

    Args:
        path: Path to convergence_mean_std.csv (written by
            scripts/aggregate_vqe_seeds.py).

    Returns:
        A mapping {optimizer: [SimpleNamespace(method, energy_history,
        energy_lower, energy_upper, stage_boundaries), ...]} where
        energy_history is the mean over seeds, so the convergence plot functions
        draw it like a single run, and energy_lower and energy_upper are the 16th
        and 84th percentiles over seeds, the shaded band. Tables written before
        the percentile columns existed give no band (both None).
    """
    acc = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = acc.setdefault(row["optimizer"], {}).setdefault(row["method"], {})
            band = "energy_p16" in row
            d[int(row["iteration"])] = (
                float(row["energy_mean"]),
                float(row["energy_p16"]) if band else None,
                float(row["energy_p84"]) if band else None,
            )
    results_by_opt = {}
    for opt, methods in acc.items():
        results_by_opt[opt] = [
            SimpleNamespace(
                method=m,
                energy_history=[d[i][0] for i in sorted(d)],
                energy_lower=_band([d[i][1] for i in sorted(d)]),
                energy_upper=_band([d[i][2] for i in sorted(d)]),
                stage_boundaries=[],
            )
            for m, d in methods.items()
        ]
    return results_by_opt


def write_json(path: str, obj) -> str:
    """Write obj to a JSON file (indented).

    Args:
        path: Destination file path.
        obj: A JSON-serializable object (dataclasses are converted).

    Returns:
        The written path.
    """
    if is_dataclass(obj):
        obj = asdict(obj)
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    return path


def code_version() -> dict:
    """Return the code and package versions a run was made with.

    git_commit is the checked-out commit and git_dirty says whether tracked files
    had uncommitted changes (untracked files such as new results do not count).
    Both are None when git is not available. The package versions are those of
    the installed PennyLane, its lightning simulator and NumPy.
    """
    import subprocess
    from importlib.metadata import PackageNotFoundError, version

    def git(*args):
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()

    def package(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "pennylane": package("pennylane"),
        "pennylane_lightning": package("pennylane_lightning"),
        "numpy": package("numpy"),
    }


def write_run_parameters(
    output_dir: str, params: dict, data_dir: str, provenance: dict | None = None
) -> str:
    """Write run_parameters.json with provenance.

    Args:
        output_dir: The run's output directory.
        params: Run parameters (config, seeds, method list, ...).
        data_dir: The data-source directory recorded for provenance.
        provenance: Extra entries for the _provenance block, for example the
            methods rerun in a seed folder. The block always holds the code
            version as well (code_version).

    Returns:
        The written path.
    """
    payload = dict(params)
    payload["_provenance"] = {
        "data_dir": os.path.abspath(data_dir),
        **code_version(),
        **(provenance or {}),
    }
    return write_json(os.path.join(output_dir, "run_parameters.json"), payload)
