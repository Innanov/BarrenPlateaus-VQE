"""Output directory management and CSV/JSON writers.

Results are written under results/<molecule>/<analysis_type>/<timestamp>/ (CSVs,
JSON summaries, run parameters, plus PDFs written by plots). Every
run_parameters.json records the data-source path and the current git commit for
provenance.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")


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
    fields = ["optimizer", "method", "iteration", "energy", "energy_global", "is_stage_boundary"]
    _ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for opt, results in results_by_opt.items():
            for r in results:
                boundaries = set(getattr(r, "stage_boundaries", []) or [])
                raw = r.energy_history
                gh = getattr(r, "energy_history_global", None) or raw
                # Right-align energy_global (see the Columns note in the docstring).
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


def save_landscape_npz(path: str, landscape) -> str:
    """Save a landscape's grid and trajectories to a compact .npz.

    Stored as named numpy arrays so the surface and contour figures can be
    rebuilt without re-sampling the expensive grid.

    Args:
        path: Destination .npz path.
        landscape: A core.landscape.Landscape instance.

    Returns:
        The written path.
    """
    arrays = {
        "a_grid": np.asarray(landscape.a_grid),
        "b_grid": np.asarray(landscape.b_grid),
        "energies": np.asarray(landscape.energies),
        "span": np.asarray(landscape.span, dtype=float),
        "param_indices": np.asarray(getattr(landscape, "param_indices", ()), dtype=int),
        "axis_labels": np.asarray(getattr(landscape, "axis_labels", ()), dtype=object),
    }
    labels = []
    for i, tr in enumerate(getattr(landscape, "trajectories", None) or []):
        labels.append(tr.label)
        arrays[f"traj{i}_ab"] = np.asarray(tr.ab)
        arrays[f"traj{i}_true_energy"] = np.asarray(tr.true_energy)
        arrays[f"traj{i}_surface_energy"] = np.asarray(tr.surface_energy)
    arrays["traj_labels"] = np.asarray(labels, dtype=object)
    _ensure_parent(path)
    np.savez_compressed(path, **arrays)
    return path


def load_landscape_npz(path: str):
    """Load a landscape saved by save_landscape_npz.

    Rebuilds a core.landscape.Landscape from the saved arrays so figures can be
    regenerated from disk.
    """
    # Imported here to avoid a core<-utils import cycle at module load.
    from src.core.analysis import Landscape, TrajectoryProjection

    data = np.load(path, allow_pickle=True)
    labels = list(data["traj_labels"]) if "traj_labels" in data else []
    trajectories = []
    for i, label in enumerate(labels):
        trajectories.append(
            TrajectoryProjection(
                label=str(label),
                ab=data[f"traj{i}_ab"],
                true_energy=data[f"traj{i}_true_energy"],
                surface_energy=data[f"traj{i}_surface_energy"],
            )
        )
    span = data["span"]
    span = (
        tuple(float(s) for s in np.asarray(span).reshape(-1))
        if np.size(span) == 2
        else float(np.asarray(span).reshape(-1)[0])
    )
    pidx = tuple(int(i) for i in data["param_indices"]) if "param_indices" in data else ()
    labels = tuple(str(lbl) for lbl in data["axis_labels"]) if "axis_labels" in data else ()
    return Landscape(
        a_grid=data["a_grid"],
        b_grid=data["b_grid"],
        energies=data["energies"],
        center=None,
        param_indices=pidx,
        axis_labels=labels,
        trajectories=trajectories,
        span=span,
    )


def save_optimization_npz(path: str, method, final_params, param_histories) -> str:
    """Persist a method's optimizer paths so landscapes can be rebuilt later.

    Sampling a landscape needs only the converged parameters (the slice centre)
    and each optimizer's trajectory, so saving those lets the plot step build
    landscapes without re-running the VQE.

    Args:
        path: Destination .npz path (per method).
        method: Method name.
        final_params: The reference (landscape-centre) parameter vector.
        param_histories: {optimizer: [param_vector, ...]} for the method.

    Returns:
        The written path.
    """
    arrays = {
        "method": np.asarray(str(method)),
        "final_params": np.asarray(final_params, dtype=float),
        "optimizers": np.asarray(list(param_histories.keys()), dtype=object),
    }
    for opt, hist in param_histories.items():
        arrays[f"hist_{opt}"] = np.asarray(hist, dtype=float)
    _ensure_parent(path)
    np.savez_compressed(path, **arrays)
    return path


def load_optimization_npz(path: str):
    """Load optimizer paths saved by save_optimization_npz.

    Args:
        path: Path to the .npz file.

    Returns:
        A (method, final_params, {optimizer: history}) tuple.
    """
    data = np.load(path, allow_pickle=True)
    method = str(data["method"])
    final_params = data["final_params"]
    opts = list(data["optimizers"])
    histories = {str(o): data[f"hist_{o}"] for o in opts}
    return method, final_params, histories


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


def write_run_parameters(output_dir: str, params: dict, data_dir: str) -> str:
    """Write run_parameters.json with provenance.

    Args:
        output_dir: The run's output directory.
        params: Run parameters (config, seeds, method list, ...).
        data_dir: The data-source directory recorded for provenance.

    Returns:
        The written path.
    """
    payload = dict(params)
    payload["_provenance"] = {"data_dir": os.path.abspath(data_dir)}
    return write_json(os.path.join(output_dir, "run_parameters.json"), payload)
