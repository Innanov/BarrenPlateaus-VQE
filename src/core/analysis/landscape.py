"""2D loss-landscape sampling over two real ansatz parameters.

The two axes are two circuit angles (i, j), chosen as the two parameters the
optimizer moved the most during training (see _select_param_indices), unless the
caller passes an explicit pair. Each is swept over a full 2*pi period centred on
its converged value, so theta* sits at the middle of the panel:

    x over [theta*[i] - pi, theta*[i] + pi]
    y over [theta*[j] - pi, theta*[j] + pi]

Every other parameter is held at theta*:

    E(x, y) = <H> evaluated at theta*, but with theta[i] := x and theta[j] := y

That is, the surface height at (x, y) is the energy of the converged parameters
with only angles i and j replaced by x and y.

Each optimizer's trajectory is projected onto those same two coordinates with no
shifting, so a path ends where it truly ended.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ...utils.helpers import angle_gap
from ..ansatze import Ansatz
from ..backend import make_device


@dataclass
class TrajectoryProjection:
    """One optimizer's path projected onto the landscape's two parameters.

    Attributes:
        label: Name of the trajectory (usually the optimizer).
        ab: (k, 2) array of the path's (theta[i], theta[j]) values.
        true_energy: (k,) true energy <H> at each full parameter vector (the path
            leaves the 2D slice, so this differs from the surface).
        surface_energy: (k,) in-plane energy at each point's (theta[i], theta[j])
            with the other parameters fixed at theta*, i.e. the surface height
            directly under the path.
    """

    label: str
    ab: np.ndarray
    true_energy: np.ndarray
    surface_energy: np.ndarray


@dataclass
class Landscape:
    """A sampled 2D energy landscape and any projected optimizer trajectories.

    Attributes:
        a_grid: 1D array of coordinates along axis 1 (values of theta[i]).
        b_grid: 1D array of coordinates along axis 2 (values of theta[j]).
        energies: 2D array of energies with shape (len(b_grid), len(a_grid)).
        center: The anchor parameter vector theta*.
        param_indices: The two swept parameter indices (i, j).
        axis_labels: Human-readable labels for the two axes, e.g.
            ("theta[3]", "theta[7]").
        trajectories: List of TrajectoryProjection, one per optimizer (empty if
            none supplied), all on these same two axes.
        span: (low, high) coordinate range of each axis.
    """

    a_grid: np.ndarray
    b_grid: np.ndarray
    energies: np.ndarray
    center: np.ndarray
    param_indices: tuple
    axis_labels: tuple
    trajectories: list
    span: tuple


def _select_param_indices(cost, center, paths_list=None, k=2, gauge_tol=0.20):
    """Pick the two parameters the optimizer made the most net progress on.

    Ranks each parameter by its net displacement over the run,
    circular|theta_start - theta_end| (averaged over the optimizers), and takes
    the top k. Net displacement (not arc length) avoids inflating the score with
    the per-step jitter of a stochastic optimizer. A gauge-tightness filter is
    applied first: only parameters whose converged value agrees across optimizers
    (spread within gauge_tol) qualify, so both paths' endpoints coincide on the
    slice rather than being split apart by the ansatz's gauge freedom.

    Args:
        cost: The energy cost callable (used only in the no-trajectory fallback).
        center: The anchor parameter vector theta*.
        paths_list: List of full parameter-trajectories (one per optimizer). If
            None or empty, falls back to ranking by cost sensitivity.
        k: Number of indices to return. Defaults to 2, the two axes of the
            landscape.
        gauge_tol: Maximum optimizer-endpoint spread (radians) to count as tight.

    Returns:
        A tuple of k parameter indices, sorted ascending.
    """
    center = np.asarray(center, dtype=float)
    n = center.size
    paths = [
        [np.asarray(p, dtype=float) for p in path]
        for path in (paths_list or [])
        if path is not None and len(path)
    ]

    # Fallback: no trajectories -> rank by finite-difference cost sensitivity.
    if not paths:
        e0 = float(cost(pnp.array(center, requires_grad=False)))
        sens = np.empty(n)
        for p in range(n):
            probe = center.copy()
            probe[p] += 0.15
            sens[p] = abs(float(cost(pnp.array(probe, requires_grad=False))) - e0)
        idx = np.argsort(sens)[::-1][:k]
        return tuple(int(i) for i in sorted(idx))

    # Net progress per angle, averaged over optimizers (circular start->end).
    progress = np.zeros(n)
    for path in paths:
        start, end = path[0], path[-1]
        for p in range(min(n, start.size, end.size)):
            progress[p] += angle_gap(start[p], end[p])
    progress /= len(paths)

    # Gauge-tightness: disagreement between optimizers' converged values.
    spread = np.zeros(n)
    if len(paths) >= 2:
        ends = [path[-1] for path in paths]
        for p in range(n):
            vals = [e[p] for e in ends if p < e.size]
            spread[p] = max(angle_gap(a, b) for a in vals for b in vals) if len(vals) > 1 else 0.0

    tight = np.where(spread <= gauge_tol)[0]
    if len(tight) >= k:
        # Most net-progress among gauge-tight angles.
        idx = tight[np.argsort(progress[tight])[::-1][:k]]
    else:
        # Too few tight angles: rank net progress damped by disagreement.
        score = progress / (1.0 + spread)
        idx = np.argsort(score)[::-1][:k]
    return tuple(int(i) for i in sorted(idx))


def _unwrap_to_endpoint(vals, ref):
    """Unwrap a sequence of angles onto the window centred on ref.

    The endpoint is mapped into [ref - pi, ref + pi), then every earlier point is
    shifted by the multiple of 2*pi keeping it within pi of the point after it. A
    shared ref (the same theta* for every optimizer) keeps two paths that
    converged to the same angle on the same side of the plotted window instead of
    splitting them to opposite edges.

    Args:
        vals: 1D array of angle values along the path.
        ref: Reference angle the window is centred on (theta* for this axis).

    Returns:
        The unwrapped 1D array, ending in [ref - pi, ref + pi).
    """
    v = np.asarray(vals, dtype=float).copy()
    if v.size == 0:
        return v
    v[-1] = ref + ((v[-1] - ref + np.pi) % (2 * np.pi) - np.pi)
    for t in range(v.size - 2, -1, -1):
        v[t] = v[t + 1] + ((v[t] - v[t + 1] + np.pi) % (2 * np.pi) - np.pi)
    return v


def _project_trajectory(label, trajectory, cost, center, i, j):
    """Project one parameter-space path onto the two swept parameters.

    The path's (theta[i], theta[j]) are read straight off each parameter vector
    (no position shifting) and unwrapped toward the endpoint so the drawn path
    never jumps across the 0<->2*pi seam (see _unwrap_to_endpoint). true_energy
    is the real <H> at the full vector. surface_energy is the in-plane height
    (others fixed at theta*).

    Args:
        label: Name for this trajectory (e.g. the optimizer).
        trajectory: Iterable of parameter vectors.
        cost: The energy cost callable.
        center: The anchor parameter vector theta*.
        i: First swept parameter index.
        j: Second swept parameter index.

    Returns:
        A TrajectoryProjection, or None if the trajectory is empty.
    """
    if trajectory is None or len(trajectory) == 0:
        return None
    traj = [np.asarray(p, dtype=float) for p in trajectory]
    ci, cj = float(center[i]), float(center[j])
    ai = _unwrap_to_endpoint(np.array([p[i] for p in traj]), ci)
    aj = _unwrap_to_endpoint(np.array([p[j] for p in traj]), cj)
    ab = np.column_stack([ai, aj])
    true_e = np.array([float(cost(pnp.array(p, requires_grad=False))) for p in traj])
    surf_e = []
    for p in traj:
        theta = np.asarray(center, dtype=float).copy()
        theta[i] = p[i]
        theta[j] = p[j]
        surf_e.append(float(cost(pnp.array(theta, requires_grad=False))))
    return TrajectoryProjection(
        label=label, ab=ab, true_energy=true_e, surface_energy=np.array(surf_e)
    )


def compute_landscape(
    ansatz: Ansatz,
    hamiltonian: qml.Hamiltonian,
    n_qubits: int,
    center,
    resolution: int = 41,
    trajectory=None,
    trajectories=None,
    param_indices=None,
    **_ignored,
) -> Landscape:
    """Sample a 2D energy landscape over two real ansatz parameters.

    Two parameter indices are each swept over a full period with all other
    parameters fixed at center (theta*), giving the true cost surface. Optimizer
    trajectories are projected onto those same two parameters with no shifting.

    Args:
        ansatz: The ansatz.
        hamiltonian: The observable Hamiltonian.
        n_qubits: Number of wires.
        center: Anchor parameter vector theta* (typically the converged
            parameters). Fixes the non-swept coordinates.
        resolution: Number of grid points per axis.
        trajectory: Optional single parameter-vector path (stored under the label
            "trajectory").
        trajectories: Optional {label: param_history} mapping, e.g.
            {"adam": [...], "qng": [...]}, overlaid on the same surface.
        param_indices: Optional explicit (i, j) to sweep. If None, the two most
            cost-sensitive parameters at center are chosen.

    Returns:
        A Landscape with the sampled energies and projected trajectories.
    """
    center = np.asarray(center, dtype=float)

    # Normalize the two trajectory inputs into an ordered {label: path} dict.
    paths = {}
    if trajectories:
        paths.update(trajectories)
    elif trajectory is not None:
        paths["trajectory"] = trajectory

    dev = make_device(n_qubits)

    @qml.qnode(dev)
    def cost(params):
        ansatz.apply(params)
        return qml.expval(hamiltonian)

    if param_indices is None:
        i, j = _select_param_indices(cost, center, paths_list=list(paths.values()))
    else:
        i, j = int(param_indices[0]), int(param_indices[1])

    # Sweep each axis over a full period centred on theta* (see module docstring).
    xs = center[i] + np.linspace(-np.pi, np.pi, resolution)  # cols -> theta[i]
    ys = center[j] + np.linspace(-np.pi, np.pi, resolution)  # rows -> theta[j]
    energies = np.empty((resolution, resolution))
    theta = center.copy()
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            theta[i] = x
            theta[j] = y
            energies[r, c] = float(cost(pnp.array(theta, requires_grad=False)))
        theta[i] = center[i]
        theta[j] = center[j]

    projected = []
    for label, path in paths.items():
        proj = _project_trajectory(label, path, cost, center, i, j)
        if proj is not None:
            projected.append(proj)

    return Landscape(
        a_grid=xs,
        b_grid=ys,
        energies=energies,
        center=center,
        param_indices=(i, j),
        axis_labels=(f"theta[{i}]", f"theta[{j}]"),
        trajectories=projected,
        span=(float(xs.min()), float(xs.max())),
    )
