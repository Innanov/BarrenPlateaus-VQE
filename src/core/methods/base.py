"""Shared method infrastructure: config and result containers, cost, loop, registry.

MethodConfig/MethodResult are the input settings and output record every method
uses. _make_cost builds the QNode cost and _optimize runs the iteration loop, so
each method is a thin wrapper that assembles a Hamiltonian and calls these. METHODS
maps each method name to its function and run_method dispatches by name.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ..ansatze import Ansatz
from ..backend import MolecularSystem, Optimizer, make_device


@dataclass
class MethodConfig:
    """Input configuration for a method run: ansatz depth, optimizer, and iteration counts.

    Attributes:
        depth: Ansatz depth (kept identical across methods for a comparison).
        optimizer: Optimizer name ('adam', 'qng', or 'adagrad').
        optimizer_kwargs: Extra keyword args for the optimizer.
        max_iters: Iterations of the main stage, the SAME for every method so their
            main-stage convergence compares iteration-for-iteration. Single-stage
            methods run this once, and adiabatic distributes it across its stages.
        seed: RNG seed for the initial parameters (reproducibility).
        warm_iters: Iterations of the warm-up stage for the two-stage methods
            (local_global's local-cost warm-up, pretrained's MPS pre-training),
            run as extra preparation ON TOP of max_iters so the main stage still
            gets the full max_iters. Ignored by single-stage methods (standard,
            sea, adiabatic).
        adiabatic_steps: Number of anneal stages for the adiabatic method. The
            Hamiltonian is held fixed within each stage, which runs
            max_iters / adiabatic_steps iterations warm-started from the previous.
            Default 10 (the reference value).
    """

    depth: int = 4
    optimizer: str = "adam"
    optimizer_kwargs: dict = field(default_factory=dict)
    max_iters: int = 200
    seed: int = 0
    warm_iters: int = 50
    adiabatic_steps: int = 10  # reference (qubap) value


@dataclass
class MethodResult:
    """What a method run returns: the converged solution plus its convergence record.

    Carries the final energy, parameters, and ansatz (for metrics like energy
    error and fidelity) and the per-iteration energy and parameter histories (for
    convergence curves and landscape overlays).

    Attributes:
        method: Method name.
        energy_history: Energy at each optimization iteration (concatenated
            across stages for multi-stage methods).
        final_energy: Energy at the returned parameters against the full H.
        params: Final parameter vector.
        ansatz: The ansatz whose apply reproduces the final state.
        n_params: Parameter count of the reported ansatz.
        stage_boundaries: Iteration indices where a new stage begins (for plots
            and per-stage reporting).
        param_history: Parameter vectors along the final stage's trajectory (in
            the reported ansatz's parameter space), for landscape overlays.
            Multi-stage methods record only the final stage, since earlier stages
            use a different space.
        energy_history_global: Per-iteration energy vs the global (full)
            Hamiltonian for the whole trajectory: continuous for two-stage methods
            (one observable throughout). energy_history may instead hold raw
            per-stage energies. Empty for single-stage methods, where it equals
            energy_history.
        runtime_s: Wall-clock optimization time in seconds, stamped by run_method.
            -1.0 if the method was called directly without timing.
    """

    method: str
    energy_history: list
    final_energy: float
    params: object
    ansatz: Ansatz
    n_params: int
    stage_boundaries: list = field(default_factory=list)
    param_history: list = field(default_factory=list)
    energy_history_global: list = field(default_factory=list)
    runtime_s: float = -1.0


def _fixed_hamiltonian(coeffs, obs) -> qml.Hamiltonian:
    """Build a simplified qml.Hamiltonian with non-trainable coefficients.

    The coefficients are fixed constants, never optimized, so they are marked
    non-trainable (requires_grad=False) to stop adjoint differentiation warning
    about them.

    Args:
        coeffs: The term coefficients.
        obs: The term observables.

    Returns:
        The simplified Hamiltonian.
    """
    return qml.Hamiltonian(pnp.array(coeffs, requires_grad=False), obs).simplify()


def _make_cost(ansatz: Ansatz, hamiltonian: qml.Hamiltonian, n_qubits: int) -> Callable:
    """Build a QNode cost function for <ansatz(params)| H |ansatz(params)>.

    Args:
        ansatz: The ansatz to evaluate.
        hamiltonian: The observable Hamiltonian.
        n_qubits: Number of device wires.

    Returns:
        A callable mapping a parameter vector to a scalar energy.
    """
    dev = make_device(n_qubits)

    @qml.qnode(dev)
    def cost(params):
        ansatz.apply(params)
        return qml.expval(hamiltonian)

    return cost


def _optimize(cost, params, optimizer: Optimizer, iters: int, reset: bool = True):
    """Run an optimization for iters steps, recording energy and parameters.

    Args:
        cost: Scalar cost callable.
        params: Initial parameter vector.
        optimizer: Optimizer conforming to the shared step API.
        iters: Number of optimization iterations.
        reset: If True (default), reset the optimizer's internal state first.
            Pass False to continue an optimizer across a stage boundary.

    Returns:
        A (final_params, history, param_history) triple, where history is the
        per-iteration energy and param_history the per-iteration parameters.
    """
    if reset:
        params = optimizer.reset(params)
    history = []
    param_history = []
    for _ in range(iters):
        params, energy = optimizer.step(cost, params)
        history.append(float(energy))
        param_history.append(np.asarray(params, dtype=float).copy())
    return params, history, param_history


def _load_methods() -> dict[str, Callable[[MolecularSystem, MethodConfig], MethodResult]]:
    """Import the method functions and return the name->function registry.

    Imported here, not at module top, to avoid a circular import: each method
    module does `from .base import ...`, so importing them while base itself is
    still loading would deadlock. By the time this runs (first METHODS access or
    a run_method call) base is fully initialized.
    """
    from .adiabatic import adiabatic
    from .local_global import local_global
    from .pretrained import pretrained
    from .sea import sea
    from .standard import standard

    return {
        "standard": standard,
        "local_global": local_global,
        "adiabatic": adiabatic,
        "sea": sea,
        "pretrained": pretrained,
    }


class _MethodRegistry(Mapping):
    """A lazy, dict-like registry of the method functions.

    Iteration and len (list(METHODS), enumerate(METHODS), choices=list(METHODS))
    work off the fixed method-name order without importing anything. Indexing
    (METHODS[name]) loads the functions on first use. This keeps METHODS a plain
    mapping to callers while breaking the base<->method import cycle.
    """

    _NAMES = ("standard", "local_global", "adiabatic", "sea", "pretrained")

    def __init__(self):
        self._funcs = None

    def _resolve(self):
        if self._funcs is None:
            self._funcs = _load_methods()
        return self._funcs

    def __getitem__(self, name):
        return self._resolve()[name]

    def __iter__(self):
        return iter(self._NAMES)

    def __len__(self):
        return len(self._NAMES)


METHODS = _MethodRegistry()


def run_method(name: str, system: MolecularSystem, config: MethodConfig) -> MethodResult:
    """Look up a method in METHODS by name, run it, and time it.

    The dispatch entry point: unlike calling a method function directly, this
    validates the name and stamps the wall-clock runtime onto result.runtime_s.

    Args:
        name: One of the method names in METHODS.
        system: The molecular system to solve.
        config: Shared method configuration.

    Returns:
        The MethodResult, with runtime_s set.

    Raises:
        ValueError: If name is not a registered method.
    """
    if name not in METHODS:
        raise ValueError(f"Unknown method '{name}'. Choices: {sorted(METHODS)}")
    t0 = time.perf_counter()
    result = METHODS[name](system, config)
    result.runtime_s = time.perf_counter() - t0
    return result
