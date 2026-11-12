"""Gradient-variance scaling: the barren-plateau measurement.

The barren-plateau quantity is the variance of a cost-function partial
derivative over parameter samples, measured versus circuit depth (the standard
BP diagnostic from McClean et al.).

For each (ansatz, depth) we draw parameter samples, take the gradient of the cost
at each, and keep a fixed reference component (component 0). Its variance across
samples is the reported mean_grad_var. mean_grad_norm reports the typical
gradient magnitude for context.

Two kinds of samples are supported, so every method can be measured where it
actually starts training:

- gradient_scaling draws random parameters (uniform [0, 2pi)) and differentiates
  one of the costs in COSTS: the molecular energy (standard, sea), the
  Hartree-Fock local cost that local_global trains on first, or the Hartree-Fock
  reference Hamiltonian that adiabatic starts its anneal on.
- warm_start_gradient_scaling runs a two-stage method's warm-up (local_global's
  local-cost stage, pretrained's MPS pre-training) from each sample's seed and
  differentiates the molecular energy at the resulting handoff point, the point
  where its energy stage begins. It reaches that point by running the method with
  max_iters=0, so the method code is used unchanged. Sample i uses seed + i, so
  the first samples coincide with the VQE seeds of the same depth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ...utils.progress import format_duration
from ..ansatze import build_ansatz
from ..backend import MolecularSystem, make_device

# Costs gradient_scaling can differentiate at random parameters.
COSTS = ("energy", "local_cost", "hf_reference")

# Two-stage methods whose handoff point warm_start_gradient_scaling measures.
WARM_START_METHODS = ("local_global", "pretrained")


@dataclass
class GradientScalingPoint:
    """One (molecule, method, depth) measurement.

    Attributes:
        molecule: Molecule name.
        method: Ansatz name for random samples, method name for warm starts.
        n_layers: Circuit depth.
        n_parameters: Parameter count at this depth.
        mean_grad_var: Variance of the reference gradient component across
            samples (the BP metric).
        std_grad_var: Spread of the per-component variances across components.
        mean_grad_norm: Mean gradient L2 norm over the samples.
        std_grad_norm: Std of the gradient L2 norm over the samples.
        n_qubits: Number of qubits.
        runtime_s: Wall-clock time (seconds) to sample all gradients at this
            depth.
        cost: The differentiated cost, one of COSTS.
        start: 'random' for random parameters, 'warm_start' for a handoff point.
        n_samples: Number of parameter samples.
        mean_energy_error: Mean of E - E_FCI at the samples (warm starts only,
            NaN otherwise), to tell a small gradient near the minimum from one on
            a plateau.
        std_energy_error: Std of E - E_FCI at the samples (warm starts only).
    """

    molecule: str
    method: str
    n_layers: int
    n_parameters: int
    mean_grad_var: float
    std_grad_var: float
    mean_grad_norm: float
    std_grad_norm: float
    n_qubits: int
    runtime_s: float = -1.0
    cost: str = "energy"
    start: str = "random"
    n_samples: int = 0
    mean_energy_error: float = float("nan")
    std_energy_error: float = float("nan")


def cost_observable(
    system: MolecularSystem, cost: str = "energy"
) -> qml.Hamiltonian:
    """Return the observable whose expectation is the named cost.

    Args:
        system: The molecular system.
        cost: 'energy' (the molecular Hamiltonian), 'local_cost' (local_global's
            first-stage cost, local_cost_observable) or 'hf_reference' (adiabatic's
            first-stage Hamiltonian, hf_reference_hamiltonian).

    Returns:
        The observable.

    Raises:
        ValueError: If cost is not in COSTS.
    """
    # Imported here so this module does not import the methods at load time.
    from ..methods.adiabatic import hf_reference_hamiltonian
    from ..methods.local_global import local_cost_observable

    if cost == "energy":
        return system.hamiltonian
    if cost == "local_cost":
        return local_cost_observable(system.hf_state, system.n_qubits)
    if cost == "hf_reference":
        return hf_reference_hamiltonian(system.hf_occupation)
    raise ValueError(f"Unknown cost '{cost}'. Choices: {list(COSTS)}")


def _grad_fn(ansatz, observable, n_qubits):
    """Return the gradient function of <ansatz(params)| observable |ansatz(params)>."""
    dev = make_device(n_qubits)

    @qml.qnode(dev)
    def cost(params):
        ansatz.apply(params)
        return qml.expval(observable)

    return qml.grad(cost)


def _energy_grads(ansatz, hamiltonian, n_qubits, n_samples, rng):
    """Sample cost gradients at random initializations.

    Args:
        ansatz: The ansatz to differentiate.
        hamiltonian: The observable whose expectation is the cost.
        n_qubits: Number of wires.
        n_samples: Number of random initializations.
        rng: NumPy random generator.

    Returns:
        An array of shape (n_samples, n_params).
    """
    grad_fn = _grad_fn(ansatz, hamiltonian, n_qubits)
    grads = []
    for _ in range(n_samples):
        params = ansatz.random_params(rng)
        grads.append(np.asarray(grad_fn(params), dtype=float))
    return np.array(grads)


def _point(
    system, method, depth, n_params, grads, runtime_s, cost, start, errors=None
):
    """Reduce sampled gradients (and energy errors) to a GradientScalingPoint."""
    var_per_component = np.var(grads, axis=0)
    norms = np.linalg.norm(grads, axis=1)
    errors = np.asarray(errors if errors is not None else [np.nan], dtype=float)
    return GradientScalingPoint(
        molecule=system.molecule,
        method=method,
        n_layers=depth,
        n_parameters=n_params,
        mean_grad_var=float(var_per_component[0]),
        std_grad_var=float(np.std(var_per_component)),
        mean_grad_norm=float(np.mean(norms)),
        std_grad_norm=float(np.std(norms)),
        n_qubits=system.n_qubits,
        runtime_s=runtime_s,
        cost=cost,
        start=start,
        n_samples=len(grads),
        mean_energy_error=float(np.mean(errors)),
        std_energy_error=float(np.std(errors)),
    )


def _progress(point, k, depths, sweep_start):
    """Print one per-depth progress line with elapsed time and ETA."""
    elapsed = time.perf_counter() - sweep_start
    done = k + 1
    eta = (elapsed / done) * (len(depths) - done)
    extra = ""
    if not np.isnan(point.mean_energy_error):
        extra = f"  E-E_FCI={point.mean_energy_error:.3e}"
    print(
        f"    depth {point.n_layers:>2}/{depths[-1]}  "
        f"grad_var={point.mean_grad_var:.3e}{extra}  "
        f"({point.runtime_s:.1f}s | elapsed {format_duration(elapsed)} | "
        f"ETA {format_duration(eta)})",
        flush=True,
    )


def gradient_scaling(
    system: MolecularSystem,
    method: str = "efficient_su2",
    depths=None,
    n_samples: int = 100,
    seed: int = 0,
    verbose: bool = False,
    cost: str = "energy",
) -> list[GradientScalingPoint]:
    """Measure gradient-variance scaling versus depth at random parameters.

    Args:
        system: The molecular system.
        method: Ansatz name ('efficient_su2', 'mps' or 'sea').
        depths: Iterable of circuit depths to sweep. Defaults to 1..10.
        n_samples: Number of random initializations per depth.
        seed: Base RNG seed (offset per depth for independent draws).
        verbose: Print a progress line per depth.
        cost: The differentiated cost, one of COSTS (see cost_observable).

    Returns:
        A list of GradientScalingPoint, one per depth.
    """
    depths = list(range(1, 11) if depths is None else depths)
    observable = cost_observable(system, cost)
    points = []
    sweep_start = time.perf_counter()
    for k, depth in enumerate(depths):
        t0 = time.perf_counter()
        ansatz = build_ansatz(method, system.n_qubits, d=depth)
        rng = np.random.default_rng(seed + depth)
        grads = _energy_grads(ansatz, observable, system.n_qubits, n_samples, rng)
        point = _point(
            system,
            method,
            depth,
            ansatz.n_params,
            grads,
            time.perf_counter() - t0,
            cost,
            "random",
        )
        points.append(point)
        if verbose:
            _progress(point, k, depths, sweep_start)
    return points


def warm_start_gradient_scaling(
    system: MolecularSystem,
    method: str,
    depths,
    n_samples: int = 20,
    seed: int = 0,
    warm_iters: int = 150,
    optimizer: str = "adam",
    verbose: bool = False,
) -> list[GradientScalingPoint]:
    """Measure the energy-gradient variance at a two-stage method's handoff point.

    For each depth and sample, runs the method's warm-up stage from seed + i (by
    calling it with max_iters=0, see the module docstring) and differentiates the
    molecular energy at the returned parameters, in the ansatz the energy stage
    trains. The energy error there is recorded too, because a warm start near the
    minimum also has small gradients, for a different reason than a plateau.

    Args:
        system: The molecular system.
        method: One of WARM_START_METHODS.
        depths: Iterable of circuit depths.
        n_samples: Number of seeds per depth.
        seed: First seed. Sample i uses seed + i at every depth.
        warm_iters: Warm-up iterations, as in the VQE runs.
        optimizer: Optimizer of the warm-up stage, as in the VQE runs.
        verbose: Print a progress line per depth.

    Returns:
        A list of GradientScalingPoint, one per depth.

    Raises:
        ValueError: If method is not in WARM_START_METHODS.
    """
    from ..methods import MethodConfig, run_method

    if method not in WARM_START_METHODS:
        raise ValueError(
            f"No warm start for '{method}'. Choices: {list(WARM_START_METHODS)}"
        )
    depths = list(depths)
    points = []
    sweep_start = time.perf_counter()
    for k, depth in enumerate(depths):
        t0 = time.perf_counter()
        grads, errors, grad_fn, n_params = [], [], None, 0
        for i in range(n_samples):
            config = MethodConfig(
                depth=depth,
                optimizer=optimizer,
                max_iters=0,
                warm_iters=warm_iters,
                seed=seed + i,
            )
            r = run_method(method, system, config)
            if grad_fn is None:
                grad_fn = _grad_fn(r.ansatz, system.hamiltonian, system.n_qubits)
                n_params = r.n_params
            params = pnp.array(
                np.asarray(r.params, dtype=float), requires_grad=True
            )
            grads.append(np.asarray(grad_fn(params), dtype=float))
            errors.append(r.final_energy - system.fci_energy)
        point = _point(
            system,
            method,
            depth,
            n_params,
            np.array(grads),
            time.perf_counter() - t0,
            "energy",
            "warm_start",
            errors,
        )
        points.append(point)
        if verbose:
            _progress(point, k, depths, sweep_start)
    return points
