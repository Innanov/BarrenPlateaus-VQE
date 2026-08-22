"""Gradient-variance scaling: the barren-plateau measurement.

The barren-plateau quantity is the variance of a cost-function partial
derivative over random parameter initializations, measured versus circuit depth
(the standard BP diagnostic from McClean et al., sampled at random points, not
around a converged optimum).

For each (ansatz, depth) we draw n_samples random parameter vectors, take the
energy gradient of each, and keep a fixed reference component (component 0). Its
variance across samples is the reported mean_grad_var. mean_grad_norm reports the
typical gradient magnitude for context.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pennylane as qml

from ..utils.helpers import format_duration
from .ansatze import build_ansatz
from .devices import make_device
from .hamiltonians import MolecularSystem


@dataclass
class GradientScalingPoint:
    """One (molecule, method, depth) measurement.

    Attributes:
        molecule: Molecule name.
        method: Ansatz/method name used for the circuit.
        n_layers: Circuit depth.
        n_parameters: Parameter count at this depth.
        mean_grad_var: Variance of the reference gradient component across
            random initializations (the BP metric).
        std_grad_var: Bootstrap/spread estimate of mean_grad_var across
            components.
        mean_grad_norm: Mean gradient L2 norm over the samples.
        std_grad_norm: Std of the gradient L2 norm over the samples.
        n_qubits: Number of qubits.
        runtime_s: Wall-clock time (seconds) to sample all gradients at this
            depth.
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


def _energy_grads(ansatz, hamiltonian, n_qubits, n_samples, rng):
    """Sample energy gradients at random initializations.

    Args:
        ansatz: The ansatz to differentiate.
        hamiltonian: The observable Hamiltonian.
        n_qubits: Number of wires.
        n_samples: Number of random initializations.
        rng: NumPy random generator.

    Returns:
        An array of shape (n_samples, n_params).
    """
    dev = make_device(n_qubits)

    @qml.qnode(dev)
    def cost(params):
        ansatz.apply(params)
        return qml.expval(hamiltonian)

    grad_fn = qml.grad(cost)
    grads = []
    for _ in range(n_samples):
        params = ansatz.random_params(rng)
        grads.append(np.asarray(grad_fn(params), dtype=float))
    return np.array(grads)


def gradient_scaling(
    system: MolecularSystem,
    method: str = "efficient_su2",
    depths=None,
    n_samples: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> list[GradientScalingPoint]:
    """Measure gradient-variance scaling versus depth for one molecule.

    Args:
        system: The molecular system.
        method: Ansatz name ('efficient_su2', 'mps' or 'sea').
        depths: Iterable of circuit depths to sweep. Defaults to 1..10.
        n_samples: Number of random initializations per depth.
        seed: Base RNG seed (offset per depth for independent draws).

    Returns:
        A list of GradientScalingPoint, one per depth.
    """
    depths = list(range(1, 11) if depths is None else depths)
    points = []
    sweep_start = time.perf_counter()
    for k, depth in enumerate(depths):
        t0 = time.perf_counter()
        ansatz = build_ansatz(method, system.n_qubits, depth=depth)
        rng = np.random.default_rng(seed + depth)
        grads = _energy_grads(ansatz, system.hamiltonian, system.n_qubits, n_samples, rng)
        # Reference component variance (the BP metric). Use component 0.
        var_per_component = np.var(grads, axis=0)
        norms = np.linalg.norm(grads, axis=1)
        dt = time.perf_counter() - t0
        points.append(
            GradientScalingPoint(
                molecule=system.molecule,
                method=method,
                n_layers=depth,
                n_parameters=ansatz.n_params,
                mean_grad_var=float(var_per_component[0]),
                std_grad_var=float(np.std(var_per_component)),
                mean_grad_norm=float(np.mean(norms)),
                std_grad_norm=float(np.std(norms)),
                n_qubits=system.n_qubits,
                runtime_s=dt,
            )
        )
        if verbose:
            # Per-depth progress line with elapsed and ETA.
            elapsed = time.perf_counter() - sweep_start
            done = k + 1
            eta = (elapsed / done) * (len(depths) - done)
            print(
                f"    depth {depth:>2}/{depths[-1]}  "
                f"grad_var={var_per_component[0]:.3e}  "
                f"({dt:.1f}s | elapsed {format_duration(elapsed)} | "
                f"ETA {format_duration(eta)})",
                flush=True,
            )
    return points
