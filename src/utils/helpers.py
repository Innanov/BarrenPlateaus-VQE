"""Shared circuit, quantum-information, and test-system utilities.

Small helpers reused across the project: run an ansatz to a statevector or gate
list (statevector, tape_ops), quantum-information operations (expectation,
single_qubit_marginals, angle_gap), a filesystem timestamp, and builders for the
small hand-made systems the tests run on (ising_hamiltonian, ising_system).
"""

import datetime

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ..core.backend import MolecularSystem, make_device


def tape_ops(ansatz, params):
    """Record the gate sequence that ansatz.apply(params) produces."""
    with qml.tape.QuantumTape() as tape:
        ansatz.apply(params)
    return list(tape.operations)


def statevector(ansatz, n_qubits, params=None, seed=0):
    """Return the statevector the ansatz produces.

    Args:
        ansatz: The ansatz.
        n_qubits: Number of wires.
        params: Parameter vector. If None, random parameters are drawn (seeded).
        seed: Seed for the random parameters when params is None.

    Returns:
        A complex numpy array of length 2**n_qubits.
    """
    if params is None:
        params = ansatz.random_params(np.random.default_rng(seed))
    dev = make_device(n_qubits)

    @qml.qnode(dev)
    def circuit(p):
        ansatz.apply(p)
        return qml.state()

    return np.asarray(circuit(pnp.array(params, requires_grad=False)))


def timestamp() -> str:
    """Return a filesystem-safe YYYY-MM-DD_HH-MM-SS timestamp."""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def angle_gap(a, b):
    """Smallest distance between two angles on the circle, in [0, pi]."""
    d = abs((a - b) % (2 * np.pi))
    return min(d, 2 * np.pi - d)


def expectation(hamiltonian, state, n_qubits):
    """Return <state| hamiltonian |state> for an explicit statevector.

    Args:
        hamiltonian: A pennylane observable.
        state: A statevector of length 2**n_qubits.
        n_qubits: Number of wires.

    Returns:
        The real expectation value as a float.
    """
    dev = make_device(n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(state, wires=range(n_qubits))
        return qml.expval(hamiltonian)

    return float(circuit())


def single_qubit_marginals(state, n_qubits):
    """Return the single-qubit reduced density matrices of a statevector.

    Args:
        state: A statevector of length 2**n_qubits.
        n_qubits: Number of qubits.

    Returns:
        A list of n_qubits 2x2 complex numpy arrays, one per qubit (each the
        partial trace over all the other qubits).
    """
    psi = np.asarray(state, dtype=complex).reshape([2] * n_qubits)
    rdms = []
    for j in range(n_qubits):
        others = [k for k in range(n_qubits) if k != j]
        rdms.append(np.tensordot(psi, psi.conj(), axes=(others, others)))
    return rdms


def ising_hamiltonian(n_qubits):
    """A small transverse-field Ising Hamiltonian for the given qubit count (tests).

    2 qubits is a single Z-Z bond plus transverse fields, and 4 qubits is a Z-Z
    ladder plus transverse fields. Only these two sizes are provided.
    """
    if n_qubits == 2:
        return qml.Hamiltonian(
            [1.0, 1.0, 0.5, 0.5],
            [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0), qml.PauliX(1), qml.PauliZ(0)],
        )
    if n_qubits == 4:
        return qml.Hamiltonian(
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5],
            [
                qml.PauliZ(0) @ qml.PauliZ(1),
                qml.PauliZ(1) @ qml.PauliZ(2),
                qml.PauliZ(2) @ qml.PauliZ(3),
                qml.PauliX(0),
                qml.PauliX(1),
                qml.PauliX(2),
                qml.PauliX(3),
            ],
        )
    raise ValueError(f"No hand-made Hamiltonian for {n_qubits} qubits (use 2 or 4).")


def ising_system(n_qubits):
    """A small MolecularSystem wrapping ising_hamiltonian(n_qubits), for tests.

    Constructs the MolecularSystem in code (fci fields unused) instead of loading a
    real molecule from the data/ cache, so tests are self-contained and fast. The
    exact ground_state and ground_energy are solved on first access.
    """
    return MolecularSystem(
        molecule="test",
        geometry="equilibrium",
        basis="sto-3g",
        bondlength=0.0,
        n_qubits=n_qubits,
        fci_energy=0.0,
        hamiltonian=ising_hamiltonian(n_qubits),
    )
