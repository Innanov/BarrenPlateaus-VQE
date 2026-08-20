"""Small shared helpers used across the project."""

import numpy as np
import pennylane as qml


def tape_ops(ansatz, params):
    """Record the gate sequence that ansatz.apply(params) produces."""
    with qml.tape.QuantumTape() as tape:
        ansatz.apply(params)
    return list(tape.operations)


def statevector(ansatz, n_qubits, seed=0):
    """Run the ansatz on random params and return the output statevector."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(p):
        ansatz.apply(p)
        return qml.state()

    return circuit(ansatz.random_params(np.random.default_rng(seed)))
