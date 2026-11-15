"""Tests for the Pauli-string conversion of scripts/fetch_hamiltonians.py.

A tapered Hamiltonian keeps the labels of the wires it did not remove (for H2O
0 to 8 and 10), and the conversion uses wire labels as string positions. These
tests check that such wires are refused, and that relabelling them gives the
same Hamiltonian back.
"""

import numpy as np
import pennylane as qml
import pytest

from scripts.fetch_hamiltonians import hamiltonian_to_pauli_terms


def _rebuild(terms):
    """Rebuild a Hamiltonian from [pauli_string, coefficient] pairs."""
    ops = [
        qml.pauli.string_to_pauli_word(s) if set(s) != {"I"} else qml.Identity(0)
        for s, _ in terms
    ]
    return qml.Hamiltonian([c for _, c in terms], ops)


def test_refuses_wires_that_are_not_positions():
    """Wires 0 and 2 on 2 qubits would drop wire 2, so it raises."""
    h = qml.Hamiltonian([0.5, -1.0], [qml.Z(0) @ qml.X(2), qml.Y(2)])
    with pytest.raises(ValueError):
        hamiltonian_to_pauli_terms(h, 2, qml)


def test_relabelled_wires_keep_the_hamiltonian():
    """After mapping sorted wires to 0..n-1 the matrix is unchanged."""
    h = qml.Hamiltonian(
        [0.5, -1.0, 0.3], [qml.Z(0) @ qml.X(2), qml.Y(2), qml.Z(0)]
    )
    wire_map = {w: i for i, w in enumerate(sorted(h.wires))}
    mapped = h.map_wires(wire_map)
    terms = hamiltonian_to_pauli_terms(mapped, 2, qml)
    np.testing.assert_allclose(
        qml.matrix(_rebuild(terms), wire_order=[0, 1]),
        qml.matrix(mapped, wire_order=[0, 1]),
    )
