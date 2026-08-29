"""Unit tests for the SEA (State Efficient Ansatz).

Pins the implementation against qubap's SCL construction: each layer is an initial
Ry on every qubit, then d repeats of CZ on even pairs, Ry on every qubit, and an
offset CZ with paired Ry. The full ansatz is three such layers (U1 on A, a CNOT
ladder A[i]->B[i], U2 on A, U3 on B), and it requires an even qubit count. The
layer uses only Ry and CZ (no Rz).

Run with: pytest tests/ansatz/test_sea_ansatz.py
"""

import numpy as np
import pennylane as qml
import pytest

from src.core.ansatze import SEA
from src.utils.helpers import statevector, tape_ops


def test_requires_even_qubits():
    for odd in (3, 5, 7):
        with pytest.raises(ValueError):
            SEA(odd)


def test_even_qubits_ok():
    for even in (2, 4, 6):
        assert SEA(even).half == even // 2


def test_total_is_three_layers():
    # The ansatz is three SCL layers with independent parameters.
    for n in (2, 4, 6):
        for d in (1, 2):
            assert SEA(n, d=d).n_params == 3 * SEA._scl_n_params(n // 2, d)


def test_scl_formula():
    # (d + 1) * half + d * max(half - 2, 0), matching the SCL layout.
    assert SEA._scl_n_params(2, 1) == 4
    assert SEA._scl_n_params(3, 1) == 7
    assert SEA._scl_n_params(2, 2) == 6


def test_uses_ry_and_cz_only_plus_ladder():
    # SCL layers are Ry + CZ (no Rz); the only other gates are the CNOT ladder.
    m = SEA(4, d=1)
    names = {op.name for op in tape_ops(m, np.arange(m.n_params, dtype=float))}
    assert names == {"RY", "CZ", "CNOT"}


def test_cnot_ladder_links_a_to_b():
    # cnot_layer Full_connect: CNOT A[i] -> B[i] for i in range(half).
    m = SEA(4, d=1)
    ops = tape_ops(m, np.zeros(m.n_params))
    cnots = [tuple(op.wires) for op in ops if op.name == "CNOT"]
    assert cnots == [(0, 2), (1, 3)]


def test_scl_matches_reference_gate_sequence():
    """The per-layer gate list must equal the reference SCL exactly, encoding
    the known sequence for half=2, d=1: Ry,Ry, CZ, Ry,Ry."""
    m = SEA(4, d=1)
    with qml.tape.QuantumTape() as tape:
        m._scl(np.arange(SEA._scl_n_params(2, 1), dtype=float), [0, 1])
    seq = [(op.name, tuple(op.wires)) for op in tape.operations]
    assert seq == [("RY", (0,)), ("RY", (1,)), ("CZ", (0, 1)), ("RY", (0,)), ("RY", (1,))]


def test_three_scl_layers_present():
    # U1 on A, U2 on A, U3 on B: three CZ-containing layers around one ladder.
    # Each SCL layer (half=2, d=1) has exactly one CZ, so three layers -> 3 CZ.
    m = SEA(4, d=1)
    ops = tape_ops(m, np.zeros(m.n_params))
    assert sum(1 for op in ops if op.name == "CZ") == 3


def test_consumes_all_params():
    for n in (2, 4, 6):
        for d in (1, 2):
            m = SEA(n, d=d)
            tape_ops(m, np.arange(m.n_params, dtype=float))  # must not raise


def test_state_is_unit_norm():
    for n in (2, 4):
        state = statevector(SEA(n, d=1), n)
        assert float(np.abs(np.vdot(state, state))) == pytest.approx(1.0, abs=1e-6)
