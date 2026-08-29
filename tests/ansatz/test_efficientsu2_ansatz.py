"""Unit tests for the EfficientSU2 ansatz.

Pins the hardware-efficient construction: per layer an Ry and Rz on every qubit
followed by a circular CNOT chain, plus a final rotation block with no trailing
entangler. Parameter count is 2 * n_qubits * (d + 1).

Run with: pytest tests/ansatz/test_efficientsu2_ansatz.py
"""

import numpy as np
import pytest

from src.core.ansatze import EfficientSU2
from src.utils.helpers import statevector, tape_ops


def test_param_count_formula():
    # 2 rotations (Ry, Rz) per qubit, over d entangled blocks plus 1 final.
    for n in (2, 4, 6):
        for d in (1, 2, 5):
            assert EfficientSU2(n, d=d).n_params == 2 * n * (d + 1)


def test_gate_counts():
    """n=4, d=1: Ry and Rz on each qubit for (d+1)=2 rotation blocks, plus one
    circular CNOT chain of n gates after the single entangled block."""
    m = EfficientSU2(4, d=1)
    names = [op.name for op in tape_ops(m, np.arange(m.n_params, dtype=float))]
    assert names.count("RY") == 4 * (1 + 1)
    assert names.count("RZ") == 4 * (1 + 1)
    assert names.count("CNOT") == 4  # one circular chain, d=1


def test_entangler_is_circular():
    # The CNOT chain wraps 0->1->...->(n-1)->0.
    m = EfficientSU2(4, d=1)
    ops = tape_ops(m, np.zeros(m.n_params))
    wires = [tuple(op.wires) for op in ops if op.name == "CNOT"]
    assert wires == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_final_block_has_no_trailing_entangler():
    # The last operations are the final rotation block (Ry, Rz), not a CNOT.
    m = EfficientSU2(4, d=2)
    ops = tape_ops(m, np.zeros(m.n_params))
    assert ops[-1].name != "CNOT"
    assert sum(1 for op in ops if op.name == "CNOT") == 2 * 4  # d circular chains


def test_single_qubit_has_no_entangler():
    # With one qubit there is no CNOT chain (n < 2).
    m = EfficientSU2(1, d=3)
    ops = tape_ops(m, np.zeros(m.n_params))
    assert sum(1 for op in ops if op.name == "CNOT") == 0


def test_consumes_all_params():
    for n in (2, 4, 6):
        for d in (1, 3):
            m = EfficientSU2(n, d=d)
            tape_ops(m, np.arange(m.n_params, dtype=float))  # must not raise


def test_state_is_unit_norm():
    for n in (2, 4):
        state = statevector(EfficientSU2(n, d=2), n)
        assert float(np.abs(np.vdot(state, state))) == pytest.approx(1.0, abs=1e-6)
