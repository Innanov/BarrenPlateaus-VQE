"""Unit tests for the MPS ansatz.

Pins the implementation against qubap's known values: the diagonal block has 15
parameters, the off-diagonal block has 3, the pretraining stage is a staircase of
diagonal blocks, and the full stage adds off-diagonal sweeps. The diagonal params
must be a prefix of the full param vector so the pretraining optimum transfers by
zero-pad. A separate check uses PennyLane's own two-qubit (KAK) decomposition as
an oracle to confirm the diagonal block is a fully general two-qubit unitary.

Run with: pytest tests/ansatz/test_mps_ansatz.py
"""

import numpy as np
import pennylane as qml
import pytest

from src.core.ansatze import MPS
from src.utils.helpers import statevector, tape_ops


def _diag_block_matrix(params):
    """4x4 unitary the diagonal block produces for the given 15 angles."""
    block = MPS(2)._diag_block
    with qml.tape.QuantumTape() as tape:
        MPS._apply_block(block, params, (0, 1))
    return qml.matrix(tape, wire_order=[0, 1])


def test_diagonal_block_has_15_params():
    # diagonal block: u,u (3 each), rxx, ryy, rzz, u,u (3 each) = 15
    assert MPS._block_n_params(MPS(2)._diag_block) == 15


def test_offdiag_block_has_3_params():
    # off-diagonal block: ry, ry, crx = 3
    assert MPS._block_n_params(MPS(2)._offdiag_block) == 3


def test_diagonal_block_gate_types():
    gates = [g for g, _n, _w in MPS(2)._diag_block]
    assert gates == [qml.U3, qml.U3, qml.IsingXX, qml.IsingYY, qml.IsingZZ, qml.U3, qml.U3]


def test_offdiag_block_gate_types():
    gates = [g for g, _n, _w in MPS(2)._offdiag_block]
    assert gates == [qml.RY, qml.RY, qml.CRX]


def test_diagonal_stage_is_nblocks_times_15():
    # diagonal stage: (n-1) diagonal blocks, 15 params each, times d
    for n in (4, 6, 8, 10):
        for d in (1, 2):
            assert MPS(n, d=d, diagonal=True).n_params == d * (n - 1) * 15


def test_full_stage_adds_offdiag_blocks():
    # full stage: diagonal params plus (# off-diagonal blocks) * 3
    for n in (4, 6, 8):
        m = MPS(n, d=1, diagonal=False)
        assert m.n_params == (n - 1) * 15 + MPS._offdiag_count(n) * 3


def test_full_has_more_params_than_diagonal():
    for n in (4, 6, 8):
        diag = MPS(n, d=1, diagonal=True).n_params
        full = MPS(n, d=1, diagonal=False).n_params
        assert full > diag


def test_offdiag_count_known_values():
    # Computed directly from the diagonal=False loop structure.
    for n, val in {4: 3, 6: 10, 8: 21}.items():
        assert MPS._offdiag_count(n) == val


def test_diagonal_is_block_staircase():
    """n=4, d=1 gives three diagonal blocks on (0,1),(1,2),(2,3), each a
    7-op block (U3,U3, IsingXX,IsingYY,IsingZZ, U3,U3), so 21 ops total."""
    m = MPS(4, d=1, diagonal=True)
    ops = tape_ops(m, np.arange(m.n_params, dtype=float))
    assert len(ops) == 3 * 7
    # first block acts on wires (0, 1)
    first = ops[:7]
    assert [op.name for op in first] == ["U3", "U3", "IsingXX", "IsingYY", "IsingZZ", "U3", "U3"]
    assert list(first[0].wires) == [0]
    assert list(first[1].wires) == [1]
    assert list(first[2].wires) == [0, 1]
    # blocks sweep left to right: second block on (1, 2), third on (2, 3)
    assert list(ops[7].wires) == [1]
    assert list(ops[14].wires) == [2]


def test_full_contains_offdiag_blocks():
    """The full stage must include CRX gates (the off-diagonal block's
    controlled rotation), which the diagonal stage never has."""
    names_full = [
        op.name
        for op in tape_ops(MPS(6, d=1, diagonal=False), np.zeros(MPS(6, diagonal=False).n_params))
    ]
    names_diag = [
        op.name
        for op in tape_ops(MPS(6, d=1, diagonal=True), np.zeros(MPS(6, diagonal=True).n_params))
    ]
    assert "CRX" in names_full
    assert "CRX" not in names_diag


def test_apply_consumes_all_params():
    # If apply sliced the wrong number of params, an index error would fire.
    for n in (4, 6, 8):
        for diag in (True, False):
            m = MPS(n, d=1, diagonal=diag)
            tape_ops(m, np.arange(m.n_params, dtype=float))  # must not raise


def test_diagonal_ops_match_full_prefix_when_offdiag_zero():
    """With the off-diagonal params set to zero its gates are trivial, so the
    diagonal-block ops in the full stage use the same parameters as the
    diagonal-only stage. Checks the U3/Ising ops and their parameters line up.
    """
    n = 6
    m_diag = MPS(n, d=1, diagonal=True)
    m_full = MPS(n, d=1, diagonal=False)
    diag_params = np.arange(m_diag.n_params, dtype=float) + 1.0
    # full params: same diagonal prefix, zeros for the off-diagonal blocks
    full_params = np.concatenate([diag_params, np.zeros(m_full.n_params - m_diag.n_params)])

    diag_ops = tape_ops(m_diag, diag_params)
    full_ops = tape_ops(m_full, full_params)

    # Extract only the diagonal-block gates (U3 and the Ising rotations).
    u_names = {"U3", "IsingXX", "IsingYY", "IsingZZ"}
    diag_u = [op for op in diag_ops if op.name in u_names]
    full_u = [op for op in full_ops if op.name in u_names]
    assert len(diag_u) == len(full_u)
    for a, b in zip(diag_u, full_u, strict=False):
        assert a.name == b.name
        assert list(a.wires) == list(b.wires)
        np.testing.assert_allclose(a.parameters, b.parameters)


def test_block_is_unitary():
    for seed in (0, 1, 2):
        p = np.random.default_rng(seed).uniform(0, 2 * np.pi, 15)
        U = _diag_block_matrix(p)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(4), atol=1e-8)


def test_kak_round_trip_reproduces_block():
    """Decompose the block's matrix with PennyLane's KAK, rebuild it from the
    returned gates, and check it matches up to a global phase."""
    p = np.random.default_rng(4).uniform(0, 2 * np.pi, 15)
    U = _diag_block_matrix(p)
    ops = qml.ops.two_qubit_decomposition(qml.math.array(U), wires=[0, 1])
    rebuilt = qml.matrix(qml.prod(*ops[::-1]), wire_order=[0, 1])
    phase = (rebuilt / U)[np.abs(U) > 1e-6]
    assert np.allclose(phase, phase[0], atol=1e-6)


def test_block_needs_three_cnots():
    """KAK uses three CNOTs only for a fully general two-qubit unitary, so a
    three-CNOT result confirms the block spans the whole two-qubit family."""
    p = np.random.default_rng(5).uniform(0, 2 * np.pi, 15)
    U = _diag_block_matrix(p)
    ops = qml.ops.two_qubit_decomposition(qml.math.array(U), wires=[0, 1])
    assert sum(1 for o in ops if o.name == "CNOT") == 3


def test_state_is_unit_norm():
    for n in (4, 6):
        for diag in (True, False):
            state = statevector(MPS(n, d=1, diagonal=diag), n)
            assert float(np.abs(np.vdot(state, state))) == pytest.approx(1.0, abs=1e-6)


def test_even_and_odd_qubit_counts_supported():
    # MPS has no even-qubit restriction (unlike SEA).
    for n in (3, 4, 5):
        assert MPS(n, d=1, diagonal=True).n_params == (n - 1) * 15
