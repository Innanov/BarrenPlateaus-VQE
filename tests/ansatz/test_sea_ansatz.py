"""Unit tests for the SEA (State Efficient Ansatz).

Pins the implementation against the construction of Liu et al., arXiv:2205.13539:
S = (U2 (x) U3) V (U1 (x) I), with V a CNOT from each A[i] to B[i] and U1, U2, U3
alternating layered ansatze (ALT) of Ry and CZ gates, one parameter per Ry gate.
The parameter count for an even half N is the paper's Table S1, 3N + 6(N-1)d.
SEA requires an even qubit count.

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


def test_total_is_three_alts():
    for n in (2, 4, 6, 14):
        for d in (1, 2):
            assert SEA(n, d=d).n_params == 3 * SEA._alt_n_params(n // 2, d)


@pytest.mark.parametrize("half", [2, 4, 6, 8])
@pytest.mark.parametrize("d", [1, 4])
def test_param_count_matches_paper_table(half, d):
    """Table S1 of the paper gives 3N + 6(N-1)l parameters for an even half N."""
    assert SEA(2 * half, d=d).n_params == 3 * half + 6 * (half - 1) * d


def test_alt_param_count_for_odd_half():
    # half=3, d=1: 3 initial Ry, 3 Ry after the even CZ, 2 Ry on the pair (1,2).
    assert SEA._alt_n_params(3, 1) == 8
    # half=7 (BeH2): 7 + d * (7 + 6).
    assert SEA._alt_n_params(7, 4) == 7 + 4 * 13


@pytest.mark.parametrize("n", [4, 6, 8, 12, 14])
def test_one_parameter_per_ry_gate(n):
    """Every Ry gate reads its own parameter, and every parameter is used once."""
    m = SEA(n, d=2)
    ops = tape_ops(m, np.arange(m.n_params, dtype=float))
    angles = [float(op.parameters[0]) for op in ops if op.name == "RY"]
    assert sorted(angles) == list(range(m.n_params))


@pytest.mark.parametrize("n", [8, 12])
def test_every_parameter_changes_the_state(n):
    """No parameter is dead: moving any one of them changes the output state."""
    m = SEA(n, d=1)
    p = np.random.default_rng(0).uniform(0, 2 * np.pi, m.n_params)
    base = statevector(m, n, params=p)
    for k in range(m.n_params):
        q = p.copy()
        q[k] += 0.7
        overlap = abs(np.vdot(statevector(m, n, params=q), base))
        assert overlap < 1 - 1e-9, f"parameter {k} does not change the state"


def test_uses_ry_and_cz_only_plus_cnots():
    m = SEA(4, d=1)
    names = {op.name for op in tape_ops(m, np.arange(m.n_params, dtype=float))}
    assert names == {"RY", "CZ", "CNOT"}


def test_entangling_layer_links_a_to_b():
    # The paper's entangling layer: CNOT on the pairs (i, N + i).
    m = SEA(6, d=1)
    ops = tape_ops(m, np.zeros(m.n_params))
    cnots = [tuple(op.wires) for op in ops if op.name == "CNOT"]
    assert cnots == [(0, 3), (1, 4), (2, 5)]


def test_alt_matches_paper_figure():
    """One 4-qubit ALT layer as drawn in the paper's ALT figure: Ry on all, CZ on
    (0,1) and (2,3), Ry on all, CZ on (1,2), then Ry on qubits 1 and 2."""
    m = SEA(8, d=1)
    with qml.tape.QuantumTape() as tape:
        m._alt(np.arange(SEA._alt_n_params(4, 1), dtype=float), [0, 1, 2, 3])
    seq = [(op.name, tuple(op.wires), op.parameters) for op in tape.operations]
    expected = (
        [("RY", (q,), [float(q)]) for q in range(4)]
        + [("CZ", (0, 1), []), ("CZ", (2, 3), [])]
        + [("RY", (q,), [float(4 + q)]) for q in range(4)]
        + [("CZ", (1, 2), []), ("RY", (1,), [8.0]), ("RY", (2,), [9.0])]
    )
    assert seq == expected


def test_schmidt_form_without_basis_change():
    """With U2 and U3 at zero (identity), the output is sum_k lambda_k |k>_A |k>_B,
    the Schmidt form of the paper's Fig. 2, with lambda_k the amplitudes U1 makes."""
    n, half = 6, 3
    m = SEA(n, d=2)
    s = SEA._alt_n_params(half, 2)
    p = np.zeros(m.n_params)
    p[:s] = np.random.default_rng(1).uniform(0, 2 * np.pi, s)
    state = statevector(m, n, params=p).reshape(2**half, 2**half)
    assert np.allclose(state, np.diag(np.diag(state)), atol=1e-12)

    with qml.tape.QuantumTape() as tape:
        m._alt(p[:s], list(range(half)))
    u1 = qml.matrix(tape, wire_order=list(range(half)))[:, 0]
    assert np.allclose(np.diag(state), u1, atol=1e-10)


def test_consumes_all_params():
    for n in (2, 4, 6, 14):
        for d in (1, 2):
            m = SEA(n, d=d)
            tape_ops(m, np.arange(m.n_params, dtype=float))  # must not raise


def test_state_is_unit_norm():
    for n in (2, 4):
        state = statevector(SEA(n, d=1), n)
        assert float(np.abs(np.vdot(state, state))) == pytest.approx(1.0, abs=1e-6)
