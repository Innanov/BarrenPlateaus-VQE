"""Parametrized quantum circuits (ansatze) used by the VQE methods.

Each ansatz is a small class. Construct it with a qubit count (and a layer count
d), then use it inside a QNode:

    ansatz = EfficientSU2(n_qubits=4, d=2)
    ansatz.n_params            # how many trainable angles it needs
    theta = ansatz.random_params(rng)
    @qml.qnode(dev)
    def circuit(theta):
        ansatz.apply(theta)    # builds the gates on wires 0..n_qubits-1
        return qml.expval(H)

To pick an ansatz from a config string instead of importing the class, use the
build_ansatz factory, which maps a name to the matching class:

    ansatz = build_ansatz("sea", n_qubits=4, d=2)   # same as SEA(4, d=2)

All three are ported from qubap [1]. SEA follows the State Efficient Ansatz of Liu
et al. [2], and MPS follows the pretraining scheme of Dborin et al. [3].

References:
    [1] github.com/jgidi/quantum-barren-plateaus
    [2] Liu et al., Mitigating barren plateaus of variational quantum
        eigensolvers, arXiv:2205.13539 (2022).
    [3] Dborin et al., Matrix product state pre-training for quantum machine
        learning, Quantum Sci. Technol. 7, 035014 (2022), arXiv:2106.05742.

Classes:
    EfficientSU2: the default hardware-efficient circuit. Per layer: Ry and Rz on
        every qubit, then a circular chain of CNOTs. depth layers plus a final
        rotation block. n_params = 2 * n_qubits * (depth + 1).
    SEA: State Efficient Ansatz. Splits the qubits into halves A and B and runs,
        in order: (1) an Ry + CZ layer on A, (2) a CNOT ladder linking each A[i]
        to B[i], (3) an Ry + CZ layer on A and another on B. The three layers
        share the same gate pattern but have independent parameters. Within a
        layer, the Ry + CZ block repeats d times (d is a small integer, default 1).
        Needs an even qubit count.
    MPS: builds either of the two circuits the pretrained method uses, chosen by
        the diagonal flag. The gates form a brick-wall circuit, where "diagonal"
        names the main-diagonal staircase that carries the compiled MPS.
        diagonal=True is the cheap pretraining circuit (that staircase only).
        diagonal=False is the fuller circuit that adds the off-diagonal blocks
        and warm-starts from the pretrained parameters.
        See methods.pretrained.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class Ansatz:
    """Base class every ansatz inherits from.

    Attributes:
        n_qubits: Number of qubits the ansatz acts on.
        n_params: Number of trainable parameters apply(params) expects.
    """

    n_qubits: int
    n_params: int

    def apply(self, params) -> None:  # pragma: no cover - interface
        """Build the ansatz gates for the given parameter vector (subclass provides)."""
        raise NotImplementedError

    def random_params(self, rng: np.random.Generator | None = None, scale: float = 2 * np.pi):
        """Draw a random parameter vector, uniform in [0, scale).

        Returns a pennylane.numpy array of length n_params with
        requires_grad=True (autodiff-aware, so gradients flow into an
        optimizer). A fresh default generator is used when rng is None.
        """
        rng = np.random.default_rng() if rng is None else rng
        return pnp.array(rng.uniform(0.0, scale, size=self.n_params), requires_grad=True)

    @staticmethod
    def _block_n_params(spec) -> int:
        """Total angles a block spec consumes (sum of each gate's angle count).

        A block spec is a list of (gate, n_angles, wire_indices) tuples.
        """
        return sum(n for _gate, n, _wires in spec)

    @staticmethod
    def _apply_block(spec, params, block_wires) -> None:
        """Apply a block spec to the given wires, slicing params gate by gate.

        See _block_n_params for the spec format.
        """
        i = 0
        for gate, n, wire_idx in spec:
            wires = [block_wires[k] for k in wire_idx]
            gate(*params[i : i + n], wires=wires if len(wires) > 1 else wires[0])
            i += n


class EfficientSU2(Ansatz):
    """Hardware-efficient ansatz (Ry, Rz + circular CNOT).

    Per layer, applies Ry+Rz on every qubit followed by a circular CNOT chain
    (0->1->...->(n-1)->0). A final rotation block closes the circuit. Total
    parameters: 2 * n_qubits * (d + 1).

    Attributes:
        n_qubits: Number of qubits.
        d: Number of entangled rotation layers.
        n_params: Number of trainable parameters (set in __init__).
    """

    def __init__(self, n_qubits: int, d: int = 1):
        self.n_qubits = n_qubits
        self.d = d
        # d entangled blocks + 1 final block, 2 rotations (Ry, Rz) per qubit.
        self.n_params = 2 * self.n_qubits * (self.d + 1)

    def apply(self, params) -> None:
        p = pnp.reshape(params, (self.d + 1, self.n_qubits, 2))
        for layer in range(self.d):
            for q in range(self.n_qubits):
                qml.RY(p[layer, q, 0], wires=q)
                qml.RZ(p[layer, q, 1], wires=q)
            self._entangle()
        # final rotation block (no entangler after it)
        for q in range(self.n_qubits):
            qml.RY(p[self.d, q, 0], wires=q)
            qml.RZ(p[self.d, q, 1], wires=q)

    def _entangle(self) -> None:
        if self.n_qubits < 2:
            return
        for q in range(self.n_qubits):
            qml.CNOT(wires=[q, (q + 1) % self.n_qubits])


class MPS(Ansatz):
    """MPS-pretraining ansatz with a warm-up stage and a full stage.

    A matrix product state trained classically is a chain of two-qubit unitary
    matrices. KAK-decomposing each matrix turns it into gates (a single-qubit
    rotation on each qubit, the three Ising interactions Rxx, Ryy, Rzz, then a
    single-qubit rotation on each qubit), and those gates land on the main diagonal
    of a brick-wall circuit. That is where the name comes from: the "diagonal"
    blocks are the ones carrying the compiled state, and any extra blocks sit on
    the diagonals beside them. This class builds that circuit directly with the
    same gates, leaving their angles trainable.

    The diagonal flag picks the stage:

    - diagonal=True (warm-up): the diagonal staircase alone, on neighbouring qubits
      (0,1),(1,2),...,(n-2,n-1), the compiled MPS.
    - diagonal=False (full): everything the warm-up has, plus off-diagonal blocks
      interleaved around the staircase. Each off-diagonal block acts on a pair of
      neighbouring qubits: an Ry on each of the two, then a controlled-X between
      them.

    The two stages are separate circuits (separate MPS instances) used in
    sequence:

    1. Optimize the warm-up from random parameters (uniform in [0, 2*pi)).
    2. Start the full stage from those trained values, placed in its first
       parameters (the shared staircase), with the rest set to zero.

    Since zero makes an off-diagonal block the identity, the full stage begins
    exactly where the warm-up ended, then trains all its parameters.

    Attributes:
        n_qubits: Number of qubits.
        d: Number of layers (staircase passes).
        diagonal: True builds the warm-up stage, False the full stage.
        n_params: Number of trainable parameters (set in __init__).
    """

    @staticmethod
    def _diagonal_block():
        """A 2-local block on one neighbouring pair, spanning every two-qubit
        unitary up to global phase (all of SU(4), 15 real degrees of freedom).

        This is the form the KAK decomposition gives: any two-qubit unitary equals
        a single-qubit rotation on each qubit, the three Ising interactions Rxx,
        Ryy, Rzz, then a single-qubit rotation on each qubit. That is 4 * 3 + 3 =
        15 angles, matching the 15 degrees of freedom, so nothing is redundant.
        """
        return [
            (qml.U3, 3, (0,)),
            (qml.U3, 3, (1,)),
            (qml.IsingXX, 1, (0, 1)),
            (qml.IsingYY, 1, (0, 1)),
            (qml.IsingZZ, 1, (0, 1)),
            (qml.U3, 3, (0,)),
            (qml.U3, 3, (1,)),
        ]

    @staticmethod
    def _offdiagonal_block():
        """A 2-local entangler on one neighbouring pair: an Ry on each qubit,
        then a CRX between them (3 angles)."""
        return [
            (qml.RY, 1, (0,)),
            (qml.RY, 1, (1,)),
            (qml.CRX, 1, (1, 0)),
        ]

    @staticmethod
    def _offdiag_count(n_qubits: int) -> int:
        """Number of off-diagonal blocks in the full ansatz.

        Counts the off-diagonal sweeps for one staircase pass: the below- and
        above-diagonal sweeps per step, plus the final sweep.
        """
        count = 0
        for i in range(n_qubits - 1):
            count += len(range(i % 2, i, 2))  # below diagonal
            count += len(range(i + 2, n_qubits - 1, 2))  # above diagonal
        count += len(range(1, n_qubits - 1, 2))  # final sweep
        return count

    def __init__(self, n_qubits: int, d: int = 1, diagonal: bool = True):
        self.n_qubits = n_qubits
        self.d = d
        self.diagonal = diagonal

        self._diag_block = self._diagonal_block()
        self._offdiag_block = self._offdiagonal_block()
        self._n_blocks = max(self.n_qubits - 1, 0)
        self._diag_block_params = self._block_n_params(self._diag_block)
        self._offdiag_block_params = self._block_n_params(self._offdiag_block)
        diag = self.d * self._n_blocks * self._diag_block_params
        offdiag = (
            0
            if self.diagonal
            else (self.d * self._offdiag_count(self.n_qubits) * self._offdiag_block_params)
        )
        self._diag_total = diag
        self.n_params = diag + offdiag

    def apply(self, params) -> None:
        # Diagonal-block params first (prefix), then off-diagonal-block params.
        dbp = self._diag_block_params
        obp = self._offdiag_block_params
        diag = params[: self._diag_total]
        offd = params[self._diag_total :]
        dp = pnp.reshape(diag, (self.d, self._n_blocks, dbp))
        oi = 0  # off-diagonal param cursor

        def offdiag_block(a, b):
            nonlocal oi
            self._apply_block(self._offdiag_block, offd[oi : oi + obp], (a, b))
            oi += obp

        for layer in range(self.d):
            for i in range(self._n_blocks):
                if not self.diagonal:
                    for j in range(i % 2, i, 2):
                        offdiag_block(j, j + 1)
                self._apply_block(self._diag_block, dp[layer, i], (i, i + 1))
                if not self.diagonal:
                    for j in range(i + 2, self.n_qubits - 1, 2):
                        offdiag_block(j, j + 1)
            if not self.diagonal:
                for j in range(1, self.n_qubits - 1, 2):
                    offdiag_block(j, j + 1)


class SEA(Ansatz):
    """State Efficient Ansatz.

    Splits the register into system A (first n/2 qubits) and system B (last n/2),
    then runs in order:

        1. an Ry + CZ layer on A,
        2. a CNOT ladder linking each A[i] to B[i],
        3. an Ry + CZ layer on A and another on B.

    Each layer starts with an Ry on every qubit, then repeats the following block
    d times: CZ on the even-indexed neighbour pairs, Ry on every qubit, then CZ on
    the odd-indexed pairs with a Ry on each of those two qubits. The three layers
    share this gate pattern but have independent parameters.

    Attributes:
        n_qubits: Number of qubits. Must be even.
        d: Number of Ry + CZ repetitions inside each layer.
        half: Qubits per subsystem (n_qubits // 2, set in __init__).
        n_params: Number of trainable parameters (set in __init__).

    Raises:
        ValueError: If n_qubits is odd (raised on construction).
    """

    def __init__(self, n_qubits: int, d: int = 1):
        self.n_qubits = n_qubits
        self.d = d
        if self.n_qubits % 2 != 0:
            raise ValueError(f"SEA requires an even number of qubits, got {self.n_qubits}.")
        self.half = self.n_qubits // 2
        self._scl_params = self._scl_n_params(self.half, self.d)
        self.n_params = 3 * self._scl_params

    @staticmethod
    def _scl_n_params(half: int, d: int) -> int:
        """Return one Ry + CZ layer's parameter count.

        Args:
            half: Number of qubits the layer acts on (half the register).
            d: Number of repetitions of the Ry + CZ body.
        """
        return (d + 1) * half + d * max(half - 2, 0)

    def _scl(self, params, wires) -> None:
        """Apply one Ry + CZ layer to the given wires.

        params is a flat vector of length self._scl_n_params(len(wires), d). wires is
        the subsystem it acts on.
        """
        qubits = len(wires)
        par = -(qubits % 2)  # parity: -1 for odd, 0 for even
        idx = 0
        for i in range(qubits):
            qml.RY(params[idx], wires=wires[i])
            idx += 1
        for _ in range(self.d):
            for i in range(0, qubits + par, 2):
                if i + 1 < qubits:
                    qml.CZ(wires=[wires[i], wires[i + 1]])
            for i in range(qubits):
                qml.RY(params[idx], wires=wires[i])
                idx += 1
            # offset CZ chain with paired Ry (the "1..qubits-1 step 2" body)
            for i in range(1, qubits - 1, 2):
                if i + 1 < qubits:
                    qml.CZ(wires=[wires[i], wires[i + 1]])
                    qml.RY(params[idx], wires=wires[i])
                    qml.RY(params[idx], wires=wires[i + 1])
                    idx += 1

    def apply(self, params) -> None:
        s = self._scl_params
        p1 = params[0:s]
        p2 = params[s : 2 * s]
        p3 = params[2 * s : 3 * s]
        wires_A = list(range(self.half))
        wires_B = list(range(self.half, self.n_qubits))

        self._scl(p1, wires_A)  # U1 on A
        for i in range(self.half):  # CNOT ladder A[i] -> B[i]
            qml.CNOT(wires=[wires_A[i], wires_B[i]])
        self._scl(p2, wires_A)  # U2 on A
        self._scl(p3, wires_B)  # U3 on B


def build_ansatz(name: str, n_qubits: int, d: int = 1) -> Ansatz:
    """Construct an ansatz by name.

    Args:
        name: One of 'efficient_su2', 'mps' or 'sea' (aliases 'su2' / 'standard'
            also map to EfficientSU2).
        n_qubits: Number of qubits in the register.
        d: Number of layers.

    Raises:
        ValueError: If name is not a recognized ansatz.
    """
    name = name.lower()
    if name in ("efficient_su2", "efficientsu2", "su2", "standard"):
        return EfficientSU2(n_qubits=n_qubits, d=d)
    if name == "mps":
        return MPS(n_qubits=n_qubits, d=d)
    if name == "sea":
        return SEA(n_qubits=n_qubits, d=d)
    raise ValueError(f"Unknown ansatz '{name}'.")
