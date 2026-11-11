"""Hamiltonian loading from the cached PennyLane JSON datasets.

The JSON cache is produced by scripts/fetch_hamiltonians.py into the gitignored
data/ directory (run it once before loading). Each file stores a qubit
Hamiltonian as [pauli_string, coefficient] pairs, the exact FCI energy, and the
FCI ground state as a determinant expansion (from PennyLane's
initial_state_dets/coeffs). This module rebuilds a pennylane.Hamiltonian and
exposes the exact ground-state energy and eigenvector (needed for a real state
fidelity).

Both the FCI energy and the ground-state statevector come straight from the
dataset. The statevector is reconstructed from the cached determinant expansion.
A sparse Lanczos solve (scipy.sparse.linalg.eigsh, never a dense 2**n x 2**n
diagonalization) is only a fallback for the rare case where a dataset ships no
ground-state expansion.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp  # noqa: F401  (re-exported for callers)
from scipy.sparse.linalg import eigsh

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", "data"))

_PAULI = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


@dataclass
class MolecularSystem:
    """One molecule/geometry: its qubit Hamiltonian, FCI energy, and ground state."""

    molecule: str
    geometry: str
    basis: str
    bondlength: float
    n_qubits: int
    fci_energy: float
    hamiltonian: qml.Hamiltonian
    electrons: int | None = None
    _ground_state: np.ndarray | None = None
    _ground_energy: float | None = None

    @property
    def ground_state(self) -> np.ndarray:
        """Exact lowest eigenvector as a length 2**n_qubits statevector."""
        if self._ground_state is None:
            self._solve_ground()
        return self._ground_state

    @property
    def hf_occupation(self) -> np.ndarray:
        """Hartree-Fock occupation vector: 1 on the lowest `electrons` qubits.

        Under Jordan-Wigner the HF determinant occupies the lowest-energy
        spin-orbitals, which are qubits 0..electrons-1. This is the easy-to-prepare
        reference state (a single product state) used to anchor the adiabatic and
        local_global warm-starts.
        """
        if self.electrons is None:
            raise ValueError(
                f"No electron count cached for {self.molecule}; re-run "
                "scripts/fetch_hamiltonians.py to add the 'electrons' field."
            )
        occ = np.zeros(self.n_qubits, dtype=int)
        occ[: self.electrons] = 1
        return occ

    @property
    def hf_state(self) -> np.ndarray:
        """Hartree-Fock reference as a length 2**n_qubits statevector.

        The computational-basis product state |1...1 0...0> with the lowest
        `electrons` qubits occupied. Being a single product state it is exactly
        preparable (a layer of X gates), so it is a cheap
        anchor for the warm-start and adiabatic schedules.
        Qubit 0 is the most significant bit, matching the
        ground_state ordering.
        """
        occ = self.hf_occupation
        idx = 0
        for bit in occ:
            idx = (idx << 1) | int(bit)
        vec = np.zeros(2**self.n_qubits, dtype=float)
        vec[idx] = 1.0
        return vec

    @property
    def ground_energy(self) -> float:
        """Exact lowest eigenvalue (verify_against_fci checks it vs fci_energy)."""
        if self._ground_energy is None:
            self._solve_ground()
        return self._ground_energy

    def _solve_ground(self) -> None:
        energy, state = _exact_ground_state(self.hamiltonian, self.n_qubits)
        self._ground_energy = energy
        self._ground_state = state


def json_path(molecule: str, geometry: str = "equilibrium") -> str:
    return os.path.join(DATA_DIR, f"{molecule}_{geometry}.json")


def _term_to_observable(pauli_string: str):
    """Convert a Pauli string to a PennyLane tensor observable.

    Identity factors are omitted from the product. An all-identity string maps
    to Identity(0). pauli_string is one character per qubit over
    {'I', 'X', 'Y', 'Z'}.
    """
    factors = [_PAULI[p](w) for w, p in enumerate(pauli_string) if p != "I"]
    if not factors:
        return qml.Identity(0)
    obs = factors[0]
    for f in factors[1:]:
        obs = obs @ f
    return obs


def load_hamiltonian(molecule: str, geometry: str = "equilibrium") -> MolecularSystem:
    """Load a cached molecule and build its PennyLane Hamiltonian.

    Args:
        molecule: Molecule name, e.g. 'H2' or 'LiH'.
        geometry: Geometry label, one of 'equilibrium', 'stretched' or
            'compressed'.

    Returns:
        A MolecularSystem with the reconstructed Hamiltonian and the cached FCI
        energy.

    Raises:
        FileNotFoundError: If no cached JSON exists for the molecule/geometry.
    """
    path = json_path(molecule, geometry)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No cached Hamiltonian for {molecule} ({geometry}).\n"
            f"Expected: {path}\n"
            f"Run scripts/fetch_hamiltonians.py to generate it."
        )

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    coeffs = [float(c) for _, c in raw["pauli_terms"]]
    obs = [_term_to_observable(str(p)) for p, _ in raw["pauli_terms"]]
    hamiltonian = qml.Hamiltonian(coeffs, obs)

    n_qubits = int(raw["n_qubits"])
    # Exact FCI ground state from the cached determinant expansion, if present.
    cached_gs = _ground_state_from_dets(raw.get("gs_dets"), raw.get("gs_coeffs"), n_qubits)

    return MolecularSystem(
        molecule=str(raw["molecule"]),
        geometry=str(raw["geometry"]),
        basis=str(raw["basis"]),
        bondlength=float(raw["bondlength"]),
        n_qubits=n_qubits,
        fci_energy=float(raw["fci_energy"]),
        hamiltonian=hamiltonian,
        electrons=(int(raw["electrons"]) if raw.get("electrons") is not None else None),
        _ground_state=cached_gs,
    )


def _ground_state_from_dets(dets, coeffs, n_qubits):
    """Build the FCI ground-state statevector from a determinant expansion.

    The state is sum_i coeff_i |det_i> with the qubit-0-is-most-significant
    convention that matches PennyLane's qml.state ordering. Returns None if the
    expansion is missing (caller then falls back to a sparse eigensolve).

    Args:
        dets: List of occupation bitstrings, or None.
        coeffs: Matching amplitudes, or None.
        n_qubits: Number of qubits (statevector length 2**n_qubits).

    Returns:
        A unit-norm statevector of length 2**n_qubits, or None.
    """
    if not dets or not coeffs:
        return None
    vec = np.zeros(2**n_qubits, dtype=float)
    for det, c in zip(dets, coeffs, strict=False):
        idx = 0
        for bit in det:  # qubit 0 = most significant bit
            idx = (idx << 1) | int(bit)
        vec[idx] += float(c)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


def _sparse_matrix(hamiltonian: qml.Hamiltonian, n_qubits: int):
    """Return the Hamiltonian's CSR matrix on wires 0..n_qubits-1."""
    return hamiltonian.sparse_matrix(wire_order=range(n_qubits))


def _exact_ground_state(hamiltonian: qml.Hamiltonian, n_qubits: int):
    """Compute the exact ground state via a sparse Lanczos solve.

    Uses scipy.sparse.linalg.eigsh on the CSR operator so it stays tractable at
    up to 32 qubits where a dense solve would OOM.

    Returns:
        A (energy, statevector) tuple with a unit-norm statevector of length
        2**n_qubits.
    """
    mat = _sparse_matrix(hamiltonian, n_qubits)
    vals, vecs = eigsh(mat, k=1, which="SA")
    energy = float(vals[0].real)
    state = np.asarray(vecs[:, 0]).ravel()
    state = state / np.linalg.norm(state)
    return energy, state


def verify_against_fci(system: MolecularSystem, tol: float = 1e-6) -> bool:
    """Check the Hamiltonian diagonalizes to the stored FCI energy.

    Returns True if the sparse ground-state energy matches the cached FCI energy
    within tol.
    """
    return abs(system.ground_energy - system.fci_energy) < tol
