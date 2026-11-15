"""Tests for loading cached Hamiltonians, including tapered ones.

Each test writes a small JSON file in the format of scripts/fetch_hamiltonians.py
into a temporary data directory, so no real molecule cache is needed.
"""

import json

import numpy as np
import pytest

from src.core.backend import hamiltonians


def _write(tmp_path, name, **extra):
    """Write a 2-qubit Hamiltonian -Z0 - Z1 as data/<name>_equilibrium.json."""
    raw = {
        "molecule": name,
        "geometry": "equilibrium",
        "basis": "STO-3G",
        "bondlength": 1.0,
        "n_qubits": 2,
        "electrons": 2,
        "fci_energy": -2.0,
        "pauli_terms": [["ZI", -1.0], ["IZ", -1.0]],
        "gs_dets": None,
        "gs_coeffs": None,
        **extra,
    }
    with open(tmp_path / f"{name}_equilibrium.json", "w", encoding="utf-8") as fh:
        json.dump(raw, fh)


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the loader at an empty temporary data directory."""
    monkeypatch.setattr(hamiltonians, "DATA_DIR", str(tmp_path))
    return tmp_path


def test_hf_occupation_fills_first_electrons_without_stored_state(data_dir):
    """Without hf_state the HF occupation is the first `electrons` qubits."""
    _write(data_dir, "Plain")
    system = hamiltonians.load_hamiltonian("Plain")
    np.testing.assert_array_equal(system.hf_occupation, [1, 1])


def test_hf_occupation_uses_stored_tapered_state(data_dir):
    """A stored hf_state (tapered Hamiltonian) overrides the electron count."""
    _write(data_dir, "Plain_tapered", hf_state=[0, 1])
    system = hamiltonians.load_hamiltonian("Plain_tapered")
    np.testing.assert_array_equal(system.hf_occupation, [0, 1])
    np.testing.assert_array_equal(system.hf_state, [0, 1, 0, 0])


def test_ground_state_from_eigensolve_without_expansion(data_dir):
    """Without gs_dets the ground state comes from the sparse eigensolve."""
    _write(data_dir, "Plain_tapered", hf_state=[0, 1])
    system = hamiltonians.load_hamiltonian("Plain_tapered")
    assert system.ground_energy == pytest.approx(-2.0)
    assert abs(system.ground_state[0]) == pytest.approx(1.0)
