"""Shared fixtures for the method tests: small hand-made systems to run on.

Defined once here (pytest auto-loads conftest for every test in this directory)
so the per-file tests can inject them without duplicating the construction. Both
wrap ising_system, which builds a MolecularSystem in code instead of loading a
real molecule from the data/ cache, so the tests are self-contained and fast.
"""

import pytest

from src.utils.helpers import ising_system


@pytest.fixture
def two_qubit_system():
    """A small 2-qubit MolecularSystem (transverse-field Ising)."""
    return ising_system(2)


@pytest.fixture
def four_qubit_system():
    """A small 4-qubit MolecularSystem (Z-Z ladder), even n so SEA is valid."""
    return ising_system(4)
