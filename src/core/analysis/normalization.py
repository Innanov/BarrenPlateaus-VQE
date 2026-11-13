"""Pauli weight of a cost, which puts gradient variances on one scale.

The gradient variance of a cost C = sum_P c_P P (Pauli strings P) at random
parameters is proportional to the Pauli weight W = sum over P != I of c_P**2,
the cost's squared size without its constant (identity) part. For a circuit
random enough to be a 2-design, Var[dC] = W * f(circuit, n_qubits), where f does
not depend on the cost. Dividing by W therefore leaves only how trainable the
circuit is, so curves of different costs and molecules become comparable:

- local_global's cost is 1/2 + H_0 / (2n) and adiabatic's first stage is H_0,
  so their raw variances differ by exactly (2n)**2, and their weights by the
  same factor (1/(4n) against n).
- Each molecule's Hamiltonian has its own weight, so only the normalized flat
  levels show how the plateau deepens with qubit count.

Rescaling a cost by a multiplies its gradient variance and its weight by a**2,
so the normalized variance equals the variance of C / sqrt(W). The costs
themselves are left as the methods define them, and with Adam a uniform scale
does not change the training steps anyway. The weight depends only on the
Hamiltonian, so it is computed from the molecule whenever runs are read, and
saved runs never need recomputing.
"""

from __future__ import annotations

from functools import cache

from ..backend import load_hamiltonian
from .gradient_scaling import cost_observable


def pauli_weight(observable) -> float:
    """Return sum over non-identity Pauli strings P of |c_P|**2."""
    return float(
        sum(abs(c) ** 2 for word, c in observable.pauli_rep.items() if len(word))
    )


@cache
def cost_pauli_weight(
    molecule: str, cost: str = "energy", geometry: str = "equilibrium"
) -> float:
    """Return the Pauli weight of a molecule's cost (see cost_observable)."""
    system = load_hamiltonian(molecule, geometry)
    return pauli_weight(cost_observable(system, cost))
