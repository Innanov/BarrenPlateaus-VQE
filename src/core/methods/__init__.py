"""Barren-plateau mitigation methods (and an unmitigated baseline) behind a
single interface.

Each method is a function (system, config) -> MethodResult that runs a VQE
optimization and returns the energy history plus the final parameters and
circuit, so metrics can compute energy error and fidelity uniformly.

All methods run at the same depth for a comparison, and a method that cannot run
raises rather than injecting placeholder numbers.

Every mitigation method generalizes a method from qubap [1]. See each method
module for its own detailed reference.

Functions:
    standard: the unmitigated baseline, EfficientSU2 optimized directly against
        the full H.
    local_global: warm-start on a local cost [2], then refine on the full H.
    adiabatic: staged anneal (1-s)*H_local + s*H_global, each stage warm-started
        from the previous [3]. Generalizes qubap's continuous ramp as a superset.
    sea: the State Efficient Ansatz [4] against the full H.
    pretrained: two-stage MPS [5]. Train the diagonal MPS stage, transfer its
        parameters (prefix zero-pad) into the full stage, then refine. Both stages
        are reported.

References:
    [1] qubap, https://github.com/jgidi/quantum-barren-plateaus
    [2] Cerezo, Sone, Volkoff, Cincio, Coles, "Cost function dependent barren
        plateaus in shallow parametrized quantum circuits", Nat. Commun. 12(1),
        1791 (2021).
    [3] Harwood, Trenev, Stober, Barkoutsos, Gujarati, Mostame, Greenberg,
        "Improving the Variational Quantum Eigensolver Using Variational Adiabatic
        Quantum Computing", ACM Trans. Quantum Comput. 3(1), 1-20 (2022),
        doi:10.1145/3479197.
    [4] Liu, Liu, Zhang, Huang, Wang, "Mitigating barren plateaus of variational
        quantum eigensolvers", IEEE Trans. Quantum Eng. 5, 1-19 (2024).
    [5] Dborin, Barratt, Wimalaweera, Wright, Green, "Matrix product state
        pre-training for quantum machine learning", Quantum Sci. Technol. 7(3),
        035014 (2022).
"""

from __future__ import annotations

from .adiabatic import adiabatic
from .base import METHODS, MethodConfig, MethodResult, run_method
from .local_global import local_cost_observable, local_global
from .pretrained import pretrained
from .sea import sea
from .standard import standard

__all__ = [
    "MethodConfig",
    "MethodResult",
    "local_cost_observable",
    "METHODS",
    "run_method",
    "standard",
    "local_global",
    "adiabatic",
    "sea",
    "pretrained",
]
