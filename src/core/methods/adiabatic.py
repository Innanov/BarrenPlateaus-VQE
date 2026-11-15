"""Adiabatic VQE, following the variational adiabatic quantum computing of [2].

The method solves a sequence of VQE problems on the interpolated Hamiltonian

    H(t) = (1 - t) H_0 + t H_global,

at t_1 < t_2 < ... < t_K = 1, each warm-started from the solution of the previous
one ("bootstrapping", [2] Sec. Bootstrapping). It differs from qubap's
VQE_adiabatic [1], which ramps t every iteration from random parameters, in the
three points where [2] is specific:

1. Start in the ground state of H_0. [2] assumes "initial optimal parameters
   theta^0 in argmin f(theta, 0)" and uses an ansatz that prepares the reference
   state at theta = 0 (main.tex lines 325 to 327 and 485). Here H_0's ground state
   is the Hartree-Fock (HF) state, which EfficientSU2 prepares exactly with every
   angle 0 except Ry = pi on the occupied qubits of the final rotation layer
   (hf_parameters). That point is a stationary point of every number-conserving
   Hamiltonian (all gradients vanish there, checked on LiH: max |grad| 1e-16), so
   a small seeded offset (HF_OFFSET) is added, as qubap's own adiabatic tutorial
   starts at 0.01. The earlier version started from random parameters and spent
   its first stage looking for the HF state, which at depths past d_sat it never
   found (the first-stage cost stayed near its random value), so the anneal never
   started from the reference.
2. H_0 on the scale of the target. [2] builds H_I from the target's own identity
   and single-Z terms (main.tex lines 487 to 503), and names the HF state and its
   Hamiltonian as an alternative. Those single-Z terms do not have the HF state as
   their ground state for every molecule here (H2O, LiH and NH3 differ), so H_0
   keeps their magnitudes and takes the signs from the HF occupation
   (hf_reference_hamiltonian with weights). Its ground state is then the HF state,
   and it equals [2]'s H_I wherever that already has the HF ground state.
3. The schedule t_k = 1 - (1 - k / K)**3 for k = 1..K, K = 20 ([2] main.tex line
   525), which takes smaller steps near the target. No step is spent on H_0
   itself, since the start is already its ground state.

[2] solves each step to a tolerance. Here every step gets an equal share of the
max_iters budget, so the total matches every other method. The optimizer state
carries across steps (no reset), so a step does not restart with a full-size Adam
step that would kick the parameters out of the previous minimum.

References:
    [1] qubap, https://github.com/jgidi/quantum-barren-plateaus
    [2] Harwood, Trenev, Stober, Barkoutsos, Gujarati, Mostame, Greenberg,
        "Improving the Variational Quantum Eigensolver Using Variational Adiabatic
        Quantum Computing", ACM Trans. Quantum Comput. (2022), arXiv:2102.02875.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ..ansatze import EfficientSU2
from ..backend import MolecularSystem, build_optimizer
from .base import (
    MethodConfig,
    MethodResult,
    _fixed_hamiltonian,
    _make_cost,
    _optimize,
)

HF_OFFSET = 0.01
"""Standard deviation of the seeded offset added to the HF parameters."""


def adiabatic(system: MolecularSystem, config: MethodConfig) -> MethodResult:
    """Adiabatic VQE: bootstrap from the HF state of H_0 along t_1..t_K to H.

    Starts at hf_parameters (the HF state plus a seeded offset), then runs
    config.adiabatic_steps steps on H(t_k) = (1 - t_k) H_0 + t_k H_global with
    t_k from schedule. The max_iters iterations are spread over the steps so they
    sum to exactly max_iters (the first max_iters % K steps get one extra), and K
    is clamped to max_iters.
    """
    n = system.n_qubits
    ansatz = EfficientSU2(n, d=config.depth)
    rng = np.random.default_rng(config.seed)
    params = hf_parameters(ansatz, system.hf_occupation, rng)

    h_0 = initial_hamiltonian(system)
    h_global = system.hamiltonian
    cost_global = _make_cost(ansatz, h_global, n)

    iters = max(config.max_iters, 1)
    n_steps = min(max(int(config.adiabatic_steps or 1), 1), iters)
    base, extra = divmod(iters, n_steps)
    step_iters = [base + (1 if k < extra else 0) for k in range(n_steps)]

    optimizer = build_optimizer(
        config.optimizer, seed=config.seed, **config.optimizer_kwargs
    )
    params = optimizer.reset(params)

    history = []
    param_history = []
    for t, n_iters in zip(schedule(n_steps), step_iters, strict=True):
        cost_t = _make_cost(ansatz, _interpolate(h_0, h_global, t), n)
        params, hist, phist = _optimize(
            cost_t, params, optimizer, n_iters, reset=False
        )
        history.extend(hist)
        param_history.extend(phist)

    history_global = [
        float(cost_global(pnp.array(p, requires_grad=False)))
        for p in param_history
    ]

    return MethodResult(
        method="adiabatic",
        energy_history=history,
        final_energy=float(cost_global(params)),
        params=params,
        ansatz=ansatz,
        n_params=ansatz.n_params,
        stage_boundaries=[],
        param_history=param_history,
        energy_history_global=history_global,
    )


def schedule(n_steps: int) -> list[float]:
    """Return t_k = 1 - (1 - k / n_steps)**3 for k = 1..n_steps.

    The schedule of [2] (with n_steps = 20 there). It rises fast at first and
    slows down near t = 1, and its last value is exactly 1.
    """
    return [1.0 - (1.0 - k / n_steps) ** 3 for k in range(1, n_steps + 1)]


def hf_parameters(ansatz: EfficientSU2, occupation, rng) -> pnp.ndarray:
    """Return EfficientSU2 parameters near the HF state.

    Every angle is 0 except Ry = pi on the occupied qubits of the final rotation
    layer, which prepares the HF basis state exactly (the CNOTs act on |0...0>
    and change nothing). A normal offset of standard deviation HF_OFFSET, drawn
    from rng, moves the start off that stationary point.

    Args:
        ansatz: The EfficientSU2 ansatz.
        occupation: Length-n array of 0/1 HF occupations.
        rng: numpy Generator for the offset (seeded by the run's seed).

    Returns:
        A trainable parameter vector of length ansatz.n_params.
    """
    p = np.zeros((ansatz.d + 1, ansatz.n_qubits, 2))
    p[ansatz.d, :, 0] = np.pi * np.asarray(occupation, dtype=float)
    p += rng.normal(0.0, HF_OFFSET, p.shape)
    return pnp.array(p.ravel(), requires_grad=True)


def initial_hamiltonian(system: MolecularSystem) -> qml.Hamiltonian:
    """Return H_0: [2]'s H_I with the signs of its Z terms set by the HF state.

    Takes the identity coefficient c_0 and the single-Z coefficients c_j of the
    molecular Hamiltonian and returns c_0 I + sum_j |c_j| s_j Z_j, with s_j = +1
    on the occupied qubits and -1 on the empty ones (hf_reference_hamiltonian).
    A qubit without a single-Z term gets weight 1 so the ground state stays
    non-degenerate.
    """
    n = system.n_qubits
    weights = np.zeros(n)
    constant = 0.0
    for coeff, op in zip(*system.hamiltonian.terms(), strict=True):
        word = op.pauli_rep
        if word is None or len(word) != 1:
            continue
        ((pauli_word, factor),) = word.items()
        c = float(np.real(coeff * factor))
        if len(pauli_word) == 0:
            constant += c
        elif len(pauli_word) == 1:
            ((wire, letter),) = pauli_word.items()
            if letter == "Z":
                weights[int(wire)] += c
    weights = np.where(np.abs(weights) > 0, np.abs(weights), 1.0)
    return hf_reference_hamiltonian(
        system.hf_occupation, weights=weights, constant=constant
    )


def hf_reference_hamiltonian(
    occupation, weights=None, constant: float = 0.0
) -> qml.Hamiltonian:
    """Hamiltonian whose non-degenerate ground state is the HF basis state.

    H = constant * I + sum_j w_j sign_j Z_j, with sign_j = +1 on the occupied
    qubits and -1 on the empty ones. Since Z|1> = -|1> and Z|0> = +|0>, each term
    is lowest when qubit j is in its HF state, so the basis state |occupation> is
    the ground state for any positive weights. The default unit weights and zero
    constant give the first-stage cost of the gradient sweeps ('hf_reference' in
    gradient_scaling.py), and adiabatic uses target-scaled weights
    (initial_hamiltonian).

    Args:
        occupation: Length-n array of 0/1 HF occupations (qubit j occupied iff 1).
        weights: Positive weight per qubit, or None for all ones.
        constant: Coefficient of the identity term.

    Returns:
        A qml.Hamiltonian on n qubits.
    """
    occupation = np.asarray(occupation, dtype=int)
    if weights is None:
        weights = np.ones(len(occupation))
    coeffs = [
        w if o == 1 else -w for w, o in zip(weights, occupation, strict=True)
    ]
    obs = [qml.PauliZ(j) for j in range(len(occupation))]
    if constant:
        coeffs.append(constant)
        obs.append(qml.Identity(0))
    return _fixed_hamiltonian(coeffs, obs)


def _interpolate(
    h_local: qml.Hamiltonian, h_global: qml.Hamiltonian, s: float
) -> qml.Hamiltonian:
    """Return (1 - s) * H_local + s * H_global.

    Args:
        h_local: The initial Hamiltonian.
        h_global: The target Hamiltonian.
        s: Interpolation parameter in [0, 1].

    Returns:
        The interpolated Hamiltonian, simplified.
    """
    coeffs = list((1.0 - s) * pnp.array(h_local.coeffs)) + list(
        s * pnp.array(h_global.coeffs)
    )
    obs = list(h_local.ops) + list(h_global.ops)
    return _fixed_hamiltonian(coeffs, obs)
