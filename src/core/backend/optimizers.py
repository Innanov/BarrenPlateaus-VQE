"""Gradient optimizers behind a shared API.

Each optimizer is step-driven: the caller loops and asks for one update at a time
via step(cost, params), which returns the updated parameters and the cost before
the step.

The cost must be a QNode (the circuit itself, not just a scalar-valued function),
because QNG reads the circuit to compute its metric tensor. Adam and QNSPSA also
accept a QNode, so passing one works for all three.

All three wrap PennyLane optimizer classes.

Classes:
    Adam: adaptive rate plus momentum.
    QNG: quantum natural gradient via the Fubini-Study metric.
    QNSPSA: stochastic estimate of the natural gradient.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class Optimizer:
    """Common optimizer interface."""

    def reset(self, params):
        """Reset internal state and return the initial parameters (autodiff-ready)."""
        return pnp.array(np.asarray(params, dtype=float), requires_grad=True)

    def step(self, cost, params):  # pragma: no cover - interface
        """Take one optimization step.

        Args:
            cost: A QNode mapping a parameter vector to a scalar energy.
            params: Current parameter vector.

        Returns:
            A (new_params, energy_before_step) tuple.
        """
        raise NotImplementedError


class Adam(Optimizer):
    """Adaptive-rate optimizer with momentum (PennyLane AdamOptimizer).

    Attributes:
        stepsize: Adam learning rate.
        beta1: First-moment decay.
        beta2: Second-moment decay.
        eps: Numerical stabilizer.
    """

    def __init__(self, stepsize=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._opt = None

    def reset(self, params):
        self._opt = qml.AdamOptimizer(
            stepsize=self.stepsize, beta1=self.beta1, beta2=self.beta2, eps=self.eps
        )
        return pnp.array(np.asarray(params, dtype=float), requires_grad=True)

    def step(self, cost, params):
        if self._opt is None:
            self.reset(params)
        new_params, energy_before = self._opt.step_and_cost(cost, params)
        return new_params, float(energy_before)


class QNG(Optimizer):
    """Quantum natural gradient (PennyLane QNGOptimizer).

    Rescales the gradient by the inverse Fubini-Study metric tensor, so steps
    follow the geometry of the state space rather than the raw parameter space.
    The metric tensor makes each step more expensive than Adam's.

    Attributes:
        stepsize: Learning rate.
        approx: Metric-tensor approximation ("block-diag" or "diag").
        lam: Tikhonov regularization added to the metric before inversion. A
            nonzero default keeps steps bounded when the Fubini-Study metric is
            near-singular (common in flat/barren regions), which otherwise causes
            occasional huge natural-gradient steps.
    """

    def __init__(self, stepsize=0.1, approx="block-diag", lam=1e-2):
        self.stepsize = stepsize
        self.approx = approx
        self.lam = lam
        self._opt = None

    def reset(self, params):
        self._opt = qml.QNGOptimizer(stepsize=self.stepsize, approx=self.approx, lam=self.lam)
        return pnp.array(np.asarray(params, dtype=float), requires_grad=True)

    def step(self, cost, params):
        if self._opt is None:
            self.reset(params)
        new_params, energy_before = self._opt.step_and_cost(cost, params)
        return new_params, float(energy_before)


class QNSPSA(Optimizer):
    """Quantum natural SPSA (PennyLane QNSPSAOptimizer).

    Estimates the natural gradient stochastically: the gradient by an SPSA
    finite-difference and the metric tensor by a second random perturbation, so
    the per-step cost is a small constant instead of scaling with n_params.

    Attributes:
        stepsize: Learning rate.
        regularization: Added to the estimated metric before inversion.
        finite_diff_step: Perturbation size for the SPSA estimates.
        resamplings: Number of estimates averaged per step. A single estimate
            (resamplings=1) is too noisy to converge at a fixed iteration budget,
            so the default averages several, which cuts the variance enough to
            reach the ground state (H2 error ~0.5 to ~0.003 across the methods).
        seed: Optional seed for the perturbation RNG (reproducibility).
    """

    def __init__(
        self,
        stepsize=0.1,
        regularization=1e-3,
        finite_diff_step=1e-2,
        resamplings=8,
        seed=None,
    ):
        self.stepsize = stepsize
        self.regularization = regularization
        self.finite_diff_step = finite_diff_step
        self.resamplings = resamplings
        self.seed = seed
        self._opt = None

    def reset(self, params):
        self._opt = qml.QNSPSAOptimizer(
            stepsize=self.stepsize,
            regularization=self.regularization,
            finite_diff_step=self.finite_diff_step,
            resamplings=self.resamplings,
            seed=self.seed,
        )
        return pnp.array(np.asarray(params, dtype=float), requires_grad=True)

    def step(self, cost, params):
        if self._opt is None:
            self.reset(params)
        new_params, energy_before = self._opt.step_and_cost(cost, params)
        return new_params, float(energy_before)


def build_optimizer(name: str, seed: int | None = None, **kwargs) -> Optimizer:
    """Construct an optimizer by name.

    Args:
        name: 'adam', 'qng', or 'qnspsa'.
        seed: RNG seed for QNSPSA's perturbations. Ignored by the deterministic
            Adam and QNG.
        **kwargs: Passed through to the optimizer constructor.

    Raises:
        ValueError: If name is not recognized.
    """
    name = name.lower()
    if name == "adam":
        return Adam(**kwargs)
    if name == "qng":
        return QNG(**kwargs)
    if name == "qnspsa":
        return QNSPSA(seed=seed, **kwargs)
    raise ValueError(f"Unknown optimizer '{name}'.")
