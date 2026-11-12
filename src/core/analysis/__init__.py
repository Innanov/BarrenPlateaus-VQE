"""Analysis tools for gradient scaling, metrics, and statistics over seeds."""

from .gradient_scaling import (
    COSTS,
    WARM_START_METHODS,
    GradientScalingPoint,
    cost_observable,
    gradient_scaling,
    warm_start_gradient_scaling,
)
from .metrics import (
    energy_error,
    fidelity,
    gradient_norm,
    gradient_variance,
    statevector,
)
from .seeds import aggregate_convergence, aggregate_performance, replace_methods

__all__ = [
    "gradient_scaling",
    "warm_start_gradient_scaling",
    "cost_observable",
    "COSTS",
    "WARM_START_METHODS",
    "GradientScalingPoint",
    "aggregate_convergence",
    "aggregate_performance",
    "replace_methods",
    "energy_error",
    "fidelity",
    "gradient_norm",
    "gradient_variance",
    "statevector",
]
