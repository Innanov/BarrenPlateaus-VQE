"""Matplotlib figures for the analysis (convergence, gradient scaling, landscapes)."""

from __future__ import annotations

from .plots import (
    plot_convergence,
    plot_convergence_grid,
    plot_convergence_optimizers,
    plot_gradient_scaling,
    plot_landscape_contour,
    plot_landscape_grid,
    plot_landscape_surface,
)

__all__ = [
    "plot_convergence",
    "plot_convergence_grid",
    "plot_convergence_optimizers",
    "plot_gradient_scaling",
    "plot_landscape_contour",
    "plot_landscape_grid",
    "plot_landscape_surface",
]
