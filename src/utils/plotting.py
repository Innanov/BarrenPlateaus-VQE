"""Matplotlib figures: convergence and gradient scaling.

Uses the Agg backend (headless) and an IEEE-conference style (serif fonts,
compact single-column sizing) so figures drop into a two-column paper. A
convergence curve that carries an energy_std (the mean over seeds) is drawn with
a shaded band of mean +- std.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

# IEEE-conference style, applied per-figure via plt.rc_context (never global).
IEEE_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.4,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Single-column IEEE figure width (~3.3 in) and a comfortable height.
_FIGSIZE = (3.3, 2.6)

# Opacity of the mean +- std band behind a seed-averaged curve.
_STD_ALPHA = 0.25


def _plot_curve(ax, r, **line_kw):
    """Draw one convergence curve, with a mean +- std band if r has energy_std.

    Args:
        ax: The matplotlib axis.
        r: Object with energy_history and, for a seed average, energy_std.
        **line_kw: Passed to ax.plot (colour, line style, label, ...).
    """
    y = np.asarray(r.energy_history, dtype=float)
    x = np.arange(len(y))
    (line,) = ax.plot(x, y, **line_kw)
    std = getattr(r, "energy_std", None)
    if std is not None:
        std = np.asarray(std, dtype=float)
        ax.fill_between(x, y - std, y + std, color=line.get_color(), alpha=_STD_ALPHA, linewidth=0)


def write_convergence_figures(
    results_by_opt,
    out_dir: str,
    fci_energy: float | None = None,
    label: str | None = None,
    max_iters: int | None = None,
) -> list[str]:
    """Write the energy-convergence figures of one run (or one seed average).

    One optimizer gives energy_convergence_all_methods.pdf as a single axes.
    Several optimizers give a panel per optimizer in that file, plus
    energy_convergence_overlay.pdf with all of them on one axes.

    Args:
        results_by_opt: {optimizer: [result, ...]} as returned by
            io.load_convergence or io.load_convergence_mean_std.
        out_dir: Folder to write the PDFs into.
        fci_energy: Optional FCI energy drawn as a reference line.
        label: Optional system label for the title, e.g. "LiH (12 qubits)".
        max_iters: Optional x-axis cap (the main-stage iteration budget).

    Returns:
        The written paths.
    """
    conv_path = os.path.join(out_dir, "energy_convergence_all_methods.pdf")
    if len(results_by_opt) == 1:
        results = next(iter(results_by_opt.values()))
        plot_convergence(
            results, conv_path, fci_energy=fci_energy, title=label, max_iters=max_iters
        )
        return [conv_path]
    opt_list = " vs ".join(o.upper() for o in results_by_opt)
    title = f"{label}: {opt_list}" if label else None
    overlay_path = os.path.join(out_dir, "energy_convergence_overlay.pdf")
    plot_convergence_grid(
        results_by_opt, conv_path, fci_energy=fci_energy, title=title, max_iters=max_iters
    )
    plot_convergence_optimizers(
        results_by_opt, overlay_path, fci_energy=fci_energy, title=title, max_iters=max_iters
    )
    return [conv_path, overlay_path]


def plot_convergence(
    results,
    path: str,
    fci_energy: float | None = None,
    title: str | None = None,
    max_iters: int | None = None,
) -> str:
    """Plot energy-convergence curves for several methods.

    Args:
        results: Iterable of objects with method, energy_history and (optionally)
            stage_boundaries and energy_std attributes. With energy_std, the
            curve is a seed mean and gets a shaded mean +- std band.
        path: Destination figure path (.pdf).
        fci_energy: Optional reference energy drawn as a horizontal line.
        title: Optional plot title (e.g. "H2 (4 qubits)"). A default is used when
            None.

    Returns:
        The written path.
    """
    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        for r in results:
            _plot_curve(ax, r, label=r.method)
        if fci_energy is not None:
            ax.axhline(fci_energy, color="black", linestyle="--", label="FCI")
        if max_iters is not None:
            ax.set_xlim(0, max_iters)
        ax.set_xlabel("iteration")
        ax.set_ylabel("energy (Ha)")
        ax.set_title(title or "VQE convergence")
        # Legend below the axes (under the x-label), clear of the title on top.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=3,
            frameon=False,
            borderaxespad=0.0,
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def plot_convergence_grid(
    results_by_opt,
    path: str,
    fci_energy: float | None = None,
    title: str | None = None,
    max_iters: int | None = None,
) -> str:
    """Plot convergence as a side-by-side grid: one panel per optimizer.

    Each optimizer gets its own panel with a shared y-axis, so the method curves
    are readable and directly comparable across panels. Method colours are
    consistent, so one legend serves the whole figure.

    Args:
        results_by_opt: Mapping {optimizer_name: [MethodResult, ...]} (e.g.
            {"adam": [...], "qng": [...], "adagrad": [...]}). Insertion order sets
            the panel order.
        path: Destination figure path (.pdf).
        fci_energy: Optional reference energy drawn as a horizontal line in each
            panel.
        title: Optional overall figure title.
        max_iters: Optional x-axis cap (e.g. to hide a warm-up tail).

    Returns:
        The written path.
    """
    from matplotlib.lines import Line2D

    opt_names = list(results_by_opt.keys())
    ncols = len(opt_names)
    if ncols == 0:
        return path

    # Stable colour per method, taken from the first optimizer's method order.
    methods = [r.method for r in next(iter(results_by_opt.values()))]
    cmap = plt.get_cmap("tab10")
    method_color = {m: cmap(i % 10) for i, m in enumerate(methods)}

    with plt.rc_context(IEEE_STYLE):
        fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 2.9), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, opt in zip(axes, opt_names, strict=False):
            for r in results_by_opt[opt]:
                _plot_curve(ax, r, color=method_color.get(r.method, "gray"), linewidth=1.0)
            if fci_energy is not None:
                ax.axhline(fci_energy, color="black", linestyle=(0, (1, 1)), linewidth=0.8)
            if max_iters is not None:
                ax.set_xlim(0, max_iters)
            ax.set_title(opt.upper(), fontsize=8)
            ax.set_xlabel("iteration")
        axes[0].set_ylabel("energy (Ha)")

        # One shared legend for the whole figure: method colours (+ FCI).
        handles = [Line2D([0], [0], color=method_color[m], lw=1.4, label=m) for m in methods]
        if fci_energy is not None:
            handles.append(
                Line2D([0], [0], color="black", lw=0.8, linestyle=(0, (1, 1)), label="FCI")
            )
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            fontsize=6,
            bbox_to_anchor=(0.5, 0.0),
        )
        if title:
            fig.suptitle(title, fontsize=10)
        # Reserve room for the shared legend (bottom) and the title (top).
        fig.tight_layout(rect=[0, 0.10, 1, 0.94 if title else 1.0])
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_convergence_optimizers(
    results_by_opt,
    path: str,
    fci_energy: float | None = None,
    title: str | None = None,
    max_iters: int | None = None,
) -> str:
    """Overlay convergence curves for several optimizers, one line style each.

    Each method keeps one colour across optimizers, while each optimizer gets its
    own line style (solid, dashed, dash-dot, dotted). Two legends sit outside the
    plot box: one for method colours, one for the line-style to optimizer
    mapping.

    Args:
        results_by_opt: Mapping {optimizer_name: [MethodResult, ...]} (e.g.
            {"adam": [...], "qng": [...], "adagrad": [...]}). Insertion order sets
            the line-style assignment.
        path: Destination figure path (.pdf).
        fci_energy: Optional reference energy drawn as a horizontal line.

    Returns:
        The written path.
    """
    from matplotlib.lines import Line2D

    opt_names = list(results_by_opt.keys())
    styles = ["-", "--", "-.", ":"]
    # Stable colour per method, taken from the first optimizer's method order.
    methods = [r.method for r in next(iter(results_by_opt.values()))]
    cmap = plt.get_cmap("tab10")
    method_color = {m: cmap(i % 10) for i, m in enumerate(methods)}

    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        for oi, opt in enumerate(opt_names):
            style = styles[oi % len(styles)]
            for r in results_by_opt[opt]:
                _plot_curve(
                    ax,
                    r,
                    color=method_color.get(r.method, "gray"),
                    linestyle=style,
                    linewidth=1.0,
                )
        if fci_energy is not None:
            ax.axhline(fci_energy, color="black", linestyle=(0, (1, 1)), linewidth=0.8)
        if max_iters is not None:
            ax.set_xlim(0, max_iters)  # cap the view (e.g. hide a warm-up tail)
        ax.set_xlabel("iteration")
        ax.set_ylabel("energy (Ha)")
        # Title above the method legend, which is itself above the axes.
        ax.set_title(
            title or "VQE convergence: " + " vs ".join(o.upper() for o in opt_names), pad=22
        )

        # Legend 1: method colours (outside, above the axes).
        method_handles = [Line2D([0], [0], color=method_color[m], lw=1.4, label=m) for m in methods]
        leg1 = ax.legend(
            handles=method_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(methods),
            frameon=False,
            fontsize=6,
            columnspacing=1.0,
            handlelength=1.6,
        )
        ax.add_artist(leg1)
        # Legend 2: line-style -> optimizer (outside, to the right).
        style_handles = [
            Line2D(
                [0],
                [0],
                color="black",
                lw=1.2,
                linestyle=styles[i % len(styles)],
                label=opt.upper(),
            )
            for i, opt in enumerate(opt_names)
        ]
        if fci_energy is not None:
            style_handles.append(
                Line2D([0], [0], color="black", lw=0.8, linestyle=(0, (1, 1)), label="FCI")
            )
        ax.legend(
            handles=style_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
            fontsize=6,
        )
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def plot_gradient_scaling(points, path: str, title: str | None = None) -> str:
    """Plot gradient variance versus depth on a log scale.

    The mean is a line with markers and mean +- std is a shaded band. Where
    mean - std is not positive, the log axis cannot show the band's lower edge,
    so the band is left open there.

    Args:
        points: Iterable of gradient-scaling points with n_layers, mean_grad_var
            and std_grad_var attributes.
        path: Destination figure path (.pdf).

    Returns:
        The written path.
    """
    depths = np.array([p.n_layers for p in points])
    var = np.array([p.mean_grad_var for p in points])
    err = np.array([p.std_grad_var for p in points])
    lower = var - err
    lower = np.where(lower > 0, lower, np.nan)
    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        (line,) = ax.plot(depths, var, marker="o", markersize=3)
        ax.fill_between(
            depths, lower, var + err, color=line.get_color(), alpha=_STD_ALPHA, linewidth=0
        )
        ax.set_yscale("log")
        ax.set_xlabel("number of layers")
        ax.set_ylabel(r"Var[$\partial_\theta E$]")
        ax.set_title(title or "Gradient-variance scaling")
        # Integer major ticks every 5 layers, minor ticks every layer.
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlim(left=0)
        ax.grid(axis="x", which="major", linewidth=0.3, alpha=0.5)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path
