"""Matplotlib figures: convergence, gradient scaling, and loss landscapes.

Uses the Agg backend (headless) and an IEEE-conference style (serif fonts,
compact single-column sizing) so figures drop into a two-column paper. Landscape
plots mark the trajectory start with a lime circle and the end with a red star.
Convergence plots carry no trajectory markers.
"""

from __future__ import annotations

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
_FIGSIZE_3D = (3.4, 3.0)

# Landscape trajectory markers: start = lime circle, end = red star.
_START_STYLE = dict(color="lime", marker="o", s=45, edgecolors="black", linewidths=0.5, zorder=5)
_END_STYLE = dict(color="red", marker="*", s=110, edgecolors="black", linewidths=0.5, zorder=5)

# Per-optimizer path-line colours (saturated, to stay visible over viridis).
_TRAJ_COLORS = {
    "adam": "orangered",
    "qng": "deepskyblue",
    "adagrad": "magenta",
    "trajectory": "magenta",
}
_TRAJ_FALLBACK = ["magenta", "orangered", "deepskyblue", "black"]


def _traj_color(label, index):
    """Return the path colour for a trajectory label.

    Args:
        label: Trajectory label (usually the optimizer name).
        index: Position in the trajectory list (for the fallback palette).

    Returns:
        A matplotlib colour.
    """
    return _TRAJ_COLORS.get(label, _TRAJ_FALLBACK[index % len(_TRAJ_FALLBACK)])


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
            stage_boundaries attributes.
        path: Destination figure path (.pdf).
        fci_energy: Optional reference energy drawn as a horizontal line.
        title: Optional plot title (e.g. "H2 (4 qubits)"). A default is used when
            None.

    Returns:
        The written path.
    """
    # No trajectory markers here, convergence curves only.
    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        for r in results:
            ax.plot(range(len(r.energy_history)), r.energy_history, label=r.method)
        if fci_energy is not None:
            ax.axhline(fci_energy, color="black", linestyle="--", label="FCI")
        if max_iters is not None:
            ax.set_xlim(0, max_iters)
        ax.set_xlabel("iteration")
        ax.set_ylabel("energy (Ha)")
        ax.set_title(title or "VQE convergence")
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, borderaxespad=0.0
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
                ax.plot(
                    range(len(r.energy_history)),
                    r.energy_history,
                    color=method_color.get(r.method, "gray"),
                    linewidth=1.0,
                )
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
                ax.plot(
                    range(len(r.energy_history)),
                    r.energy_history,
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

    Args:
        points: Iterable of gradient-scaling points with n_layers, mean_grad_var
            and std_grad_var attributes.
        path: Destination figure path (.pdf).

    Returns:
        The written path.
    """
    depths = [p.n_layers for p in points]
    var = np.array([p.mean_grad_var for p in points])
    err = np.array([p.std_grad_var for p in points])
    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        ax.errorbar(depths, var, yerr=err, marker="o", markersize=3, capsize=2)
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


def plot_landscape_contour(landscape, path: str, title: str | None = None) -> str:
    """Plot a filled contour of a loss landscape with trajectory overlay.

    Args:
        landscape: A core.landscape.Landscape instance.
        path: Destination figure path (.pdf).

    Returns:
        The written path.
    """
    with plt.rc_context(IEEE_STYLE):
        fig, ax = plt.subplots(figsize=_FIGSIZE)
        cs = ax.contourf(
            landscape.a_grid, landscape.b_grid, landscape.energies, levels=40, cmap="viridis"
        )
        fig.colorbar(cs, ax=ax, label="energy (Ha)")
        has_traj = _overlay_trajectory(ax, landscape)
        xl, yl = _axis_labels(landscape)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title or "Convergence landscape (contour)")
        if has_traj:
            handles = _trajectory_legend_handles(landscape)
            # Reserve bottom margin for a figure-level legend below the x-label.
            fig.subplots_adjust(bottom=0.28)
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=len(handles),
                frameon=False,
                fontsize=6,
                columnspacing=1.0,
                handlelength=1.4,
                bbox_to_anchor=(0.5, 0.0),
            )
        fig.savefig(path)
        plt.close(fig)
    return path


def _draw_fci_plane(ax, landscape, fci_energy):
    """Draw a translucent horizontal plane at the FCI energy on a 3D surface.

    Shows the exact ground-state energy level, so paths can be seen settling onto
    the same energy even when they end at different in-plane positions.

    Args:
        ax: The 3D axis.
        landscape: The landscape (for the (a,b) extent).
        fci_energy: The FCI energy level to draw, or None to skip.
    """
    if fci_energy is None:
        return
    a0, a1 = float(landscape.a_grid.min()), float(landscape.a_grid.max())
    b0, b1 = float(landscape.b_grid.min()), float(landscape.b_grid.max())
    PA, PB = np.meshgrid(np.linspace(a0, a1, 2), np.linspace(b0, b1, 2))
    PZ = np.full_like(PA, float(fci_energy))
    # Red FCI plane plus red edge frame.
    ax.plot_surface(
        PA, PB, PZ, color="red", alpha=0.35, linewidth=0, antialiased=False, shade=False, zorder=5
    )
    ax.plot(
        [a0, a1, a1, a0, a0],
        [b0, b0, b1, b1, b0],
        [fci_energy] * 5,
        color="red",
        linewidth=1.2,
        alpha=0.9,
        zorder=6,
    )


def plot_landscape_surface(
    landscape, path: str, title: str | None = None, fci_energy: float | None = None
) -> str:
    """Plot a 3D surface of a loss landscape.

    Args:
        landscape: A core.landscape.Landscape instance.
        path: Destination figure path (.pdf).
        title: Optional plot title.
        fci_energy: If given, draw a translucent plane at this energy so paths
            are seen settling onto the true ground-state level.

    Returns:
        The written path.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    A, B = np.meshgrid(landscape.a_grid, landscape.b_grid)
    with plt.rc_context(IEEE_STYLE):
        fig = plt.figure(figsize=_FIGSIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            A, B, landscape.energies, cmap="viridis", linewidth=0, antialiased=True, alpha=0.6
        )
        _draw_fci_plane(ax, landscape, fci_energy)
        has_traj = _overlay_trajectory_3d(ax, landscape)
        xl, yl = _axis_labels(landscape)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_zlabel("energy (Ha)")
        ax.set_title(title or "Convergence landscape (surface)")
        if has_traj:
            handles = _trajectory_legend_handles(landscape)
            # Figure-level legend in a reserved bottom strip, clear of the 3D axes.
            fig.subplots_adjust(bottom=0.16)
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=len(handles),
                frameon=False,
                fontsize=6,
                columnspacing=1.0,
                handlelength=1.4,
                bbox_to_anchor=(0.5, 0.0),
            )
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_landscape_grid(
    items, path: str, title: str | None = None, fci_energy: float | None = None
) -> str:
    """Plot all method landscapes as a grid: methods in columns, views in rows.

    Columns are methods, rows are the two landscape views (top: 3D surface,
    bottom: contour). Each panel carries its optimizer-path overlays, with one
    shared trajectory legend below.

    Args:
        items: Iterable of (label, landscape) pairs, one per method (column order
            is preserved).
        path: Destination figure path (.pdf).
        title: Optional overall figure title.
        fci_energy: If given, draw a translucent FCI-energy plane on each surface
            so paths are seen settling onto the true ground-state level.

    Returns:
        The written path.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    items = list(items)
    ncols = len(items)
    if ncols == 0:
        return path

    with plt.rc_context(IEEE_STYLE):
        fig = plt.figure(figsize=(3.0 * ncols, 5.8))
        legend_src = None
        for c, (label, ls) in enumerate(items):
            A, B = np.meshgrid(ls.a_grid, ls.b_grid)

            # Paths lie ON the surface, so the surface's range bounds them, padded.
            zmin = float(ls.energies.min())
            zmax = float(ls.energies.max())
            pad = 0.08 * (zmax - zmin + 1e-9)
            zlo, zhi = zmin - pad, zmax + pad

            # Row 1 (top): 3D surface, with the contour projected onto the floor.
            axs = fig.add_subplot(2, ncols, c + 1, projection="3d")
            axs.plot_surface(
                A, B, ls.energies, cmap="viridis", linewidth=0, antialiased=True, alpha=0.6
            )
            axs.contourf(
                A, B, ls.energies, levels=20, cmap="viridis", zdir="z", offset=zlo, alpha=0.7
            )
            axs.set_zlim(zlo, zhi)
            _draw_fci_plane(axs, ls, fci_energy)
            _overlay_trajectory_3d(axs, ls)
            xl, yl = _axis_labels(ls)
            axs.set_title(label, fontsize=8)
            axs.set_xlabel(xl, fontsize=6, labelpad=-4)
            axs.set_ylabel(yl, fontsize=6, labelpad=-4)
            axs.set_zlabel("energy (Ha)", fontsize=6, labelpad=-4)
            axs.tick_params(labelsize=4, pad=-2)

            # Row 2 (bottom): 2D contour.
            axc = fig.add_subplot(2, ncols, ncols + c + 1)
            axc.contourf(ls.a_grid, ls.b_grid, ls.energies, levels=30, cmap="viridis")
            if _overlay_trajectory(axc, ls) and legend_src is None:
                legend_src = ls
            axc.set_xlabel(xl, fontsize=6, labelpad=1)
            axc.tick_params(labelsize=5)
            if c == 0:
                axc.set_ylabel(yl, fontsize=6)

        if title:
            fig.suptitle(title, fontsize=11)
        if legend_src is not None:
            handles = _trajectory_legend_handles(legend_src)
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=len(handles),
                frameon=False,
                fontsize=7,
                bbox_to_anchor=(0.5, 0.0),
            )
        fig.tight_layout(rect=[0, 0.05, 1, 0.95 if title else 1.0])
        fig.savefig(path)
        plt.close(fig)
    return path


def _axis_labels(landscape):
    """Return (xlabel, ylabel) for a landscape's two axes.

    Uses the swept-parameter labels when present (e.g. theta[3]), else generic
    names for landscapes saved before parameter axes existed.
    """
    labels = getattr(landscape, "axis_labels", None)
    if labels and len(labels) == 2:
        return str(labels[0]), str(labels[1])
    return "param 1", "param 2"


def _trajectory_legend_handles(landscape):
    """Build legend handles for the landscape trajectories.

    One coloured line per optimizer path, plus shared start/end marker entries.
    """
    from matplotlib.lines import Line2D

    handles = []
    for i, tr in enumerate(getattr(landscape, "trajectories", None) or []):
        handles.append(Line2D([0], [0], color=_traj_color(tr.label, i), lw=1.4, label=tr.label))
    handles.append(
        Line2D(
            [0],
            [0],
            color="lime",
            marker="o",
            linestyle="none",
            markeredgecolor="black",
            markersize=5,
            label="start",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color="red",
            marker="*",
            linestyle="none",
            markeredgecolor="black",
            markersize=8,
            label="end",
        )
    )
    return handles


def _overlay_trajectory(ax, landscape) -> bool:
    """Overlay every projected optimizer trajectory on a 2D axis.

    Each path is drawn in its optimizer colour with start (lime circle) and end
    (red star) markers. Returns True if at least one trajectory was drawn.
    """
    trajs = getattr(landscape, "trajectories", None) or []
    drawn = False
    for i, tr in enumerate(trajs):
        ab = np.asarray(tr.ab)
        if ab.size == 0:
            continue
        color = _traj_color(tr.label, i)
        ax.plot(ab[:, 0], ab[:, 1], color=color, linewidth=1.2, alpha=0.9, label=tr.label)
        ax.scatter([ab[0, 0]], [ab[0, 1]], **_START_STYLE)
        ax.scatter([ab[-1, 0]], [ab[-1, 1]], **_END_STYLE)
        drawn = True
    # Pin each axis to its sampled range so a stray point cannot stretch it.
    if drawn:
        a = np.asarray(landscape.a_grid, dtype=float)
        b = np.asarray(landscape.b_grid, dtype=float)
        ax.set_xlim(float(a.min()), float(a.max()))
        ax.set_ylim(float(b.min()), float(b.max()))
    return drawn


def _overlay_trajectory_3d(ax, landscape) -> bool:
    """Overlay every projected optimizer trajectory on a 3D surface axis.

    Uses each point's surface height as the z-coordinate (the in-plane energy at
    (theta[i], theta[j]) with the others fixed at theta*), so the path lies on
    the plotted surface. depthshade=False and a high zorder keep the paths
    visible over the surface. Returns True if at least one trajectory was drawn.
    """
    trajs = getattr(landscape, "trajectories", None) or []
    drawn = False
    for i, tr in enumerate(trajs):
        ab = np.asarray(tr.ab)
        # Surface height under the path, so the curve lies ON the surface.
        z = getattr(tr, "surface_energy", None)
        z = np.asarray(z if z is not None else tr.true_energy, dtype=float)
        if ab.size == 0:
            continue
        mask = ~np.isnan(z)
        if not mask.any():
            continue
        a, b, zz = ab[mask, 0], ab[mask, 1], z[mask]
        color = _traj_color(tr.label, i)
        ax.plot(a, b, zz, color=color, linewidth=1.6, alpha=0.95, zorder=10, label=tr.label)
        start = dict(_START_STYLE, depthshade=False, zorder=11)
        end = dict(_END_STYLE, depthshade=False, zorder=11)
        ax.scatter([a[0]], [b[0]], [zz[0]], **start)
        ax.scatter([a[-1]], [b[-1]], [zz[-1]], **end)
        drawn = True
    return drawn
