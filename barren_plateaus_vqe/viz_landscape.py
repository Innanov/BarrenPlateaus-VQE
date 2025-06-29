#!/usr/bin/env python3
"""
Loss Landscape and Convergence Visualization Module (`viz_landscape.py`)
======================================================================

This module provides comprehensive visualization tools for analyzing barren plateau
phenomena in Variational Quantum Eigensolver (VQE) algorithms using molecular Hamiltonians.
It implements state-of-the-art visualization techniques adapted for quantum chemistry applications.

Visualization Techniques Implemented:

1. Energy Convergence Analysis:
   - Time-series plots of energy vs optimization iteration
   - Comparison across multiple VQE variants for specific molecules
   - Exact ground state energy reference lines with molecular context

2. Loss Landscape Visualization:
   - PCA-based 2D projections of high-dimensional molecular landscapes
   - 3D surface plots showing molecular energy topology
   - Optimization trajectory overlays for molecular systems

3. Gradient Landscape Analysis:
   - Heat maps of gradient magnitudes across molecular parameter space
   - Logarithmic scaling for barren plateau identification in molecules

4. Statistical Distribution Analysis:
   - Histograms of gradient norms, cost values, and curvature measures
   - Statistical overlays for molecular systems
   - Multi-method comparison for different molecules

5. Performance Comparison Visualizations:
   - Gradient variance bar charts for molecular systems
   - Energy error and state fidelity comparisons across molecules
   - Scaling analysis plots for layer and system size studies
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Core scientific libraries
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print(
        "Warning: matplotlib not available. Visualization functionality will be limited."
    )
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns

    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Optional PCA for landscape analysis
try:
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Quantum computing libraries
try:
    from qiskit_aer import AerSimulator

    QISKIT_AER_AVAILABLE = True
except ImportError:
    QISKIT_AER_AVAILABLE = False

warnings.filterwarnings("ignore")

# Set default plotting style
if MATPLOTLIB_AVAILABLE:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": [12, 8],
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )


# ============================================================================
# Landscape Computation Utilities
# ============================================================================


def compute_loss_landscape_pca(
    cost_function,
    trajectory_params: List[np.ndarray],
    grid_size: int = 30,
    scale_factor: float = 1.0,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compute 2D loss landscape using PCA projection of optimization trajectory.

    Args:
        `cost_function`: Quantum cost function C(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
        `trajectory_params`: List of parameter vectors from optimization trajectory
        `grid_size`: Resolution of the visualization grid (default: 30)
        `scale_factor`: Expansion factor for visualization bounds (default: 1.0)

    Returns:
        Dictionary containing landscape data or None if computation fails

    Theory:
        Given trajectory points {θ₁, θ₂, ..., θₜ}, PCA finds orthogonal directions
        v₁, v₂ that maximize the variance of projected points:

        max Var[θᵢ · v₁] subject to ||v₁|| = 1
        max Var[θᵢ · v₂] subject to ||v₂|| = 1, v₁ · v₂ = 0

        The 2D landscape is then C(θ₀ + α₁v₁ + α₂v₂) where θ₀ is the trajectory center.
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: `sklearn` not available, skipping PCA landscape computation")
        return None

    if not trajectory_params or len(trajectory_params) < 1:
        print("Warning: Empty trajectory provided")
        return None

    try:
        # Convert trajectory to consistent format
        trajectory_arrays = []
        for params in trajectory_params:
            if isinstance(params, (list, tuple)):
                trajectory_arrays.append(np.array(params))
            else:
                trajectory_arrays.append(params)

        # Handle mixed dimensions
        param_lengths = [len(params) for params in trajectory_arrays]
        unique_lengths = list(set(param_lengths))

        if len(unique_lengths) > 1:
            print(f"Warning: Mixed parameter dimensions: {unique_lengths}")
            most_common_length = max(set(param_lengths), key=param_lengths.count)
            trajectory_arrays = [
                params
                for params in trajectory_arrays
                if len(params) == most_common_length
            ]

        if len(trajectory_arrays) < 2:
            # Create second point for PCA
            original_point = trajectory_arrays[0]
            perturbed_point = original_point + np.random.normal(
                0, 0.01, len(original_point)
            )
            trajectory_arrays = [original_point, perturbed_point]

        # Convert to numpy array
        trajectory = np.array(trajectory_arrays)

        # Perform PCA
        n_components = min(2, trajectory.shape[0], trajectory.shape[1])
        pca = PCA(n_components=n_components)
        trajectory_projected = pca.fit_transform(trajectory)

        if trajectory_projected.shape[1] == 1:
            # Add artificial second component
            second_component = np.random.normal(0, 0.1, trajectory_projected.shape[0])
            trajectory_projected = np.column_stack(
                [trajectory_projected[:, 0], second_component]
            )
            explained_variance_ratio = np.array([pca.explained_variance_ratio_[0], 0.0])
            pca_components = np.vstack(
                [pca.components_[0], np.zeros(trajectory.shape[1])]
            )
        else:
            explained_variance_ratio = pca.explained_variance_ratio_
            pca_components = pca.components_

    except Exception as e:
        print(f"PCA computation failed: {e}")
        # Fallback to first two dimensions
        if trajectory.shape[1] >= 2:
            trajectory_projected = trajectory[:, :2]
        else:
            trajectory_projected = np.column_stack(
                [trajectory[:, 0], np.random.normal(0, 0.1, trajectory.shape[0])]
            )
        explained_variance_ratio = np.array([1.0, 0.0])
        pca_components = np.eye(2, trajectory.shape[1])

    # Define grid bounds
    x_min, x_max = trajectory_projected[:, 0].min(), trajectory_projected[:, 0].max()
    y_min, y_max = trajectory_projected[:, 1].min(), trajectory_projected[:, 1].max()

    # Handle single point case
    if x_max - x_min < 1e-6:
        x_range = 1.0
        x_center = x_min
    else:
        x_range = (x_max - x_min) * scale_factor
        x_center = (x_max + x_min) / 2

    if y_max - y_min < 1e-6:
        y_range = 1.0
        y_center = y_min
    else:
        y_range = (y_max - y_min) * scale_factor
        y_center = (y_max + y_min) / 2

    # Create grid
    x_vals = np.linspace(x_center - x_range / 2, x_center + x_range / 2, grid_size)
    y_vals = np.linspace(y_center - y_range / 2, y_center + y_range / 2, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    print(f"Computing PCA-based loss landscape ({grid_size}x{grid_size})...")

    # Evaluate cost function over grid
    trajectory_center = np.mean(trajectory, axis=0)

    for i in range(grid_size):
        for j in range(grid_size):
            try:
                pca_point = np.array([X[i, j], Y[i, j]])

                # Transform back to original parameter space
                if pca_components.shape[0] >= 2:
                    original_params = (
                        np.dot(pca_point, pca_components) + trajectory_center
                    )
                else:
                    original_params = (
                        pca_point[0] * pca_components[0] + trajectory_center
                    )

                Z[i, j] = cost_function(original_params)
            except Exception:
                Z[i, j] = np.inf

        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{grid_size} rows")

    return {
        "X": X,
        "Y": Y,
        "Z": Z,
        "trajectory_projected": trajectory_projected,
        "pca_components": pca_components,
        "explained_variance_ratio": explained_variance_ratio,
        "grid_size": grid_size,
        "trajectory_center": trajectory_center,
    }


def safe_pca_transform(
    trajectory: List[np.ndarray], n_components: int = 2
) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Safely perform PCA transformation on trajectory data with robust error handling.

    Args:
        `trajectory`: List of parameter vectors
        `n_components`: Number of PCA components to compute

    Returns:
        Tuple of (`projected_trajectory`, `pca_object`)
    """
    if not SKLEARN_AVAILABLE:
        print("Warning: sklearn not available for PCA transformation")
        return np.array([[0, 0], [1, 1]]), None

    if not trajectory or len(trajectory) == 0:
        raise ValueError("Empty trajectory provided")

    # Convert to numpy array with dimension consistency
    trajectory_array = np.array(trajectory)

    # Handle single point case
    if len(trajectory) == 1:
        # Add a perturbed point for PCA
        perturbation = np.random.normal(0, 0.01, trajectory_array.shape[1])
        trajectory_array = np.vstack(
            [trajectory_array, trajectory_array + perturbation]
        )

    # Perform PCA with error handling
    try:
        pca = PCA(
            n_components=min(
                n_components, trajectory_array.shape[0], trajectory_array.shape[1]
            )
        )
        projected = pca.fit_transform(trajectory_array)

        # Ensure we have 2D output
        if projected.shape[1] == 1:
            # Add artificial second dimension
            second_dim = np.random.normal(0, 0.1, projected.shape[0])
            projected = np.column_stack([projected, second_dim])

        return projected, pca

    except Exception as e:
        print(f"PCA failed: {e}, using fallback method")
        # Fallback: use first two dimensions or create artificial ones
        if trajectory_array.shape[1] >= 2:
            return trajectory_array[:, :2], None
        else:
            # Create 2D projection artificially
            projected = np.column_stack(
                [
                    trajectory_array[:, 0],
                    np.random.normal(0, 0.1, len(trajectory_array)),
                ]
            )
            return projected, None


# ============================================================================
# Main Visualization Class
# ============================================================================


class MolecularLandscapeVisualizer:
    """
    Comprehensive visualization class for molecular barren plateau analysis.

    This class provides publication-ready visualizations for studying barren plateau
    phenomena in VQE algorithms applied to molecular systems.
    """

    def __init__(self, analyzer, output_dir: str = "./plots"):
        """
        Initialize the molecular landscape visualizer.

        Args:
            `analyzer`: MolecularVQEAnalyzer instance
            `output_dir`: Directory to save plots
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for visualization")

        self.analyzer = analyzer
        self.output_dir = output_dir
        self.ensure_output_dir()

        # Set up molecular context for plots
        self.setup_molecular_context()

    def ensure_output_dir(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Plots will be saved to: {self.output_dir}")

    def setup_molecular_context(self):
        """Setup molecular context for plot titles and labels."""
        if hasattr(self.analyzer, "molecule_name") and not getattr(
            self.analyzer, "use_test_hamiltonian", False
        ):
            self.molecular_context = {
                "title_prefix": f"{self.analyzer.molecule_name} ({self.analyzer.geometry_type})",
                "system_info": f"{self.analyzer.basis_set}, {self.analyzer.num_qubits} qubits",
                "filename_prefix": f"{self.analyzer.molecule_name}_{self.analyzer.geometry_type}_{self.analyzer.basis_set}".replace(
                    "-", ""
                ),
            }
        else:
            self.molecular_context = {
                "title_prefix": f"Test System",
                "system_info": f"{self.analyzer.num_qubits} qubits",
                "filename_prefix": f"test_{self.analyzer.num_qubits}qubits",
            }

    def plot_energy_convergence(self, save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot energy convergence curves for all VQE methods with molecular context.

        Args:
            `save_name`: Custom filename for saving plot

        Returns:
            Matplotlib figure object
        """
        if save_name is None:
            save_name = (
                f"{self.molecular_context['filename_prefix']}_energy_convergence.pdf"
            )

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        # Plot energy convergence for each method
        for i, (method_name, data) in enumerate(self.analyzer.results.items()):
            try:
                energies = data["method_results"]["energies"]
                ax.plot(
                    range(len(energies)),
                    energies,
                    color=colors[i % len(colors)],
                    label=method_name,
                    linewidth=2.5,
                    alpha=0.8,
                    marker="o" if len(energies) < 50 else None,
                    markersize=4 if len(energies) < 50 else 0,
                )
            except Exception as e:
                print(f"Warning: Could not plot {method_name}: {e}")
                continue

        # Add exact ground state reference
        if hasattr(self.analyzer, "exact_min_energy"):
            ax.axhline(
                y=self.analyzer.exact_min_energy,
                color="black",
                linestyle="--",
                label="Exact Ground State",
                linewidth=2,
                alpha=0.8,
            )

        # Formatting
        ax.set_xlabel("Iterations", fontsize=14, fontweight="bold")
        ax.set_ylabel(
            (
                r"Energy (Hartree)"
                if not getattr(self.analyzer, "use_test_hamiltonian", False)
                else r"$\langle H \rangle$"
            ),
            fontsize=14,
            fontweight="bold",
        )

        title = f"Energy Convergence: {self.molecular_context['title_prefix']}"
        subtitle = f"{self.molecular_context['system_info']}"
        ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight="bold")

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")

        return fig

    def create_performance_table(self) -> pd.DataFrame:
        """
        Create and display performance metrics table with molecular context.

        Returns:
            DataFrame with performance metrics
        """
        # Collect data
        methods = []
        energy_errors = []
        fidelities = []
        gradient_variances = []
        gradient_norms = []

        for method_name, data in self.analyzer.results.items():
            try:
                methods.append(method_name)
                energy_errors.append(data["performance_metrics"]["final_energy_error"])
                fidelities.append(data["performance_metrics"]["state_fidelity"])
                gradient_variances.append(data["bp_diagnostics"]["gradient_variance"])
                gradient_norms.append(data["bp_diagnostics"]["gradient_norm_mean"])
            except Exception as e:
                print(f"Warning: Could not extract data for {method_name}: {e}")
                continue

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Method": methods,
                "Energy Error": energy_errors,
                "State Fidelity": fidelities,
                "Gradient Variance": gradient_variances,
                "Gradient Norm": gradient_norms,
            }
        )

        # Display formatted table
        df_display = df.copy()
        df_display["Energy Error"] = df_display["Energy Error"].apply(
            lambda x: f"{x:.2e}"
        )
        df_display["State Fidelity"] = df_display["State Fidelity"].apply(
            lambda x: f"{x:.3f}"
        )
        df_display["Gradient Variance"] = df_display["Gradient Variance"].apply(
            lambda x: f"{x:.2e}"
        )
        df_display["Gradient Norm"] = df_display["Gradient Norm"].apply(
            lambda x: f"{x:.2e}"
        )

        print("\n" + "=" * 80)
        print(f"PERFORMANCE METRICS: {self.molecular_context['title_prefix']}")
        print(f"{self.molecular_context['system_info']}")
        print("=" * 80)
        print(df_display.to_string(index=False))
        print("=" * 80 + "\n")

        # Save files
        csv_path = os.path.join(
            self.output_dir,
            f"{self.molecular_context['filename_prefix']}_performance_table.csv",
        )
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")

        # Save LaTeX table
        latex_path = os.path.join(
            self.output_dir,
            f"{self.molecular_context['filename_prefix']}_performance_table.tex",
        )

        latex_content = f"""\\begin{{table}}[h!]
\\centering
\\caption{{Performance Metrics for {self.molecular_context['title_prefix']} ({self.molecular_context['system_info']})}}
\\label{{tab:performance_metrics_{self.molecular_context['filename_prefix']}}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Method}} & \\textbf{{Energy Error}} & \\textbf{{State Fidelity}} & \\textbf{{Gradient Variance}} & \\textbf{{Gradient Norm}} \\\\
\\hline
"""

        for _, row in df.iterrows():
            method = row["Method"].replace("_", r"\_")
            latex_content += f"{method} & ${row['Energy Error']:.2e}$ & ${row['State Fidelity']:.3f}$ & ${row['Gradient Variance']:.2e}$ & ${row['Gradient Norm']:.2e}$ \\\\\n"

        latex_content += r"""\hline
\end{tabular}
\end{table}"""

        with open(latex_path, "w") as f:
            f.write(latex_content)
        print(f"Saved LaTeX: {latex_path}")

        return df

    def plot_gradient_diagnostics(self, save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive gradient diagnostics with molecular context.

        Args:
            `save_name`: Custom filename for saving plot

        Returns:
            Matplotlib figure object
        """
        if save_name is None:
            save_name = (
                f"{self.molecular_context['filename_prefix']}_gradient_diagnostics.pdf"
            )

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Gradient Diagnostics: {self.molecular_context['title_prefix']}\n{self.molecular_context['system_info']}",
            fontsize=16,
            fontweight="bold",
        )

        # Collect data
        methods = list(self.analyzer.results.keys())
        gradient_variances = []
        gradient_norms = []
        energy_errors = []
        fidelities = []

        for method in methods:
            try:
                data = self.analyzer.results[method]
                gradient_variances.append(data["bp_diagnostics"]["gradient_variance"])
                gradient_norms.append(data["bp_diagnostics"]["gradient_norm_mean"])
                energy_errors.append(data["performance_metrics"]["final_energy_error"])
                fidelities.append(data["performance_metrics"]["state_fidelity"])
            except Exception as e:
                print(f"Warning: Could not extract data for {method}: {e}")
                gradient_variances.append(0)
                gradient_norms.append(0)
                energy_errors.append(1)
                fidelities.append(0)

        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))

        # Plot 1: Gradient Variance Bar Chart
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(methods)), gradient_variances, color=colors, alpha=0.7)
        ax1.set_yscale("log")
        ax1.set_title("Gradient Variance by Method", fontweight="bold", fontsize=12)
        ax1.set_ylabel("Gradient Variance")
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Gradient Norm vs Energy Error
        ax2 = axes[0, 1]
        scatter = ax2.scatter(
            gradient_norms,
            energy_errors,
            s=100,
            alpha=0.7,
            c=range(len(methods)),
            cmap="viridis",
        )
        for i, method in enumerate(methods):
            if (
                gradient_norms[i] > 0 and energy_errors[i] > 0
            ):  # Only annotate valid points
                ax2.annotate(
                    method,
                    (gradient_norms[i], energy_errors[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax2.set_xlabel("Gradient Norm Mean")
        ax2.set_ylabel("Final Energy Error")
        ax2.set_title("Gradient Norm vs Energy Error", fontweight="bold", fontsize=12)
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Variance vs Fidelity
        ax3 = axes[1, 0]
        scatter2 = ax3.scatter(
            gradient_variances,
            fidelities,
            s=100,
            alpha=0.7,
            c=range(len(methods)),
            cmap="viridis",
        )
        for i, method in enumerate(methods):
            if gradient_variances[i] > 0:  # Only annotate valid points
                ax3.annotate(
                    method,
                    (gradient_variances[i], fidelities[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        ax3.set_xlabel("Gradient Variance")
        ax3.set_ylabel("State Fidelity")
        ax3.set_title(
            "Gradient Variance vs State Fidelity", fontweight="bold", fontsize=12
        )
        ax3.set_xscale("log")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Performance Summary (normalized)
        ax4 = axes[1, 1]
        if gradient_variances and max(gradient_variances) > 0:
            norm_variances = np.array(gradient_variances) / max(gradient_variances)
            norm_errors = (
                np.array(energy_errors) / max(energy_errors)
                if max(energy_errors) > 0
                else energy_errors
            )
            norm_fidelities = 1 - np.array(fidelities)  # Invert so lower is better

            x = np.arange(len(methods))
            width = 0.25

            ax4.bar(
                x - width,
                norm_variances,
                width,
                label="Grad Variance (norm)",
                alpha=0.7,
            )
            ax4.bar(x, norm_errors, width, label="Energy Error (norm)", alpha=0.7)
            ax4.bar(x + width, norm_fidelities, width, label="1 - Fidelity", alpha=0.7)

            ax4.set_title(
                "Normalized Performance Comparison", fontweight="bold", fontsize=12
            )
            ax4.set_ylabel("Normalized Metric (lower is better)")
            ax4.set_xticks(x)
            ax4.set_xticklabels(methods, rotation=45, ha="right")
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")

        return fig

    def plot_loss_landscapes(
        self,
        num_methods_to_plot: Optional[int] = None,
        grid_size: int = 25,
        save_name: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Plot 2D and 3D loss landscapes for molecular systems.

        Args:
            `num_methods_to_plot`: Number of methods to include (None for all)
            `grid_size`: Resolution of the landscape grid
            `save_name`: Custom filename for saving plot

        Returns:
            Matplotlib figure object or None if computation fails
        """
        if not QISKIT_AER_AVAILABLE:
            print("Warning: `qiskit-aer` not available for loss landscape computation")
            return None

        if save_name is None:
            save_name = (
                f"{self.molecular_context['filename_prefix']}_loss_landscapes.pdf"
            )

        method_names = list(self.analyzer.results.keys())
        selected_methods = (
            method_names[:num_methods_to_plot] if num_methods_to_plot else method_names
        )
        print(f"Plotting loss landscapes for {len(selected_methods)} methods")

        n_methods = len(selected_methods)
        if n_methods == 0:
            print("No methods available for landscape plotting")
            return None

        fig = plt.figure(figsize=(6 * n_methods, 12))

        # Add main title
        fig.suptitle(
            f"Loss Landscapes: {self.molecular_context['title_prefix']}\n{self.molecular_context['system_info']}",
            fontsize=12,
            fontweight="bold",
            y=0.95,
        )

        for idx, method_name in enumerate(selected_methods):
            print(f"Processing method {idx+1}/{n_methods}: {method_name}")

            try:
                method_data = self.analyzer.results[method_name]["method_results"]
                ansatz = method_data["ansatz"]
                trajectory = method_data["trajectory"]
                final_params = method_data["final_params"]

                # Define cost function
                backend = AerSimulator(method="statevector")

                def cost_function(params):
                    try:
                        # Import here to avoid circular imports
                        from qubap.qiskit.variational_algorithms import (
                            energy_evaluation,
                        )

                        return energy_evaluation(
                            self.analyzer.H, ansatz, params, backend
                        )
                    except Exception as e:
                        print(f"Warning: Cost function evaluation failed: {e}")
                        return np.inf

                # Compute landscape
                landscape_data = compute_loss_landscape_pca(
                    cost_function, trajectory, grid_size=grid_size, scale_factor=1.5
                )

                if landscape_data is None:
                    print(f"  Warning: Could not compute landscape for {method_name}")
                    self._create_error_subplot(
                        fig,
                        2,
                        n_methods,
                        idx + 1,
                        f"Error: {method_name}",
                        "Landscape computation failed",
                    )
                    self._create_error_subplot(
                        fig,
                        2,
                        n_methods,
                        idx + 1 + n_methods,
                        f"Error: {method_name}",
                        "3D visualization failed",
                    )
                    continue

                # 2D Contour Plot
                ax1 = plt.subplot(2, n_methods, idx + 1)

                Z_plot = landscape_data["Z"]
                finite_mask = np.isfinite(Z_plot)
                if np.any(finite_mask):
                    Z_plot = np.where(
                        finite_mask, Z_plot, np.nanmax(Z_plot[finite_mask])
                    )
                else:
                    Z_plot = np.zeros_like(Z_plot)

                try:
                    contour = ax1.contourf(
                        landscape_data["X"],
                        landscape_data["Y"],
                        Z_plot,
                        levels=50,
                        cmap="viridis",
                        alpha=0.8,
                    )
                    ax1.contour(
                        landscape_data["X"],
                        landscape_data["Y"],
                        Z_plot,
                        levels=20,
                        colors="white",
                        alpha=0.5,
                        linewidths=0.8,
                    )
                except Exception as e:
                    print(f"  Warning: Could not create contour plot: {e}")

                # Plot trajectory
                traj_proj = landscape_data["trajectory_projected"]
                if len(traj_proj) > 1:
                    ax1.plot(
                        traj_proj[:, 0],
                        traj_proj[:, 1],
                        "r-",
                        linewidth=2,
                        alpha=0.8,
                        label="Path",
                    )
                    ax1.scatter(
                        traj_proj[0, 0],
                        traj_proj[0, 1],
                        c="red",
                        s=100,
                        marker="o",
                        label="Start",
                        zorder=5,
                    )
                    ax1.scatter(
                        traj_proj[-1, 0],
                        traj_proj[-1, 1],
                        c="red",
                        s=100,
                        marker="*",
                        label="End",
                        zorder=5,
                    )
                else:
                    ax1.scatter(
                        traj_proj[0, 0],
                        traj_proj[0, 1],
                        c="red",
                        s=100,
                        marker="*",
                        label="Final",
                        zorder=5,
                    )

                ax1.set_title(
                    f"{method_name}\nLoss Landscape (PCA)",
                    fontsize=12,
                    fontweight="bold",
                )

                if "explained_variance_ratio" in landscape_data:
                    ax1.set_xlabel(
                        f'PC1 ({landscape_data["explained_variance_ratio"][0]:.1%} variance)'
                    )
                    ax1.set_ylabel(
                        f'PC2 ({landscape_data["explained_variance_ratio"][1]:.1%} variance)'
                    )
                else:
                    ax1.set_xlabel("PC1")
                    ax1.set_ylabel("PC2")

                ax1.legend(fontsize=8)

                # Add colorbar
                try:
                    cbar = plt.colorbar(contour, ax=ax1)
                    cbar.set_label("Energy", fontsize=8)
                except:
                    pass

                # 3D Surface Plot
                ax2 = plt.subplot(2, n_methods, idx + 1 + n_methods, projection="3d")

                try:
                    surf = ax2.plot_surface(
                        landscape_data["X"],
                        landscape_data["Y"],
                        Z_plot,
                        cmap="viridis",
                        alpha=0.7,
                        linewidth=0,
                        antialiased=True,
                    )

                    # Plot 3D trajectory
                    if len(trajectory) > 1:
                        traj_sample_indices = np.linspace(
                            0, len(trajectory) - 1, min(20, len(trajectory)), dtype=int
                        )
                        traj_sample_params = [
                            trajectory[i] for i in traj_sample_indices
                        ]
                        traj_sample_proj = (
                            traj_proj[traj_sample_indices]
                            if len(traj_proj) > len(traj_sample_indices)
                            else traj_proj
                        )

                        traj_energies = []
                        for params in traj_sample_params:
                            try:
                                energy = cost_function(params)
                                traj_energies.append(
                                    energy
                                    if np.isfinite(energy)
                                    else np.nanmean(Z_plot[finite_mask])
                                )
                            except:
                                traj_energies.append(
                                    np.nanmean(Z_plot[finite_mask])
                                    if np.any(finite_mask)
                                    else 0
                                )

                        if len(traj_sample_proj) >= len(traj_energies):
                            ax2.plot(
                                traj_sample_proj[: len(traj_energies), 0],
                                traj_sample_proj[: len(traj_energies), 1],
                                traj_energies,
                                "r-",
                                linewidth=3,
                                alpha=0.9,
                            )
                            ax2.scatter(
                                traj_sample_proj[0, 0],
                                traj_sample_proj[0, 1],
                                traj_energies[0],
                                c="red",
                                s=100,
                                marker="o",
                            )
                            ax2.scatter(
                                traj_sample_proj[len(traj_energies) - 1, 0],
                                traj_sample_proj[len(traj_energies) - 1, 1],
                                traj_energies[-1],
                                c="red",
                                s=100,
                                marker="*",
                            )

                except Exception as e:
                    print(f"  Error creating 3D surface: {e}")

                ax2.set_title(
                    f"{method_name}\n3D Loss Surface", fontsize=12, fontweight="bold"
                )
                if "explained_variance_ratio" in landscape_data:
                    ax2.set_xlabel(
                        f'PC1 ({landscape_data["explained_variance_ratio"][0]:.1%})'
                    )
                    ax2.set_ylabel(
                        f'PC2 ({landscape_data["explained_variance_ratio"][1]:.1%})'
                    )
                else:
                    ax2.set_xlabel("PC1")
                    ax2.set_ylabel("PC2")
                ax2.set_zlabel("Energy")

                print(f"  ✓ Landscape computed successfully for {method_name}")

            except Exception as e:
                print(f"  ✗ Error computing landscape for {method_name}: {e}")
                # Create error plots
                self._create_error_subplot(
                    fig, 2, n_methods, idx + 1, f"Error: {method_name}", str(e)
                )
                self._create_error_subplot(
                    fig,
                    2,
                    n_methods,
                    idx + 1 + n_methods,
                    f"Error: {method_name}",
                    "3D computation failed",
                )

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}")

        return fig

    def _create_error_subplot(self, fig, nrows, ncols, subplot_num, title, error_msg):
        """Helper method to create error message plots."""
        ax = plt.subplot(nrows, ncols, subplot_num)
        ax.text(
            0.5,
            0.5,
            f'{title}\n\nError: {error_msg[:100]}{"..." if len(error_msg) > 100 else ""}',
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7),
        )
        ax.set_title(title, fontsize=11, color="red")
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_layer_variance_scaling(
        self, scaling_data: pd.DataFrame, save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot variance scaling with ansatz layers.

        Args:
            `scaling_data`: DataFrame with layer scaling results
            `save_name`: Custom filename for saving plot

        Returns:
            Matplotlib figure object
        """
        if save_name is None:
            save_name = f"{self.molecular_context['filename_prefix']}_layer_variance_scaling.pdf"

        fig, ax = plt.subplots(figsize=(12, 8))

        methods = scaling_data["method"].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
        base_markers = ["o", "s", "^", "D", "v", "*", "p", "h", "+", "x"]
        markers = [base_markers[i % len(base_markers)] for i in range(len(methods))]

        for i, (method, color) in enumerate(zip(methods, colors)):
            method_data = scaling_data[scaling_data["method"] == method].sort_values(
                "num_layers"
            )
            if len(method_data) > 0:
                ax.semilogy(
                    method_data["num_layers"],
                    method_data["gradient_variance"],
                    marker=markers[i % len(markers)],
                    color=color,
                    label=method,
                    markersize=10,
                    linewidth=2.5,
                    alpha=0.8,
                )

        ax.set_xlabel("Number of Ansatz Layers", fontsize=14, fontweight="bold")
        ax.set_ylabel("Gradient Variance", fontsize=14, fontweight="bold")

        title = f'Gradient Variance vs Ansatz Depth: {self.molecular_context["title_prefix"]}'
        subtitle = f'{self.molecular_context["system_info"]}'
        ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight="bold")

        ax.set_xticks(sorted(scaling_data["num_layers"].unique()))
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        return fig

    def generate_all_plots(self, grid_size: int = 20):
        """
        Generate all visualization plots for molecular analysis.

        Args:
            `grid_size`: Grid resolution for landscape plots
        """
        print(
            f"Generating all visualizations for {self.molecular_context['title_prefix']}..."
        )

        generated_plots = []

        try:
            print("1. Energy convergence comparison...")
            self.plot_energy_convergence()
            generated_plots.append(
                f"{self.molecular_context['filename_prefix']}_energy_convergence.pdf"
            )
        except Exception as e:
            print(f"   Error: {e}")

        try:
            print("2. Performance metrics table...")
            self.create_performance_table()
            generated_plots.extend(
                [
                    f"{self.molecular_context['filename_prefix']}_performance_table.csv",
                    f"{self.molecular_context['filename_prefix']}_performance_table.tex",
                ]
            )
        except Exception as e:
            print(f"   Error: {e}")

        try:
            print("3. Gradient diagnostics...")
            self.plot_gradient_diagnostics()
            generated_plots.append(
                f"{self.molecular_context['filename_prefix']}_gradient_diagnostics.pdf"
            )
        except Exception as e:
            print(f"   Error: {e}")

        try:
            print("4. Variance scaling theory...")
            self.plot_variance_scaling_theory()
            generated_plots.append(
                f"{self.molecular_context['filename_prefix']}_variance_scaling_theory.pdf"
            )
        except Exception as e:
            print(f"   Error: {e}")

        # Only plot landscapes for reasonable system sizes
        if self.analyzer.num_qubits <= 8:
            try:
                print("5. Loss landscapes (selected methods)...")
                self.plot_loss_landscapes(num_methods_to_plot=3, grid_size=grid_size)
                generated_plots.append(
                    f"{self.molecular_context['filename_prefix']}_loss_landscapes.pdf"
                )
            except Exception as e:
                print(f"   Error in loss landscapes: {e}")
        else:
            print("5. Skipping loss landscapes (system too large)")

        print(f"\nPlots saved to: {self.output_dir}")
        print("Generated files:")

        for filename in generated_plots:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                print(f"   ✅ {filename}")
            else:
                print(f"   ❌ {filename} (not generated)")

        return True


# ============================================================================
# Standalone Plotting Functions
# ============================================================================


def plot_energy_convergence(
    analyzer, output_dir: str = "./plots", save_name: Optional[str] = None
) -> plt.Figure:
    """
    Standalone function to plot energy convergence.

    Args:
        `analyzer`: MolecularVQEAnalyzer instance
        `output_dir`: Directory to save plot
        `save_name`: Custom filename

    Returns:
        Matplotlib figure object
    """
    visualizer = MolecularLandscapeVisualizer(analyzer, output_dir)
    return visualizer.plot_energy_convergence(save_name)


def plot_loss_landscapes(
    analyzer,
    output_dir: str = "./plots",
    num_methods: Optional[int] = None,
    grid_size: int = 25,
    save_name: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Standalone function to plot loss landscapes.

    Args:
        `analyzer`: `MolecularVQEAnalyzer` instance
        `output_dir`: Directory to save plot
        `num_methods`: Number of methods to plot
        `grid_size`: Landscape resolution
        `save_name`: Custom filename

    Returns:
        Matplotlib figure object or None
    """
    visualizer = MolecularLandscapeVisualizer(analyzer, output_dir)
    return visualizer.plot_loss_landscapes(num_methods, grid_size, save_name)


def plot_gradient_diagnostics(
    analyzer, output_dir: str = "./plots", save_name: Optional[str] = None
) -> plt.Figure:
    """
    Standalone function to plot gradient diagnostics.

    Args:
        `analyzer`: MolecularVQEAnalyzer instance
        `output_dir`: Directory to save plot
        `save_name`: Custom filename

    Returns:
        Matplotlib figure object
    """
    visualizer = MolecularLandscapeVisualizer(analyzer, output_dir)
    return visualizer.plot_gradient_diagnostics(save_name)


def create_performance_table(analyzer, output_dir: str = "./plots") -> pd.DataFrame:
    """
    Standalone function to create performance table.

    Args:
        `analyzer`: `MolecularVQEAnalyzer` instance
        `output_dir`: Directory to save files

    Returns:
        Performance metrics DataFrame
    """
    visualizer = MolecularLandscapeVisualizer(analyzer, output_dir)
    return visualizer.create_performance_table()

