"""
Visualization Module
====================

This module provides comprehensive visualization tools for analyzing barren plateau 
phenomena in Variational Quantum Eigensolver (VQE) algorithms. It implements 
state-of-the-art visualization techniques adapted from machine learning and 
optimization literature for quantum computing applications.

Visualization Techniques Implemented:

1. Energy Convergence Analysis:
   - Time-series plots of energy vs optimization iteration
   - Comparison across multiple VQE variants
   - Exact ground state energy reference lines

2. Loss Landscape Visualization:
   - PCA-based 2D projections of high-dimensional landscapes
   - 3D surface plots showing energy topology
   - Optimization trajectory overlays showing actual paths taken

3. Gradient Landscape Analysis:
   - Heat maps of gradient magnitudes across parameter space
   - Logarithmic scaling for barren plateau identification

4. Statistical Distribution Analysis:
   - Histograms of gradient norms, cost values, and curvature measures
   - Statistical overlays (mean, standard deviation markers)
   - Multi-method comparison in unified grid layout

5. Barren Plateau Diagnostics:
   - Gradient variance bar charts 
   - Energy error and state fidelity comparisons
   - Performance correlation analysis

Author: Mostafa Atallah and Nouhaila Innan
Date: 2025
Version: 1.1.0 (Fixed)
License: Apache
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import os
from qiskit_aer import AerSimulator
from qubap.qiskit.variational_algorithms import energy_evaluation

try:
    from .landscape_utils import (
        compute_loss_landscape_2d, 
        compute_loss_landscape_pca, 
        create_compatible_grid,
        compute_gradient_norm_function,
        safe_pca_transform
    )
except ImportError:
    # Handle case where landscape_utils is not available
    print("Warning: landscape_utils not available, some functionality will be limited")
    compute_loss_landscape_pca = None
    compute_loss_landscape_2d = None
    create_compatible_grid = None
    compute_gradient_norm_function = None
    safe_pca_transform = None


class BarrenPlateauVisualizer:
    """Fixed visualization class for barren plateau analysis."""
    
    def __init__(self, analyzer, output_dir: str = "./plots"):
        self.analyzer = analyzer
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Plots will be saved to: {self.output_dir}")
    
    def plot_energy_convergence(self, save_name: str = "energy_convergence_comparison.pdf"):
        """Plot energy convergence curves for all methods."""
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, (method_name, data) in enumerate(self.analyzer.results.items()):
            energies = data['method_results']['energies']
            plt.plot(energies, color=colors[i % len(colors)], label=method_name, linewidth=2)
        
        plt.axhline(y=self.analyzer.exact_min_energy, color='black', linestyle='--', 
                   label='Exact Ground State', linewidth=2)
        
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel(r'$\langle H \rangle$', fontsize=12)
        plt.title('Energy Convergence Comparison', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        
        return plt.gcf()
    
    def create_performance_table(self):
        """Create and display performance metrics table, save as LaTeX."""
        # Collect data for the table
        methods = []
        energy_errors = []
        fidelities = []
        gradient_variances = []
        gradient_norms = []
        
        for method_name, data in self.analyzer.results.items():
            methods.append(method_name)
            energy_errors.append(data['performance_metrics']['final_energy_error'])
            fidelities.append(data['performance_metrics']['state_fidelity'])
            gradient_variances.append(data['bp_diagnostics']['gradient_variance'])
            gradient_norms.append(data['bp_diagnostics']['gradient_norm_mean'])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Method': methods,
            'Energy Error': energy_errors,
            'State Fidelity': fidelities,
            'Gradient Variance': gradient_variances,
            'Gradient Norm': gradient_norms
        })
        
        # Create formatted DataFrame for display
        df_display = df.copy()
        df_display['Energy Error'] = df_display['Energy Error'].apply(lambda x: f'{x:.2e}')
        df_display['State Fidelity'] = df_display['State Fidelity'].apply(lambda x: f'{x:.3f}')
        df_display['Gradient Variance'] = df_display['Gradient Variance'].apply(lambda x: f'{x:.2e}')
        df_display['Gradient Norm'] = df_display['Gradient Norm'].apply(lambda x: f'{x:.2e}')
        
        # Print to console
        print("\n" + "="*80)
        print("PERFORMANCE METRICS TABLE")
        print("="*80)
        print(df_display.to_string(index=False))
        print("="*80 + "\n")
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, "performance_metrics_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Save as LaTeX
        latex_path = os.path.join(self.output_dir, "performance_metrics_table.tex")
        
        # Create LaTeX table with better formatting
        latex_content = r"""\begin{table}[h!]
\centering
\caption{Performance Metrics Summary for VQE Methods}
\label{tab:performance_metrics}
\begin{tabular}{lcccc}
\hline
\textbf{Method} & \textbf{Energy Error} & \textbf{State Fidelity} & \textbf{Gradient Variance} & \textbf{Gradient Norm} \\
\hline
"""
        
        for _, row in df.iterrows():
            method = row['Method'].replace('_', r'\_')
            latex_content += f"{method} & ${row['Energy Error']:.2e}$ & ${row['State Fidelity']:.3f}$ & ${row['Gradient Variance']:.2e}$ & ${row['Gradient Norm']:.2e}$ \\\\\n"
        
        latex_content += r"""\hline
\end{tabular}
\end{table}"""
        
        with open(latex_path, 'w') as f:
            f.write(latex_content)
        
        print(f"Saved LaTeX: {latex_path}")
        
        return df
    
    def plot_variance_vs_layers(self, save_name: str = "variance_vs_layers.pdf"):
        """Plot gradient variance vs number of layers."""
        plt.figure(figsize=(8, 6))
        
        # Extract layer information from method names/ansatzes if possible
        methods_with_layers = []
        for method_name, data in self.analyzer.results.items():
            ansatz = data['method_results']['ansatz']
            # Try to extract layer count
            if hasattr(ansatz, 'reps'):
                n_layers = ansatz.reps
            elif 'SEA' in method_name:
                n_layers = 1  # From the ansatz_constructor deep parameter
            else:
                n_layers = 0  # Default for simple ansatzes
            
            variance = data['bp_diagnostics']['gradient_variance']
            methods_with_layers.append((method_name, n_layers, variance))
        
        # Group by layer count
        layer_data = {}
        for method, layers, var in methods_with_layers:
            if layers not in layer_data:
                layer_data[layers] = []
            layer_data[layers].append((method, var))
        
        if layer_data:
            layers = sorted(layer_data.keys())
            mean_vars = []
            std_vars = []
            
            for l in layers:
                vars_at_layer = [v for _, v in layer_data[l]]
                mean_vars.append(np.mean(vars_at_layer))
                std_vars.append(np.std(vars_at_layer) if len(vars_at_layer) > 1 else 0)
            
            plt.errorbar(layers, mean_vars, yerr=std_vars, marker='o', markersize=10, 
                        capsize=5, capthick=2, linewidth=2, color='darkblue')
            
            # Add method labels
            for l in layers:
                for i, (method, var) in enumerate(layer_data[l]):
                    offset = i * 0.02 - 0.01 * (len(layer_data[l]) - 1) / 2
                    plt.annotate(method, (l + offset, var), fontsize=8, ha='center', va='bottom')
        
        plt.xlabel('Number of Layers', fontsize=14)
        plt.ylabel('Gradient Variance', fontsize=14)
        plt.title('Gradient Variance vs Circuit Depth', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        
        return plt.gcf()
    
    def plot_variance_scaling_theory(self, save_name: str = "variance_scaling_theory.pdf"):
        """Plot theoretical variance scaling with number of qubits."""
        plt.figure(figsize=(8, 6))
        
        # Show theoretical exponential scaling
        n_qubits_range = np.arange(2, 12)
        
        # Different theoretical scaling rates
        alpha_values = [0.5, 0.7, 1.0]
        colors = ['blue', 'green', 'red']
        
        for alpha, color in zip(alpha_values, colors):
            theoretical_variance = 1.0 * np.exp(-alpha * n_qubits_range)
            plt.plot(n_qubits_range, theoretical_variance, '--', 
                    label=f'Theory: exp(-{alpha}n)', linewidth=2, color=color)
        
        # Add our actual data point
        actual_variance = np.mean([data['bp_diagnostics']['gradient_variance'] 
                                  for data in self.analyzer.results.values()])
        plt.scatter([self.analyzer.num_qubits], [actual_variance], 
                   s=200, c='red', marker='*', label=f'Actual ({self.analyzer.num_qubits} qubits)', 
                   zorder=5, edgecolors='black', linewidth=2)
        
        # Add error bars for actual variance
        variance_std = np.std([data['bp_diagnostics']['gradient_variance'] 
                              for data in self.analyzer.results.values()])
        plt.errorbar([self.analyzer.num_qubits], [actual_variance], yerr=variance_std,
                    fmt='none', ecolor='red', capsize=5, capthick=2)
        
        plt.xlabel('Number of Qubits', fontsize=14)
        plt.ylabel('Gradient Variance', fontsize=14)
        plt.title('Theoretical Variance Scaling with System Size', fontsize=16, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Add annotation about barren plateau
        plt.annotate('Barren Plateau Region', 
                    xy=(10, 1e-6), xytext=(8, 1e-4),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                    fontsize=11, color='gray')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        
        return plt.gcf()
    
    def plot_variance_scaling_comparison(self, scaling_data: pd.DataFrame, 
                                       save_name: str = "variance_scaling_comparison.pdf"):
        """Plot variance scaling comparison across different qubit numbers."""
        plt.figure(figsize=(12, 8))
        
        # Get unique methods
        methods = scaling_data['method'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h']
        
        # Plot each method
        for i, (method, color) in enumerate(zip(methods, colors)):
            method_data = scaling_data[scaling_data['method'] == method].sort_values('num_qubits')
            
            plt.semilogy(method_data['num_qubits'], 
                        method_data['gradient_variance'], 
                        marker=markers[i % len(markers)],
                        color=color,
                        markersize=10,
                        linewidth=2.5,
                        label=method,
                        alpha=0.8)
        
        # Add theoretical scaling references
        qubits = scaling_data['num_qubits'].unique()
        qubits.sort()
        
        # Plot theoretical lines with different alphas
        alpha_values = [0.5, 0.7, 1.0]
        linestyles = [':', '--', '-.']
        
        for alpha, ls in zip(alpha_values, linestyles):
            theoretical = 0.01 * np.exp(-alpha * qubits)
            plt.semilogy(qubits, theoretical, 
                        linestyle=ls, 
                        color='gray',
                        linewidth=2,
                        alpha=0.6,
                        label=f'exp(-{alpha}n)')
        
        plt.xlabel('Number of Qubits', fontsize=16)
        plt.ylabel('Gradient Variance', fontsize=16)
        plt.title('Barren Plateau Scaling: Gradient Variance vs System Size', 
                 fontsize=18, fontweight='bold')
        plt.legend(fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.3, which='both')
        plt.xticks(qubits)
        
        # Add annotation
        plt.annotate('Exponential suppression\nof gradients', 
                    xy=(max(qubits)-1, min(scaling_data['gradient_variance'])*2),
                    xytext=(max(qubits)-2, min(scaling_data['gradient_variance'])*20),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=12, color='red')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        
        return plt.gcf()
    
    def plot_loss_landscapes(self, num_methods_to_plot: int = None, grid_size: int = 25,
                           save_name: str = "loss_landscapes.pdf"):
        """Plot 2D loss landscapes for all methods (with robust error handling)."""
        
        if compute_loss_landscape_pca is None:
            print("Loss landscape computation not available (landscape_utils not imported)")
            return None
            
        # Get all methods
        method_names = list(self.analyzer.results.keys())
        
        selected_methods = method_names[:num_methods_to_plot] if num_methods_to_plot else method_names
        print(f"Plotting loss landscapes for {len(selected_methods)} methods")
        
        n_methods = len(selected_methods)
        fig = plt.figure(figsize=(6*n_methods, 12))
        
        for idx, method_name in enumerate(selected_methods):
            print(f"Processing method {idx+1}/{n_methods}: {method_name}")
            
            method_data = self.analyzer.results[method_name]['method_results']
            ansatz = method_data['ansatz']
            trajectory = method_data['trajectory']
            final_params = method_data['final_params']
            
            # Define cost function
            backend = AerSimulator(method="statevector")
            def cost_function(params):
                try:
                    return energy_evaluation(self.analyzer.H, ansatz, params, backend)
                except Exception as e:
                    print(f"Warning: Cost function evaluation failed for {method_name}: {e}")
                    return np.inf
            
            # Compute 2D loss landscape using PCA projection
            try:
                print(f"  Computing PCA-based landscape for {method_name}...")
                
                # Handle empty trajectory case
                if not trajectory or len(trajectory) == 0:
                    print(f"  Warning: Empty trajectory for {method_name}, using final params only")
                    trajectory = [final_params]
                
                landscape_data = compute_loss_landscape_pca(
                    cost_function, trajectory, grid_size=grid_size, scale_factor=1.5
                )
                
                # Create 2D contour plot
                ax1 = plt.subplot(2, n_methods, idx + 1)
                
                # Plot contour map
                Z_plot = landscape_data['Z']
                # Handle infinite values more robustly
                finite_mask = np.isfinite(Z_plot)
                if np.any(finite_mask):
                    Z_plot = np.where(finite_mask, Z_plot, np.nanmax(Z_plot[finite_mask]))
                else:
                    print(f"  Warning: No finite values in landscape for {method_name}")
                    Z_plot = np.zeros_like(Z_plot)
                
                try:
                    contour = ax1.contourf(landscape_data['X'], landscape_data['Y'], Z_plot, 
                                         levels=50, cmap='viridis', alpha=0.8)
                    
                    # Add contour lines
                    ax1.contour(landscape_data['X'], landscape_data['Y'], Z_plot, 
                               levels=20, colors='white', alpha=0.5, linewidths=0.8)
                except Exception as e:
                    print(f"  Warning: Could not create contour plot for {method_name}: {e}")
                
                # Plot optimization trajectory
                traj_proj = landscape_data['trajectory_projected']
                if len(traj_proj) > 1:
                    ax1.plot(traj_proj[:, 0], traj_proj[:, 1], 'r-', linewidth=2, alpha=0.8, 
                            label='Optimization Path')
                    ax1.scatter(traj_proj[0, 0], traj_proj[0, 1], c='red', s=100, marker='o', 
                              label='Start', zorder=5)
                    ax1.scatter(traj_proj[-1, 0], traj_proj[-1, 1], c='red', s=100, marker='*', 
                              label='End', zorder=5)
                else:
                    # Single point
                    ax1.scatter(traj_proj[0, 0], traj_proj[0, 1], c='red', s=100, marker='*', 
                              label='Final', zorder=5)
                
                ax1.set_title(f'{method_name}\nLoss Landscape (PCA Projection)', 
                            fontsize=12, fontweight='bold')
                
                if 'explained_variance_ratio' in landscape_data:
                    ax1.set_xlabel(f'PC1 ({landscape_data["explained_variance_ratio"][0]:.1%} variance)')
                    ax1.set_ylabel(f'PC2 ({landscape_data["explained_variance_ratio"][1]:.1%} variance)')
                else:
                    ax1.set_xlabel('PC1')
                    ax1.set_ylabel('PC2')
                    
                ax1.legend(fontsize=10)
                
                # Add colorbar
                try:
                    cbar = plt.colorbar(contour, ax=ax1)
                    cbar.set_label('Energy', fontsize=10)
                except:
                    print(f"  Could not add colorbar for {method_name}")
                
                # Create 3D surface plot
                ax2 = plt.subplot(2, n_methods, idx + 1 + n_methods, projection='3d')
                
                # Plot 3D surface
                try:
                    surf = ax2.plot_surface(landscape_data['X'], landscape_data['Y'], Z_plot, 
                                          cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
                    
                    # Plot trajectory in 3D if we have multiple points
                    if len(trajectory) > 1:
                        # Sample trajectory points for 3D visualization
                        traj_sample_indices = np.linspace(0, len(trajectory)-1, min(20, len(trajectory)), dtype=int)
                        traj_sample_params = [trajectory[i] for i in traj_sample_indices]
                        traj_sample_proj = traj_proj[traj_sample_indices] if len(traj_proj) > len(traj_sample_indices) else traj_proj
                        
                        traj_energies = []
                        for params in traj_sample_params:
                            try:
                                energy = cost_function(params)
                                traj_energies.append(energy if np.isfinite(energy) else np.nanmean(Z_plot[finite_mask]))
                            except:
                                traj_energies.append(np.nanmean(Z_plot[finite_mask]) if np.any(finite_mask) else 0)
                        
                        if len(traj_sample_proj) >= len(traj_energies):
                            ax2.plot(traj_sample_proj[:len(traj_energies), 0], 
                                    traj_sample_proj[:len(traj_energies), 1], 
                                    traj_energies, 
                                    'r-', linewidth=3, alpha=0.9, label='Optimization Path')
                            ax2.scatter(traj_sample_proj[0, 0], traj_sample_proj[0, 1], traj_energies[0], 
                                      c='red', s=100, marker='o', label='Start')
                            ax2.scatter(traj_sample_proj[len(traj_energies)-1, 0], 
                                      traj_sample_proj[len(traj_energies)-1, 1], 
                                      traj_energies[-1], 
                                      c='red', s=100, marker='*', label='End')
                    
                except Exception as e:
                    print(f"  Error creating 3D surface for {method_name}: {e}")
                
                ax2.set_title(f'{method_name}\n3D Loss Surface', fontsize=12, fontweight='bold')
                if 'explained_variance_ratio' in landscape_data:
                    ax2.set_xlabel(f'PC1 ({landscape_data["explained_variance_ratio"][0]:.1%})')
                    ax2.set_ylabel(f'PC2 ({landscape_data["explained_variance_ratio"][1]:.1%})')
                else:
                    ax2.set_xlabel('PC1')
                    ax2.set_ylabel('PC2')
                ax2.set_zlabel('Energy')
                
                print(f"  ✓ Loss landscape computed successfully for {method_name}")
                
            except Exception as e:
                print(f"  ✗ Error computing loss landscape for {method_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Create placeholder plots with error message
                ax1 = plt.subplot(2, n_methods, idx + 1)
                self._create_error_plot(ax1, f'Error computing\nlandscape for\n{method_name}', str(e))
                
                ax2 = plt.subplot(2, n_methods, idx + 1 + n_methods, projection='3d')
                ax2.text(0.5, 0.5, 0.5, f'Error computing\nlandscape for\n{method_name}\n\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
                ax2.set_title(f'{method_name}\n3D Loss Surface (Error)', fontsize=12, color='red')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        
        return fig
    
    def _create_error_plot(self, ax, title: str, error_msg: str):
        """Helper method to create error message plots."""
        ax.text(0.5, 0.5, f'{title}\n\nError: {error_msg[:100]}{"..." if len(error_msg) > 100 else ""}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
        ax.set_title(title, fontsize=11, color='red')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def plot_gradient_diagnostics(self, save_name: str = "gradient_diagnostics.pdf"):
        """Plot comprehensive gradient diagnostics for all methods."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Collect data
        methods = list(self.analyzer.results.keys())
        gradient_variances = [self.analyzer.results[m]['bp_diagnostics']['gradient_variance'] for m in methods]
        gradient_norms = [self.analyzer.results[m]['bp_diagnostics']['gradient_norm_mean'] for m in methods]
        energy_errors = [self.analyzer.results[m]['performance_metrics']['final_energy_error'] for m in methods]
        fidelities = [self.analyzer.results[m]['performance_metrics']['state_fidelity'] for m in methods]
        
        # Plot 1: Gradient Variance Bar Chart
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(methods)), gradient_variances, color=['blue', 'green', 'red', 'orange', 'purple'][:len(methods)])
        ax1.set_yscale('log')
        ax1.set_title('Gradient Variance by Method', fontweight='bold')
        ax1.set_ylabel('Gradient Variance')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Gradient Norm vs Energy Error
        ax2 = axes[0, 1]
        scatter = ax2.scatter(gradient_norms, energy_errors, s=100, alpha=0.7, 
                             c=range(len(methods)), cmap='viridis')
        for i, method in enumerate(methods):
            ax2.annotate(method, (gradient_norms[i], energy_errors[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Gradient Norm Mean')
        ax2.set_ylabel('Final Energy Error')
        ax2.set_title('Gradient Norm vs Energy Error', fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Variance vs Fidelity
        ax3 = axes[1, 0]
        scatter2 = ax3.scatter(gradient_variances, fidelities, s=100, alpha=0.7,
                              c=range(len(methods)), cmap='viridis')
        for i, method in enumerate(methods):
            ax3.annotate(method, (gradient_variances[i], fidelities[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax3.set_xlabel('Gradient Variance')
        ax3.set_ylabel('State Fidelity')
        ax3.set_title('Gradient Variance vs State Fidelity', fontweight='bold')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Summary
        ax4 = axes[1, 1]
        # Normalize metrics for comparison
        norm_variances = np.array(gradient_variances) / max(gradient_variances)
        norm_errors = np.array(energy_errors) / max(energy_errors)
        norm_fidelities = 1 - np.array(fidelities)  # Invert so lower is better
        
        x = np.arange(len(methods))
        width = 0.25
        
        ax4.bar(x - width, norm_variances, width, label='Grad Variance (norm)', alpha=0.7)
        ax4.bar(x, norm_errors, width, label='Energy Error (norm)', alpha=0.7)
        ax4.bar(x + width, norm_fidelities, width, label='1 - Fidelity', alpha=0.7)
        
        ax4.set_title('Normalized Performance Comparison', fontweight='bold')
        ax4.set_ylabel('Normalized Metric (lower is better)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(methods, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
        
        return fig
    
    def generate_all_plots(self, num_iters: int = 200, grid_size: int = 20):
        """Generate all visualization plots (fixed version)."""
        print("Generating all visualizations...")
        
        generated_plots = []
        
        try:
            print("1. Energy convergence comparison...")
            self.plot_energy_convergence()
            generated_plots.append("energy_convergence_comparison.pdf")
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("2. Performance metrics table (console + LaTeX)...")
            self.create_performance_table()
            generated_plots.extend(["performance_metrics_table.csv", "performance_metrics_table.tex"])
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("3. Variance vs layers...")
            self.plot_variance_vs_layers()
            generated_plots.append("variance_vs_layers.pdf")
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("4. Variance scaling theory...")
            self.plot_variance_scaling_theory()
            generated_plots.append("variance_scaling_theory.pdf")
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("5. Gradient diagnostics...")
            self.plot_gradient_diagnostics()
            generated_plots.append("gradient_diagnostics.pdf")
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("6. Loss landscapes (selected methods)...")
            self.plot_loss_landscapes(num_methods_to_plot=3, grid_size=grid_size)  # Limit to 3 methods
            generated_plots.append("loss_landscapes.pdf")
        except Exception as e:
            print(f"   Error in loss landscapes: {e}")
        
        print(f"\nPlots saved to: {self.output_dir}")
        print("\nGenerated files:")
        
        for filename in generated_plots:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                print(f"   ✅ {filename}")
            else:
                print(f"   ❌ {filename} (not generated)")
        
        return True
    
    def generate_basic_plots(self):
        """Generate basic visualization plots only (most robust)."""
        print("Generating basic visualizations...")
        
        generated_plots = []
        
        try:
            print("1. Energy convergence comparison...")
            self.plot_energy_convergence()
            generated_plots.append("energy_convergence_comparison.pdf")
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("2. Performance metrics table...")
            self.create_performance_table()
            generated_plots.extend(["performance_metrics_table.csv", "performance_metrics_table.tex"])
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("3. Variance vs layers...")
            self.plot_variance_vs_layers()
            generated_plots.append("variance_vs_layers.pdf")
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("4. Variance scaling theory...")
            self.plot_variance_scaling_theory()
            generated_plots.append("variance_scaling_theory.pdf")
        except Exception as e:
            print(f"   Error: {e}")
        
        try:
            print("5. Gradient diagnostics...")
            self.plot_gradient_diagnostics()
            generated_plots.append("gradient_diagnostics.pdf")
        except Exception as e:
            print(f"   Error: {e}")
        
        print(f"\nBasic plots saved to: {self.output_dir}")
        print("Generated files:")
        
        for filename in generated_plots:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                print(f"   ✅ {filename}")
            else:
                print(f"   ❌ {filename} (not generated)")
        
        return True