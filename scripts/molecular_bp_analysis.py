#!/usr/bin/env python3
"""
Molecular Barren Plateau Analysis Script
========================================

Comprehensive analysis of barren plateau mitigation techniques for molecular VQE systems.
Generates all plots, data, and analysis results for specified molecules.

Usage:
    python3 scripts/molecular_bp_analysis.py --molecule "H2"
    python3 scripts/molecular_bp_analysis.py --molecule "H2" --basis sto-3g --geometry equilibrium --iterations 1000
    python3 scripts/molecular_bp_analysis.py --molecule "H2O" --active-space 8 6 --iterations 800
    python3 scripts/molecular_bp_analysis.py --molecule "H2" --layer-scaling --max-layers 4
    python3 scripts/molecular_bp_analysis.py --molecule "LiH" --geometry stretched --basis 6-31g --freeze-core --iterations 1200

Features:
- Complete VQE method comparison
- Layer scaling analysis
- Loss landscape visualization
- Comprehensive diagnostic plots
- Statistical analysis and summaries
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add the package to Python path
script_dir = Path(__file__).parent.absolute()
root_dir = script_dir.parent
package_dir = root_dir / "barren_plateaus_vqe"
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(package_dir))

# Import package modules
try:
    from barren_plateaus_vqe.hamiltonian_builder import validate_molecular_system
    from barren_plateaus_vqe.molecular_analyzer import (
        create_molecular_analyzer,
        create_test_analyzer,
    )
    from barren_plateaus_vqe.viz_landscape import MolecularLandscapeVisualizer

    print("✅ Successfully imported barren_plateaus_vqe package")
except ImportError as e:
    print(f"❌ Error importing package: {e}")
    print("Make sure the barren_plateaus_vqe package is in the correct location")
    print("Required package structure:")
    print("  barren_plateaus_vqe/")
    print("    ├── molecular_analyzer.py")
    print("    ├── viz_landscape.py")
    print("    └── hamiltonian_builder.py")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive molecular barren plateau analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--molecule",
        type=str,
        required=True,
        help='Molecule to analyze ("H2", "LiH", "BeH2", "H2O", "N2", "CO")',
    )

    parser.add_argument(
        "--geometry",
        default="equilibrium",
        help="Molecular geometry (equilibrium, stretched, etc.)",
    )

    parser.add_argument("--basis", default="sto-3g", help="Basis set for calculation")

    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of VQE iterations per method",
    )

    parser.add_argument(
        "--active-space",
        nargs=2,
        type=int,
        metavar=("electrons", "orbitals"),
        help="Active space (num_electrons num_orbitals)",
    )

    parser.add_argument(
        "--freeze-core", action="store_true", help="Use frozen core approximation"
    )

    parser.add_argument(
        "--layer-scaling", action="store_true", help="Run layer scaling analysis"
    )

    parser.add_argument(
        "--max-layers", type=int, default=6, help="Maximum layers for scaling analysis"
    )

    parser.add_argument(
        "--landscape-grid",
        type=int,
        default=20,
        help="Grid size for loss landscape plots",
    )

    parser.add_argument(
        "--skip-landscapes",
        action="store_true",
        help="Skip loss landscape computation (faster)",
    )

    parser.add_argument(
        "--test-hamiltonian",
        action="store_true",
        help="Use test Hamiltonian instead of molecular",
    )

    parser.add_argument(
        "--output-suffix", default="", help="Additional suffix for output directory"
    )

    return parser.parse_args()


def setup_directories(args):
    """Setup output directories for data and plots."""

    # Create timestamp for unique runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine molecule name for directories
    if args.test_hamiltonian:
        mol_name = "test_hamiltonian"
    else:
        mol_name = f"{args.molecule}_{args.geometry}_{args.basis}"

    if args.output_suffix:
        mol_name += f"_{args.output_suffix}"

    # Setup directory structure - separate folders in root
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data" / mol_name / timestamp
    plots_dir = root_dir / "plots" / mol_name / timestamp

    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Plots directory: {plots_dir}")

    return data_dir, plots_dir


def create_analyzer(args):
    """Create molecular VQE analyzer based on arguments."""
    print("⚛️  Creating molecular VQE analyzer...")

    if args.test_hamiltonian:
        # Use test Hamiltonian
        analyzer = create_test_analyzer(num_qubits=6, num_layers=1)
        print(f"✅ Test analyzer created: {analyzer.num_qubits} qubits")
    else:
        # Convert active space to tuple if provided
        active_space = tuple(args.active_space) if args.active_space else None

        # Validate system first
        validation = validate_molecular_system(args.molecule, args.geometry, args.basis)
        if not validation["valid"]:
            print(f"❌ Invalid molecular system: {validation['warnings']}")
            return None

        if validation.get("estimated_qubits", 0) > 16:
            print(
                f"⚠️  Large system ({validation['estimated_qubits']} qubits) - consider active space"
            )
            if active_space is None:
                print("❌ Active space required for large systems")
                return None

        # Create analyzer
        analyzer = create_molecular_analyzer(
            molecule=args.molecule,
            geometry=args.geometry,
            basis=args.basis,
            freeze_core=args.freeze_core,
            active_space=active_space,
            num_layers=1,  # Will be modified for layer scaling
        )

        print(f"✅ Molecular analyzer created:")
        print(f"   Molecule: {analyzer.molecule_name}")
        print(f"   Qubits: {analyzer.num_qubits}")
        print(f"   Active space: {active_space}")

    return analyzer


def run_basic_analysis(analyzer, args, data_dir, plots_dir):
    """Run basic VQE method comparison analysis."""
    print("\n" + "=" * 70)
    print("BASIC VQE METHOD COMPARISON ANALYSIS")
    print("=" * 70)

    start_time = time.time()

    # Run complete analysis
    results = analyzer.run_complete_analysis(num_iters=args.iterations)

    analysis_time = time.time() - start_time
    print(f"\n✅ Analysis completed in {analysis_time:.1f} seconds")

    if not results:
        print("❌ No results generated")
        return None

    # Save results
    results_file = data_dir / "basic_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Results saved: {results_file}")

    # Generate summary table
    summary_data = []
    for method_name, result in results.items():
        try:
            bp_data = result["bp_diagnostics"]
            perf_data = result["performance_metrics"]

            summary_data.append(
                {
                    "Method": method_name,
                    "Energy_Error": perf_data["final_energy_error"],
                    "State_Fidelity": perf_data["state_fidelity"],
                    "Gradient_Variance": bp_data["gradient_variance"],
                    "Gradient_Norm_Mean": bp_data["gradient_norm_mean"],
                    "Status": "Success",
                }
            )
        except Exception as e:
            summary_data.append(
                {
                    "Method": method_name,
                    "Energy_Error": np.nan,
                    "State_Fidelity": np.nan,
                    "Gradient_Variance": np.nan,
                    "Gradient_Norm_Mean": np.nan,
                    "Status": f"Failed: {str(e)}",
                }
            )

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_file = data_dir / "method_comparison_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"💾 Summary saved: {summary_file}")

    # Print summary
    print("\n📊 RESULTS SUMMARY:")
    print("=" * 60)
    print(summary_df.to_string(index=False, float_format="%.2e"))

    # Generate visualizations
    print("\n📈 Generating visualizations...")
    try:
        visualizer = MolecularLandscapeVisualizer(analyzer, str(plots_dir))

        # Energy convergence
        visualizer.plot_energy_convergence()
        print("   ✅ Energy convergence plot")

        # Performance table
        visualizer.create_performance_table()
        print("   ✅ Performance table")

        # Gradient diagnostics
        visualizer.plot_gradient_diagnostics()
        print("   ✅ Gradient diagnostics")

        # Loss landscapes (if not skipped and system not too large)
        if not args.skip_landscapes and analyzer.num_qubits <= 10:
            visualizer.plot_loss_landscapes(
                # num_methods_to_plot=3,
                grid_size=args.landscape_grid
            )
            print("   ✅ Loss landscapes")
        else:
            print("   ⏭️  Loss landscapes skipped")

    except Exception as e:
        print(f"   ❌ Visualization error: {e}")

    return summary_df


def run_layer_scaling_analysis(base_analyzer, args, data_dir, plots_dir):
    """Run layer scaling analysis."""
    print("\n" + "=" * 70)
    print("LAYER SCALING ANALYSIS")
    print("=" * 70)

    layer_range = list(range(1, args.max_layers + 1))
    print(f"Layer range: {layer_range}")
    print(f"Base system: {base_analyzer.num_qubits} qubits")

    all_results = {}
    scaling_data = []

    for num_layers in layer_range:
        print(f"\n🔄 Analyzing {num_layers} layers...")

        try:
            # Create new analyzer with specified layers
            if args.test_hamiltonian:
                analyzer = create_test_analyzer(
                    num_qubits=base_analyzer.num_qubits, num_layers=num_layers
                )
            else:
                active_space = tuple(args.active_space) if args.active_space else None
                analyzer = create_molecular_analyzer(
                    molecule=args.molecule,
                    geometry=args.geometry,
                    basis=args.basis,
                    freeze_core=args.freeze_core,
                    active_space=active_space,
                    num_layers=num_layers,
                )

            # Run analysis with fewer iterations for speed
            iterations = min(args.iterations, 500)  # Cap at 500 for scaling studies
            results = analyzer.run_complete_analysis(num_iters=iterations)

            if results:
                all_results[num_layers] = results

                # Extract metrics for scaling analysis
                for method_name, result in results.items():
                    try:
                        bp_data = result["bp_diagnostics"]
                        perf_data = result["performance_metrics"]

                        scaling_data.append(
                            {
                                "num_layers": num_layers,
                                "method": method_name,
                                "num_qubits": analyzer.num_qubits,
                                "gradient_variance": bp_data["gradient_variance"],
                                "gradient_norm_mean": bp_data["gradient_norm_mean"],
                                "final_energy_error": perf_data["final_energy_error"],
                                "state_fidelity": perf_data["state_fidelity"],
                                "num_parameters": len(
                                    result["method_results"]["final_params"]
                                ),
                            }
                        )
                    except Exception as e:
                        print(f"     ⚠️  Error extracting {method_name}: {e}")

                print(f"   ✅ Success: {len(results)} methods")
            else:
                print(f"   ❌ Failed")
                all_results[num_layers] = None

        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_results[num_layers] = None

    # Convert to DataFrame
    scaling_df = pd.DataFrame(scaling_data)

    if scaling_df.empty:
        print("❌ No scaling data generated")
        return None

    # Save scaling data
    scaling_file = data_dir / "layer_scaling_data.csv"
    scaling_df.to_csv(scaling_file, index=False)
    print(f"💾 Scaling data saved: {scaling_file}")

    # Save detailed results
    detailed_file = data_dir / "layer_scaling_detailed.json"
    with open(detailed_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"💾 Detailed results saved: {detailed_file}")

    # Generate scaling plots
    print("\n📈 Generating scaling plots...")

    # Plot 1: Gradient variance vs layers
    plt.figure(figsize=(12, 8))
    methods = scaling_df["method"].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    markers = ["o", "s", "^", "D", "v", "*", "p", "h"]

    for i, method in enumerate(methods):
        method_data = scaling_df[scaling_df["method"] == method].sort_values(
            "num_layers"
        )
        if len(method_data) > 0:
            plt.semilogy(
                method_data["num_layers"],
                method_data["gradient_variance"],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=method,
                markersize=8,
                linewidth=2,
                alpha=0.8,
            )

    plt.xlabel("Number of Ansatz Layers", fontsize=14, fontweight="bold")
    plt.ylabel("Gradient Variance", fontsize=14, fontweight="bold")

    if args.test_hamiltonian:
        title = f"Gradient Variance vs Ansatz Depth (Test Hamiltonian, {base_analyzer.num_qubits} qubits)"
    else:
        title = f"Gradient Variance vs Ansatz Depth ({args.molecule}, {base_analyzer.num_qubits} qubits)"

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xticks(layer_range)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    variance_plot = plots_dir / "layer_variance_scaling.pdf"
    plt.savefig(variance_plot, dpi=300, bbox_inches="tight")
    print(f"   ✅ Variance scaling plot: {variance_plot}")
    plt.close()

    # Plot 2: Parameter count vs layers
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        method_data = scaling_df[scaling_df["method"] == method].sort_values(
            "num_layers"
        )
        if len(method_data) > 0:
            plt.plot(
                method_data["num_layers"],
                method_data["num_parameters"],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=method,
                markersize=8,
                linewidth=2,
                alpha=0.8,
            )

    plt.xlabel("Number of Ansatz Layers", fontsize=14)
    plt.ylabel("Number of Parameters", fontsize=14)
    plt.title(f"Parameter Count vs Ansatz Depth", fontsize=16)
    plt.xticks(layer_range)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    param_plot = plots_dir / "parameter_scaling.pdf"
    plt.savefig(param_plot, dpi=300, bbox_inches="tight")
    print(f"   ✅ Parameter scaling plot: {param_plot}")
    plt.close()

    # Plot 3: Energy error vs layers
    plt.figure(figsize=(10, 6))
    for i, method in enumerate(methods):
        method_data = scaling_df[scaling_df["method"] == method].sort_values(
            "num_layers"
        )
        if len(method_data) > 0:
            plt.semilogy(
                method_data["num_layers"],
                method_data["final_energy_error"],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=method,
                markersize=8,
                linewidth=2,
                alpha=0.8,
            )

    plt.xlabel("Number of Ansatz Layers", fontsize=14)
    plt.ylabel("Final Energy Error", fontsize=14)
    plt.title(f"Energy Error vs Ansatz Depth", fontsize=16)
    plt.xticks(layer_range)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    energy_plot = plots_dir / "energy_error_scaling.pdf"
    plt.savefig(energy_plot, dpi=300, bbox_inches="tight")
    print(f"   ✅ Energy error scaling plot: {energy_plot}")
    plt.close()

    # Create summary table
    try:
        variance_pivot = scaling_df.pivot(
            index="num_layers", columns="method", values="gradient_variance"
        )
        summary_file = plots_dir / "layer_scaling_summary.csv"
        variance_pivot.to_csv(summary_file)
        print(f"   ✅ Summary table: {summary_file}")

        print(f"\n📊 LAYER SCALING SUMMARY:")
        print("=" * 60)
        print(variance_pivot.to_string(float_format="%.2e"))

    except Exception as e:
        print(f"   ⚠️  Summary table error: {e}")

    return scaling_df


def generate_final_report(args, data_dir, basic_summary=None, scaling_summary=None):
    """Generate final analysis report."""
    print("\n📋 Generating final report...")

    report_file = data_dir / "analysis_report.md"

    with open(report_file, "w") as f:
        f.write(f"# Molecular Barren Plateau Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # System information
        f.write(f"## System Information\n\n")
        if args.test_hamiltonian:
            f.write(f"- **System:** Test Hamiltonian\n")
        else:
            f.write(f"- **Molecule:** {args.molecule}\n")
            f.write(f"- **Geometry:** {args.geometry}\n")
            f.write(f"- **Basis Set:** {args.basis}\n")
            if args.active_space:
                f.write(
                    f"- **Active Space:** ({args.active_space[0]}e, {args.active_space[1]}o)\n"
                )
            f.write(f"- **Frozen Core:** {args.freeze_core}\n")
        f.write(f"- **Iterations:** {args.iterations}\n\n")

        # Analysis configuration
        f.write(f"## Analysis Configuration\n\n")
        f.write(f"- **Layer Scaling:** {args.layer_scaling}\n")
        if args.layer_scaling:
            f.write(f"- **Max Layers:** {args.max_layers}\n")
        f.write(f"- **Landscape Grid:** {args.landscape_grid}\n")
        f.write(f"- **Skip Landscapes:** {args.skip_landscapes}\n\n")

        # Basic analysis results
        if basic_summary is not None:
            f.write(f"## VQE Method Comparison\n\n")
            successful_methods = basic_summary[basic_summary["Status"] == "Success"]
            f.write(
                f"**Successful Methods:** {len(successful_methods)}/{len(basic_summary)}\n\n"
            )

            if len(successful_methods) > 0:
                best_idx = successful_methods["Energy_Error"].idxmin()
                best_method = successful_methods.loc[best_idx]
                f.write(f"**Best Performing Method:** {best_method['Method']}\n")
                f.write(f"- Energy Error: {best_method['Energy_Error']:.2e}\n")
                f.write(f"- State Fidelity: {best_method['State_Fidelity']:.3f}\n")
                f.write(
                    f"- Gradient Variance: {best_method['Gradient_Variance']:.2e}\n\n"
                )

        # Scaling analysis results
        if scaling_summary is not None:
            f.write(f"## Layer Scaling Analysis\n\n")
            f.write(
                f"**Layers Analyzed:** {sorted(scaling_summary['num_layers'].unique())}\n"
            )
            f.write(f"**Methods:** {list(scaling_summary['method'].unique())}\n\n")

            # Analyze scaling trends
            f.write(f"### Scaling Trends\n\n")
            for method in scaling_summary["method"].unique():
                method_data = scaling_summary[
                    scaling_summary["method"] == method
                ].sort_values("num_layers")
                if len(method_data) > 1:
                    first_var = method_data.iloc[0]["gradient_variance"]
                    last_var = method_data.iloc[-1]["gradient_variance"]
                    ratio = last_var / first_var
                    trend = "decreasing (better)" if ratio < 1 else "increasing (worse)"
                    f.write(f"- **{method}:** {ratio:.1f}x change ({trend})\n")

        # Files generated
        f.write(f"\n## Generated Files\n\n")
        f.write(f"### Data Files\n")
        for file_path in data_dir.glob("*.csv"):
            f.write(f"- `{file_path.name}`\n")
        for file_path in data_dir.glob("*.json"):
            f.write(f"- `{file_path.name}`\n")

        f.write(f"\n### Plot Files\n")
        plots_dir = (
            data_dir.parent.parent.parent
            / "plots"
            / data_dir.parent.name
            / data_dir.name
        )
        if plots_dir.exists():
            for file_path in plots_dir.glob("*.pdf"):
                f.write(f"- `{file_path.name}`\n")
            for file_path in plots_dir.glob("*.png"):
                f.write(f"- `{file_path.name}`\n")

    print(f"📋 Report saved: {report_file}")


def main():
    """Main analysis function."""
    print("🧪 Molecular Barren Plateau Analysis")
    print("=" * 70)

    # Parse arguments
    args = parse_arguments()

    # Display configuration
    print(f"\n📋 Configuration:")
    if args.test_hamiltonian:
        print(f"   System: Test Hamiltonian")
    else:
        print(f"   Molecule: {args.molecule}")
        print(f"   Geometry: {args.geometry}")
        print(f"   Basis: {args.basis}")
        if args.active_space:
            print(
                f"   Active space: ({args.active_space[0]}e, {args.active_space[1]}o)"
            )
    print(f"   Iterations: {args.iterations}")
    print(f"   Layer scaling: {args.layer_scaling}")
    if args.layer_scaling:
        print(f"   Max layers: {args.max_layers}")

    # Setup directories
    data_dir, plots_dir = setup_directories(args)

    # Save configuration
    config_file = data_dir / "analysis_config.json"
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"💾 Configuration saved: {config_file}")

    # Create analyzer
    analyzer = create_analyzer(args)
    if analyzer is None:
        print("❌ Failed to create analyzer")
        return 1

    basic_summary = None
    scaling_summary = None

    try:
        # Run basic analysis
        basic_summary = run_basic_analysis(analyzer, args, data_dir, plots_dir)

        # Run layer scaling analysis if requested
        if args.layer_scaling:
            scaling_summary = run_layer_scaling_analysis(
                analyzer, args, data_dir, plots_dir
            )

        # Generate final report
        generate_final_report(args, data_dir, basic_summary, scaling_summary)

        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Data: {data_dir}")
        print(f"📊 Plots: {plots_dir}")

        return 0

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
