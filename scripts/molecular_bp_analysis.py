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
from typing import Any, List, Optional, Tuple

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


def apply_molecule_name_patch():
    """Apply comprehensive patch to fix molecule name case sensitivity issues."""
    try:
        # Import the modules we need to patch
        from barren_plateaus_vqe import hamiltonian_builder, molecular_analyzer

        # Patch 1: Fix get_molecular_hamiltonian_pyscf function
        original_get_hamiltonian = hamiltonian_builder.get_molecular_hamiltonian_pyscf

        def fixed_get_molecular_hamiltonian_pyscf(molecule, **kwargs):
            """Fixed version that handles case issues."""
            # Case normalization map
            case_fixes = {
                "LIH": "LiH",
                "BEH2": "BeH2",
                "H2O": "H2O",
                "NH3": "NH3",
                "CH4": "CH4",
                "lih": "LiH",
                "beh2": "BeH2",
                "h2o": "H2O",
                "nh3": "NH3",
                "ch4": "CH4",
            }

            if molecule in case_fixes:
                original_molecule = molecule
                molecule = case_fixes[molecule]
                print(
                    f"🔧 [PATCH] Fixed molecule case: {original_molecule} -> {molecule}"
                )

            return original_get_hamiltonian(molecule, **kwargs)

        # Apply hamiltonian patch
        hamiltonian_builder.get_molecular_hamiltonian_pyscf = (
            fixed_get_molecular_hamiltonian_pyscf
        )

        # Patch 2: Fix create_molecular_analyzer function
        original_create_analyzer = molecular_analyzer.create_molecular_analyzer

        def fixed_create_molecular_analyzer(molecule, **kwargs):
            """Fixed version that preserves correct molecule name case."""
            # Normalize molecule name first
            case_fixes = {"LIH": "LiH", "BEH2": "BeH2", "lih": "LiH", "beh2": "BeH2"}

            if molecule in case_fixes:
                original_molecule = molecule
                molecule = case_fixes[molecule]
                print(
                    f"🔧 [PATCH] Analyzer molecule case fix: {original_molecule} -> {molecule}"
                )

            # Create analyzer with fixed molecule name
            analyzer = original_create_analyzer(molecule, **kwargs)

            # Force correct molecule name in the analyzer object
            if hasattr(analyzer, "molecule_name"):
                analyzer.molecule_name = molecule
            if hasattr(analyzer, "molecule"):
                analyzer.molecule = molecule

            return analyzer

        # Apply analyzer patch
        molecular_analyzer.create_molecular_analyzer = fixed_create_molecular_analyzer

        # Patch 3: Try to patch any other potential sources of case conversion
        try:
            # If there's a MolecularVQEAnalyzer class, patch its methods too
            if hasattr(molecular_analyzer, "MolecularVQEAnalyzer"):
                MolecularVQEAnalyzer = molecular_analyzer.MolecularVQEAnalyzer

                # Patch the setup_hamiltonian method if it exists
                if hasattr(MolecularVQEAnalyzer, "setup_hamiltonian"):
                    original_setup_hamiltonian = MolecularVQEAnalyzer.setup_hamiltonian

                    def fixed_setup_hamiltonian(self):
                        """Fixed setup_hamiltonian that preserves molecule name case."""
                        # Ensure molecule name is correct before setup
                        case_fixes = {"LIH": "LiH", "BEH2": "BeH2"}

                        if (
                            hasattr(self, "molecule_name")
                            and self.molecule_name in case_fixes
                        ):
                            original_name = self.molecule_name
                            self.molecule_name = case_fixes[original_name]
                            print(
                                f"🔧 [PATCH] Fixed setup molecule name: {original_name} -> {self.molecule_name}"
                            )

                        if hasattr(self, "molecule") and self.molecule in case_fixes:
                            original_name = self.molecule
                            self.molecule = case_fixes[original_name]
                            print(
                                f"🔧 [PATCH] Fixed setup molecule: {original_name} -> {self.molecule}"
                            )

                        return original_setup_hamiltonian(self)

                    MolecularVQEAnalyzer.setup_hamiltonian = fixed_setup_hamiltonian

        except Exception as e:
            print(f"🔧 [PATCH] Warning: Could not patch MolecularVQEAnalyzer: {e}")

        print("🔧 [PATCH] Applied comprehensive molecule name case fixes")

    except Exception as e:
        print(f"🔧 [PATCH] Warning: Could not apply all patches: {e}")
        print("🔧 [PATCH] Will attempt to continue with basic fixes")


# Import package modules with error handling and apply critical patches
try:
    from barren_plateaus_vqe.hamiltonian_builder import get_available_molecules 
    from barren_plateaus_vqe.molecular_analyzer import (
        create_molecular_analyzer,
        create_test_analyzer,
    )
    from barren_plateaus_vqe.viz_landscape import MolecularLandscapeVisualizer

    print("✅ Successfully imported barren_plateaus_vqe package")

    # Apply critical patch to fix case sensitivity issues
    apply_molecule_name_patch()

except ImportError as e:
    print(f"❌ Error importing package: {e}")
    print("Make sure the barren_plateaus_vqe package is in the correct location")
    print("Required package structure:")
    print("  barren_plateaus_vqe/")
    print("    ├── molecular_analyzer.py")
    print("    ├── viz_landscape.py")
    print("    └── hamiltonian_builder.py")
    sys.exit(1)


class MoleculeNameHandler:
    """Handle molecule name normalization and case sensitivity issues."""

    # Standard molecule name mappings to fix case issues
    MOLECULE_CASE_MAP = {
        # Input variations -> Correct case
        "h2": "H2",
        "H2": "H2",
        "lih": "LiH",
        "LIH": "LiH",  # This is the problematic case
        "Lih": "LiH",
        "LiH": "LiH",
        "beh2": "BeH2",
        "BEH2": "BeH2",
        "Beh2": "BeH2",
        "BeH2": "BeH2",
        "h2o": "H2O",
        "H2O": "H2O",
        "n2": "N2",
        "N2": "N2",
        "co": "CO",
        "CO": "CO",
        "nh3": "NH3",
        "NH3": "NH3",
        "ch4": "CH4",
        "CH4": "CH4",
    }

    @classmethod
    def normalize_molecule_name(
        cls, molecule_name: str, available_molecules: List[str]
    ) -> str:
        """
        Normalize molecule name to match available molecules exactly.

        Args:
            molecule_name: Input molecule name
            available_molecules: List of available molecule names

        Returns:
            Normalized molecule name that matches available list
        """
        # First try exact match
        if molecule_name in available_molecules:
            return molecule_name

        # Try case mapping
        if molecule_name in cls.MOLECULE_CASE_MAP:
            normalized = cls.MOLECULE_CASE_MAP[molecule_name]
            if normalized in available_molecules:
                print(f"🔧 Normalized molecule name: {molecule_name} -> {normalized}")
                return normalized

        # Try case-insensitive match
        molecule_lower = molecule_name.lower()
        for available in available_molecules:
            if available.lower() == molecule_lower:
                print(f"🔧 Case-insensitive match: {molecule_name} -> {available}")
                return available

        # No match found
        return molecule_name

    @classmethod
    def validate_molecule_name(
        cls, molecule_name: str, available_molecules: List[str]
    ) -> bool:
        """Check if molecule name is valid after normalization."""
        normalized = cls.normalize_molecule_name(molecule_name, available_molecules)
        return normalized in available_molecules


def parse_arguments() -> argparse.Namespace:
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

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def setup_directories(args: argparse.Namespace) -> Tuple[Path, Path]:
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


def get_available_molecules_safe() -> List[str]:
    """Safely get available molecules list."""
    try:
        available_molecules = get_available_molecules()
        if isinstance(available_molecules, dict):
            return list(available_molecules.keys())
        elif isinstance(available_molecules, list):
            return available_molecules
        else:
            # Fallback to known molecules
            return ["H2", "LiH", "BeH2", "H2O", "N2", "CO", "NH3", "CH4"]
    except Exception as e:
        print(f"   Warning: Could not get available molecules: {e}")
        # Return fallback list
        return ["H2", "LiH", "BeH2", "H2O", "N2", "CO", "NH3", "CH4"]


def create_analyzer_safe(args: argparse.Namespace) -> Optional[Any]:
    """Create molecular VQE analyzer with robust error handling."""
    print("⚛️  Creating molecular VQE analyzer...")

    if args.test_hamiltonian:
        # Use test Hamiltonian
        try:
            analyzer = create_test_analyzer(num_qubits=6, num_layers=1)
            print(f"✅ Test analyzer created: {analyzer.num_qubits} qubits")
            return analyzer
        except Exception as e:
            print(f"❌ Failed to create test analyzer: {e}")
            return None

    # Get available molecules
    available_molecules = get_available_molecules_safe()
    print(f"   Available molecules: {available_molecules}")

    # Normalize molecule name
    normalized_molecule = MoleculeNameHandler.normalize_molecule_name(
        args.molecule, available_molecules
    )

    # Validate molecule exists
    if not MoleculeNameHandler.validate_molecule_name(
        args.molecule, available_molecules
    ):
        print(f"❌ Molecule '{args.molecule}' not found in available molecules")
        print(f"   Normalized to: '{normalized_molecule}'")
        print(f"   Available: {available_molecules}")
        print(f"   Note: Try exactly one of: {', '.join(available_molecules)}")
        return None

    print(f"   Creating analyzer for {normalized_molecule}...")

    # Convert active space to tuple if provided
    active_space = tuple(args.active_space) if args.active_space else None

    try:
        # Create analyzer with normalized molecule name
        analyzer = create_molecular_analyzer(
            molecule=normalized_molecule,  # Use normalized name
            geometry=args.geometry,
            basis=args.basis,
            freeze_core=args.freeze_core,
            active_space=active_space,
            num_layers=1,  # Will be modified for layer scaling
        )

        print(f"✅ Molecular analyzer created:")
        print(f"   Molecule: {getattr(analyzer, 'molecule_name', normalized_molecule)}")
        print(f"   Qubits: {analyzer.num_qubits}")
        print(f"   Active space: {active_space}")

        # Verify the analyzer molecule name matches our normalized name
        if hasattr(analyzer, "molecule_name"):
            if analyzer.molecule_name != normalized_molecule:
                print(f"🔧 Analyzer molecule name mismatch detected:")
                print(f"   Expected: {normalized_molecule}")
                print(f"   Got: {analyzer.molecule_name}")
                print(f"   Forcing correct name...")
                analyzer.molecule_name = normalized_molecule

        return analyzer

    except Exception as e:
        print(f"❌ Analyzer creation failed: {e}")

        # Provide helpful suggestions
        if "qubits" in str(e).lower() or "large" in str(e).lower():
            print(f"   💡 This might be a large system requiring active space")
            print(f"   💡 Try adding: --active-space 4 6  (or similar)")
        elif "molecule" in str(e).lower():
            print(f"   💡 Molecule name issue. Available: {available_molecules}")

        return None


def run_basic_analysis(
    analyzer: Any, args: argparse.Namespace, data_dir: Path, plots_dir: Path
) -> Optional[pd.DataFrame]:
    """Run basic VQE method comparison analysis."""
    print("\n" + "=" * 70)
    print("BASIC VQE METHOD COMPARISON ANALYSIS")
    print("=" * 70)

    start_time = time.time()

    try:
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
                bp_data = result.get("bp_diagnostics", {})
                perf_data = result.get("performance_metrics", {})

                summary_data.append(
                    {
                        "Method": method_name,
                        "Energy_Error": perf_data.get("final_energy_error", np.nan),
                        "State_Fidelity": perf_data.get("state_fidelity", np.nan),
                        "Gradient_Variance": bp_data.get("gradient_variance", np.nan),
                        "Gradient_Norm_Mean": bp_data.get("gradient_norm_mean", np.nan),
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
        try:
            print(summary_df.to_string(index=False, float_format="%.2e"))
        except Exception as e:
            print(f"Error displaying summary: {e}")
            print(summary_df)

        # Generate visualizations
        print("\n📈 Generating visualizations...")
        try:
            visualizer = MolecularLandscapeVisualizer(analyzer, str(plots_dir))

            # Energy convergence
            try:
                visualizer.plot_energy_convergence()
                print("   ✅ Energy convergence plot")
            except Exception as e:
                print(f"   ❌ Energy convergence plot failed: {e}")

            # Performance table
            try:
                visualizer.create_performance_table()
                print("   ✅ Performance table")
            except Exception as e:
                print(f"   ❌ Performance table failed: {e}")

            # Gradient diagnostics
            try:
                visualizer.plot_gradient_diagnostics()
                print("   ✅ Gradient diagnostics")
            except Exception as e:
                print(f"   ❌ Gradient diagnostics failed: {e}")

            # # Variance scaling theory
            # try:
            #     visualizer.plot_variance_scaling_theory()
            #     print("   ✅ Variance scaling theory")
            # except Exception as e:
            #     print(f"   ❌ Variance scaling theory failed: {e}")

            # Loss landscapes (if not skipped and system not too large)
            if not args.skip_landscapes and analyzer.num_qubits <= 10:
                try:
                    visualizer.plot_loss_landscapes(grid_size=args.landscape_grid)
                    print("   ✅ Loss landscapes")
                except Exception as e:
                    print(f"   ❌ Loss landscapes failed: {e}")
            else:
                print("   ⏭️  Loss landscapes skipped")

        except Exception as e:
            print(f"   ❌ Visualization setup error: {e}")

        return summary_df

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return None


def create_layer_analyzer(
    base_analyzer: Any, args: argparse.Namespace, num_layers: int
) -> Optional[Any]:
    """Create analyzer with specific number of layers."""
    try:
        if args.test_hamiltonian:
            return create_test_analyzer(
                num_qubits=base_analyzer.num_qubits, num_layers=num_layers
            )
        else:
            # Get normalized molecule name
            available_molecules = get_available_molecules_safe()
            normalized_molecule = MoleculeNameHandler.normalize_molecule_name(
                args.molecule, available_molecules
            )

            active_space = tuple(args.active_space) if args.active_space else None

            analyzer = create_molecular_analyzer(
                molecule=normalized_molecule,
                geometry=args.geometry,
                basis=args.basis,
                freeze_core=args.freeze_core,
                active_space=active_space,
                num_layers=num_layers,
            )

            # Ensure molecule name consistency
            if hasattr(analyzer, "molecule_name"):
                analyzer.molecule_name = normalized_molecule

            return analyzer

    except Exception as e:
        print(f"     ❌ Layer analyzer creation failed: {e}")
        return None


def run_layer_scaling_analysis(
    base_analyzer: Any, args: argparse.Namespace, data_dir: Path, plots_dir: Path
) -> Optional[pd.DataFrame]:
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

        # Create analyzer for this layer count
        analyzer = create_layer_analyzer(base_analyzer, args, num_layers)
        if analyzer is None:
            print(f"   ❌ Failed to create analyzer")
            all_results[num_layers] = None
            continue

        try:
            # Run analysis with fewer iterations for speed
            iterations = min(args.iterations, 500)  # Cap at 500 for scaling studies
            results = analyzer.run_complete_analysis(num_iters=iterations)

            if results:
                all_results[num_layers] = results

                # Extract metrics for scaling analysis
                for method_name, result in results.items():
                    try:
                        bp_data = result.get("bp_diagnostics", {})
                        perf_data = result.get("performance_metrics", {})
                        method_results = result.get("method_results", {})

                        scaling_data.append(
                            {
                                "num_layers": num_layers,
                                "method": method_name,
                                "num_qubits": analyzer.num_qubits,
                                "gradient_variance": bp_data.get(
                                    "gradient_variance", np.nan
                                ),
                                "gradient_norm_mean": bp_data.get(
                                    "gradient_norm_mean", np.nan
                                ),
                                "final_energy_error": perf_data.get(
                                    "final_energy_error", np.nan
                                ),
                                "state_fidelity": perf_data.get(
                                    "state_fidelity", np.nan
                                ),
                                "num_parameters": len(
                                    method_results.get("final_params", [])
                                ),
                            }
                        )
                    except Exception as e:
                        print(f"     ⚠️  Error extracting {method_name}: {e}")

                print(f"   ✅ Success: {len(results)} methods")
            else:
                print(f"   ❌ No results")
                all_results[num_layers] = None

        except Exception as e:
            print(f"   ❌ Error: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()
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
    generate_scaling_plots(scaling_df, args, base_analyzer, plots_dir, layer_range)

    return scaling_df


def generate_scaling_plots(
    scaling_df: pd.DataFrame,
    args: argparse.Namespace,
    base_analyzer: Any,
    plots_dir: Path,
    layer_range: List[int],
) -> None:
    """Generate layer scaling visualization plots."""

    methods = scaling_df["method"].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    markers = ["o", "s", "^", "D", "v", "*", "p", "h"]

    # Plot 1: Gradient variance vs layers
    try:
        plt.figure(figsize=(12, 8))

        for i, method in enumerate(methods):
            method_data = scaling_df[scaling_df["method"] == method].sort_values(
                "num_layers"
            )
            if (
                len(method_data) > 0
                and not method_data["gradient_variance"].isna().all()
            ):
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
    except Exception as e:
        print(f"   ❌ Variance scaling plot failed: {e}")

    # Plot 2: Parameter count vs layers
    try:
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methods):
            method_data = scaling_df[scaling_df["method"] == method].sort_values(
                "num_layers"
            )
            if len(method_data) > 0 and not method_data["num_parameters"].isna().all():
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
    except Exception as e:
        print(f"   ❌ Parameter scaling plot failed: {e}")

    # Plot 3: Energy error vs layers
    try:
        plt.figure(figsize=(10, 6))
        for i, method in enumerate(methods):
            method_data = scaling_df[scaling_df["method"] == method].sort_values(
                "num_layers"
            )
            if (
                len(method_data) > 0
                and not method_data["final_energy_error"].isna().all()
            ):
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
    except Exception as e:
        print(f"   ❌ Energy error scaling plot failed: {e}")

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


def generate_final_report(
    args: argparse.Namespace,
    data_dir: Path,
    basic_summary: Optional[pd.DataFrame] = None,
    scaling_summary: Optional[pd.DataFrame] = None,
) -> None:
    """Generate final analysis report."""
    print("\n📋 Generating final report...")

    report_file = data_dir / "analysis_report.md"

    try:
        with open(report_file, "w") as f:
            f.write(f"# Molecular Barren Plateau Analysis Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

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
                    # Find best method by energy error
                    valid_errors = successful_methods["Energy_Error"].dropna()
                    if not valid_errors.empty:
                        best_idx = valid_errors.idxmin()
                        best_method = successful_methods.loc[best_idx]
                        f.write(
                            f"**Best Performing Method:** {best_method['Method']}\n"
                        )
                        f.write(f"- Energy Error: {best_method['Energy_Error']:.2e}\n")
                        if not pd.isna(best_method["State_Fidelity"]):
                            f.write(
                                f"- State Fidelity: {best_method['State_Fidelity']:.3f}\n"
                            )
                        if not pd.isna(best_method["Gradient_Variance"]):
                            f.write(
                                f"- Gradient Variance: {best_method['Gradient_Variance']:.2e}\n"
                            )
                        f.write("\n")

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
                        if (
                            not (pd.isna(first_var) or pd.isna(last_var))
                            and first_var != 0
                        ):
                            ratio = last_var / first_var
                            trend = (
                                "decreasing (better)"
                                if ratio < 1
                                else "increasing (worse)"
                            )
                            f.write(f"- **{method}:** {ratio:.1f}x change ({trend})\n")

            # Files generated
            f.write(f"\n## Generated Files\n\n")
            f.write(f"### Data Files\n")
            for file_path in sorted(data_dir.glob("*.csv")):
                f.write(f"- `{file_path.name}`\n")
            for file_path in sorted(data_dir.glob("*.json")):
                f.write(f"- `{file_path.name}`\n")

            f.write(f"\n### Plot Files\n")
            plots_dir = (
                data_dir.parent.parent.parent
                / "plots"
                / data_dir.parent.name
                / data_dir.name
            )
            if plots_dir.exists():
                for file_path in sorted(plots_dir.glob("*.pdf")):
                    f.write(f"- `{file_path.name}`\n")
                for file_path in sorted(plots_dir.glob("*.png")):
                    f.write(f"- `{file_path.name}`\n")

        print(f"📋 Report saved: {report_file}")
    except Exception as e:
        print(f"❌ Report generation failed: {e}")


def main() -> int:
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
    print(f"   Verbose: {args.verbose}")

    # Setup directories
    data_dir, plots_dir = setup_directories(args)

    # Save configuration
    config_file = data_dir / "analysis_config.json"
    try:
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"💾 Configuration saved: {config_file}")
    except Exception as e:
        print(f"⚠️  Configuration save failed: {e}")

    # Create analyzer
    analyzer = create_analyzer_safe(args)
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
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print(f"\n⚠️  Analysis interrupted by user")
        return 130


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
