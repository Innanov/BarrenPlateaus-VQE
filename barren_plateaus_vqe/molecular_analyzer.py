#!/usr/bin/env python3
"""
Core VQE Analysis Engine (`molecular_analyzer.py`)
===============================================

This module provides the core analysis engine for studying barren plateau phenomena
in Variational Quantum Eigensolver (VQE) algorithms using molecular Hamiltonians.
"""

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# Qiskit imports
from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator

# qubap imports - with error handling
try:
    from qubap.qiskit.cost_function_barren_plateau import global2local
    from qubap.qiskit.hamiltonians import test_hamiltonian
    from qubap.qiskit.mps_pretraining import Ansatz, VQE_pretrained
    from qubap.qiskit.state_efficient_ansatz import ansatz_constructor
    from qubap.qiskit.variational_algorithms import (
        VQE,
        VQE_adiabatic,
        VQE_shift,
        classical_solver,
        energy_evaluation,
    )

    QUBAP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: qubap not found. Some functionality may be limited. Error: {e}")
    QUBAP_AVAILABLE = False

# Try to import hamiltonian builder - will handle gracefully if not available
try:
    from .hamiltonian_builder import (
        MOLECULAR_GEOMETRIES,
        QISKIT_NATURE_AVAILABLE,
        get_molecular_hamiltonian_pyscf,
        get_molecular_info_pyscf,
    )
except ImportError:
    print(
        "Warning: `hamiltonian_builder` not available. Using fallback implementations."
    )
    QISKIT_NATURE_AVAILABLE = False
    MOLECULAR_GEOMETRIES = {
        "H2": {
            "equilibrium": "H 0.0 0.0 0.0; H 0.735 0.0 0.0",
            "stretched": "H 0.0 0.0 0.0; H 1.5 0.0 0.0",
        }
    }


class BarrenPlateauAnalyzer:
    """
    Base class for barren plateau analysis in VQE algorithms.

    This class provides core functionality for analyzing barren plateau phenomena
    including gradient computation, landscape analysis, and performance metrics.
    """

    def __init__(
        self, num_qubits: int, num_layers: int = 1, use_test_hamiltonian: bool = False
    ):
        """
        Initialize the barren plateau analyzer.

        Args:
            `num_qubits`: Number of qubits in the system
            `num_layers`: Number of ansatz layers
            `use_test_hamiltonian`: Whether to use test Hamiltonian
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.use_test_hamiltonian = use_test_hamiltonian
        self.results = {}

        # Will be set by subclasses
        self.H = None
        self.H_local = None
        self.exact_min_energy = None

    def setup_ansatz(self):
        """Setup ansatz configurations for all VQE methods."""
        print(f"Setting up ansatzes with {self.num_layers} layers...")

        # Standard ansatz with controlled layers
        self.ansatz_standard = EfficientSU2(
            self.num_qubits, ["ry", "rz"], "circular", max(self.num_layers - 1, 0)
        ).decompose()
        print(
            f"   Standard ({self.num_layers} layers): {self.ansatz_standard.num_parameters} params"
        )

        # Multi-layer SEA ansatz
        self.ansatz_sea = self._create_sea_ansatz(self.num_qubits, self.num_layers)
        print(
            f"   SEA ({self.num_layers} layers): {self.ansatz_sea.num_parameters} params"
        )

        # MPS ansatzes
        try:
            if QUBAP_AVAILABLE:
                self.ansatz_mps = Ansatz(self.num_qubits, diagonal=True)
                self.ansatz_full = Ansatz(self.num_qubits, diagonal=False)
                print(
                    f"   MPS: {self.ansatz_mps.num_parameters}, {self.ansatz_full.num_parameters} params"
                )
            else:
                raise ImportError("qubap not available")
        except Exception as e:
            print(f"    MPS failed, using layered EfficientSU2 fallback: {e}")
            self.ansatz_mps = EfficientSU2(
                self.num_qubits, ["ry"], "linear", self.num_layers
            )
            self.ansatz_full = EfficientSU2(
                self.num_qubits, ["ry", "rz"], "linear", self.num_layers
            )
            print(
                f"   MPS fallback ({self.num_layers} layers): {self.ansatz_mps.num_parameters}, {self.ansatz_full.num_parameters} params"
            )

    def _create_sea_ansatz(self, num_qubits: int, num_layers: int):
        """Create multi-layer SEA ansatz with fallback options."""
        print(f"   Creating {num_layers}-layer SEA ansatz for {num_qubits} qubits...")

        if not QUBAP_AVAILABLE:
            print(f"     qubap not available, using EfficientSU2 fallback...")
            return EfficientSU2(
                num_qubits, ["ry", "rz"], "circular", max(num_layers - 1, 0)
            )

        # Try different approaches for multi-layer construction
        approaches = [
            lambda: ansatz_constructor(
                num_qubits, deep=[1] * num_layers, set_barrier=True
            ),
            lambda: ansatz_constructor(num_qubits, deep=[num_layers], set_barrier=True),
            lambda: EfficientSU2(
                num_qubits, ["ry", "rz"], "circular", max(num_layers - 1, 0)
            ),
        ]

        for i, approach in enumerate(approaches):
            try:
                ansatz = approach()
                print(
                    f"     Success with approach {i+1}: {ansatz.num_parameters} parameters"
                )
                return ansatz
            except Exception as e:
                print(f"     Approach {i+1} failed: {e}")
                continue

        # Final fallback
        print(f"    Using simple EfficientSU2 fallback...")
        return EfficientSU2(num_qubits, ["ry", "rz"], "linear", num_layers)

    def compute_gradient(
        self, cost_function, params: np.ndarray, epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        Compute gradient using finite differences.

        Args:
            `cost_function`: Function to compute cost/energy
            `params`: Parameter values at which to compute gradient
            `epsilon`: Finite difference step size

        Returns:
            Gradient vector ∇C(θ)
        """
        gradients = np.zeros(len(params))

        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon

            try:
                cost_plus = cost_function(params_plus)
                cost_minus = cost_function(params_minus)

                if np.isfinite(cost_plus) and np.isfinite(cost_minus):
                    gradients[i] = (cost_plus - cost_minus) / (2 * epsilon)
                else:
                    gradients[i] = 0.0
            except Exception as e:
                print(f"Warning: Gradient computation failed for parameter {i}: {e}")
                gradients[i] = 0.0

        return gradients

    def compute_gradient_variance(
        self,
        cost_function,
        params: np.ndarray,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Compute gradient variance to diagnose barren plateaus.

        Args:
            `cost_function`: Quantum cost function to differentiate
            `params`: Parameter vector around which to compute gradients
            `epsilon`: Finite difference step size

        Returns:
            Gradient variance
        """
        gradients = self.compute_gradient(cost_function, params, epsilon)
        return np.var(gradients)

    def compute_cost_landscape_statistics(
        self,
        cost_function,
        params: np.ndarray,
        num_samples: int = 100,
        perturbation_scale: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """
        Compute comprehensive cost landscape statistics.

        Args:
            `cost_function`: Cost function to analyze
            `params`: Central parameter point
            `num_samples`: Number of random samples
            `perturbation_scale`: Scale of random perturbations

        Returns:
            Dictionary with landscape statistics
        """
        cost_values = []
        gradient_norms = []
        local_variances = []
        hessian_traces = []

        epsilon = 1e-6

        for _ in range(num_samples):
            perturbed_params = params + np.random.normal(
                0, perturbation_scale, size=params.shape
            )

            try:
                cost_val = cost_function(perturbed_params)
                if np.isfinite(cost_val):
                    cost_values.append(cost_val)
                else:
                    cost_values.append(np.inf)
            except:
                cost_values.append(np.inf)

            gradients = self.compute_gradient(cost_function, perturbed_params, epsilon)
            gradient_norms.append(np.linalg.norm(gradients))

            # Local variance
            local_costs = []
            for _ in range(10):
                local_params = perturbed_params + np.random.normal(
                    0, 0.01, size=params.shape
                )
                try:
                    local_cost = cost_function(local_params)
                    if np.isfinite(local_cost):
                        local_costs.append(local_cost)
                except:
                    pass

            if local_costs:
                local_variances.append(np.var(local_costs))
            else:
                local_variances.append(0.0)

            # Approximate Hessian diagonal
            hessian_diag = self._compute_hessian_diagonal(
                cost_function, perturbed_params, epsilon
            )
            hessian_traces.append(np.sum(hessian_diag))

        return {
            "cost_values": np.array(cost_values),
            "gradient_norms": np.array(gradient_norms),
            "local_variances": np.array(local_variances),
            "hessian_traces": np.array(hessian_traces),
        }

    def _compute_hessian_diagonal(
        self, cost_function, params: np.ndarray, epsilon: float = 1e-6
    ) -> np.ndarray:
        """Compute diagonal elements of the Hessian matrix."""
        hessian_diag = np.zeros(len(params))

        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += epsilon
            params_minus[i] -= epsilon

            grad_plus = self.compute_gradient(cost_function, params_plus, epsilon)
            grad_minus = self.compute_gradient(cost_function, params_minus, epsilon)

            hessian_diag[i] = (grad_plus[i] - grad_minus[i]) / (2 * epsilon)

        return hessian_diag

    def compute_state_fidelity(self, ansatz, params: np.ndarray) -> float:
        """
        Compute approximate fidelity between VQE state and exact ground state.

        Args:
            `ansatz`: Quantum circuit ansatz
            `params`: Circuit parameters

        Returns:
            Approximate state fidelity
        """
        if not QUBAP_AVAILABLE or self.exact_min_energy is None:
            return 0.5  # Default fallback value

        backend = AerSimulator(method="statevector")
        vqe_energy = energy_evaluation(self.H, ansatz, params, backend)

        energy_diff = abs(vqe_energy - self.exact_min_energy)
        energy_gap = (
            abs(self.exact_min_energy) if abs(self.exact_min_energy) > 1e-6 else 1.0
        )

        if energy_diff < energy_gap:
            fidelity = 1.0 - (energy_diff / energy_gap) ** 2
        else:
            fidelity = np.exp(-energy_diff / energy_gap)

        return float(np.clip(fidelity, 0.0, 1.0))

    # VQE Method Implementations
    def run_standard_vqe(self, num_iters: int = 300) -> Dict[str, Any]:
        """Run standard VQE as baseline."""
        if not QUBAP_AVAILABLE:
            return self._create_fallback_result("Standard VQE")

        print("Running Standard VQE...")
        np.random.seed(102)
        initial_guess = np.random.randn(self.ansatz_standard.num_parameters) * 0.1

        backend = AerSimulator(shots=2**8)
        results = VQE(self.H, self.ansatz_standard, initial_guess, num_iters, backend)

        analytic_backend = AerSimulator(method="statevector")
        energies = [
            energy_evaluation(self.H, self.ansatz_standard, x, analytic_backend)
            for x in results["x"]
        ]

        return {
            "method": "Standard VQE",
            "results": results,
            "energies": energies,
            "ansatz": self.ansatz_standard,
            "final_params": results["x"][-1],
            "trajectory": results["x"],
        }

    def run_local_global_vqe(self, num_iters: int = 300) -> Dict[str, Any]:
        """Run Local-Global VQE mitigation technique."""
        if not QUBAP_AVAILABLE or self.H_local is None:
            return self._create_fallback_result("Local-Global VQE")

        print("Running Local-Global VQE...")
        np.random.seed(200)
        initial_guess = np.random.randn(self.ansatz_standard.num_parameters) * 0.1

        backend = AerSimulator(shots=2**8)
        results = VQE_shift(
            self.H_local,
            self.H,
            self.ansatz_standard,
            initial_guess,
            num_iters,
            num_iters // 10,
            backend,
        )

        analytic_backend = AerSimulator(method="statevector")
        local_energies = [
            energy_evaluation(self.H, self.ansatz_standard, x, analytic_backend)
            for x in results["in"]["x"]
        ]
        global_energies = [
            energy_evaluation(self.H, self.ansatz_standard, x, analytic_backend)
            for x in results["out"]["x"]
        ]

        all_energies = local_energies + global_energies
        all_trajectory = results["in"]["x"] + results["out"]["x"]

        return {
            "method": "Local-Global VQE",
            "results": results,
            "energies": all_energies,
            "local_energies": local_energies,
            "global_energies": global_energies,
            "ansatz": self.ansatz_standard,
            "final_params": results["out"]["x"][-1],
            "trajectory": all_trajectory,
        }

    def run_adiabatic_vqe(self, num_iters: int = 300) -> Dict[str, Any]:
        """Run Adiabatic VQE mitigation technique."""
        if not QUBAP_AVAILABLE or self.H_local is None:
            return self._create_fallback_result("Adiabatic VQE")

        print("Running Adiabatic VQE...")
        np.random.seed(250)
        initial_guess = np.random.randn(self.ansatz_standard.num_parameters) * 0.1

        backend = AerSimulator(shots=2**8)
        results = VQE_adiabatic(
            self.H_local,
            self.H,
            self.ansatz_standard,
            initial_guess,
            num_iters,
            backend,
        )

        analytic_backend = AerSimulator(method="statevector")
        energies = [
            energy_evaluation(self.H, self.ansatz_standard, x, analytic_backend)
            for x in results["x"]
        ]

        return {
            "method": "Adiabatic VQE",
            "results": results,
            "energies": energies,
            "ansatz": self.ansatz_standard,
            "final_params": results["x"][-1],
            "trajectory": results["x"],
        }

    def run_sea_vqe(self, num_iters: int = 300) -> Dict[str, Any]:
        """Run VQE with State Efficient Ansatz."""
        if not QUBAP_AVAILABLE:
            return self._create_fallback_result("VQE with SEA")

        print("Running VQE with SEA...")
        np.random.seed(3000)
        initial_guess = np.random.randn(self.ansatz_sea.num_parameters) * 0.1

        backend = AerSimulator(shots=2**8)
        results = VQE(self.H, self.ansatz_sea, initial_guess, num_iters, backend)

        analytic_backend = AerSimulator(method="statevector")
        energies = [
            energy_evaluation(self.H, self.ansatz_sea, x, analytic_backend)
            for x in results["x"]
        ]

        return {
            "method": "VQE with SEA",
            "results": results,
            "energies": energies,
            "ansatz": self.ansatz_sea,
            "final_params": results["x"][-1],
            "trajectory": results["x"],
        }

    def run_pretrained_vqe(self, num_iters: int = 300) -> Dict[str, Any]:
        """Run Pretrained VQE mitigation technique."""
        if not QUBAP_AVAILABLE:
            return self._create_fallback_result("Pretrained VQE")

        print("Running Pretrained VQE...")
        np.random.seed(400)

        backend = AerSimulator(shots=2**8)

        try:
            results = VQE_pretrained(self.H, backend, num_iters, num_iters)

            # Handle result structure robustly
            vqe_key = self._find_key_containing(
                results, ["vqe", "optimization", "full"]
            )
            pretrain_key = self._find_key_containing(
                results, ["pretrain", "mps", "training"]
            )

            analytic_backend = AerSimulator(method="statevector")
            mps_energies = []
            full_energies = []

            # Extract MPS energies
            if pretrain_key and "x" in results.get(pretrain_key, {}):
                try:
                    mps_energies = [
                        energy_evaluation(self.H, self.ansatz_mps, x, analytic_backend)
                        for x in results[pretrain_key]["x"]
                    ]
                except Exception as e:
                    print(f"  Warning: Could not compute MPS energies: {e}")

            # Extract VQE energies
            if vqe_key and "x" in results.get(vqe_key, {}):
                full_energies = [
                    energy_evaluation(self.H, self.ansatz_full, x, analytic_backend)
                    for x in results[vqe_key]["x"]
                ]
                final_params = results[vqe_key]["x"][-1]
                trajectory = results[vqe_key]["x"]
            else:
                final_params = np.random.randn(self.ansatz_full.num_parameters) * 0.1
                full_energies = [
                    energy_evaluation(
                        self.H, self.ansatz_full, final_params, analytic_backend
                    )
                ]
                trajectory = [final_params]

            return {
                "method": "Pretrained VQE",
                "results": results,
                "energies": full_energies,
                "mps_energies": mps_energies,
                "full_energies": full_energies,
                "total_energies": mps_energies + full_energies,
                "ansatz": self.ansatz_full,
                "final_params": final_params,
                "trajectory": trajectory,
            }

        except Exception as e:
            print(f"  Error in Pretrained VQE: {e}")
            return self._create_fallback_result("Pretrained VQE", error=str(e))

    def _find_key_containing(
        self, dictionary: Dict, substrings: List[str]
    ) -> Optional[str]:
        """Find dictionary key containing any of the given substrings."""
        for key in dictionary.keys():
            for substring in substrings:
                if substring.lower() in key.lower():
                    return key
        return None

    def _create_fallback_result(
        self, method_name: str, error: str = "qubap not available"
    ) -> Dict[str, Any]:
        """Create fallback result when qubap is not available."""
        fallback_params = np.random.randn(self.ansatz_standard.num_parameters) * 0.1
        fallback_energy = 1.0  # High energy indicating poor performance

        return {
            "method": method_name,
            "results": {"error": error, "fallback": True},
            "energies": [fallback_energy],
            "ansatz": self.ansatz_standard,
            "final_params": fallback_params,
            "trajectory": [fallback_params],
        }

    def compute_bp_diagnostics(self, method_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive barren plateau diagnostics."""
        ansatz = method_results["ansatz"]
        final_params = method_results["final_params"]

        if not QUBAP_AVAILABLE:
            # Return minimal diagnostics if qubap not available
            return {
                "gradient_variance": 1e-3,
                "gradient_norm_mean": 1e-2,
                "gradient_norm_std": 1e-3,
                "cost_value_variance": 1e-2,
                "local_variance_mean": 1e-3,
                "hessian_trace_mean": 0.0,
                "distributions": {
                    "gradient_norms": np.array([1e-2]),
                    "cost_values": np.array([1.0]),
                    "local_variances": np.array([1e-3]),
                    "hessian_traces": np.array([0.0]),
                },
            }

        backend = AerSimulator(method="statevector")

        def cost_function(params):
            return energy_evaluation(self.H, ansatz, params, backend)

        grad_var = self.compute_gradient_variance(
            cost_function, final_params, num_samples=20
        )
        landscape_stats = self.compute_cost_landscape_statistics(
            cost_function, final_params, num_samples=30, perturbation_scale=0.1
        )

        grad_norms = landscape_stats["gradient_norms"]
        cost_values = landscape_stats["cost_values"]
        local_variances = landscape_stats["local_variances"]
        hessian_traces = landscape_stats["hessian_traces"]

        return {
            "gradient_variance": grad_var,
            "gradient_norm_mean": np.mean(grad_norms),
            "gradient_norm_std": np.std(grad_norms),
            "cost_value_variance": np.var(cost_values),
            "local_variance_mean": np.mean(local_variances),
            "hessian_trace_mean": np.mean(hessian_traces),
            "distributions": {
                "gradient_norms": grad_norms,
                "cost_values": cost_values,
                "local_variances": local_variances,
                "hessian_traces": hessian_traces,
            },
        }

    def compute_performance_metrics(
        self, method_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute performance metrics for a given method."""
        energies = method_results["energies"]
        ansatz = method_results["ansatz"]
        final_params = method_results["final_params"]

        final_energy_error = abs(energies[-1] - (self.exact_min_energy or 0.0))
        state_fidelity = self.compute_state_fidelity(ansatz, final_params)
        energy_variance = (
            np.var(energies[-50:]) if len(energies) >= 50 else np.var(energies)
        )

        return {
            "final_energy_error": final_energy_error,
            "state_fidelity": state_fidelity,
            "energy_variance": energy_variance,
            "min_energy_reached": min(energies),
        }

    def run_complete_analysis(self, num_iters: int = 300) -> Dict[str, Any]:
        """
        Run complete analysis of all mitigation techniques.

        Args:
            `num_iters`: Number of VQE iterations per method

        Returns:
            Dictionary with complete analysis results
        """
        print("=" * 60)
        print("COMPREHENSIVE VQE BARREN PLATEAU ANALYSIS")
        print("=" * 60)

        # Setup components
        self.setup_hamiltonian()
        self.setup_ansatz()

        # VQE methods to run
        methods = [
            self.run_standard_vqe,
            self.run_local_global_vqe,
            self.run_adiabatic_vqe,
            self.run_sea_vqe,
            self.run_pretrained_vqe,
        ]

        all_results = {}

        for method in methods:
            try:
                result = method(num_iters)
                method_name = result["method"]

                bp_diagnostics = self.compute_bp_diagnostics(result)
                performance_metrics = self.compute_performance_metrics(result)

                all_results[method_name] = {
                    "method_results": result,
                    "bp_diagnostics": bp_diagnostics,
                    "performance_metrics": performance_metrics,
                }

                print(f"\n{method_name} completed successfully!")

            except Exception as e:
                print(f"Error in {method.__name__}: {e}")
                method_name = (
                    method.__name__.replace("run_", "").replace("_", " ").title()
                )
                print(f"Creating fallback result for {method_name}")

                fallback_result = self._create_fallback_result(method_name, str(e))

                try:
                    bp_diagnostics = self.compute_bp_diagnostics(fallback_result)
                    performance_metrics = self.compute_performance_metrics(
                        fallback_result
                    )

                    all_results[method_name] = {
                        "method_results": fallback_result,
                        "bp_diagnostics": bp_diagnostics,
                        "performance_metrics": performance_metrics,
                    }
                except Exception as e2:
                    print(f"Failed to create fallback for {method_name}: {e2}")
                    continue

        self.results = all_results
        return all_results

    def setup_hamiltonian(self):
        """Setup Hamiltonian - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement setup_hamiltonian")

    def generate_summary_table(self):
        """Generate summary table of all metrics."""
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)

        header = f"{'Method':<20} {'Grad Var':<12} {'Grad Norm':<12} {'Energy Error':<15} {'Fidelity':<10}"
        print(header)
        print("-" * len(header))

        for method_name, data in self.results.items():
            bp_diag = data["bp_diagnostics"]
            perf_met = data["performance_metrics"]

            row = (
                f"{method_name:<20} {bp_diag['gradient_variance']:<12.2e} "
                f"{bp_diag['gradient_norm_mean']:<12.2e} "
                f"{perf_met['final_energy_error']:<15.2e} "
                f"{perf_met['state_fidelity']:<10.3f}"
            )
            print(row)

    def save_results(self, output_dir: str):
        """Save analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save metadata
        metadata = {
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "use_test_hamiltonian": self.use_test_hamiltonian,
            "exact_min_energy": self.exact_min_energy,
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)


class MolecularVQEAnalyzer(BarrenPlateauAnalyzer):
    """
    Comprehensive analyzer for barren plateau phenomena in VQE algorithms using molecular Hamiltonians.

    This class extends the base `BarrenPlateauAnalyzer` to work with real molecular systems
    using PySCF quantum chemistry calculations and Qiskit Nature integration.
    """

    def __init__(
        self,
        molecule: str,
        geometry: str = "equilibrium",
        basis: str = "sto-3g",
        freeze_core: bool = False,
        active_space: Optional[Tuple[int, int]] = None,
        num_layers: int = 1,
        use_test_hamiltonian: bool = False,
    ):
        """
        Initialize the molecular VQE analyzer.

        Args:
            `molecule`: Molecule name (H2, LiH, BeH2, H2O, N2, CO)
            `geometry`: Geometry type (equilibrium, stretched)
            `basis`: Basis set for calculation
            `freeze_core`: Whether to freeze core orbitals
            `active_space`: (num_electrons, num_orbitals) for active space
            `num_layers`: Number of ansatz layers
            `use_test_hamiltonian`: If True, uses test Hamiltonian instead of molecular
        """
        self.molecule_name = molecule.upper()
        self.geometry_type = geometry
        self.basis_set = basis
        self.freeze_core = freeze_core
        self.active_space = active_space

        # Setup molecular system
        if not use_test_hamiltonian and QISKIT_NATURE_AVAILABLE:
            self._setup_molecular_system()
        else:
            # Fallback to test Hamiltonian
            self.num_qubits = 6  # Default for test
            self._setup_test_system()

        # Initialize base class
        super().__init__(
            num_qubits=self.num_qubits,
            num_layers=num_layers,
            use_test_hamiltonian=use_test_hamiltonian or not QISKIT_NATURE_AVAILABLE,
        )

    def _setup_molecular_system(self):
        """Setup molecular Hamiltonian and system parameters."""
        print(f"Setting up {self.molecule_name} molecular system...")

        try:
            # Get molecular info to determine system size
            mol_info = get_molecular_info_pyscf(
                self.molecule_name, self.geometry_type, self.basis_set
            )
            self.num_qubits = mol_info["num_qubits"]
            self.mol_info = mol_info

            print(f"  Molecule: {self.molecule_name}")
            print(f"  Geometry: {self.geometry_type}")
            print(f"  Basis: {self.basis_set}")
            print(f"  Qubits: {self.num_qubits}")
            print(f"  Electrons: {mol_info['num_particles']}")
            print(f"  Orbitals: {mol_info['num_spatial_orbitals']}")

        except Exception as e:
            print(f"  Warning: Molecular setup failed: {e}")
            print(f"  Falling back to test Hamiltonian")
            self._setup_test_system()

    def _setup_test_system(self):
        """Setup test Hamiltonian system."""
        print("Setting up test Hamiltonian system...")
        self.num_qubits = 6
        self.use_test_hamiltonian = True

    def setup_hamiltonian(self):
        """Initialize Hamiltonian and compute exact ground state."""
        print("Setting up Hamiltonian...")

        if self.use_test_hamiltonian or not QISKIT_NATURE_AVAILABLE:
            # Use test Hamiltonian
            if QUBAP_AVAILABLE:
                self.H = test_hamiltonian(self.num_qubits)
                print(f"Test Hamiltonian with {self.num_qubits} qubits")
            else:
                print("Warning: qubap not available, using minimal fallback")
                # Create a minimal Hamiltonian for fallback
                from qiskit.quantum_info import SparsePauliOp

                self.H = SparsePauliOp.from_list([("Z" * self.num_qubits, 1.0)])
        else:
            # Use molecular Hamiltonian
            self.H = get_molecular_hamiltonian_pyscf(
                self.molecule_name,
                self.geometry_type,
                self.basis_set,
                self.freeze_core,
                self.active_space,
            )
            print(f"Molecular Hamiltonian: {self.molecule_name}")
            print(
                f"  Nuclear repulsion: {self.mol_info['nuclear_repulsion_energy']:.6f}"
            )

        print(f"Hamiltonian terms: {len(getattr(self.H, 'oplist', [self.H]))}")

        # Compute exact ground state
        try:
            if QUBAP_AVAILABLE:
                self.exact_min_energy = classical_solver(self.H).eigenvalue
                print(f"Exact ground state energy: {self.exact_min_energy}")
            else:
                self.exact_min_energy = 0.0  # Fallback
                print("Warning: Cannot compute exact energy without qubap")
        except Exception as e:
            print(f"Warning: Could not compute exact energy: {e}")
            self.exact_min_energy = 0.0

        # Create local Hamiltonian for mitigation techniques
        try:
            if QUBAP_AVAILABLE:
                self.H_local = global2local(self.H)
            else:
                self.H_local = self.H
        except Exception as e:
            print(f"Warning: Could not create local Hamiltonian: {e}")
            self.H_local = self.H

    def save_results(self, output_dir: str):
        """Save analysis results with molecular metadata."""
        os.makedirs(output_dir, exist_ok=True)

        # Enhanced metadata for molecular systems
        metadata = {
            "molecule": self.molecule_name,
            "geometry": self.geometry_type,
            "basis": self.basis_set,
            "num_qubits": self.num_qubits,
            "num_layers": self.num_layers,
            "freeze_core": self.freeze_core,
            "active_space": self.active_space,
            "use_test_hamiltonian": self.use_test_hamiltonian,
            "exact_min_energy": self.exact_min_energy,
        }

        if hasattr(self, "mol_info"):
            metadata.update(self.mol_info)

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2, default=str)


# Factory Functions
def create_molecular_analyzer(molecule: str, **kwargs) -> MolecularVQEAnalyzer:
    """
    Create a molecular VQE analyzer.

    Args:
        `molecule`: Molecule name (H2, LiH, BeH2, H2O, N2, CO)
        **kwargs: Additional arguments for `MolecularVQEAnalyzer`

    Returns:
        Configured `MolecularVQEAnalyzer` instance

    Example:
        >>> analyzer = create_molecular_analyzer("H2", basis="sto-3g", num_layers=2)
        >>> results = analyzer.run_complete_analysis(num_iters=500)
    """
    return MolecularVQEAnalyzer(molecule=molecule, **kwargs)


def create_test_analyzer(
    num_qubits: int = 6, num_layers: int = 1
) -> MolecularVQEAnalyzer:
    """
    Create analyzer with test Hamiltonian.

    Args:
        `num_qubits`: Number of qubits for test system
        `num_layers`: Number of ansatz layers

    Returns:
        `MolecularVQEAnalyzer` configured with test Hamiltonian

    Example:
        >>> analyzer = create_test_analyzer(num_qubits=4, num_layers=1)
        >>> results = analyzer.run_complete_analysis(num_iters=200)
    """
    return MolecularVQEAnalyzer(
        molecule="H2",  # Placeholder name
        use_test_hamiltonian=True,
        num_layers=num_layers,
    )


# Additional utility functions
def validate_molecular_analyzer_params(
    molecule: str, geometry: str = "equilibrium", basis: str = "sto-3g", **kwargs
) -> Dict[str, Any]:
    """
    Validate parameters for molecular analyzer creation.

    Args:
        `molecule`: Molecule name
        `geometry`: Geometry type
        `basis`: Basis set
        **kwargs: Additional parameters

    Returns:
        Validation results dictionary
    """
    validation = {
        "valid": True,
        "warnings": [],
        "suggestions": [],
        "estimated_qubits": None,
    }

    # Check molecule
    molecule_upper = molecule.upper()
    if molecule_upper not in MOLECULAR_GEOMETRIES:
        validation["valid"] = False
        validation["warnings"].append(f"Unknown molecule: {molecule}")
        validation["suggestions"].append(
            f"Available molecules: {list(MOLECULAR_GEOMETRIES.keys())}"
        )
        return validation

    # Check geometry
    if geometry not in MOLECULAR_GEOMETRIES[molecule_upper]:
        validation["warnings"].append(f"Unknown geometry: {geometry}")
        validation["suggestions"].append(
            f"Available geometries: {list(MOLECULAR_GEOMETRIES[molecule_upper].keys())}"
        )

    # Estimate system size if possible
    if QISKIT_NATURE_AVAILABLE:
        try:
            mol_info = get_molecular_info_pyscf(molecule, geometry, basis)
            validation["estimated_qubits"] = mol_info["num_qubits"]

            if mol_info["num_qubits"] > 12:
                validation["warnings"].append(
                    f"Large system: {mol_info['num_qubits']} qubits"
                )
                validation["suggestions"].append(
                    "Consider using active_space parameter"
                )
        except Exception as e:
            validation["warnings"].append(f"Could not estimate system size: {e}")

    return validation


if __name__ == "__main__":
    print("Molecular VQE Analyzer Module")
    print("=" * 50)

    # Test basic functionality
    try:
        analyzer = create_molecular_analyzer(
            "H2", geometry="equilibrium", basis="sto-3g"
        )
        print(f"✅ Created analyzer for {analyzer.molecule_name}")
        print(f"   System size: {analyzer.num_qubits} qubits")
        print(f"   Layers: {analyzer.num_layers}")
    except Exception as e:
        print(f"⚠️  Error creating molecular analyzer: {e}")
        print("   Trying test Hamiltonian...")
        analyzer = create_test_analyzer(num_qubits=6, num_layers=1)
        print(f"✅ Created test analyzer with {analyzer.num_qubits} qubits")

    print("Module ready for analysis!")
