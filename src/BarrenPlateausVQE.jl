"""
# BarrenPlateausVQE.jl

A high-performance Julia implementation for studying barren plateau phenomena 
in Variational Quantum Eigensolver (VQE) algorithms using molecular Hamiltonians.
"""
module BarrenPlateausVQE

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using Yao
using YaoBlocks
using YaoArrayRegister

# ============================================================================
# Core Data Structures 
# ============================================================================

"""
    VQEResult

Container for VQE optimization results.
"""
struct VQEResult
    method_name::String
    energy_history::Vector{Float64}
    parameter_history::Vector{Vector{Float64}}
    final_energy::Float64
    final_parameters::Vector{Float64}
    num_iterations::Int
    converged::Bool
    execution_time::Float64
end

"""
    BarrenPlateauDiagnostics

Container for barren plateau analysis metrics.
"""
struct BarrenPlateauDiagnostics
    gradient_variance::Float64
    gradient_norm_mean::Float64
    gradient_norm_std::Float64
    cost_variance::Float64
    local_variance_mean::Float64
    hessian_trace_mean::Float64
    gradient_norms::Vector{Float64}
    cost_values::Vector{Float64}
    local_variances::Vector{Float64}
end

"""
    MolecularSystem

Container for molecular system information.
"""
struct MolecularSystem
    name::String
    geometry_type::String
    basis_set::String
    charge::Int
    spin::Int
    n_electrons::Int
    n_qubits::Int
    geometry_string::String
    hamiltonian::AbstractBlock
    exact_energy::Float64
    nuclear_repulsion_energy::Float64
end

# ============================================================================
# Include Core Modules (order matters due to dependencies)
# ============================================================================

# Core modules
include("core/dataset_loader.jl")
include("core/hamiltonian_builder.jl")
include("core/ansatz_library.jl")

# Utility modules
include("utils/quantum_utils.jl")

# Include VQE methods (order matters due to dependencies)
include("core/methods/standard_vqe.jl")
include("core/methods/sea_vqe.jl")
include("core/methods/local_global_vqe.jl")
include("core/methods/adiabatic_vqe.jl")
include("core/methods/pretrained_vqe.jl")

# Include analysis and visualization
include("core/molecular_analyzer.jl")
include("utils/visualization.jl")
include("utils/circuit_visualization.jl")

# ============================================================================
# Utility Function Definitions
# ============================================================================

"""
    mps_ansatz(n_qubits::Int; bond_dimension::Int=2)

Create a simplified MPS-inspired ansatz circuit.
"""
function mps_ansatz(n_qubits::Int; bond_dimension::Int=2)
    circuit_blocks = AbstractBlock[]
    param_count = 0

    # Initial single-qubit rotations
    for i in 1:n_qubits
        push!(circuit_blocks, put(n_qubits, i => Ry(0.0)))
        push!(circuit_blocks, put(n_qubits, i => Rz(0.0)))
        param_count += 2
    end

    # MPS-like nearest-neighbor structure
    for layer in 1:bond_dimension
        # Forward sweep
        for i in 1:(n_qubits - 1)
            push!(circuit_blocks, cnot(n_qubits, i, i+1))
            push!(circuit_blocks, put(n_qubits, i+1 => Ry(0.0)))
            param_count += 1
        end

        # Backward sweep (if more than one layer)
        if layer < bond_dimension && n_qubits > 2
            for i in (n_qubits - 1):-1:2
                push!(circuit_blocks, cnot(n_qubits, i, i-1))
                push!(circuit_blocks, put(n_qubits, i-1 => Ry(0.0)))
                param_count += 1
            end
        end
    end

    full_circuit = chain(n_qubits, circuit_blocks...)
    return full_circuit, param_count
end

"""
    run_all_vqe_methods(hamiltonian::AbstractBlock, n_qubits::Int;
                       max_iterations::Int=1000,
                       n_layers::Int=1,
                       verbose::Bool=true)

Run all VQE methods for comparison.
"""
function run_all_vqe_methods(
    hamiltonian::AbstractBlock,
    n_qubits::Int;
    max_iterations::Int=1000,
    n_layers::Int=1,
    verbose::Bool=true,
)
    results = Dict{String,Any}()

    # Test each method individually with error handling
    methods = [
        ("StandardVQE", () -> StandardVQE(n_qubits, n_layers)),
        ("SEAVQE", () -> SEAVQE(n_qubits)),
        ("LocalGlobalVQE", () -> LocalGlobalVQE(n_qubits, n_layers)),
        ("AdiabaticVQE", () -> AdiabaticVQE(n_qubits, n_layers)),
        ("PretrainedVQE", () -> PretrainedVQE(n_qubits)),
    ]

    for (method_name, constructor) in methods
        if verbose
            println("\n" * "="^60)
            println("Running $method_name...")
            println("="^60)
        end

        try
            vqe = constructor()
            initial_params = random_initial_parameters(vqe.n_parameters)

            if method_name == "PretrainedVQE"
                # Special handling for PretrainedVQE
                result = run_vqe(
                    vqe, hamiltonian, div(max_iterations, 2), div(max_iterations, 4)
                )
                # Convert to standard format
                final_energy = haskey(result, "full") ? result["full"]["final_energy"] : 0.0
                results[method_name] = VQEResult(
                    method_name,
                    Float64[final_energy],
                    Vector{Float64}[],
                    final_energy,
                    Float64[],
                    max_iterations,
                    true,
                    0.0,
                )
            else
                result = run_vqe(
                    vqe, hamiltonian, initial_params, max_iterations; verbose=verbose
                )
                results[method_name] = result
            end

        catch e
            if verbose
                @warn "$method_name failed: $e"
            end
            # Create fallback result
            results[method_name] = VQEResult(
                method_name, [1.0], [rand(10)], 1.0, rand(10), 1, false, 0.0
            )
        end
    end

    if verbose
        println("\n" * "="^60)
        println("All VQE methods completed!")
        println("="^60)

        # Print summary
        for (method_name, result) in results
            println("$method_name: Final energy = $(result.final_energy)")
        end
    end

    return results
end

# ============================================================================
# Core Exports
# ============================================================================

# Data structures
export VQEResult, BarrenPlateauDiagnostics, MolecularSystem

# Hamiltonian builder exports
export create_molecular_hamiltonian, global2local, test_hamiltonian
export h2_hamiltonian, lih_hamiltonian, h2o_hamiltonian
export classical_solver, energy_evaluation
export pauli_string, create_pauli_hamiltonian

# VQE methods exports
export StandardVQE, AdiabaticVQE, LocalGlobalVQE, SEAVQE, PretrainedVQE
export run_vqe, run_all_vqe_methods

# Utility exports
export random_initial_parameters, efficient_su2_ansatz, mps_ansatz
export expectation_value, apply_circuit, cost_function
export gradient_finite_diff, gradient_variance
export exact_ground_state_energy

# Analysis exports  
export MolecularVQEAnalyzer, run_complete_analysis
export run_standard_vqe,
    run_local_global_vqe, run_adiabatic_vqe, run_sea_vqe, run_pretrained_vqe
export compute_bp_diagnostics, compute_performance_metrics, generate_summary_table

# Visualization exports
export plot_energy_convergence, plot_gradient_diagnostics, create_performance_table
export plot_method_comparison, generate_all_plots, quick_analysis_plot

# Module initialization
function __init__()
    println("🚀 BarrenPlateausVQE.jl initialized")
    println("   High-performance VQE barren plateau analysis")
    println("   Julia $(VERSION) with Yao.jl quantum computing backend")
end

end # module BarrenPlateausVQE
