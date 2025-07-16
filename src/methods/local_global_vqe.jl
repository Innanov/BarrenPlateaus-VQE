"""
# Local-Global VQE Implementation (`methods/local_global_vqe.jl`)

Local-Global VQE implementation with two-stage optimization strategy.

The method first optimizes on a local Hamiltonian, then uses those parameters
as initialization for optimization on the global (target) Hamiltonian.
"""

using Yao
using YaoBlocks
using Random
using Statistics

# ============================================================================
# Local-Global VQE Implementation
# ============================================================================

"""
    LocalGlobalVQE

Local-Global VQE method configuration.
"""
struct LocalGlobalVQE
    ansatz::AbstractBlock
    n_parameters::Int
    n_qubits::Int
    n_layers::Int
    
    function LocalGlobalVQE(n_qubits::Int, n_layers::Int=1; 
                           rotation_gates::Vector{Symbol}=[:Ry, :Rz],
                           entanglement::String="circular")
        ansatz, n_params = efficient_su2_ansatz(n_qubits, n_layers; 
                                               rotation_gates=rotation_gates,
                                               entanglement=entanglement)
        new(ansatz, n_params, n_qubits, n_layers)
    end
end

"""
    create_local_hamiltonian(global_hamiltonian::AbstractBlock, n_qubits::Int)

Create a local Hamiltonian by extracting single-qubit terms from global Hamiltonian.
"""
function create_local_hamiltonian(global_hamiltonian::AbstractBlock, n_qubits::Int)
    return global2local(global_hamiltonian, n_qubits)
end

"""
    run_vqe(vqe::LocalGlobalVQE, hamiltonian_in::AbstractBlock, hamiltonian_out::AbstractBlock,
           initial_guess::Vector{Float64}, max_iter::Int, shift_iter::Int;
           iter_start::Int=0,
           optimizer::Symbol=:SPSA,
           verbose::Bool=false,
           callback=nothing)

Run Local-Global VQE optimization.
Replicates the VQE_shift function from qubap's VQEs.py.
"""
function run_vqe(vqe::LocalGlobalVQE, hamiltonian_in::AbstractBlock, hamiltonian_out::AbstractBlock,
                initial_guess::Vector{Float64}, max_iter::Int, shift_iter::Int;
                iter_start::Int=0,
                optimizer::Symbol=:SPSA,
                verbose::Bool=false,
                callback=nothing)
    
    if length(initial_guess) != vqe.n_parameters
        throw(ArgumentError("Initial guess length must match number of parameters"))
    end
    
    if shift_iter >= max_iter
        throw(ArgumentError("shift_iter must be less than max_iter"))
    end
    
    # ========================================================================
    # Stage 1: Local Hamiltonian Optimization
    # ========================================================================
    
    if verbose
        println("Stage 1: Local Hamiltonian optimization ($shift_iter iterations)")
    end
    
    # Create StandardVQE for local optimization
    local_vqe = StandardVQE(vqe.ansatz, vqe.n_parameters, vqe.n_qubits)
    
    # Run local optimization
    local_result = run_vqe(local_vqe, hamiltonian_in, initial_guess, shift_iter;
                          optimizer=optimizer, verbose=false, callback=callback)
    
    # ========================================================================
    # Stage 2: Global Hamiltonian Optimization
    # ========================================================================
    
    remaining_iters = max_iter - shift_iter
    
    if verbose
        println("Stage 2: Global Hamiltonian optimization ($remaining_iters iterations)")
        println("  Starting from local optimum: $(local_result.final_energy)")
    end
    
    # Create StandardVQE for global optimization
    global_vqe = StandardVQE(vqe.ansatz, vqe.n_parameters, vqe.n_qubits)
    
    # Use final parameters from local optimization as initial guess for global
    global_initial_guess = local_result.final_parameters
    
    # Run global optimization
    global_result = run_vqe(global_vqe, hamiltonian_out, global_initial_guess, remaining_iters;
                           optimizer=optimizer, verbose=verbose, callback=callback)
    
    # ========================================================================
    # Combine Results
    # ========================================================================
    
    if verbose
        println("Local-Global VQE completed:")
        println("  Local stage final energy: $(local_result.final_energy)")
        println("  Global stage final energy: $(global_result.final_energy)")
        println("  Total iterations: $(local_result.num_iterations + global_result.num_iterations)")
        println("  Total time: $(round(local_result.execution_time + global_result.execution_time, digits=2))s")
    end
    
    # Return results in qubap-compatible format
    results = Dict(
        "in" => Dict(
            "x" => local_result.parameter_history,
            "fx" => local_result.energy_history,
            "final_energy" => local_result.final_energy,
            "final_parameters" => local_result.final_parameters,
            "num_iterations" => local_result.num_iterations,
            "converged" => local_result.converged,
            "execution_time" => local_result.execution_time
        ),
        "out" => Dict(
            "x" => global_result.parameter_history,
            "fx" => global_result.energy_history,
            "final_energy" => global_result.final_energy,
            "final_parameters" => global_result.final_parameters,
            "num_iterations" => global_result.num_iterations,
            "converged" => global_result.converged,
            "execution_time" => global_result.execution_time
        )
    )
    
    return results
end

"""
    run_vqe(vqe::LocalGlobalVQE, hamiltonian::AbstractBlock, 
           initial_guess::Vector{Float64}, num_iters::Int;
           optimizer::Symbol=:SPSA, verbose::Bool=false, kwargs...)

Simplified run_vqe interface that automatically creates local Hamiltonian.
"""
function run_vqe(vqe::LocalGlobalVQE, hamiltonian::AbstractBlock, 
                initial_guess::Vector{Float64}, num_iters::Int;
                optimizer::Symbol=:SPSA, verbose::Bool=false, kwargs...)
    
    # Create local Hamiltonian automatically
    local_hamiltonian = create_local_hamiltonian(hamiltonian, vqe.n_qubits)
    
    # Use 30% of iterations for local optimization
    shift_iter = div(num_iters, 3)
    
    # Run the two-stage optimization
    results = run_vqe(vqe, local_hamiltonian, hamiltonian, initial_guess, 
                     num_iters, shift_iter; optimizer=optimizer, verbose=verbose, kwargs...)
    
    # Convert to unified VQEResult format for compatibility
    combined_energies = vcat(results["in"]["fx"], results["out"]["fx"])
    combined_params = vcat(results["in"]["x"], results["out"]["x"])
    
    return VQEResult(
        "Local-Global VQE",
        combined_energies,
        combined_params,
        results["out"]["final_energy"],
        results["out"]["final_parameters"],
        length(combined_energies),
        results["out"]["converged"],
        results["in"]["execution_time"] + results["out"]["execution_time"]
    )
end

"""
    run_vqe_combined(vqe::LocalGlobalVQE, hamiltonian_in::AbstractBlock, hamiltonian_out::AbstractBlock,
                    initial_guess::Vector{Float64}, max_iter::Int, shift_iter::Int;
                    kwargs...)

Run Local-Global VQE and return a single combined VQEResult.
"""
function run_vqe_combined(vqe::LocalGlobalVQE, hamiltonian_in::AbstractBlock, hamiltonian_out::AbstractBlock,
                         initial_guess::Vector{Float64}, max_iter::Int, shift_iter::Int;
                         kwargs...)
    
    # Run the two-stage optimization
    results = run_vqe(vqe, hamiltonian_in, hamiltonian_out, initial_guess, max_iter, shift_iter; kwargs...)
    
    # Combine energy and parameter histories
    combined_energy_history = vcat(results["in"]["fx"], results["out"]["fx"])
    combined_parameter_history = vcat(results["in"]["x"], results["out"]["x"])
    
    # Use global stage results for final values
    final_energy = results["out"]["final_energy"]
    final_parameters = results["out"]["final_parameters"]
    total_iterations = results["in"]["num_iterations"] + results["out"]["num_iterations"]
    converged = results["out"]["converged"]
    total_execution_time = results["in"]["execution_time"] + results["out"]["execution_time"]
    
    return VQEResult(
        "Local-Global VQE",
        combined_energy_history,
        combined_parameter_history,
        final_energy,
        final_parameters,
        total_iterations,
        converged,
        total_execution_time
    )
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    create_local_global_vqe(n_qubits::Int; kwargs...)

Factory function for creating LocalGlobalVQE instances.
"""
function create_local_global_vqe(n_qubits::Int; kwargs...)
    return LocalGlobalVQE(n_qubits; kwargs...)
end

"""
    run_local_global_vqe(hamiltonian_local::AbstractBlock, hamiltonian_global::AbstractBlock,
                         n_qubits::Int, initial_guess::Vector{Float64}, 
                         max_iter::Int, shift_iter::Int;
                         kwargs...)

Convenience function for running Local-Global VQE with automatic ansatz creation.
"""
function run_local_global_vqe(hamiltonian_local::AbstractBlock, hamiltonian_global::AbstractBlock,
                              n_qubits::Int, initial_guess::Vector{Float64}, 
                              max_iter::Int, shift_iter::Int;
                              n_layers::Int=1, combined_result::Bool=false, kwargs...)
    
    vqe = LocalGlobalVQE(n_qubits, n_layers)
    
    # Generate initial guess if not provided correctly
    if length(initial_guess) != vqe.n_parameters
        @warn "Initial guess length ($(length(initial_guess))) doesn't match ansatz parameters ($(vqe.n_parameters)). Generating new initial guess."
        initial_guess = random_initial_parameters(vqe.n_parameters)
    end
    
    if combined_result
        return run_vqe_combined(vqe, hamiltonian_local, hamiltonian_global, 
                               initial_guess, max_iter, shift_iter; kwargs...)
    else
        return run_vqe(vqe, hamiltonian_local, hamiltonian_global, 
                      initial_guess, max_iter, shift_iter; kwargs...)
    end
end

"""
    analyze_local_global_transition(results::Dict{String, Dict})

Analyze the transition from local to global optimization.
"""
function analyze_local_global_transition(results::Dict{String, Dict})
    local_final_energy = results["in"]["final_energy"]
    global_final_energy = results["out"]["final_energy"]
    
    local_final_params = results["in"]["final_parameters"]
    global_final_params = results["out"]["final_parameters"]
    
    energy_drop = local_final_energy - global_final_energy
    parameter_distance = norm(local_final_params - global_final_params)
    
    local_convergence = results["in"]["converged"]
    global_convergence = results["out"]["converged"]
    
    # Analyze energy evolution
    local_energies = results["in"]["fx"]
    global_energies = results["out"]["fx"]
    
    local_improvement = length(local_energies) > 1 ? local_energies[1] - local_energies[end] : 0.0
    global_improvement = length(global_energies) > 1 ? global_energies[1] - global_energies[end] : 0.0
    
    return Dict(
        "energy_drop" => energy_drop,
        "parameter_distance" => parameter_distance,
        "local_convergence" => local_convergence,
        "global_convergence" => global_convergence,
        "local_improvement" => local_improvement,
        "global_improvement" => global_improvement,
        "total_improvement" => local_improvement + global_improvement,
        "local_final_energy" => local_final_energy,
        "global_final_energy" => global_final_energy
    )
end

# ============================================================================
# Exports
# ============================================================================

export LocalGlobalVQE, run_vqe, run_vqe_combined, create_local_global_vqe, run_local_global_vqe
export create_local_hamiltonian, analyze_local_global_transition