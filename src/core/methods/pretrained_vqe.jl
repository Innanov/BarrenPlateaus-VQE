"""
# Pretrained VQE Implementation (`methods/pretrained_vqe.jl`)

Pretrained VQE implementation with MPS-based parameter initialization.

The method first trains on an MPS ansatz, then transfers parameters to a full ansatz.
"""

using Yao
using YaoBlocks
using Random
using Statistics

# ============================================================================
# MPS Ansatz Implementation
# ============================================================================

"""
    create_mps_ansatz(n_qubits::Int; bond_dimension::Int=2)

Create MPS-inspired ansatz circuit.
"""
function create_mps_ansatz(n_qubits::Int; bond_dimension::Int=2)
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

# ============================================================================
# Pretrained VQE Implementation
# ============================================================================

"""
    PretrainedVQE

Pretrained VQE method configuration.
"""
struct PretrainedVQE
    mps_ansatz::AbstractBlock
    full_ansatz::AbstractBlock
    mps_n_parameters::Int
    full_n_parameters::Int
    n_qubits::Int

    # Add convenience field for compatibility
    n_parameters::Int

    function PretrainedVQE(n_qubits::Int; bond_dimension::Int=2, n_layers::Int=2)
        try
            # Create MPS ansatz for pretraining
            mps_circuit, mps_params = create_mps_ansatz(
                n_qubits; bond_dimension=bond_dimension
            )

            # Create full ansatz (more complex circuit)
            full_circuit, full_params = efficient_su2_ansatz(n_qubits, n_layers)

            # Use full ansatz parameters as the main parameter count
            new(mps_circuit, full_circuit, mps_params, full_params, n_qubits, full_params)

        catch e
            @warn "PretrainedVQE creation failed: $e, using simplified version"

            # Fallback: create two simple ansatzes of different depths
            simple_circuit, simple_params = efficient_su2_ansatz(n_qubits, 1)  # 1 layer
            full_circuit, full_params = efficient_su2_ansatz(n_qubits, 2)      # 2 layers

            new(
                simple_circuit,
                full_circuit,
                simple_params,
                full_params,
                n_qubits,
                full_params,
            )
        end
    end
end

"""
    run_vqe(vqe::PretrainedVQE, hamiltonian::AbstractBlock, 
           iters_vqe::Int, iters_train::Int;
           optimizer::Symbol=:SPSA,
           verbose::Bool=false,
           callback=nothing)

Run Pretrained VQE optimization with two stages.
"""
function run_vqe(
    vqe::PretrainedVQE,
    hamiltonian::AbstractBlock,
    iters_vqe::Int,
    iters_train::Int;
    optimizer::Symbol=:SPSA,
    verbose::Bool=false,
    callback=nothing,
)
    if verbose
        println("Running Pretrained VQE:")
        println("  Stage 1: MPS pretraining ($iters_train iterations)")
        println("  Stage 2: Full optimization ($iters_vqe iterations)")
    end

    # Stage 1: MPS pretraining
    mps_vqe = StandardVQE(vqe.mps_ansatz, vqe.mps_n_parameters, vqe.n_qubits)
    mps_initial_params = random_initial_parameters(vqe.mps_n_parameters)

    mps_result = run_vqe(
        mps_vqe,
        hamiltonian,
        mps_initial_params,
        iters_train;
        optimizer=optimizer,
        verbose=false,
    )

    # Stage 2: Transfer parameters and optimize full ansatz
    full_initial_params = random_initial_parameters(vqe.full_n_parameters)

    # Simple parameter transfer: use MPS parameters for first part of full ansatz
    n_transfer = min(vqe.mps_n_parameters, vqe.full_n_parameters)
    if n_transfer > 0
        full_initial_params[1:n_transfer] = mps_result.final_parameters[1:n_transfer]
    end

    full_vqe = StandardVQE(vqe.full_ansatz, vqe.full_n_parameters, vqe.n_qubits)
    full_result = run_vqe(
        full_vqe,
        hamiltonian,
        full_initial_params,
        iters_vqe;
        optimizer=optimizer,
        verbose=verbose,
        callback=callback,
    )

    # Return combined result in qubap-style format
    results = Dict(
        "pretrain" => Dict(
            "x" => mps_result.parameter_history,
            "fx" => mps_result.energy_history,
            "final_parameters" => mps_result.final_parameters,
            "final_energy" => mps_result.final_energy,
            "n_parameters" => vqe.mps_n_parameters,
        ),
        "full" => Dict(
            "x" => full_result.parameter_history,
            "fx" => full_result.energy_history,
            "final_parameters" => full_result.final_parameters,
            "final_energy" => full_result.final_energy,
            "n_parameters" => vqe.full_n_parameters,
        ),
    )

    return results
end

"""
    run_vqe_unified(vqe::PretrainedVQE, hamiltonian::AbstractBlock, 
                   iters_vqe::Int, iters_train::Int; kwargs...)

Run pretrained VQE and return unified VQEResult.
"""
function run_vqe_unified(
    vqe::PretrainedVQE,
    hamiltonian::AbstractBlock,
    iters_vqe::Int,
    iters_train::Int;
    kwargs...,
)
    results = run_vqe(vqe, hamiltonian, iters_vqe, iters_train; kwargs...)

    # Combine into unified result
    combined_energy_history = vcat(results["pretrain"]["fx"], results["full"]["fx"])
    combined_parameter_history = vcat(results["pretrain"]["x"], results["full"]["x"])

    return VQEResult(
        "Pretrained VQE",
        combined_energy_history,
        combined_parameter_history,
        results["full"]["final_energy"],
        results["full"]["final_parameters"],
        length(combined_energy_history),
        true,  # Assume converged if completed both stages
        0.0,    # Would need proper timing
    )
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    create_pretrained_vqe(n_qubits::Int; kwargs...)

Factory function for creating PretrainedVQE instances.
"""
function create_pretrained_vqe(n_qubits::Int; kwargs...)
    return PretrainedVQE(n_qubits; kwargs...)
end

"""
    run_pretrained_vqe(hamiltonian::AbstractBlock, n_qubits::Int,
                      iters_vqe::Int, iters_train::Int; kwargs...)

Convenience function for running pretrained VQE.
"""
function run_pretrained_vqe(
    hamiltonian::AbstractBlock, n_qubits::Int, iters_vqe::Int, iters_train::Int; kwargs...
)
    vqe = PretrainedVQE(n_qubits)
    return run_vqe_unified(vqe, hamiltonian, iters_vqe, iters_train; kwargs...)
end

# ============================================================================
# Exports
# ============================================================================

export PretrainedVQE, create_mps_ansatz, run_vqe, run_vqe_unified
export create_pretrained_vqe, run_pretrained_vqe
