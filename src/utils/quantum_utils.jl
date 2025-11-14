"""
# Core Quantum Computing Utilities (`quantum_utils.jl`)

This module provides the foundational quantum computing operations using Yao.jl,
including quantum state manipulation and measurement utilities optimized for VQE applications.
"""

using Yao
using YaoBlocks
using YaoArrayRegister
using LinearAlgebra
using SparseArrays

# Note: Core data structures and main utility functions are defined in other modules
# This module only contains functions that are unique to quantum_utils

# ============================================================================
# Quantum State Operations
# ============================================================================

"""
    apply_circuit(circuit, parameters::Vector{Float64}, n_qubits::Int)

Apply parameterized circuit to |0⟩ state.
"""
function apply_circuit(circuit, parameters::Vector{Float64}, n_qubits::Int)
    # Dispatch circuit with parameters
    dispatched_circuit = dispatch(circuit, parameters)

    # Apply to |0⟩ state
    state = zero_state(n_qubits)
    return apply!(state, dispatched_circuit)
end

"""
    cost_function(hamiltonian, circuit, parameters::Vector{Float64}, n_qubits::Int)

Compute VQE cost function C(θ) = ⟨ψ(θ)|H|ψ(θ)⟩.
"""
function cost_function(hamiltonian, circuit, parameters::Vector{Float64}, n_qubits::Int)
    try
        state = apply_circuit(circuit, parameters, n_qubits)
        return expectation_value(state, hamiltonian)
    catch e
        @warn "Cost function evaluation failed: $e"
        return Inf
    end
end

# ============================================================================
# Parameter Shift Gradient Computation
# ============================================================================

"""
    parameter_shift_gradient(hamiltonian, circuit, parameters::Vector{Float64}, n_qubits::Int)

Compute gradient using parameter shift rule (when applicable).
"""
function parameter_shift_gradient(
    hamiltonian, circuit, parameters::Vector{Float64}, n_qubits::Int
)
    n_params = length(parameters)
    gradient = zeros(n_params)
    shift = π/2

    for i in 1:n_params
        params_plus = copy(parameters)
        params_minus = copy(parameters)
        params_plus[i] += shift
        params_minus[i] -= shift

        try
            cost_plus = cost_function(hamiltonian, circuit, params_plus, n_qubits)
            cost_minus = cost_function(hamiltonian, circuit, params_minus, n_qubits)

            gradient[i] = 0.5 * (cost_plus - cost_minus)
        catch e
            @warn "Parameter shift gradient failed at parameter $i: $e"
            gradient[i] = 0.0
        end
    end

    return gradient
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    exact_ground_state_energy(hamiltonian::AbstractBlock, n_qubits::Int)

Compute exact ground state energy using classical diagonalization.
"""
function exact_ground_state_energy(hamiltonian::AbstractBlock, n_qubits::Int)
    try
        # Convert to matrix representation
        H_matrix = mat(hamiltonian)
        eigenvals = eigvals(Hermitian(H_matrix))
        return minimum(real.(eigenvals))
    catch e
        @warn "Exact diagonalization failed: $e"
        return 0.0
    end
end

"""
    fidelity_with_ground_state(state::AbstractRegister, hamiltonian::AbstractBlock)

Compute approximate fidelity with exact ground state.
"""
function fidelity_with_ground_state(state::AbstractRegister, hamiltonian::AbstractBlock)
    try
        n_qubits = nqubits(state)
        exact_energy = exact_ground_state_energy(hamiltonian, n_qubits)
        vqe_energy = expectation_value(state, hamiltonian)

        energy_diff = abs(vqe_energy - exact_energy)
        energy_scale = abs(exact_energy) > 1e-6 ? abs(exact_energy) : 1.0

        # Simple energy-based fidelity estimate
        return exp(-energy_diff / energy_scale)
    catch e
        @warn "Fidelity computation failed: $e"
        return 0.5
    end
end

"""
    random_parameters(n_params::Int; scale::Float64=0.1)

Generate random initial parameters for VQE.
"""
function random_parameters(n_params::Int; scale::Float64=0.1)
    return randn(n_params) * scale
end

"""
    normalize_parameters!(parameters::Vector{Float64})

Normalize parameters to [0, 2π) range.
"""
function normalize_parameters!(parameters::Vector{Float64})
    for i in eachindex(parameters)
        parameters[i] = mod(parameters[i], 2π)
    end
    return parameters
end

"""
    check_convergence(energy_history::Vector{Float64}; 
                     tolerance::Float64=1e-6,
                     window::Int=10)

Check if VQE optimization has converged.
"""
function check_convergence(
    energy_history::Vector{Float64}; tolerance::Float64=1e-6, window::Int=10
)
    if length(energy_history) < window
        return false
    end

    recent_energies = energy_history[(end - window + 1):end]
    energy_variance = var(recent_energies)

    return energy_variance < tolerance^2
end

# ============================================================================
# Exports
# ============================================================================

export apply_circuit, cost_function
export parameter_shift_gradient
export exact_ground_state_energy, fidelity_with_ground_state
export random_parameters, normalize_parameters!, check_convergence
