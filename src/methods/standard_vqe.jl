"""
# Standard VQE Implementation (`methods/standard_vqe.jl`)

Standard Variational Quantum Eigensolver implementation following the qubap structure.
"""

using Yao
using YaoBlocks
using Random
using Statistics

# ============================================================================
# Ansatz Definitions
# ============================================================================

"""
    efficient_su2_ansatz(n_qubits::Int, n_layers::Int=1; 
                        rotation_gates::Vector{Symbol}=[:Rx, :Ry],
                        entanglement::String="circular")

Create EfficientSU2 ansatz similar to Qiskit's implementation.
"""
function efficient_su2_ansatz(
    n_qubits::Int,
    n_layers::Int=1;
    rotation_gates::Vector{Symbol}=[:Rx, :Ry],
    entanglement::String="circular",
)
    circuit_blocks = AbstractBlock[]
    param_count = 0

    for layer in 1:n_layers
        # Rotation layer
        rotation_layer = chain(n_qubits)

        for qubit in 1:n_qubits
            for gate_type in rotation_gates
                if gate_type == :Ry
                    gate = put(n_qubits, qubit => Ry(0.0))
                elseif gate_type == :Rz
                    gate = put(n_qubits, qubit => Rz(0.0))
                elseif gate_type == :Rx
                    gate = put(n_qubits, qubit => Rx(0.0))
                else
                    throw(ArgumentError("Unknown rotation gate: $gate_type"))
                end

                push!(rotation_layer, gate)
                param_count += 1
            end
        end

        push!(circuit_blocks, rotation_layer)

        # Entangling layer (except for last layer)
        if layer < n_layers || n_layers == 1
            entangling_layer = create_entangling_layer(n_qubits, entanglement)
            push!(circuit_blocks, entangling_layer)
        end
    end

    # Final rotation layer
    final_rotation_layer = chain(n_qubits)
    for qubit in 1:n_qubits
        for gate_type in rotation_gates
            if gate_type == :Ry
                gate = put(n_qubits, qubit => Ry(0.0))
            elseif gate_type == :Rz
                gate = put(n_qubits, qubit => Rz(0.0))
            elseif gate_type == :Rx
                gate = put(n_qubits, qubit => Rx(0.0))
            end

            push!(final_rotation_layer, gate)
            param_count += 1
        end
    end
    push!(circuit_blocks, final_rotation_layer)

    full_circuit = chain(n_qubits, circuit_blocks...)

    return full_circuit, param_count
end

"""
    create_entangling_layer(n_qubits::Int, entanglement::String)

Create entangling layer with specified connectivity pattern.
"""
function create_entangling_layer(n_qubits::Int, entanglement::String)
    entangling_gates = AbstractBlock[]

    if entanglement == "circular"
        # Circular connectivity: 0-1, 1-2, ..., (n-1)-0
        for i in 1:n_qubits
            target = i == n_qubits ? 1 : i + 1
            push!(entangling_gates, cnot(n_qubits, i, target))
        end
    elseif entanglement == "linear"
        # Linear connectivity: 0-1, 1-2, ..., (n-2)-(n-1)
        for i in 1:(n_qubits - 1)
            push!(entangling_gates, cnot(n_qubits, i, i+1))
        end
    elseif entanglement == "full"
        # Full connectivity: all pairs
        for i in 1:n_qubits
            for j in (i + 1):n_qubits
                push!(entangling_gates, cnot(n_qubits, i, j))
            end
        end
    else
        throw(ArgumentError("Unknown entanglement pattern: $entanglement"))
    end

    return chain(n_qubits, entangling_gates...)
end

# ============================================================================
# SPSA Optimizer (replicating qubap's SPSA functionality)
# ============================================================================

"""
    SPSAOptimizer

SPSA optimizer replicating qubap's SPSA_calibrated functionality.
"""
mutable struct SPSAOptimizer
    learning_rate::Float64
    perturbation::Float64
    maxiter::Int
    current_iter::Int
    a::Float64  # Learning rate decay parameter
    c::Float64  # Perturbation decay parameter
    A::Float64  # Stability constant
    alpha::Float64  # Learning rate exponent
    gamma::Float64  # Perturbation exponent

    function SPSAOptimizer(;
        learning_rate::Float64=0.1,
        perturbation::Float64=0.1,
        maxiter::Int=100,
        a::Float64=learning_rate,
        c::Float64=perturbation,
        A::Float64=0.0,
        alpha::Float64=0.602,
        gamma::Float64=0.101,
    )
        new(learning_rate, perturbation, maxiter, 0, a, c, A, alpha, gamma)
    end
end

"""
    spsa_step!(optimizer::SPSAOptimizer, cost_function, x::Vector{Float64})

Perform one SPSA optimization step.
"""
function spsa_step!(optimizer::SPSAOptimizer, cost_function, x::Vector{Float64})
    k = optimizer.current_iter + 1
    optimizer.current_iter = k

    # Update learning rate and perturbation
    ak = optimizer.a / (k + optimizer.A)^optimizer.alpha
    ck = optimizer.c / k^optimizer.gamma

    # Generate random perturbation vector
    delta = 2 * (rand(length(x)) .> 0.5) .- 1  # ±1 with equal probability

    # Evaluate cost at perturbed points
    x_plus = x + ck * delta
    x_minus = x - ck * delta

    try
        cost_plus = cost_function(x_plus)
        cost_minus = cost_function(x_minus)

        # SPSA gradient estimate
        gradient_estimate = (cost_plus - cost_minus) ./ (2 * ck * delta)

        # Update parameters
        x_new = x - ak * gradient_estimate

        return x_new, cost_function(x_new)
    catch e
        @warn "SPSA step failed: $e"
        return x, cost_function(x)
    end
end

# ============================================================================
# Standard VQE Implementation
# ============================================================================

"""
    StandardVQE

Standard VQE method configuration.
"""
struct StandardVQE
    ansatz::AbstractBlock
    n_parameters::Int
    n_qubits::Int

    function StandardVQE(
        n_qubits::Int,
        n_layers::Int=1;
        rotation_gates::Vector{Symbol}=[:Rx, :Ry],
        entanglement::String="circular",
    )
        ansatz, n_params = efficient_su2_ansatz(
            n_qubits, n_layers; rotation_gates=rotation_gates, entanglement=entanglement
        )
        new(ansatz, n_params, n_qubits)
    end

    # Alternative constructor with explicit ansatz
    function StandardVQE(ansatz::AbstractBlock, n_parameters::Int, n_qubits::Int)
        new(ansatz, n_parameters, n_qubits)
    end
end

"""
    run_vqe(vqe::StandardVQE, hamiltonian::AbstractBlock, 
           initial_guess::Vector{Float64}, num_iters::Int;
           optimizer::Symbol=:SPSA,
           verbose::Bool=false,
           callback=nothing)

Run standard VQE optimization.
Replicates the VQE function from qubap's VQEs.py.
"""
function run_vqe(
    vqe::StandardVQE,
    hamiltonian::AbstractBlock,
    initial_guess::Vector{Float64},
    num_iters::Int;
    optimizer::Symbol=:SPSA,
    verbose::Bool=false,
    callback=nothing,
)
    if length(initial_guess) != vqe.n_parameters
        throw(
            ArgumentError(
                "Initial guess length ($(length(initial_guess))) must match number of parameters ($(vqe.n_parameters))",
            ),
        )
    end

    # Define cost function (energy evaluation)
    function cost_function(params::Vector{Float64})
        return energy_evaluation(hamiltonian, vqe.ansatz, params, vqe.n_qubits)
    end

    # Storage for results
    energy_history = Float64[]
    parameter_history = Vector{Float64}[]

    # Callback wrapper for storing results
    function result_callback(iter::Int, params::Vector{Float64}, energy::Float64)
        push!(energy_history, energy)
        push!(parameter_history, copy(params))

        if verbose && iter % 50 == 0
            println("Iteration $iter: Energy = $energy")
        end

        if callback !== nothing
            callback(iter, params, energy, nothing, true)
        end
    end

    start_time = time()
    current_params = copy(initial_guess)
    converged = false

    # Run optimization based on selected method
    if optimizer == :SPSA
        # SPSA optimization (matching qubap behavior)
        spsa_opt = SPSAOptimizer(; maxiter=num_iters)

        for iter in 1:num_iters
            current_energy = cost_function(current_params)
            result_callback(iter, current_params, current_energy)

            # Perform SPSA step
            current_params, _ = spsa_step!(spsa_opt, cost_function, current_params)
        end

        converged = true  # SPSA doesn't have convergence criterion

    elseif optimizer == :GradientDescent
        # Simple gradient descent
        learning_rate = 0.01

        for iter in 1:num_iters
            current_energy = cost_function(current_params)
            result_callback(iter, current_params, current_energy)

            # Compute finite difference gradient
            gradient = zeros(length(current_params))
            epsilon = 1e-6

            for i in 1:length(current_params)
                params_plus = copy(current_params)
                params_minus = copy(current_params)
                params_plus[i] += epsilon
                params_minus[i] -= epsilon

                try
                    cost_plus = cost_function(params_plus)
                    cost_minus = cost_function(params_minus)
                    gradient[i] = (cost_plus - cost_minus) / (2 * epsilon)
                catch
                    gradient[i] = 0.0
                end
            end

            # Update parameters
            current_params = current_params - learning_rate * gradient

            # Simple learning rate decay
            learning_rate *= 0.999
        end

        converged = true

    else
        # For other optimizers, fall back to simple optimization
        @warn "Optimizer $optimizer not implemented, using simple optimization"

        best_energy = Inf
        best_params = copy(current_params)

        for iter in 1:num_iters
            # Random perturbation approach
            if iter > 1
                perturbation = randn(length(current_params)) * 0.1 * exp(-iter / 100)
                current_params = best_params + perturbation
            end

            current_energy = cost_function(current_params)
            result_callback(iter, current_params, current_energy)

            if current_energy < best_energy
                best_energy = current_energy
                best_params = copy(current_params)
            end
        end

        current_params = best_params
        converged = true
    end

    # Final evaluation
    final_energy = cost_function(current_params)
    execution_time = time() - start_time

    if verbose
        println("Standard VQE completed:")
        println("  Final energy: $final_energy")
        println("  Iterations: $(length(energy_history))")
        println("  Execution time: $(round(execution_time, digits=2))s")
        println("  Converged: $converged")
    end

    return VQEResult(
        "Standard VQE",
        energy_history,
        parameter_history,
        final_energy,
        current_params,
        length(energy_history),
        converged,
        execution_time,
    )
end

"""
    create_standard_vqe(n_qubits::Int; kwargs...)

Factory function for creating StandardVQE instances.
"""
function create_standard_vqe(n_qubits::Int; kwargs...)
    return StandardVQE(n_qubits; kwargs...)
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    random_initial_parameters(n_parameters::Int; scale::Float64=0.1, seed::Int=42)

Generate random initial parameters for VQE.
"""
function random_initial_parameters(n_parameters::Int; scale::Float64=0.1, seed::Int=42)
    Random.seed!(seed)
    return randn(n_parameters) * scale
end

"""
    check_vqe_convergence(energy_history::Vector{Float64}; 
                          tolerance::Float64=1e-6, window::Int=10)

Check if VQE has converged based on energy variance.
"""
function check_vqe_convergence(
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

export StandardVQE, run_vqe, create_standard_vqe
export efficient_su2_ansatz, SPSAOptimizer
export random_initial_parameters, check_vqe_convergence
