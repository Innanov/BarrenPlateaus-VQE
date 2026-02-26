"""
# Adiabatic VQE Implementation (`methods/adiabatic_vqe.jl`)

Adiabatic VQE implementation with gradual transition from local to global Hamiltonian.
"""

using Yao
using YaoBlocks
using Random
using Statistics

# ============================================================================
# Adiabatic VQE Implementation
# ============================================================================

"""
    AdiabaticVQE

Adiabatic VQE method configuration.
"""
struct AdiabaticVQE
    ansatz::AbstractBlock
    n_parameters::Int
    n_qubits::Int
    n_layers::Int
    adiabatic_steps::Int

    function AdiabaticVQE(
        n_qubits::Int,
        n_layers::Int=1;
        rotation_gates::Vector{Symbol}=[:Rx, :Ry],
        entanglement::String="circular",
        adiabatic_steps::Int=10,
    )
        ansatz, n_params = efficient_su2_ansatz(
            n_qubits, n_layers; rotation_gates=rotation_gates, entanglement=entanglement
        )
        new(ansatz, n_params, n_qubits, n_layers, adiabatic_steps)
    end
end

"""
    interpolate_hamiltonian(H_local::AbstractBlock, H_global::AbstractBlock, s::Float64)

Interpolate between local and global Hamiltonians: H(s) = (1-s)H_local + s*H_global.
"""
function interpolate_hamiltonian(
    H_local::AbstractBlock, H_global::AbstractBlock, s::Float64
)
    if !(0.0 <= s <= 1.0)
        throw(ArgumentError("Interpolation parameter s must be in [0,1]"))
    end

    # Simple linear interpolation
    return (1.0 - s) * H_local + s * H_global
end

"""
    make_adiabatic_cost_function(H_local::AbstractBlock, H_global::AbstractBlock,
                                 ansatz::AbstractBlock, n_qubits::Int, s::Float64)

Create adiabatic cost function for a specific interpolation parameter s.
"""
function make_adiabatic_cost_function(
    H_local::AbstractBlock,
    H_global::AbstractBlock,
    ansatz::AbstractBlock,
    n_qubits::Int,
    s::Float64,
)

    # Create interpolated Hamiltonian
    H_interpolated = interpolate_hamiltonian(H_local, H_global, s)

    function cost_function(params::Vector{Float64})
        return energy_evaluation(H_interpolated, ansatz, params, n_qubits)
    end

    return cost_function
end

"""
    run_vqe(vqe::AdiabaticVQE, hamiltonian_in::AbstractBlock, hamiltonian_out::AbstractBlock,
           initial_guess::Vector{Float64}, num_iters::Int;
           transition_lims::Tuple{Float64,Float64}=(0.0, 1.0),
           optimizer::Symbol=:SPSA,
           verbose::Bool=false,
           callback=nothing)

Run adiabatic VQE optimization.
"""
function run_vqe(
    vqe::AdiabaticVQE,
    hamiltonian_in::AbstractBlock,
    hamiltonian_out::AbstractBlock,
    initial_guess::Vector{Float64},
    num_iters::Int;
    transition_lims::Tuple{Float64,Float64}=(0.0, 1.0),
    optimizer::Symbol=:SPSA,
    verbose::Bool=false,
    callback=nothing,
)
    if length(initial_guess) != vqe.n_parameters
        throw(ArgumentError("Initial guess length must match number of parameters"))
    end

    # Adiabatic schedule
    s_values = collect(
        range(transition_lims[1], transition_lims[2]; length=vqe.adiabatic_steps)
    )
    iterations_per_step = div(num_iters, vqe.adiabatic_steps)

    total_energy_history = Float64[]
    total_parameter_history = Vector{Float64}[]
    current_params = copy(initial_guess)
    total_execution_time = 0.0

    if verbose
        println(
            "Adiabatic VQE: $(vqe.adiabatic_steps) steps, $iterations_per_step iterations per step",
        )
    end

    for (step, s) in enumerate(s_values)
        if verbose
            println(
                "Adiabatic step $step/$(vqe.adiabatic_steps): s = $(round(s, digits=3))"
            )
        end

        # Create cost function for this interpolation parameter
        cost_function = make_adiabatic_cost_function(
            hamiltonian_in, hamiltonian_out, vqe.ansatz, vqe.n_qubits, s
        )

        # Create StandardVQE for this step
        step_vqe = StandardVQE(vqe.ansatz, vqe.n_parameters, vqe.n_qubits)

        # Run optimization for this step
        step_result = run_vqe_with_cost_function(
            step_vqe,
            cost_function,
            current_params,
            iterations_per_step;
            optimizer=optimizer,
            verbose=false,
        )

        # Accumulate results
        append!(total_energy_history, step_result.energy_history)
        append!(total_parameter_history, step_result.parameter_history)
        current_params = step_result.final_parameters
        total_execution_time += step_result.execution_time
    end

    # Final energy with respect to target Hamiltonian
    final_energy = energy_evaluation(
        hamiltonian_out, vqe.ansatz, current_params, vqe.n_qubits
    )

    if verbose
        println("Adiabatic VQE completed:")
        println("  Final energy (target Hamiltonian): $final_energy")
        println("  Total iterations: $(length(total_energy_history))")
        println("  Execution time: $(round(total_execution_time, digits=2))s")
    end

    return VQEResult(
        "Adiabatic VQE",
        total_energy_history,
        total_parameter_history,
        final_energy,
        current_params,
        length(total_energy_history),
        true,  # Assume converged if completed all steps
        total_execution_time,
    )
end

"""
    run_vqe(vqe::AdiabaticVQE, hamiltonian::AbstractBlock, 
           initial_guess::Vector{Float64}, num_iters::Int;
           kwargs...)

Simplified interface that automatically creates local Hamiltonian.
"""
function run_vqe(
    vqe::AdiabaticVQE,
    hamiltonian::AbstractBlock,
    initial_guess::Vector{Float64},
    num_iters::Int;
    kwargs...,
)

    # Create local Hamiltonian automatically
    local_hamiltonian = global2local(hamiltonian, vqe.n_qubits)

    return run_vqe(vqe, local_hamiltonian, hamiltonian, initial_guess, num_iters; kwargs...)
end

"""
    run_vqe_with_cost_function(vqe::StandardVQE, cost_function, 
                               initial_guess::Vector{Float64}, num_iters::Int;
                               optimizer::Symbol=:SPSA, verbose::Bool=false)

Run VQE optimization with a custom cost function.
"""
function run_vqe_with_cost_function(
    vqe::StandardVQE,
    cost_function,
    initial_guess::Vector{Float64},
    num_iters::Int;
    optimizer::Symbol=:SPSA,
    verbose::Bool=false,
)

    # Storage for results
    energy_history = Float64[]
    parameter_history = Vector{Float64}[]

    # Callback for storing results
    function result_callback(iter::Int, params::Vector{Float64}, energy::Float64)
        push!(energy_history, energy)
        push!(parameter_history, copy(params))

        if verbose && iter % 50 == 0
            println("Iteration $iter: Energy = $energy")
        end
    end

    start_time = time()
    current_params = copy(initial_guess)
    converged = false

    spsa_opt = SPSAOptimizer(; maxiter=num_iters)

    for iter in 1:num_iters
        current_energy = cost_function(current_params)
        result_callback(iter, current_params, current_energy)
        current_params, _ = spsa_step!(spsa_opt, cost_function, current_params)
    end

    converged = true
    final_energy = cost_function(current_params)
    execution_time = time() - start_time

    return VQEResult(
        "Adiabatic Step",
        energy_history,
        parameter_history,
        final_energy,
        current_params,
        length(energy_history),
        converged,
        execution_time,
    )
end

# ============================================================================
# Adiabatic Schedules
# ============================================================================

"""
    linear_schedule(iteration::Int, total_iterations::Int)

Linear adiabatic schedule: s(t) = t/T
"""
function linear_schedule(iteration::Int, total_iterations::Int)
    return iteration / total_iterations
end

"""
    polynomial_schedule(iteration::Int, total_iterations::Int, power::Float64=3.0)

Polynomial adiabatic schedule: s(t) = (t/T)^power
"""
function polynomial_schedule(iteration::Int, total_iterations::Int, power::Float64=3.0)
    return (iteration / total_iterations)^power
end

"""
    custom_adiabatic_schedule(schedule_function::Function, 
                             transition_lims::Tuple{Float64,Float64}=(0.0, 1.0))

Create custom adiabatic schedule function.
"""
function custom_adiabatic_schedule(
    schedule_function::Function, transition_lims::Tuple{Float64,Float64}=(0.0, 1.0)
)
    function get_a(iteration_fraction::Float64)
        a1, a2 = minimum(transition_lims), maximum(transition_lims)

        if iteration_fraction <= a1
            return 0.0
        elseif iteration_fraction >= a2
            return 1.0
        else
            # Map [a1, a2] to [0, 1] and apply schedule
            normalized_fraction = (iteration_fraction - a1) / (a2 - a1)
            return schedule_function(normalized_fraction, 1.0)
        end
    end

    return get_a
end

# ============================================================================
# Convenience Functions
# ============================================================================

"""
    create_adiabatic_vqe(n_qubits::Int; kwargs...)

Factory function for creating AdiabaticVQE instances.
"""
function create_adiabatic_vqe(n_qubits::Int; kwargs...)
    return AdiabaticVQE(n_qubits; kwargs...)
end

"""
    run_adiabatic_vqe(hamiltonian_local::AbstractBlock, hamiltonian_global::AbstractBlock,
                      n_qubits::Int, initial_guess::Vector{Float64}, num_iters::Int;
                      kwargs...)

Convenience function for running adiabatic VQE with automatic ansatz creation.
"""
function run_adiabatic_vqe(
    hamiltonian_local::AbstractBlock,
    hamiltonian_global::AbstractBlock,
    n_qubits::Int,
    initial_guess::Vector{Float64},
    num_iters::Int;
    n_layers::Int=1,
    kwargs...,
)
    vqe = AdiabaticVQE(n_qubits, n_layers)

    # Generate initial guess if not provided correctly
    if length(initial_guess) != vqe.n_parameters
        @warn "Initial guess length ($(length(initial_guess))) doesn't match ansatz parameters ($(vqe.n_parameters)). Generating new initial guess."
        initial_guess = random_initial_parameters(vqe.n_parameters)
    end

    return run_vqe(
        vqe, hamiltonian_local, hamiltonian_global, initial_guess, num_iters; kwargs...
    )
end

# ============================================================================
# Exports
# ============================================================================

export AdiabaticVQE, run_vqe, create_adiabatic_vqe, run_adiabatic_vqe
export interpolate_hamiltonian, make_adiabatic_cost_function
export linear_schedule, polynomial_schedule, custom_adiabatic_schedule
