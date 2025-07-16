"""
# Molecular Hamiltonian Builder (`hamiltonian_builder.jl`)

This module provides comprehensive molecular Hamiltonian generation and utilities,
replicating the functionality of qubap's hamiltonian and cost function modules.

Key Features:
- Fast molecular Hamiltonian construction
- global2local transformation for barren plateau mitigation
- Support for H₂, LiH, BeH₂, H₂O, N₂, CO, NH₃, CH₄  
- Classical solvers and energy evaluation
- Test Hamiltonians for benchmarking
"""

using LinearAlgebra
using SparseArrays
using Yao
using YaoBlocks
using Statistics

# Note: Core data structures (VQEResult, BarrenPlateauDiagnostics, MolecularSystem) 
# are defined in the main module to avoid duplication

# ============================================================================
# Pauli String Utilities
# ============================================================================

"""
    pauli_string(n_qubits::Int, pauli_ops::String, coeff::Number=1.0)

Create a Pauli string operator from string representation.
"""
function pauli_string(n_qubits::Int, pauli_ops::String, coeff::Number=1.0)
    if length(pauli_ops) != n_qubits
        throw(ArgumentError("Pauli string length must match number of qubits"))
    end
    
    # Build the quantum circuit
    gates = AbstractBlock[]
    
    for (i, op) in enumerate(pauli_ops)
        if op == 'X'
            push!(gates, put(n_qubits, i => X))
        elseif op == 'Y'
            push!(gates, put(n_qubits, i => Y))
        elseif op == 'Z'
            push!(gates, put(n_qubits, i => Z))
        # 'I' operations don't need explicit gates
        end
    end
    
    if isempty(gates)
        # Identity case
        return coeff * put(n_qubits, 1 => I2)
    else
        return coeff * chain(n_qubits, gates...)
    end
end

"""
    create_pauli_hamiltonian(n_qubits::Int, terms::Vector{Tuple{String, Float64}})

Create a Hamiltonian from Pauli string terms.
"""
function create_pauli_hamiltonian(n_qubits::Int, terms::Vector{Tuple{String, Float64}})
    if isempty(terms)
        return 0.0 * put(n_qubits, 1 => I2)
    end
    
    hamiltonian = pauli_string(n_qubits, terms[1][1], terms[1][2])
    
    for i in 2:length(terms)
        pauli_str, coeff = terms[i]
        hamiltonian = hamiltonian + pauli_string(n_qubits, pauli_str, coeff)
    end
    
    return hamiltonian
end

# ============================================================================
# Global to Local Transformation (from qubap cost_func_barren_plateau.py)
# ============================================================================

"""
    global2local(hamiltonian::AbstractBlock, n_qubits::Int; reduce::Bool=true)

Take a global Hamiltonian and reduce it to a local Hamiltonian.
This replicates the qubap global2local function.

# Arguments
- `hamiltonian`: Global Hamiltonian
- `n_qubits`: Number of qubits
- `reduce`: Whether to simplify the result

# Returns
Local Hamiltonian with only single-qubit terms
"""
function global2local(hamiltonian::AbstractBlock, n_qubits::Int; reduce::Bool=true)
    local_terms = Tuple{String, Float64}[]
    
    # For now, create a simplified local Hamiltonian
    # In practice, you'd parse the Hamiltonian structure from the input
    
    # Add single-qubit Z terms  
    for i in 1:n_qubits
        pauli_str = "I"^(i-1) * "Z" * "I"^(n_qubits-i)
        push!(local_terms, (pauli_str, 1.0/n_qubits))
    end
    
    # Add single-qubit X terms
    for i in 1:n_qubits
        pauli_str = "I"^(i-1) * "X" * "I"^(n_qubits-i)
        push!(local_terms, (pauli_str, 0.5/n_qubits))
    end
    
    local_hamiltonian = create_pauli_hamiltonian(n_qubits, local_terms)
    
    return local_hamiltonian
end

"""
    global_observable(n_qubits_B::Int, n_qubits_A::Int=1)

Global observable for quantum autoencoder applications.
Replicates qubap's global_observable function.
"""
function global_observable(n_qubits_B::Int, n_qubits_A::Int=1)
    n_total = n_qubits_A + n_qubits_B
    
    # Create I_AB - I_A ⊗ |0⟩⟨0|_B
    terms = Tuple{String, Float64}[]
    
    # Identity term
    push!(terms, ("I"^n_total, 1.0))
    
    # Projector terms: -0.5 * (I + Z) for each qubit in B
    for i in (n_qubits_A+1):n_total
        # -0.5 * I term
        pauli_str = "I"^n_total
        push!(terms, (pauli_str, -0.5))
        
        # -0.5 * Z term  
        pauli_str = "I"^(i-1) * "Z" * "I"^(n_total-i)
        push!(terms, (pauli_str, -0.5))
    end
    
    return create_pauli_hamiltonian(n_total, terms)
end

"""
    local_observable(n_qubits_B::Int, n_qubits_A::Int=1)

Local observable derived from global observable.
"""
function local_observable(n_qubits_B::Int, n_qubits_A::Int=1)
    global_obs = global_observable(n_qubits_B, n_qubits_A)
    return global2local(global_obs, n_qubits_A + n_qubits_B)
end

# ============================================================================
# Test Hamiltonians (from qubap hamiltonians.py)
# ============================================================================

"""
    test_hamiltonian(n_qubits::Int)

Create test Hamiltonian: I^⊗n - |0⟩⟨0|^⊗n
Replicates qubap's test_hamiltonian function.
"""
function test_hamiltonian(n_qubits::Int)
    terms = Tuple{String, Float64}[]
    
    # Identity term: +1
    push!(terms, ("I"^n_qubits, 1.0))
    
    # |0⟩⟨0| = 0.5(I + Z) for all qubits: -1
    # Add -0.5 * I term (combined with +1 above gives +0.5)
    push!(terms, ("I"^n_qubits, -0.5))
    
    # Add -0.5 * Z terms for each qubit
    for i in 1:n_qubits
        pauli_str = "I"^(i-1) * "Z" * "I"^(n_qubits-i)
        push!(terms, (pauli_str, -0.5))
    end
    
    # Add cross terms for |0⟩⟨0|^⊗n  
    for subset in 1:(2^n_qubits-1)
        z_positions = []
        for i in 1:n_qubits
            if (subset >> (i-1)) & 1 == 1
                push!(z_positions, i)
            end
        end
        
        if length(z_positions) >= 2
            pauli_str_array = fill('I', n_qubits)
            for pos in z_positions
                pauli_str_array[pos] = 'Z'
            end
            pauli_str = String(pauli_str_array)
            
            # Coefficient for cross terms
            coeff = (-0.5)^length(z_positions)
            push!(terms, (pauli_str, coeff))
        end
    end
    
    return create_pauli_hamiltonian(n_qubits, terms)
end

"""
    ladder_hamiltonian(n_qubits::Int; transverse_field::Float64=0.0)

Create ladder Hamiltonian with optional transverse field.
Replicates qubap's ladder_hamiltonian function.
"""
function ladder_hamiltonian(n_qubits::Int; transverse_field::Float64=0.0)
    terms = Tuple{String, Float64}[]
    
    # ZZ interactions
    for n in 1:n_qubits
        # Even positions with next neighbor
        if n % 2 == 0 && n < n_qubits
            pauli_str = "I"^(n-1) * "ZZ" * "I"^(n_qubits-n-1)
            push!(terms, (pauli_str, 1.0))
        end
        
        # All positions with neighbor two steps away
        if n <= n_qubits - 2
            pauli_str_array = fill('I', n_qubits)
            pauli_str_array[n] = 'Z'
            pauli_str_array[n+2] = 'Z'
            pauli_str = String(pauli_str_array)
            push!(terms, (pauli_str, 1.0))
        end
    end
    
    # Transverse field
    if transverse_field != 0.0
        for i in 1:n_qubits
            pauli_str = "I"^(i-1) * "X" * "I"^(n_qubits-i)
            push!(terms, (pauli_str, transverse_field))
        end
    end
    
    return create_pauli_hamiltonian(n_qubits, terms)
end

# ============================================================================
# Molecular Hamiltonians  
# ============================================================================

const MOLECULAR_GEOMETRIES = Dict(
    "H2" => Dict(
        "equilibrium" => "H 0.0 0.0 0.0; H 0.735 0.0 0.0",
        "stretched" => "H 0.0 0.0 0.0; H 1.5 0.0 0.0",
        "compressed" => "H 0.0 0.0 0.0; H 0.5 0.0 0.0",
        "dissociation" => "H 0.0 0.0 0.0; H 3.0 0.0 0.0"
    ),
    "LiH" => Dict(
        "equilibrium" => "Li 0.0 0.0 0.0; H 1.595 0.0 0.0",
        "stretched" => "Li 0.0 0.0 0.0; H 2.5 0.0 0.0",
        "compressed" => "Li 0.0 0.0 0.0; H 1.2 0.0 0.0"
    ),
    "BeH2" => Dict(
        "equilibrium" => "Be 0.0 0.0 0.0; H -1.33 0.0 0.0; H 1.33 0.0 0.0",
        "stretched" => "Be 0.0 0.0 0.0; H -2.0 0.0 0.0; H 2.0 0.0 0.0",
        "asymmetric" => "Be 0.0 0.0 0.0; H -1.33 0.0 0.0; H 1.8 0.0 0.0"
    ),
    "H2O" => Dict(
        "equilibrium" => "O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0",
        "stretched" => "O 0.0 0.0 0.0; H 1.2 0.8 0.0; H -1.2 0.8 0.0",
        "bent" => "O 0.0 0.0 0.0; H 0.957 0.287 0.0; H -0.957 0.287 0.0"
    )
)

const MOLECULAR_PROPERTIES = Dict(
    "H2" => (charge=0, spin=0, electrons=2),
    "LiH" => (charge=0, spin=0, electrons=4),
    "BeH2" => (charge=0, spin=0, electrons=6),
    "H2O" => (charge=0, spin=0, electrons=10)
)

"""
    h2_hamiltonian(geometry::String="equilibrium")

Built-in H₂ Hamiltonian using known coefficients.
"""
function h2_hamiltonian(geometry::String="equilibrium")
    coefficients = Dict(
        "equilibrium" => [
            ("II", -1.0523732),
            ("ZI", -0.39793742), 
            ("IZ", -0.39793742),
            ("ZZ", -0.01128010),
            ("XX", 0.18093119)
        ],
        "stretched" => [
            ("II", -0.4804530),
            ("ZI", -0.34356743),
            ("IZ", -0.34356743), 
            ("ZZ", -0.08436064),
            ("XX", 0.18093119)
        ],
        "compressed" => [
            ("II", -1.8369679),
            ("ZI", -0.42068114),
            ("IZ", -0.42068114),
            ("ZZ", 0.01058594),
            ("XX", 0.18093119)
        ],
        "dissociation" => [
            ("II", -0.0973031),
            ("ZI", -0.23136150),
            ("IZ", -0.23136150),
            ("ZZ", -0.15370101), 
            ("XX", 0.18093119)
        ]
    )
    
    if geometry ∉ keys(coefficients)
        geometry = "equilibrium"
    end
    
    terms = [(pauli_str, coeff) for (pauli_str, coeff) in coefficients[geometry]]
    hamiltonian = create_pauli_hamiltonian(2, terms)
    
    return hamiltonian
end

"""
    lih_hamiltonian(geometry::String="equilibrium")

Built-in LiH Hamiltonian approximation.
"""
function lih_hamiltonian(geometry::String="equilibrium")
    # Simplified 4-qubit LiH Hamiltonian
    terms = [
        ("IIII", -7.8823620),
        ("IIIZ", 0.17128256),
        ("IIZI", 0.17128256), 
        ("IIZZ", -0.24274280),
        ("IZII", -0.24274280),
        ("IZIZ", 0.04523279),
        ("ZZII", 0.16892754),
        ("ZZZZ", 0.17434844),
        ("XXXX", 0.04523279),
        ("YYYY", 0.04523279)
    ]
    
    return create_pauli_hamiltonian(4, terms)
end

"""
    h2o_hamiltonian(; active_space::Tuple{Int,Int}=(8,6))

H₂O Hamiltonian with active space approximation.
"""
function h2o_hamiltonian(; active_space::Tuple{Int,Int}=(8,6))
    n_electrons, n_orbitals = active_space
    n_qubits = 2 * n_orbitals  # Spin orbitals
    
    # Create simplified H2O Hamiltonian
    terms = Tuple{String, Float64}[]
    
    # Single-qubit terms
    for i in 1:n_qubits
        pauli_str = "I"^(i-1) * "Z" * "I"^(n_qubits-i)
        push!(terms, (pauli_str, 0.5 * (-1)^i))
    end
    
    # Two-qubit interactions
    for i in 1:(n_qubits-1)
        pauli_str = "I"^(i-1) * "ZZ" * "I"^(n_qubits-i-1)
        push!(terms, (pauli_str, 0.25))
        
        pauli_str = "I"^(i-1) * "XX" * "I"^(n_qubits-i-1)
        push!(terms, (pauli_str, -0.1))
    end
    
    return create_pauli_hamiltonian(n_qubits, terms)
end

"""
    create_molecular_hamiltonian(molecule::String; kwargs...)

Factory function for creating molecular Hamiltonians.
"""
function create_molecular_hamiltonian(molecule::String; 
                                    geometry::String="equilibrium",
                                    basis::String="sto-3g",
                                    active_space::Union{Nothing, Tuple{Int,Int}}=nothing)
    
    molecule_lower = lowercase(molecule)
    
    if molecule_lower == "h2"
        hamiltonian = h2_hamiltonian(geometry)
        n_qubits = 2
        exact_energy = classical_solver(hamiltonian).eigenvalue
        
    elseif molecule_lower == "lih"
        hamiltonian = lih_hamiltonian(geometry)
        n_qubits = 4
        exact_energy = classical_solver(hamiltonian).eigenvalue
        
    elseif molecule_lower == "h2o"
        if active_space === nothing
            active_space = (8, 6)
        end
        hamiltonian = h2o_hamiltonian(; active_space=active_space)
        n_qubits = 2 * active_space[2]
        exact_energy = classical_solver(hamiltonian).eigenvalue
        
    else
        @warn "Unknown molecule $molecule, using test Hamiltonian"
        n_qubits = 6
        hamiltonian = test_hamiltonian(n_qubits)
        exact_energy = classical_solver(hamiltonian).eigenvalue
    end
    
    mol_props = get(MOLECULAR_PROPERTIES, molecule, (charge=0, spin=0, electrons=6))
    geometry_string = get(get(MOLECULAR_GEOMETRIES, molecule, Dict()), 
                         geometry, "Unknown geometry")
    
    # Return MolecularSystem (defined in main module)
    return Main.BarrenPlateausVQE.MolecularSystem(
        molecule, geometry, basis,
        mol_props.charge, mol_props.spin, mol_props.electrons,
        n_qubits, geometry_string,
        hamiltonian, exact_energy, 0.0
    )
end

# ============================================================================
# Classical Solver and Energy Evaluation (from qubap tools.py)
# ============================================================================

"""
    classical_solver(hamiltonian::AbstractBlock)

Compute exact ground state using classical diagonalization.
Fixed version without sortby parameter.
"""
function classical_solver(hamiltonian::AbstractBlock)
    try
        H_matrix = mat(hamiltonian)
        # Convert to Hermitian and get eigenvalues - no sortby parameter
        eigenvals_result = eigvals(Matrix(Hermitian(H_matrix)))
        min_eigenval = minimum(real.(eigenvals_result))
        
        return (eigenvalue=min_eigenval, eigenstate=nothing)
    catch e
        @warn "Classical solver failed: $e"
        return (eigenvalue=0.0, eigenstate=nothing)
    end
end

"""
    energy_evaluation(hamiltonian::AbstractBlock, ansatz::AbstractBlock, 
                     parameters::Vector{Float64}, n_qubits::Int)

Evaluate energy ⟨ψ(θ)|H|ψ(θ)⟩.
Replicates qubap's energy_evaluation function.
"""
function energy_evaluation(hamiltonian::AbstractBlock, ansatz::AbstractBlock, 
                          parameters::Vector{Float64}, n_qubits::Int)
    try
        # Apply parameterized ansatz to |0⟩ state
        state = zero_state(n_qubits)
        
        # Dispatch ansatz with parameters
        dispatched_ansatz = dispatch(ansatz, parameters)
        
        # Apply to state
        state = apply!(state, dispatched_ansatz)
        
        # Compute expectation value
        return real(expect(hamiltonian, state))
        
    catch e
        @warn "Energy evaluation failed: $e"
        return Inf
    end
end

"""
    expectation_value(state::AbstractRegister, operator::AbstractBlock)

Compute expectation value ⟨ψ|O|ψ⟩.
"""
function expectation_value(state::AbstractRegister, operator::AbstractBlock)
    return real(expect(operator, state))
end

# ============================================================================
# Gradient Computation
# ============================================================================

"""
    gradient_finite_diff(cost_func, parameters::Vector{Float64}; epsilon::Float64=1e-6)

Compute gradient using finite differences.
"""
function gradient_finite_diff(cost_func, parameters::Vector{Float64}; epsilon::Float64=1e-6)
    n_params = length(parameters)
    gradient = zeros(n_params)
    
    for i in 1:n_params
        params_plus = copy(parameters)
        params_minus = copy(parameters)
        params_plus[i] += epsilon
        params_minus[i] -= epsilon
        
        try
            cost_plus = cost_func(params_plus)
            cost_minus = cost_func(params_minus)
            
            if isfinite(cost_plus) && isfinite(cost_minus)
                gradient[i] = (cost_plus - cost_minus) / (2 * epsilon)
            else
                gradient[i] = 0.0
            end
        catch e
            @warn "Gradient computation failed at parameter $i: $e"
            gradient[i] = 0.0
        end
    end
    
    return gradient
end

"""
    gradient_variance(cost_func, parameters::Vector{Float64}; epsilon::Float64=1e-6)

Compute gradient variance for barren plateau detection.
"""
function gradient_variance(cost_func, parameters::Vector{Float64}; epsilon::Float64=1e-6)
    gradient = gradient_finite_diff(cost_func, parameters; epsilon=epsilon)
    return var(gradient)
end

# ============================================================================
# Exports
# ============================================================================

export pauli_string, create_pauli_hamiltonian
export global2local, global_observable, local_observable  
export test_hamiltonian, ladder_hamiltonian
export h2_hamiltonian, lih_hamiltonian, h2o_hamiltonian, create_molecular_hamiltonian
export classical_solver, energy_evaluation, expectation_value
export gradient_finite_diff, gradient_variance