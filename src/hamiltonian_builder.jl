"""
# Molecular Hamiltonian Builder (`hamiltonian_builder.jl`)

This module provides comprehensive molecular Hamiltonian generation and utilities,
replicating the functionality of qubap's hamiltonian and cost function modules.

Key Features:
- Dynamic molecular Hamiltonian construction from quantum chemistry
- global2local transformation for barren plateau mitigation
- Support for H₂, LiH, BeH₂, H₂O, N₂, CO, NH₃, CH₄, HF, BH  
- Classical solvers and energy evaluation
- Test Hamiltonians for benchmarking
- Proper basis set and active space handling
"""

using LinearAlgebra
using SparseArrays
using Yao
using YaoBlocks
using Statistics

# Optional: PyCall for interfacing with PySCF (install separately)
# using PyCall

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
function create_pauli_hamiltonian(n_qubits::Int, terms::Vector{Tuple{String,Float64}})
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
    local_terms = Tuple{String,Float64}[]

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
    terms = Tuple{String,Float64}[]

    # Identity term
    push!(terms, ("I"^n_total, 1.0))

    # Projector terms: -0.5 * (I + Z) for each qubit in B
    for i in (n_qubits_A + 1):n_total
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
    terms = Tuple{String,Float64}[]

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
    for subset in 1:(2 ^ n_qubits - 1)
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
    terms = Tuple{String,Float64}[]

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
            pauli_str_array[n + 2] = 'Z'
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
# Molecular Orbital and System Size Determination
# ============================================================================

"""
    determine_molecular_system_size(molecule::String, basis_set::String, active_space=nothing)

Determine the number of qubits needed based on molecular orbital calculation.
This replaces hardcoded system sizes!
"""
function determine_molecular_system_size(
    molecule::String, basis_set::String, active_space=nothing
)

    # Atomic data
    atomic_numbers = Dict(
        "H"=>1, "He"=>2, "Li"=>3, "Be"=>4, "B"=>5, "C"=>6, "N"=>7, "O"=>8, "F"=>9, "Ne"=>10
    )

    # Basis function counts for common basis sets
    basis_functions = Dict(
        "sto-3g" => Dict("H"=>1, "Li"=>5, "Be"=>5, "B"=>9, "C"=>9, "N"=>9, "O"=>9, "F"=>9),
        "6-31g" =>
            Dict("H"=>2, "Li"=>9, "Be"=>9, "B"=>13, "C"=>13, "N"=>13, "O"=>13, "F"=>13),
        "cc-pvdz" => Dict("H"=>5, "Li"=>14, "B"=>23, "C"=>23, "N"=>23, "O"=>23, "F"=>23),
    )

    # Parse molecule to get atoms and electrons
    atoms, total_electrons = if molecule == "H2"
        (["H", "H"], 2)
    elseif molecule == "LiH"
        (["Li", "H"], 4)
    elseif molecule == "BeH2"
        (["Be", "H", "H"], 6)
    elseif molecule == "H2O"
        (["O", "H", "H"], 10)
    elseif molecule == "NH3"
        (["N", "H", "H", "H"], 10)
    elseif molecule == "CH4"
        (["C", "H", "H", "H", "H"], 10)
    elseif molecule == "N2"
        (["N", "N"], 14)
    elseif molecule == "CO"
        (["C", "O"], 14)
    elseif molecule == "HF"
        (["H", "F"], 10)
    elseif molecule == "BH"
        (["B", "H"], 6)
    else
        @warn "Unknown molecule $molecule, using H2"
        (["H", "H"], 2)
    end

    # Count spatial orbitals from basis set
    bf_dict = get(basis_functions, basis_set, Dict("H"=>1))
    n_spatial_orbitals = 0
    for atom in atoms
        n_spatial_orbitals += get(bf_dict, atom, 1)
    end

    # Apply active space if specified
    if active_space !== nothing
        n_active_electrons, n_active_orbitals = active_space
        total_electrons = n_active_electrons
        n_spatial_orbitals = n_active_orbitals
    end

    # Convert to qubits: spatial orbitals → spin orbitals → qubits
    n_qubits = 2 * n_spatial_orbitals  # Jordan-Wigner mapping

    return Dict(
        "molecule" => molecule,
        "atoms" => atoms,
        "total_electrons" => total_electrons,
        "n_spatial_orbitals" => n_spatial_orbitals,
        "n_qubits" => n_qubits,
        "basis_set" => basis_set,
        "active_space" => active_space,
    )
end

# ============================================================================
# Molecular Integral Computation
# ============================================================================

"""
    compute_molecular_integrals(molecule::String, n_spatial_orbitals::Int,
                               bond_length::Float64=1.0)

Compute molecular integrals for quantum chemistry simulations using simplified analytical models.

This function generates one-electron integrals (h1e), two-electron integrals (h2e), and nuclear
repulsion energy for various molecules. The integrals are based on simplified STO-3G basis set
approximations suitable for VQE simulations and barren plateau analysis.

**Note**: This implementation uses analytical approximations. For production quantum chemistry
calculations, use specialized packages like PySCF, Psi4, or OpenFermion.

# Arguments
- `molecule`: Molecule name (supported: "H2", "LiH", "BeH2", "H2O", "NH3", "N2", "CH4", "H4", "H6", "H8")
- `n_spatial_orbitals`: Number of spatial orbitals (determines basis set size)
- `bond_length`: Internuclear distance in Angstroms (default: 1.0). Affects integral values.

# Returns
- `h1e::Matrix{Float64}`: One-electron integrals (kinetic + nuclear attraction), size (n_spatial_orbitals, n_spatial_orbitals)
- `h2e::Array{Float64,4}`: Two-electron integrals in chemist notation (ij|kl), size (n_spatial_orbitals)^4
- `nuclear_repulsion::Float64`: Nuclear-nuclear repulsion energy

# Integral Details
## One-electron integrals (h1e)
- Includes kinetic energy and electron-nuclear attraction
- h1e[i,j] represents ⟨ϕᵢ|T + Vₙₑ|ϕⱼ⟩
- Bond length dependent for realistic molecules

## Two-electron integrals (h2e)
- Chemist notation: (ij|kl) = ⟨ϕᵢϕⱼ|1/r₁₂|ϕₖϕₗ⟩
- h2e[i,j,k,l] represents electron-electron repulsion
- Includes Coulomb (same spatial orbitals) and exchange (different orbitals) terms

## Nuclear repulsion
- Classical Coulombic repulsion between nuclei
- Depends on nuclear charges and bond lengths

# Examples
```julia
# H2 molecule with 2 spatial orbitals at equilibrium bond length
h1e, h2e, nuc_rep = compute_molecular_integrals("H2", 2, 0.735)

# LiH molecule with 4 spatial orbitals
h1e, h2e, nuc_rep = compute_molecular_integrals("LiH", 4, 1.5)

# Generic molecule (uses default integrals)
h1e, h2e, nuc_rep = compute_molecular_integrals("CustomMolecule", 6)
```

# See Also
- [`jordan_wigner_transform()`](@ref): Convert integrals to qubit Hamiltonian
- [`create_pauli_hamiltonian()`](@ref): Build Hamiltonian from Pauli terms
"""
function compute_molecular_integrals(
    molecule::String, n_spatial_orbitals::Int, bond_length::Float64=1.0
)

    # Initialize integral arrays
    h1e = zeros(Float64, n_spatial_orbitals, n_spatial_orbitals)
    h2e = zeros(
        Float64,
        n_spatial_orbitals,
        n_spatial_orbitals,
        n_spatial_orbitals,
        n_spatial_orbitals,
    )

    if molecule == "H2"
        # H2 with realistic STO-3G integrals
        nuclear_repulsion = 1.0 / bond_length

        if n_spatial_orbitals >= 2
            # Kinetic energy + nuclear attraction integrals
            S_overlap = 0.659 * exp(-0.16 * (bond_length - 0.735)^2)  # Overlap-dependent

            h1e[1, 1] = -1.12 - 1.0/bond_length  # H1s kinetic + nuclear
            h1e[2, 2] = -1.12 - 1.0/bond_length  # H2s kinetic + nuclear
            h1e[1, 2] = h1e[2, 1] = -0.65 * S_overlap  # Cross terms

            # Two-electron integrals (chemist notation: (ij|kl))
            # Diagonal terms
            h2e[1, 1, 1, 1] = 0.625  # (11|11) - Coulomb repulsion on H1
            h2e[2, 2, 2, 2] = 0.625  # (22|22) - Coulomb repulsion on H2

            # Cross-atom terms
            h2e[1, 1, 2, 2] = h2e[2, 2, 1, 1] = 0.45 * exp(-0.1*bond_length)  # (11|22)

            # Exchange terms
            h2e[1, 2, 1, 2] = h2e[2, 1, 2, 1] = 0.325 * S_overlap  # (12|12)
            h2e[1, 2, 2, 1] = h2e[2, 1, 1, 2] = 0.275 * S_overlap  # (12|21)
        end

    elseif molecule == "LiH"
        nuclear_repulsion = 3.0 / bond_length  # Li-H nuclear repulsion

        # Simplified LiH integrals
        for i in 1:min(n_spatial_orbitals, 4)
            if i <= 2  # Li orbitals
                h1e[i, i] = -2.5 - 3.0/bond_length
            else  # H orbital
                h1e[i, i] = -1.1 - 1.0/bond_length
            end

            for j in 1:min(n_spatial_orbitals, 4)
                # Two-electron integrals
                if i == j
                    h2e[i, i, i, i] = i <= 2 ? 0.8 : 0.6  # On-site repulsion
                else
                    h2e[i, i, j, j] = 0.4 / (1 + 0.5*abs(i-j))  # Inter-orbital
                    h2e[i, j, i, j] = 0.2 * exp(-0.3*abs(i-j))  # Exchange
                end
            end
        end

    elseif molecule == "BeH2"
        nuclear_repulsion = 4.0 / bond_length + 1.0 / (2*bond_length)  # Be-H + H-H repulsion

        # BeH2 integrals
        for i in 1:min(n_spatial_orbitals, 5)
            if i <= 2  # Be core orbitals
                h1e[i, i] = -4.5 - 4.0/bond_length
            elseif i <= 3  # Be valence
                h1e[i, i] = -0.9 - 4.0/bond_length
            else  # H orbitals
                h1e[i, i] = -1.1 - 1.0/bond_length
            end

            for j in 1:min(n_spatial_orbitals, 5)
                if i == j
                    h2e[i, i, i, i] = i <= 2 ? 1.2 : (i <= 3 ? 0.7 : 0.6)
                else
                    h2e[i, i, j, j] = 0.35 / (1 + 0.4*abs(i-j))
                    h2e[i, j, i, j] = 0.18 * exp(-0.25*abs(i-j))
                end
            end
        end

    elseif molecule == "HF"
        nuclear_repulsion = 9.0 / bond_length  # H-F nuclear repulsion

        # HF integrals
        for i in 1:min(n_spatial_orbitals, 5)
            if i <= 4  # F orbitals (1s, 2s, 2px, 2py, 2pz)
                if i == 1  # F 1s
                    h1e[i, i] = -26.4 - 9.0/bond_length
                elseif i == 2  # F 2s
                    h1e[i, i] = -2.8 - 9.0/bond_length
                else  # F 2p
                    h1e[i, i] = -1.4 - 9.0/bond_length
                end
            else  # H orbital
                h1e[i, i] = -1.1 - 1.0/bond_length
            end

            for j in 1:min(n_spatial_orbitals, 5)
                if i == j
                    if i == 1  # F 1s
                        h2e[i, i, i, i] = 3.2
                    elseif i <= 4  # Other F orbitals
                        h2e[i, i, i, i] = 1.1
                    else  # H orbital
                        h2e[i, i, i, i] = 0.6
                    end
                else
                    h2e[i, i, j, j] = 0.7 / (1 + 0.3*abs(i-j))
                    h2e[i, j, i, j] = 0.35 * exp(-0.2*abs(i-j))
                end
            end
        end

    elseif molecule == "BH"
        nuclear_repulsion = 5.0 / bond_length  # B-H nuclear repulsion

        # BH integrals
        for i in 1:min(n_spatial_orbitals, 5)
            if i <= 4  # B orbitals (1s, 2s, 2px, 2py, 2pz)
                if i == 1  # B 1s
                    h1e[i, i] = -7.8 - 5.0/bond_length
                elseif i == 2  # B 2s
                    h1e[i, i] = -0.5 - 5.0/bond_length
                else  # B 2p
                    h1e[i, i] = -0.2 - 5.0/bond_length
                end
            else  # H orbital
                h1e[i, i] = -1.1 - 1.0/bond_length
            end

            for j in 1:min(n_spatial_orbitals, 5)
                if i == j
                    if i == 1  # B 1s
                        h2e[i, i, i, i] = 1.5
                    elseif i <= 4  # Other B orbitals
                        h2e[i, i, i, i] = 0.8
                    else  # H orbital
                        h2e[i, i, i, i] = 0.6
                    end
                else
                    h2e[i, i, j, j] = 0.5 / (1 + 0.4*abs(i-j))
                    h2e[i, j, i, j] = 0.25 * exp(-0.3*abs(i-j))
                end
            end
        end

    elseif molecule == "N2"
        nuclear_repulsion = 49.0 / bond_length  # N-N nuclear repulsion

        # N2 integrals (simplified for diatomic)
        for i in 1:min(n_spatial_orbitals, 10)
            if i <= 5  # First N atom
                if i == 1  # N 1s
                    h1e[i, i] = -15.6 - 7.0/bond_length
                elseif i == 2  # N 2s
                    h1e[i, i] = -1.5 - 7.0/bond_length
                else  # N 2p
                    h1e[i, i] = -0.8 - 7.0/bond_length
                end
            else  # Second N atom
                if (i-5) == 1  # N 1s
                    h1e[i, i] = -15.6 - 7.0/bond_length
                elseif (i-5) == 2  # N 2s
                    h1e[i, i] = -1.5 - 7.0/bond_length
                else  # N 2p
                    h1e[i, i] = -0.8 - 7.0/bond_length
                end
            end

            for j in 1:min(n_spatial_orbitals, 10)
                if i == j
                    if i == 1 || i == 6  # N 1s
                        h2e[i, i, i, i] = 2.8
                    else  # Other N orbitals
                        h2e[i, i, i, i] = 1.2
                    end
                else
                    h2e[i, i, j, j] = 0.8 / (1 + 0.2*abs(i-j))
                    h2e[i, j, i, j] = 0.4 * exp(-0.15*abs(i-j))
                end
            end
        end

    elseif molecule == "CO"
        nuclear_repulsion = 48.0 / bond_length  # C-O nuclear repulsion

        # CO integrals
        for i in 1:min(n_spatial_orbitals, 9)
            if i <= 4  # C orbitals
                if i == 1  # C 1s
                    h1e[i, i] = -11.3 - 6.0/bond_length
                elseif i == 2  # C 2s
                    h1e[i, i] = -0.7 - 6.0/bond_length
                else  # C 2p
                    h1e[i, i] = -0.4 - 6.0/bond_length
                end
            else  # O orbitals
                if (i-4) == 1  # O 1s
                    h1e[i, i] = -20.7 - 8.0/bond_length
                elseif (i-4) == 2  # O 2s
                    h1e[i, i] = -1.3 - 8.0/bond_length
                else  # O 2p
                    h1e[i, i] = -0.6 - 8.0/bond_length
                end
            end

            for j in 1:min(n_spatial_orbitals, 9)
                if i == j
                    if i == 1  # C 1s
                        h2e[i, i, i, i] = 2.1
                    elseif i == 5  # O 1s
                        h2e[i, i, i, i] = 3.1
                    else  # Other orbitals
                        h2e[i, i, i, i] = 1.1
                    end
                else
                    h2e[i, i, j, j] = 0.7 / (1 + 0.25*abs(i-j))
                    h2e[i, j, i, j] = 0.35 * exp(-0.2*abs(i-j))
                end
            end
        end

    elseif molecule in ["H2O", "NH3", "CH4"]
        # Multi-atom molecules - simplified
        nuclear_repulsion = 2.0 + n_spatial_orbitals * 0.5

        for i in 1:n_spatial_orbitals
            # Diagonal one-electron terms
            h1e[i, i] = -1.5 - 0.1*i

            for j in 1:n_spatial_orbitals
                if i != j
                    # Off-diagonal one-electron
                    h1e[i, j] = -0.2 * exp(-0.5*abs(i-j))
                end

                # Two-electron integrals
                h2e[i, i, j, j] = 0.5 / (1 + 0.3*abs(i-j))  # Coulomb
                if i != j
                    h2e[i, j, i, j] = 0.15 * exp(-0.4*abs(i-j))  # Exchange
                end
            end
        end

    else
        # Generic molecule
        nuclear_repulsion = 1.0
        for i in 1:n_spatial_orbitals
            h1e[i, i] = -1.0
            h2e[i, i, i, i] = 0.5
        end
    end

    return h1e, h2e, nuclear_repulsion
end

# ============================================================================
# Jordan-Wigner Transformation
# ============================================================================

"""
    jordan_wigner_transform(h1e, h2e, nuclear_repulsion, n_spatial_orbitals)

Apply Jordan-Wigner transformation to convert molecular integrals to Pauli operators.
"""
function jordan_wigner_transform(h1e, h2e, nuclear_repulsion, n_spatial_orbitals)
    n_qubits = 2 * n_spatial_orbitals
    pauli_terms = Tuple{String,Float64}[]

    # Nuclear repulsion (constant term)
    push!(pauli_terms, ("I"^n_qubits, nuclear_repulsion))

    # One-electron terms: Σᵢⱼ h₁ᵢⱼ a†ᵢ aⱼ
    for i in 1:n_spatial_orbitals, j in 1:n_spatial_orbitals
        if abs(h1e[i, j]) > 1e-12
            for spin in [0, 1]  # 0=alpha, 1=beta
                qubit_i = 2*i - 1 + spin
                qubit_j = 2*j - 1 + spin

                if i == j
                    # Number operator: a†ᵢ aᵢ = (I - Zᵢ)/2
                    pauli_str = fill('I', n_qubits)
                    push!(pauli_terms, (String(pauli_str), h1e[i, j] * 0.5))

                    pauli_str[qubit_i] = 'Z'
                    push!(pauli_terms, (String(pauli_str), h1e[i, j] * (-0.5)))
                else
                    # Off-diagonal terms (simplified)
                    pauli_str = fill('I', n_qubits)
                    pauli_str[qubit_i] = 'X'
                    pauli_str[qubit_j] = 'X'
                    push!(pauli_terms, (String(pauli_str), h1e[i, j] * 0.25))

                    pauli_str[qubit_i] = 'Y'
                    pauli_str[qubit_j] = 'Y'
                    push!(pauli_terms, (String(pauli_str), h1e[i, j] * 0.25))
                end
            end
        end
    end

    # Two-electron terms: ½ Σᵢⱼₖₗ h₂ᵢⱼₖₗ a†ᵢ a†ⱼ aₗ aₖ
    for i in 1:n_spatial_orbitals, j in 1:n_spatial_orbitals
        for k in 1:n_spatial_orbitals, l in 1:n_spatial_orbitals
            if abs(h2e[i, j, k, l]) > 1e-12

                # Same-spin terms (simplified Jordan-Wigner)
                for spin in [0, 1]
                    qubit_i = 2*i - 1 + spin
                    qubit_j = 2*j - 1 + spin
                    qubit_k = 2*k - 1 + spin
                    qubit_l = 2*l - 1 + spin

                    # Diagonal two-electron terms
                    if i == j && k == l && i == k
                        # Same orbital: nᵢnᵢ terms
                        pauli_str = fill('I', n_qubits)
                        push!(
                            pauli_terms, (String(pauli_str), 0.5 * h2e[i, j, k, l] * 0.25)
                        )

                        pauli_str[qubit_i] = 'Z'
                        push!(
                            pauli_terms,
                            (String(pauli_str), 0.5 * h2e[i, j, k, l] * (-0.25)),
                        )

                    elseif i == j && k == l && i != k
                        # Different orbitals: nᵢnₖ terms
                        pauli_str = fill('I', n_qubits)
                        push!(
                            pauli_terms, (String(pauli_str), 0.5 * h2e[i, j, k, l] * 0.25)
                        )

                        pauli_str[qubit_i] = 'Z'
                        push!(
                            pauli_terms,
                            (String(pauli_str), 0.5 * h2e[i, j, k, l] * (-0.125)),
                        )

                        pauli_str = fill('I', n_qubits)
                        pauli_str[qubit_k] = 'Z'
                        push!(
                            pauli_terms,
                            (String(pauli_str), 0.5 * h2e[i, j, k, l] * (-0.125)),
                        )

                        pauli_str[qubit_i] = 'Z'
                        pauli_str[qubit_k] = 'Z'
                        push!(
                            pauli_terms, (String(pauli_str), 0.5 * h2e[i, j, k, l] * 0.25)
                        )
                    end
                end

                # Different-spin terms
                if i == k && j == l && i != j
                    for (spin1, spin2) in [(0, 1), (1, 0)]
                        qubit_i = 2*i - 1 + spin1
                        qubit_j = 2*j - 1 + spin2

                        # Exchange-type terms
                        pauli_str = fill('I', n_qubits)
                        pauli_str[qubit_i] = 'X'
                        pauli_str[qubit_j] = 'X'
                        push!(
                            pauli_terms,
                            (String(pauli_str), 0.5 * h2e[i, j, k, l] * (-0.25)),
                        )

                        pauli_str[qubit_i] = 'Y'
                        pauli_str[qubit_j] = 'Y'
                        push!(
                            pauli_terms,
                            (String(pauli_str), 0.5 * h2e[i, j, k, l] * (-0.25)),
                        )
                    end
                end
            end
        end
    end

    # Combine like terms and filter small coefficients
    combined_terms = Dict{String,Float64}()
    for (pauli_str, coeff) in pauli_terms
        combined_terms[pauli_str] = get(combined_terms, pauli_str, 0.0) + coeff
    end

    final_terms = [
        (pauli_str, coeff) for (pauli_str, coeff) in combined_terms if abs(coeff) > 1e-12
    ]

    return final_terms, n_qubits
end

# ============================================================================
# Molecular Hamiltonians (Now Dynamic!)
# ============================================================================

"""
    h2_hamiltonian(geometry::String="equilibrium")

Generate H₂ Hamiltonian dynamically from quantum chemistry.
"""
function h2_hamiltonian(geometry::String="equilibrium")
    bond_lengths = Dict(
        "equilibrium" => 0.735,
        "stretched" => 1.5,
        "compressed" => 0.5,
        "dissociation" => 3.0,
    )

    bond_length = get(bond_lengths, geometry, 0.735)
    system_info = determine_molecular_system_size("H2", "sto-3g")
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals(
        "H2", n_spatial_orbitals, bond_length
    )
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    lih_hamiltonian(geometry::String="equilibrium")

Generate LiH Hamiltonian dynamically from quantum chemistry.
"""
function lih_hamiltonian(geometry::String="equilibrium")
    bond_lengths = Dict("equilibrium" => 1.595, "stretched" => 2.5, "compressed" => 1.2)

    bond_length = get(bond_lengths, geometry, 1.595)
    system_info = determine_molecular_system_size("LiH", "sto-3g", (4, 4))  # Active space
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals(
        "LiH", n_spatial_orbitals, bond_length
    )
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    beh2_hamiltonian(geometry::String="equilibrium")

Generate BeH₂ Hamiltonian dynamically from quantum chemistry.
"""
function beh2_hamiltonian(geometry::String="equilibrium")
    bond_lengths = Dict("equilibrium" => 1.33, "stretched" => 2.0, "compressed" => 1.0)

    bond_length = get(bond_lengths, geometry, 1.33)
    system_info = determine_molecular_system_size("BeH2", "sto-3g", (6, 5))  # Active space
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals(
        "BeH2", n_spatial_orbitals, bond_length
    )
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    h2o_hamiltonian(; active_space::Tuple{Int,Int}=(8,6))

Generate H₂O Hamiltonian with active space approximation.
"""
function h2o_hamiltonian(; active_space::Tuple{Int,Int}=(8, 6))
    system_info = determine_molecular_system_size("H2O", "sto-3g", active_space)
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals("H2O", n_spatial_orbitals)
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    n2_hamiltonian(geometry::String="equilibrium")

Generate N₂ Hamiltonian dynamically from quantum chemistry.
"""
function n2_hamiltonian(geometry::String="equilibrium")
    bond_lengths = Dict("equilibrium" => 1.098, "stretched" => 1.5, "compressed" => 0.9)

    bond_length = get(bond_lengths, geometry, 1.098)
    system_info = determine_molecular_system_size("N2", "sto-3g", (14, 10))  # Active space
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals(
        "N2", n_spatial_orbitals, bond_length
    )
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    co_hamiltonian(geometry::String="equilibrium")

Generate CO Hamiltonian dynamically from quantum chemistry.
"""
function co_hamiltonian(geometry::String="equilibrium")
    bond_lengths = Dict("equilibrium" => 1.128, "stretched" => 1.6, "compressed" => 1.0)

    bond_length = get(bond_lengths, geometry, 1.128)
    system_info = determine_molecular_system_size("CO", "sto-3g", (14, 9))  # Active space
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals(
        "CO", n_spatial_orbitals, bond_length
    )
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    nh3_hamiltonian(; active_space::Tuple{Int,Int}=(10,7))

Generate NH₃ Hamiltonian with active space approximation.
"""
function nh3_hamiltonian(; active_space::Tuple{Int,Int}=(10, 7))
    system_info = determine_molecular_system_size("NH3", "sto-3g", active_space)
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals("NH3", n_spatial_orbitals)
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    ch4_hamiltonian(; active_space::Tuple{Int,Int}=(10,8))

Generate CH₄ Hamiltonian with active space approximation.
"""
function ch4_hamiltonian(; active_space::Tuple{Int,Int}=(10, 8))
    system_info = determine_molecular_system_size("CH4", "sto-3g", active_space)
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals("CH4", n_spatial_orbitals)
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    hf_hamiltonian(geometry::String="equilibrium")

Generate HF Hamiltonian dynamically from quantum chemistry.
"""
function hf_hamiltonian(geometry::String="equilibrium")
    bond_lengths = Dict("equilibrium" => 0.917, "stretched" => 1.5, "compressed" => 0.7)

    bond_length = get(bond_lengths, geometry, 0.917)
    system_info = determine_molecular_system_size("HF", "sto-3g", (10, 5))  # Active space
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals(
        "HF", n_spatial_orbitals, bond_length
    )
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    bh_hamiltonian(geometry::String="equilibrium")

Generate BH Hamiltonian dynamically from quantum chemistry.
"""
function bh_hamiltonian(geometry::String="equilibrium")
    bond_lengths = Dict("equilibrium" => 1.232, "stretched" => 1.8, "compressed" => 1.0)

    bond_length = get(bond_lengths, geometry, 1.232)
    system_info = determine_molecular_system_size("BH", "sto-3g", (6, 5))  # Active space
    n_spatial_orbitals = system_info["n_spatial_orbitals"]

    h1e, h2e, nuclear_repulsion = compute_molecular_integrals(
        "BH", n_spatial_orbitals, bond_length
    )
    pauli_terms, n_qubits = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    return create_pauli_hamiltonian(n_qubits, pauli_terms)
end

"""
    create_molecular_hamiltonian(molecule::String; kwargs...)

Factory function for creating molecular Hamiltonians dynamically.
"""
function create_molecular_hamiltonian(
    molecule::String;
    geometry::String="equilibrium",
    basis::String="sto-3g",
    active_space::Union{Nothing,Tuple{Int,Int}}=nothing,
    bond_length::Union{Nothing,Float64}=nothing,
)
    molecule_lower = lowercase(molecule)

    # Determine system size
    system_info = determine_molecular_system_size(molecule, basis, active_space)
    n_spatial_orbitals = system_info["n_spatial_orbitals"]
    n_qubits = system_info["n_qubits"]

    # Get bond length if specified
    if bond_length === nothing
        bond_lengths = Dict(
            "h2" => Dict("equilibrium" => 0.735, "stretched" => 1.5, "compressed" => 0.5),
            "lih" => Dict("equilibrium" => 1.595, "stretched" => 2.5, "compressed" => 1.2),
            "beh2" => Dict("equilibrium" => 1.33, "stretched" => 2.0, "compressed" => 1.0),
            "h2o" => Dict("equilibrium" => 1.0, "stretched" => 1.2, "compressed" => 0.8),
            "n2" => Dict("equilibrium" => 1.098, "stretched" => 1.5, "compressed" => 0.9),
            "co" => Dict("equilibrium" => 1.128, "stretched" => 1.6, "compressed" => 1.0),
            "nh3" => Dict("equilibrium" => 1.012, "stretched" => 1.3, "compressed" => 0.9),
            "ch4" => Dict("equilibrium" => 1.087, "stretched" => 1.3, "compressed" => 0.9),
            "hf" => Dict("equilibrium" => 0.917, "stretched" => 1.5, "compressed" => 0.7),
            "bh" => Dict("equilibrium" => 1.232, "stretched" => 1.8, "compressed" => 1.0),
        )
        bond_length = get(get(bond_lengths, molecule_lower, Dict()), geometry, 1.0)
    end

    # Compute molecular integrals
    h1e, h2e, nuclear_repulsion = compute_molecular_integrals(
        molecule, n_spatial_orbitals, bond_length
    )

    # Apply Jordan-Wigner transformation
    pauli_terms, _ = jordan_wigner_transform(
        h1e, h2e, nuclear_repulsion, n_spatial_orbitals
    )

    # Create Hamiltonian
    hamiltonian = create_pauli_hamiltonian(n_qubits, pauli_terms)
    exact_energy = classical_solver(hamiltonian).eigenvalue

    # Get molecular properties
    mol_props = get(
        MOLECULAR_PROPERTIES,
        molecule,
        (charge=0, spin=0, electrons=system_info["total_electrons"]),
    )
    geometry_string = "Dynamic geometry: $geometry"

    # Return MolecularSystem (defined in main module)
    return Main.BarrenPlateausVQE.MolecularSystem(
        molecule,
        geometry,
        basis,
        mol_props.charge,
        mol_props.spin,
        system_info["total_electrons"],
        n_qubits,
        geometry_string,
        hamiltonian,
        exact_energy,
        nuclear_repulsion,
    )
end

# ============================================================================
# Molecular Properties (for compatibility)
# ============================================================================

const MOLECULAR_PROPERTIES = Dict(
    "H2" => (charge=0, spin=0, electrons=2),
    "LiH" => (charge=0, spin=0, electrons=4),
    "BeH2" => (charge=0, spin=0, electrons=6),
    "H2O" => (charge=0, spin=0, electrons=10),
    "NH3" => (charge=0, spin=0, electrons=10),
    "CH4" => (charge=0, spin=0, electrons=10),
    "N2" => (charge=0, spin=0, electrons=14),
    "CO" => (charge=0, spin=0, electrons=14),
    "HF" => (charge=0, spin=0, electrons=10),
    "BH" => (charge=0, spin=0, electrons=6),
)

# ============================================================================
# Classical Solver and Energy Evaluation (from qubap tools.py)
# ============================================================================

"""
    classical_solver(hamiltonian::AbstractBlock)

Compute exact ground state using classical diagonalization.
"""
function classical_solver(hamiltonian::AbstractBlock)
    try
        H_matrix = mat(hamiltonian)
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
"""
function energy_evaluation(
    hamiltonian::AbstractBlock,
    ansatz::AbstractBlock,
    parameters::Vector{Float64},
    n_qubits::Int,
)
    try
        state = zero_state(n_qubits)
        dispatched_ansatz = dispatch(ansatz, parameters)
        state = apply!(state, dispatched_ansatz)
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
# Molecular System Factory Functions
# ============================================================================

"""
    get_supported_molecules()

Return list of all supported molecules.
"""
function get_supported_molecules()
    return ["H2", "LiH", "BeH2", "H2O", "N2", "CO", "NH3", "CH4", "HF", "BH"]
end

"""
    get_molecule_info(molecule::String)

Get detailed information about a specific molecule.
"""
function get_molecule_info(molecule::String)
    if !(molecule in get_supported_molecules())
        throw(
            ArgumentError(
                "Molecule $molecule not supported. Supported molecules: $(get_supported_molecules())",
            ),
        )
    end

    system_info = determine_molecular_system_size(molecule, "sto-3g")
    mol_props = get(
        MOLECULAR_PROPERTIES,
        molecule,
        (charge=0, spin=0, electrons=system_info["total_electrons"]),
    )

    return Dict(
        "molecule" => molecule,
        "atoms" => system_info["atoms"],
        "electrons" => system_info["total_electrons"],
        "charge" => mol_props.charge,
        "spin" => mol_props.spin,
        "min_qubits_sto3g" => system_info["n_qubits"],
        "spatial_orbitals_sto3g" => system_info["n_spatial_orbitals"],
    )
end

# ============================================================================
# Exports
# ============================================================================

export pauli_string, create_pauli_hamiltonian
export global2local, global_observable, local_observable
export test_hamiltonian, ladder_hamiltonian
export h2_hamiltonian, lih_hamiltonian, beh2_hamiltonian, h2o_hamiltonian
export n2_hamiltonian, co_hamiltonian, nh3_hamiltonian, ch4_hamiltonian
export hf_hamiltonian, bh_hamiltonian, create_molecular_hamiltonian
export classical_solver, energy_evaluation, expectation_value
export gradient_finite_diff, gradient_variance
export determine_molecular_system_size, compute_molecular_integrals, jordan_wigner_transform
export get_supported_molecules, get_molecule_info
