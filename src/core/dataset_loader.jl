"""
# Dataset Loader (`dataset_loader.jl`)

Loads pre-fetched molecular Hamiltonian data from JSON files produced by
`scripts/fetch_hamiltonians.py` (PennyLane Datasets → JSON cache).

This bypasses the expensive runtime pipeline:
    compute_molecular_integrals → jordan_wigner_transform → eigvals(diagonalize)

Each JSON file contains:
    - molecule, geometry, basis, bondlength
    - n_qubits
    - fci_energy   (exact ground-state energy, replaces eigvals diagonalization)
    - hf_energy    (Hartree-Fock reference, optional)
    - pauli_terms  [[pauli_string, coefficient], ...]

Lookup priority when `create_molecular_hamiltonian` is called:
    1. JSON cache in DATA_DIR  → instant load, exact integrals
    2. Analytical approximation in hamiltonian_builder.jl  → fallback
"""

using JSON3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""Default directory for cached Hamiltonian JSON files."""
const DATA_DIR = joinpath(dirname(@__DIR__), "..", "data")

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

"""
    CachedHamiltonian

Holds all data loaded from a pre-fetched JSON file.
"""
struct CachedHamiltonian
    molecule::String
    geometry::String
    basis::String
    bondlength::Float64
    n_qubits::Int
    fci_energy::Float64
    hf_energy::Union{Float64,Nothing}
    pauli_terms::Vector{Tuple{String,Float64}}
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _json_path(molecule, geometry; data_dir) -> String

Construct the expected JSON file path for a molecule/geometry pair.
"""
function _json_path(
    molecule::String, geometry::String; data_dir::String=DATA_DIR
)
    filename = "$(molecule)_$(geometry).json"
    return joinpath(data_dir, filename)
end

"""
    _normalize_molecule(name) -> String

Normalize molecule name to match JSON filenames (e.g. "h2" → "H2", "lih" → "LiH").
"""
function _normalize_molecule(name::String)
    lookup = Dict(
        # diatomics
        "h2"   => "H2",
        "he2"  => "He2",
        "li2"  => "Li2",
        "c2"   => "C2",
        "n2"   => "N2",
        "o2"   => "O2",
        "co"   => "CO",
        "hf"   => "HF",
        # ions
        "heh+" => "HeH+",
        "neh+" => "NeH+",
        "oh-"  => "OH-",
        # hydrogen chains
        "h3+"  => "H3+",
        "h4"   => "H4",
        "h5"   => "H5",
        "h6"   => "H6",
        "h7"   => "H7",
        "h8"   => "H8",
        "h10"  => "H10",
        # small polyatomics
        "lih"  => "LiH",
        "beh2" => "BeH2",
        "bh3"  => "BH3",
        "ch2"  => "CH2",
        "nh3"  => "NH3",
        "h2o"  => "H2O",
        "h2o2" => "H2O2",
        "ch4"  => "CH4",
        "ch2o" => "CH2O",
        "hcn"  => "HCN",
        "o3"   => "O3",
        "co2"  => "CO2",
        # larger molecules
        "n2h2" => "N2H2",
        "n2h4" => "N2H4",
        "c2h2" => "C2H2",
        "c2h4" => "C2H4",
        "c2h6" => "C2H6",
    )
    return get(lookup, lowercase(name), name)
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    dataset_available(molecule, geometry; data_dir) -> Bool

Check whether a cached JSON file exists for the given molecule and geometry.
"""
function dataset_available(
    molecule::String,
    geometry::String="equilibrium";
    data_dir::String=DATA_DIR,
)
    mol = _normalize_molecule(molecule)
    return isfile(_json_path(mol, geometry; data_dir=data_dir))
end

"""
    load_cached_hamiltonian(molecule, geometry; data_dir) -> CachedHamiltonian

Load a pre-fetched Hamiltonian from the JSON cache.

# Arguments
- `molecule`: Molecule name (e.g. "H2", "LiH")
- `geometry`: Geometry label (e.g. "equilibrium", "stretched")
- `data_dir`: Directory containing JSON files (default: `data/`)

# Returns
A `CachedHamiltonian` with Pauli terms and exact FCI energy.

# Throws
- `ErrorException` if the JSON file does not exist (use `dataset_available` to check first).
"""
function load_cached_hamiltonian(
    molecule::String,
    geometry::String="equilibrium";
    data_dir::String=DATA_DIR,
)
    mol = _normalize_molecule(molecule)
    path = _json_path(mol, geometry; data_dir=data_dir)

    if !isfile(path)
        error(
            "No cached Hamiltonian found for $mol ($geometry).\n" *
            "Run: python scripts/fetch_hamiltonians.py --molecules $mol\n" *
            "Expected file: $path"
        )
    end

    raw = JSON3.read(read(path, String))

    # Parse pauli_terms: JSON array of [string, float] pairs
    pauli_terms = Tuple{String,Float64}[]
    for term in raw["pauli_terms"]
        push!(pauli_terms, (String(term[1]), Float64(term[2])))
    end

    hf_energy = isnothing(get(raw, "hf_energy", nothing)) ||
                raw["hf_energy"] === nothing ? nothing : Float64(raw["hf_energy"])

    return CachedHamiltonian(
        String(raw["molecule"]),
        String(raw["geometry"]),
        String(raw["basis"]),
        Float64(raw["bondlength"]),
        Int(raw["n_qubits"]),
        Float64(raw["fci_energy"]),
        hf_energy,
        pauli_terms,
    )
end

"""
    list_cached_molecules(; data_dir) -> Vector{NamedTuple}

List all available cached molecule/geometry combinations.
"""
function list_cached_molecules(; data_dir::String=DATA_DIR)
    results = NamedTuple[]
    isdir(data_dir) || return results

    for fname in readdir(data_dir)
        endswith(fname, ".json") || continue
        path = joinpath(data_dir, fname)
        try
            raw = JSON3.read(read(path, String))
            push!(results, (
                molecule   = String(raw["molecule"]),
                geometry   = String(raw["geometry"]),
                n_qubits   = Int(raw["n_qubits"]),
                fci_energy = Float64(raw["fci_energy"]),
                n_terms    = length(raw["pauli_terms"]),
                file       = fname,
            ))
        catch
            # skip malformed files
        end
    end

    return results
end
