#!/usr/bin/env python3
"""
fetch_hamiltonians.py
---------------------
Downloads molecular Hamiltonians from PennyLane Datasets (qchem collection)
and saves them as JSON files that the Julia package can load directly,
bypassing the expensive compute_molecular_integrals + jordan_wigner_transform
pipeline at runtime.

Output (one file per molecule/geometry):
    data/<MOLECULE>_<geometry>.json

Each JSON contains:
    {
      "molecule":    "H2",
      "geometry":    "equilibrium",
      "basis":       "STO-3G",
      "bondlength":  0.742,
      "n_qubits":    4,
      "fci_energy":  -1.1361894540879054,
      "hf_energy":   -1.117...,          (if available)
      "pauli_terms": [
          ["IIII", 0.2345],
          ["ZZII", -0.4567],
          ...
      ]
    }

Usage:
    python scripts/fetch_hamiltonians.py
    python scripts/fetch_hamiltonians.py --molecules H2 LiH --output data/hamiltonians
    python scripts/fetch_hamiltonians.py --all-bondlengths --molecules H2
    python scripts/fetch_hamiltonians.py --list

Bond lengths are discovered dynamically via qml.data.list_datasets() at runtime,
so the script always reflects what PennyLane actually has available.

Requirements:
    pip install pennylane pennylane-datasets
"""

import argparse
import json
import os
import sys


# ---------------------------------------------------------------------------
# Molecule list
# All molecules confirmed from XanaduAI/pennylane-datasets/content/qchem/
# (nh2 excluded: no dataset.json, not loadable via qml.data.load)
# ---------------------------------------------------------------------------

ALL_MOLECULES = [
    # diatomics
    "H2", "He2", "Li2", "C2", "N2", "O2", "CO", "HF",
    # ions
    "HeH+", "NeH+", "OH-",
    # hydrogen chains
    "H3+", "H4", "H5", "H6", "H7", "H8", "H10",
    # small polyatomics
    "LiH", "BeH2", "BH3", "CH2", "NH3", "H2O", "H2O2",
    "CH4", "CH2O", "HCN", "O3", "CO2",
    # larger molecules
    "N2H2", "N2H4", "C2H2", "C2H4", "C2H6",
]

DEFAULT_BASIS = "STO-3G"


# ---------------------------------------------------------------------------
# Dynamic bond length discovery
# ---------------------------------------------------------------------------

def get_available_bondlengths(molname, basis, qml):
    """
    Query PennyLane's dataset registry and return (actual_basis, sorted bond lengths).

    If the requested basis is not available for this molecule, falls back to the
    first available basis (e.g. He2 only has 6-31G, not STO-3G).
    Returns (None, []) if the molecule is not found at all.
    """
    try:
        registry = qml.data.list_datasets()
        mol_data = registry.get("qchem", {}).get(molname, {})
        if not mol_data:
            return None, []
        if basis in mol_data:
            return basis, sorted(float(b) for b in mol_data[basis])
        # Fallback: use first available basis
        fallback_basis = next(iter(mol_data))
        return fallback_basis, sorted(float(b) for b in mol_data[fallback_basis])
    except Exception:
        return None, []


def pick_geometries(bondlengths):
    """
    Given a sorted list of available bond lengths, return a list of
    (geometry_label, bondlength) pairs to fetch:

      - Single entry  → [("equilibrium", bl)]
      - Multiple      → equilibrium (middle/marked value) + stretched (last) + compressed (first)

    The 'equilibrium' is the value closest to the physical equilibrium, which
    PennyLane marks by embedding a non-round number in an otherwise round scan
    (e.g. 0.742 amid 0.5, 0.54, 0.58...). We detect this as the value whose
    distance to the nearest round-step neighbour is smallest *and* whose decimal
    representation is non-trivially long — or simply the median if all values
    look round.
    """
    if not bondlengths:
        return []

    if len(bondlengths) == 1:
        return [("equilibrium", bondlengths[0])]

    # Find the equilibrium: the "special" non-round value embedded in the scan.
    # Heuristic: the value with the most significant decimal digits that sits
    # roughly in the lower half of the scan (equilibrium is never at the extremes).
    def decimal_places(v):
        s = f"{v:.10f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 0

    candidates = bondlengths[1:-1] if len(bondlengths) > 2 else bondlengths
    equil = max(candidates, key=decimal_places)

    geometries = [("equilibrium", equil)]

    last = bondlengths[-1]
    if last != equil:
        geometries.append(("stretched", last))

    first = bondlengths[0]
    if first != equil:
        geometries.append(("compressed", first))

    return geometries


# ---------------------------------------------------------------------------
# Pauli string helpers
# ---------------------------------------------------------------------------

# Maps PennyLane single-qubit Pauli class/name to letter
_PAULI_NAMES = {
    "PauliX": "X", "X": "X",
    "PauliY": "Y", "Y": "Y",
    "PauliZ": "Z", "Z": "Z",
    "Identity": "I", "I": "I",
}


def _op_to_label(op, n_qubits):
    """
    Recursively convert any PennyLane operator to a fixed-width Pauli string.
    Handles: Identity, PauliX/Y/Z, Prod (tensor product), SProd (scaled).
    """
    label = ["I"] * n_qubits
    cls = type(op).__name__

    # Scaled product — unwrap to the base operator
    if cls == "SProd" or (hasattr(op, "scalar") and hasattr(op, "base")):
        return _op_to_label(op.base, n_qubits)

    # Tensor product (Prod) — recurse into each factor
    if cls == "Prod" or (hasattr(op, "operands") and not hasattr(op, "coeffs")):
        for factor in op.operands:
            sub = _op_to_label(factor, n_qubits)
            for i, ch in enumerate(sub):
                if ch != "I":
                    label[i] = ch
        return "".join(label)

    # Single-qubit Pauli or Identity
    op_name = getattr(op, "name", cls)
    letter = _PAULI_NAMES.get(op_name, "I")
    wires = list(op.wires) if hasattr(op, "wires") else []
    for w in wires:
        if 0 <= w < n_qubits:
            label[w] = letter

    return "".join(label)


def hamiltonian_to_pauli_terms(hamiltonian, n_qubits, qml):
    """
    Extract list of [pauli_string, coefficient] from any PennyLane Hamiltonian.

    Strategy (in order):
      1. Legacy .coeffs/.ops  — qml.Hamiltonian (fast, no matrix needed)
      2. Modern Sum of SProd  — LinearCombination / Sum (fast, no matrix needed)
      3. qml.pauli_decompose  — last resort via full matrix (slow, small molecules only)
    """
    # --- Strategy 1: legacy qml.Hamiltonian ---
    if hasattr(hamiltonian, "coeffs") and hasattr(hamiltonian, "ops"):
        return [
            [_op_to_label(op, n_qubits), float(coeff)]
            for coeff, op in zip(hamiltonian.coeffs, hamiltonian.ops)
        ]

    # --- Strategy 2: modern Sum of SProd ---
    if hasattr(hamiltonian, "operands"):
        terms = []
        for summand in hamiltonian.operands:
            if hasattr(summand, "scalar") and hasattr(summand, "base"):
                coeff, op = float(summand.scalar), summand.base
            else:
                coeff, op = 1.0, summand
            terms.append([_op_to_label(op, n_qubits), coeff])
        return terms

    # --- Strategy 3: last resort — full matrix decomposition (exponential cost) ---
    try:
        import numpy as np
        wire_order = list(range(n_qubits))
        mat = qml.matrix(hamiltonian, wire_order=wire_order)
        pl_terms = qml.pauli_decompose(mat, wire_order=wire_order)
        return [
            [_op_to_label(op, n_qubits), float(coeff)]
            for coeff, op in zip(pl_terms.coeffs, pl_terms.ops)
        ]
    except Exception:
        pass

    raise ValueError(f"Cannot parse Hamiltonian of type {type(hamiltonian)}")


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

def fetch_molecule(molname, bondlength, basis, output_dir, geometry_label, verbose, qml):
    """Download one molecule/bondlength and save to JSON. Returns True on success."""
    out_path = os.path.join(output_dir, f"{molname}_{geometry_label}.json")

    if os.path.exists(out_path):
        if verbose:
            print(f"  [skip] {out_path} already exists")
        return True

    if verbose:
        print(f"  Fetching {molname} ({geometry_label}, bond={bondlength} Å, basis={basis}) ...")

    try:
        datasets = qml.data.load("qchem", molname=molname, basis=basis, bondlength=bondlength)
    except Exception as e:
        print(f"  [ERROR] qml.data.load failed: {e}")
        return False

    if not datasets:
        print(f"  [ERROR] No dataset returned for {molname} bondlength={bondlength}")
        return False

    data = datasets[0]

    # number of qubits
    try:
        n_qubits = len(data.hamiltonian.wires)
    except Exception:
        try:
            n_qubits = 2 * data.molecule.n_orbitals
        except Exception:
            n_qubits = None

    if n_qubits is None:
        print(f"  [ERROR] Cannot determine n_qubits for {molname}")
        return False

    # Pauli terms
    try:
        pauli_terms = hamiltonian_to_pauli_terms(data.hamiltonian, n_qubits, qml)
    except Exception as e:
        print(f"  [ERROR] Failed to extract Pauli terms: {e}")
        return False

    # energies
    fci_energy = None
    hf_energy = None
    try:
        fci_energy = float(data.fci_energy)
    except Exception:
        pass
    try:
        hf_energy = float(data.hf_energy)
    except Exception:
        pass

    # fallback: diagonalise to get FCI energy
    if fci_energy is None:
        try:
            import numpy as np
            eigvals = np.linalg.eigvalsh(
                qml.matrix(data.hamiltonian, wire_order=range(n_qubits))
            )
            fci_energy = float(eigvals.min())
            if verbose:
                print(f"  Computed FCI energy via diagonalization: {fci_energy:.8f}")
        except Exception as e:
            print(f"  [WARN] Could not compute FCI energy: {e}")

    result = {
        "molecule":    molname,
        "geometry":    geometry_label,
        "basis":       basis,
        "bondlength":  bondlength,
        "n_qubits":    n_qubits,
        "fci_energy":  fci_energy,
        "hf_energy":   hf_energy,
        "pauli_terms": pauli_terms,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"  Saved {out_path}  ({n_qubits} qubits, {len(pauli_terms)} Pauli terms, "
              f"FCI={fci_energy:.6f})")
    return True


# ---------------------------------------------------------------------------
# List helper
# ---------------------------------------------------------------------------

def list_molecules(basis, qml):
    """Print all available molecules and their bond lengths from the live registry."""
    print(f"Querying PennyLane dataset registry (preferred basis={basis}) ...")
    print(f"\n  {'Molecule':<10} {'Basis':<10} {'# bond lengths':<18} {'Range (Å)'}")
    print("  " + "-" * 62)
    for mol in ALL_MOLECULES:
        actual_basis, bls = get_available_bondlengths(mol, basis, qml)
        if bls:
            note = f" (fallback)" if actual_basis != basis else ""
            print(f"  {mol:<10} {actual_basis:<10} {len(bls):<18} {bls[0]} – {bls[-1]}{note}")
        else:
            print(f"  {mol:<10} {'(not found)'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch molecular Hamiltonians from PennyLane Datasets → JSON cache"
    )
    parser.add_argument(
        "--molecules", "-m",
        nargs="+",
        default=ALL_MOLECULES,
        help="Molecules to fetch (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--basis", "-b",
        default=DEFAULT_BASIS,
        help=f"Basis set (default: {DEFAULT_BASIS})",
    )
    parser.add_argument(
        "--all-bondlengths",
        action="store_true",
        help="Fetch every available bond length instead of just equilibrium/stretched/compressed",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available molecules and bond lengths, then exit",
    )
    parser.add_argument(
        "--skip", "-s",
        nargs="+",
        default=[],
        help="Molecules to skip (e.g. --skip H7 H8 H10 for large files)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    try:
        import pennylane as qml
        print(f"PennyLane version: {qml.__version__}")
    except ImportError:
        print("ERROR: PennyLane not installed. Run: pip install pennylane")
        sys.exit(1)

    if args.list:
        list_molecules(args.basis, qml)
        return

    verbose = not args.quiet
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # Normalise molecule names (case-insensitive match against ALL_MOLECULES)
    mol_lookup = {m.upper(): m for m in ALL_MOLECULES}
    skip_set = {m.upper() for m in args.skip}
    requested = []
    for name in args.molecules:
        canonical = mol_lookup.get(name.upper())
        if canonical is None:
            print(f"[WARN] Unknown molecule '{name}', skipping. Use --list to see options.")
        elif name.upper() in skip_set:
            print(f"[SKIP] {canonical}")
        else:
            requested.append(canonical)

    if not requested:
        print("No valid molecules to fetch.")
        sys.exit(1)

    success_count = 0
    fail_count = 0

    for molname in requested:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Molecule: {molname}")
            print(f"{'='*60}")

        # Discover bond lengths from the live registry (auto-fallback basis)
        actual_basis, bondlengths = get_available_bondlengths(molname, args.basis, qml)

        if not bondlengths:
            print(f"  [WARN] No bond lengths found for {molname}, skipping.")
            fail_count += 1
            continue

        if actual_basis != args.basis and verbose:
            print(f"  [INFO] {molname} not available in {args.basis}, using {actual_basis}")

        if args.all_bondlengths:
            geometries = [(f"bl_{bl}", bl) for bl in bondlengths]
        else:
            geometries = pick_geometries(bondlengths)

        if verbose:
            print(f"  Available bond lengths ({len(bondlengths)}): {bondlengths[0]} – {bondlengths[-1]}")
            print(f"  Fetching {len(geometries)} geometries: "
                  f"{[g[0] for g in geometries]}")

        for geometry_label, bondlength in geometries:
            ok = fetch_molecule(
                molname=molname,
                bondlength=bondlength,
                basis=actual_basis,
                output_dir=output_dir,
                geometry_label=geometry_label,
                verbose=verbose,
                qml=qml,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1

    print(f"\nDone: {success_count} saved, {fail_count} failed.")
    print(f"JSON files written to: {output_dir}/")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
