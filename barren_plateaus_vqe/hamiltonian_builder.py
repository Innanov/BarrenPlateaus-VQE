#!/usr/bin/env python3
"""
Hamiltonian Builder Module (`hamiltonian_builder.py`)
================================================

This module provides comprehensive molecular Hamiltonian generation using PySCF
quantum chemistry calculations integrated with Qiskit. It supports multiple
molecular systems, basis sets, and approximation techniques for studying
barren plateau phenomena in realistic chemical systems.

Key Features:
- PySCF-Qiskit integration for accurate molecular Hamiltonians
- Multiple molecules: H₂, LiH, BeH₂, H₂O, N₂, CO with realistic geometries
- Flexible basis sets: STO-3G, 6-31G, cc-pVDZ, etc.
- Active space and frozen core approximations
- Multiple fermion-to-qubit transformations
- Graceful fallback to test Hamiltonians
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

try:
    import qiskit.opflow as of
    from qiskit.opflow import PauliSumOp
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )
    from qiskit_nature.second_q.problems import ElectronicStructureProblem
    from qiskit_nature.second_q.transformers import (
        ActiveSpaceTransformer,
        FreezeCoreTransformer,
    )
    from qiskit_nature.units import DistanceUnit

    QISKIT_NATURE_AVAILABLE = True
    logger.info("Qiskit Nature and PySCF integration available")
except ImportError as e:
    QISKIT_NATURE_AVAILABLE = False
    logger.warning(f"Qiskit Nature not available: {e}")
    logger.warning(
        "Molecular Hamiltonians will not be available. Install with: qiskit-nature[pyscf]==0.6.0"
    )

# Try to import qubap for fallback test Hamiltonians
try:
    from qubap.qiskit.hamiltonians import test_hamiltonian

    QUBAP_AVAILABLE = True
    logger.info("qubap test Hamiltonians available")
except ImportError:
    QUBAP_AVAILABLE = False
    logger.warning("qubap not available. Test Hamiltonians will be limited.")


# Predefined molecular geometries in Angstroms
MOLECULAR_GEOMETRIES = {
    "H2": {
        "equilibrium": "H 0.0 0.0 0.0; H 0.735 0.0 0.0",
        "stretched": "H 0.0 0.0 0.0; H 1.5 0.0 0.0",
        "compressed": "H 0.0 0.0 0.0; H 0.5 0.0 0.0",
        "dissociation": "H 0.0 0.0 0.0; H 3.0 0.0 0.0",
    },
    "LiH": {
        "equilibrium": "Li 0.0 0.0 0.0; H 1.595 0.0 0.0",
        "stretched": "Li 0.0 0.0 0.0; H 2.5 0.0 0.0",
        "compressed": "Li 0.0 0.0 0.0; H 1.2 0.0 0.0",
    },
    "BeH2": {
        "equilibrium": "Be 0.0 0.0 0.0; H -1.33 0.0 0.0; H 1.33 0.0 0.0",
        "stretched": "Be 0.0 0.0 0.0; H -2.0 0.0 0.0; H 2.0 0.0 0.0",
        "asymmetric": "Be 0.0 0.0 0.0; H -1.33 0.0 0.0; H 1.8 0.0 0.0",
    },
    "H2O": {
        "equilibrium": "O 0.0 0.0 0.0; H 0.757 0.587 0.0; H -0.757 0.587 0.0",
        "stretched": "O 0.0 0.0 0.0; H 1.2 0.8 0.0; H -1.2 0.8 0.0",
        "bent": "O 0.0 0.0 0.0; H 0.957 0.287 0.0; H -0.957 0.287 0.0",
    },
    "N2": {
        "equilibrium": "N 0.0 0.0 0.0; N 1.098 0.0 0.0",
        "stretched": "N 0.0 0.0 0.0; N 2.5 0.0 0.0",
        "dissociation": "N 0.0 0.0 0.0; N 4.0 0.0 0.0",
    },
    "CO": {
        "equilibrium": "C 0.0 0.0 0.0; O 1.128 0.0 0.0",
        "stretched": "C 0.0 0.0 0.0; O 2.0 0.0 0.0",
        "dissociation": "C 0.0 0.0 0.0; O 3.5 0.0 0.0",
    },
    "NH3": {
        "equilibrium": "N 0.0 0.0 0.0; H 0.0 1.017 0.0; H 0.880 -0.508 0.0; H -0.880 -0.508 0.0",
        "planar": "N 0.0 0.0 0.0; H 1.017 0.0 0.0; H -0.508 0.880 0.0; H -0.508 -0.880 0.0",
    },
    "CH4": {
        "equilibrium": "C 0.0 0.0 0.0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629; H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629"
    },
}

# Default molecular properties
MOLECULAR_PROPERTIES = {
    "H2": {"charge": 0, "spin": 0, "electrons": 2},
    "LiH": {"charge": 0, "spin": 0, "electrons": 4},
    "BeH2": {"charge": 0, "spin": 0, "electrons": 6},
    "H2O": {"charge": 0, "spin": 0, "electrons": 10},
    "N2": {"charge": 0, "spin": 0, "electrons": 14},
    "CO": {"charge": 0, "spin": 0, "electrons": 14},
    "NH3": {"charge": 0, "spin": 0, "electrons": 10},
    "CH4": {"charge": 0, "spin": 0, "electrons": 10},
}

# Default active spaces for large molecules
DEFAULT_ACTIVE_SPACES = {
    "H2O": (8, 6),  # 8 electrons, 6 orbitals
    "LiH": (2, 4),  # 2 electrons, 4 orbitals
    "BeH2": (4, 6),  # 4 electrons, 6 orbitals
    "N2": (6, 8),  # 6 electrons, 8 orbitals
    "CO": (6, 8),  # 6 electrons, 8 orbitals
    "NH3": (8, 6),  # 8 electrons, 6 orbitals
    "CH4": (8, 6),  # 8 electrons, 6 orbitals
}


def normalize_molecule_name(molecule: str, available_molecules: List[str]) -> str:
    """
    Normalize molecule name to match available molecules exactly.
    Fixes case sensitivity issues (e.g., "LIH" -> "LiH").

    Args:
        molecule: Input molecule name
        available_molecules: List of available molecule names

    Returns:
        Normalized molecule name that matches available list
    """
    # First try exact match
    if molecule in available_molecules:
        return molecule

    # Try case-insensitive match
    for available in available_molecules:
        if available.lower() == molecule.lower():
            return available

    # No match found - return original
    return molecule


class MolecularHamiltonianGenerator:
    """
    Generate molecular Hamiltonians using PySCF and Qiskit Nature.

    This class provides a comprehensive interface for creating accurate molecular
    Hamiltonians from quantum chemistry calculations. It supports multiple basis
    sets, active space approximations, and fermion-to-qubit transformations.
    """

    def __init__(
        self,
        basis: str = "sto-3g",
        charge: int = 0,
        spin: int = 0,
        transformation: str = "jordan_wigner",
    ):
        """
        Initialize the molecular Hamiltonian generator.

        Args:
            `basis`: Basis set for quantum chemistry calculation (e.g., "sto-3g", "6-31g")
            `charge`: Molecular charge
            `spin`: Molecular spin (2S)
            `transformation`: Fermion-to-qubit transformation
                          ("jordan_wigner", "parity", "bravyi_kitaev")
        """
        if not QISKIT_NATURE_AVAILABLE:
            raise ImportError(
                "Qiskit Nature is required for molecular Hamiltonians. "
                "Install with: pip install qiskit-nature[pyscf]==0.6.0"
            )

        self.basis = basis.lower()
        self.charge = charge
        self.spin = spin
        self.transformation = transformation.lower()

        # Set up fermion-to-qubit mapper
        if self.transformation == "jordan_wigner":
            self.mapper = JordanWignerMapper()
        elif self.transformation == "parity":
            self.mapper = ParityMapper()
        elif self.transformation == "bravyi_kitaev":
            self.mapper = BravyiKitaevMapper()
        else:
            raise ValueError(f"Unknown transformation: {transformation}")

        logger.info(
            f"Initialized MolecularHamiltonianGenerator: {basis}, {transformation}"
        )

    def create_molecule_problem(self, geometry: str) -> "ElectronicStructureProblem":
        """
        Create molecule and perform quantum chemistry calculation.

        Args:
            `geometry`: Molecular geometry string in PySCF format

        Returns:
            `ElectronicStructureProblem` object with quantum chemistry results
        """
        try:
            driver = PySCFDriver(
                atom=geometry,
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                unit=DistanceUnit.ANGSTROM,
            )

            problem = driver.run()
            logger.info(
                f"Successfully created molecule problem with {problem.num_spatial_orbitals} orbitals"
            )
            return problem

        except Exception as e:
            logger.error(f"Failed to create molecule problem: {e}")
            raise RuntimeError(f"PySCF calculation failed: {e}")

    def apply_transformations(
        self,
        problem: "ElectronicStructureProblem",
        freeze_core: bool = False,
        active_space: Optional[Tuple[int, int]] = None,
        remove_orbitals: Optional[List[int]] = None,
    ) -> "ElectronicStructureProblem":
        """
        Apply orbital transformations to reduce problem size.

        Args:
            `problem`: Original electronic structure problem
            `freeze_core`: Whether to freeze core orbitals
            `active_space`: (num_electrons, num_orbitals) for active space
            `remove_orbitals`: List of orbital indices to remove

        Returns:
            Transformed electronic structure problem
        """
        transformers = []

        if freeze_core:
            transformers.append(FreezeCoreTransformer())
            logger.info("Applied frozen core approximation")

        if remove_orbitals:
            transformers.append(
                ActiveSpaceTransformer(
                    num_electrons=problem.num_particles[0] + problem.num_particles[1],
                    num_spatial_orbitals=problem.num_spatial_orbitals,
                    remove_orbitals=remove_orbitals,
                )
            )
            logger.info(f"Removed orbitals: {remove_orbitals}")

        if active_space:
            num_electrons, num_orbitals = active_space
            transformers.append(
                ActiveSpaceTransformer(
                    num_electrons=num_electrons, num_spatial_orbitals=num_orbitals
                )
            )
            logger.info(f"Applied active space: ({num_electrons}e, {num_orbitals}o)")

        # Apply transformations sequentially
        transformed_problem = problem
        for transformer in transformers:
            try:
                transformed_problem = transformer.transform(transformed_problem)
            except Exception as e:
                logger.warning(f"Transformation failed: {e}")
                # Continue with previous problem if transformation fails
                continue

        return transformed_problem

    def get_hamiltonian(
        self,
        geometry: str,
        freeze_core: bool = False,
        active_space: Optional[Tuple[int, int]] = None,
        remove_orbitals: Optional[List[int]] = None,
    ) -> of.OperatorBase:
        """
        Generate molecular Hamiltonian as `qiskit.opflow` operator.

        Args:
            `geometry`: Molecular geometry string
            `freeze_core`: Whether to freeze core orbitals
            `active_space`: (`num_electrons`, `num_orbitals`) for active space
            `remove_orbitals`: List of orbital indices to remove

        Returns:
            Molecular Hamiltonian as `qiskit.opflow` operator
        """
        try:
            # Create the electronic structure problem
            problem = self.create_molecule_problem(geometry)

            # Apply transformations
            transformed_problem = self.apply_transformations(
                problem, freeze_core, active_space, remove_orbitals
            )

            # Get the second-quantized Hamiltonian
            second_q_ham = transformed_problem.hamiltonian.second_q_op()

            # Convert to qubit operator
            qubit_ham = self.mapper.map(second_q_ham)

            # Convert to qiskit.opflow format
            if hasattr(qubit_ham, "primitive"):
                # For newer versions of Qiskit
                pauli_sum = PauliSumOp(qubit_ham.primitive)
            else:
                # For older versions
                pauli_sum = PauliSumOp(qubit_ham)

            logger.info(f"Generated Hamiltonian with {pauli_sum.num_qubits} qubits")
            return pauli_sum.reduce()

        except Exception as e:
            logger.error(f"Hamiltonian generation failed: {e}")
            raise RuntimeError(f"Failed to generate molecular Hamiltonian: {e}")

    def get_molecule_info(self, geometry: str) -> Dict[str, Any]:
        """
        Get comprehensive information about the molecular system.

        Args:
            `geometry`: Molecular geometry string

        Returns:
            Dictionary with molecular system information
        """
        try:
            problem = self.create_molecule_problem(geometry)
            ham = self.get_hamiltonian(geometry)

            return {
                "num_particles": problem.num_particles,
                "num_spatial_orbitals": problem.num_spatial_orbitals,
                "num_qubits": ham.num_qubits,
                "nuclear_repulsion_energy": problem.nuclear_repulsion_energy,
                "basis_set": self.basis,
                "charge": self.charge,
                "spin": self.spin,
                "transformation": self.transformation,
                "hamiltonian_terms": (
                    len(ham.oplist) if hasattr(ham, "oplist") else "N/A"
                ),
                "molecule_geometry": geometry,
            }

        except Exception as e:
            logger.error(f"Failed to get molecule info: {e}")
            return {
                "error": str(e),
                "basis_set": self.basis,
                "charge": self.charge,
                "spin": self.spin,
                "transformation": self.transformation,
            }


def get_molecular_hamiltonian_pyscf(
    molecule: str,
    geometry: str = "equilibrium",
    basis: str = "sto-3g",
    freeze_core: bool = False,
    active_space: Optional[Tuple[int, int]] = None,
    transformation: str = "jordan_wigner",
    auto_active_space: bool = True,
) -> of.OperatorBase:
    """
    Get molecular Hamiltonian using PySCF backend with automatic optimizations.

    Args:
        `molecule`: Molecule name (H2, LiH, BeH2, H2O, N2, CO, NH3, CH4)
        `geometry`: Geometry type ("equilibrium", "stretched", etc.)
        `basis`: Basis set for calculation (default: "sto-3g")
        `freeze_core`: Whether to freeze core orbitals
        `active_space`: (`num_electrons`, `num_orbitals`) for active space
        `transformation`: Fermion-to-qubit transformation
        `auto_active_space`: Automatically apply active space for large molecules

    Returns:
        Molecular Hamiltonian as `qiskit.opflow` operator

    Examples:
        >>> h2_ham = get_molecular_hamiltonian_pyscf("H2", "equilibrium", "sto-3g")
        >>> h2o_ham = get_molecular_hamiltonian_pyscf("H2O", active_space=(8, 6))
        >>> lih_ham = get_molecular_hamiltonian_pyscf("LiH", freeze_core=True)
    """
    if not QISKIT_NATURE_AVAILABLE:
        raise ImportError(
            "Qiskit Nature is required for molecular Hamiltonians. "
            "Install with: pip install qiskit-nature[pyscf]==0.6.0"
        )

    # FIXED: Case-insensitive molecule name handling
    available_molecules = list(MOLECULAR_GEOMETRIES.keys())
    normalized_molecule = normalize_molecule_name(molecule, available_molecules)

    if normalized_molecule not in MOLECULAR_GEOMETRIES:
        raise ValueError(
            f"Unknown molecule: {molecule}. Available: {available_molecules}"
        )

    # Use the normalized (correctly cased) molecule name
    molecule = normalized_molecule

    if geometry not in MOLECULAR_GEOMETRIES[molecule]:
        available = list(MOLECULAR_GEOMETRIES[molecule].keys())
        raise ValueError(
            f"Unknown geometry: {geometry}. Available for {molecule}: {available}"
        )

    mol_geometry = MOLECULAR_GEOMETRIES[molecule][geometry]
    mol_props = MOLECULAR_PROPERTIES[molecule]

    # Auto-apply active space for large molecules if not specified
    if auto_active_space and active_space is None and molecule in DEFAULT_ACTIVE_SPACES:
        # First check system size without active space
        try:
            temp_generator = MolecularHamiltonianGenerator(
                basis=basis,
                charge=mol_props["charge"],
                spin=mol_props["spin"],
                transformation=transformation,
            )
            temp_info = temp_generator.get_molecule_info(mol_geometry)

            if temp_info.get("num_qubits", 0) > 12:  # Large system threshold
                active_space = DEFAULT_ACTIVE_SPACES[molecule]
                logger.info(f"Auto-applied active space for {molecule}: {active_space}")

        except Exception as e:
            logger.warning(
                f"Could not determine system size for auto active space: {e}"
            )

    generator = MolecularHamiltonianGenerator(
        basis=basis,
        charge=mol_props["charge"],
        spin=mol_props["spin"],
        transformation=transformation,
    )

    return generator.get_hamiltonian(
        mol_geometry, freeze_core=freeze_core, active_space=active_space
    )


def get_molecular_info_pyscf(
    molecule: str, geometry: str = "equilibrium", basis: str = "sto-3g"
) -> Dict[str, Any]:
    """
    Get detailed information about molecular Hamiltonian from PySCF.

    Args:
        `molecule`: Molecule name
        `geometry`: Geometry type
        `basis`: Basis set

    Returns:
        Dictionary with detailed molecular information
    """
    if not QISKIT_NATURE_AVAILABLE:
        raise ImportError(
            "Qiskit Nature is required for molecular information. "
            "Install with: pip install qiskit-nature[pyscf]==0.6.0"
        )

    # FIXED: Case-insensitive molecule name handling
    available_molecules = list(MOLECULAR_GEOMETRIES.keys())
    normalized_molecule = normalize_molecule_name(molecule, available_molecules)

    if normalized_molecule not in MOLECULAR_GEOMETRIES:
        raise ValueError(
            f"Unknown molecule: {molecule}. Available: {available_molecules}"
        )

    # Use the normalized (correctly cased) molecule name
    molecule = normalized_molecule

    if geometry not in MOLECULAR_GEOMETRIES[molecule]:
        available = list(MOLECULAR_GEOMETRIES[molecule].keys())
        raise ValueError(
            f"Unknown geometry: {geometry}. Available for {molecule}: {available}"
        )

    mol_geometry = MOLECULAR_GEOMETRIES[molecule][geometry]
    mol_props = MOLECULAR_PROPERTIES[molecule]

    generator = MolecularHamiltonianGenerator(
        basis=basis, charge=mol_props["charge"], spin=mol_props["spin"]
    )

    return generator.get_molecule_info(mol_geometry)


def create_test_hamiltonian(num_qubits: int = 6) -> of.OperatorBase:
    """
    Create test Hamiltonian for fallback when PySCF is not available.

    Args:
        `num_qubits`: Number of qubits for test Hamiltonian

    Returns:
        Test Hamiltonian as `qiskit.opflow` operator
    """
    if QUBAP_AVAILABLE:
        logger.info(f"Creating `qubap` test Hamiltonian with {num_qubits} qubits")
        return test_hamiltonian(num_qubits)
    else:
        # Simple fallback Hamiltonian
        logger.warning("Creating simple fallback Hamiltonian (`qubap` not available)")

        # Create simple Ising-like Hamiltonian
        I, X, Y, Z = of.I, of.X, of.Y, of.Z

        # Start with identity
        hamiltonian = 0.0 * I
        for _ in range(num_qubits - 1):
            hamiltonian = hamiltonian ^ I

        # Add Z terms
        for i in range(num_qubits):
            z_op = I
            for j in range(num_qubits):
                if j == i:
                    z_op = z_op ^ Z if j == 0 else z_op ^ Z
                else:
                    z_op = z_op ^ I if j == 0 else z_op ^ I
            hamiltonian += 0.5 * z_op

        # Add ZZ interactions
        for i in range(num_qubits - 1):
            zz_op = I
            for j in range(num_qubits):
                if j == i or j == i + 1:
                    zz_op = zz_op ^ Z if j == 0 else zz_op ^ Z
                else:
                    zz_op = zz_op ^ I if j == 0 else zz_op ^ I
            hamiltonian += 0.25 * zz_op

        return hamiltonian.reduce()


def get_available_molecules() -> Dict[str, List[str]]:
    """
    Get list of available molecules and their geometries.

    Returns:
        Dictionary mapping molecule names to available geometries
    """
    return {
        mol: list(geometries.keys()) for mol, geometries in MOLECULAR_GEOMETRIES.items()
    }


def validate_molecular_system(
    molecule: str, geometry: str = "equilibrium", basis: str = "sto-3g"
) -> Dict[str, Any]:
    """
    Validate molecular system parameters and provide recommendations.

    Args:
        `molecule`: Molecule name
        `geometry`: Geometry type
        `basis`: Basis set

    Returns:
        Validation results with warnings and suggestions
    """
    validation = {
        "valid": True,
        "warnings": [],
        "suggestions": [],
        "estimated_qubits": None,
        "recommended_active_space": None,
        "molecular_properties": None,
    }

    # FIXED: Case-insensitive molecule validation
    available_molecules = list(MOLECULAR_GEOMETRIES.keys())
    normalized_molecule = normalize_molecule_name(molecule, available_molecules)

    # Check molecule
    if normalized_molecule not in MOLECULAR_GEOMETRIES:
        validation["valid"] = False
        validation["warnings"].append(f"Unknown molecule: {molecule}")
        validation["suggestions"].append(f"Available molecules: {available_molecules}")
        return validation

    # Use normalized molecule name for further validation
    molecule = normalized_molecule

    # Check geometry
    if geometry not in MOLECULAR_GEOMETRIES[molecule]:
        validation["warnings"].append(f"Unknown geometry: {geometry}")
        available_geoms = list(MOLECULAR_GEOMETRIES[molecule].keys())
        validation["suggestions"].append(
            f"Available geometries for {molecule}: {available_geoms}"
        )
        # Use first available geometry for estimation
        geometry = available_geoms[0]

    # Get molecular properties
    if molecule in MOLECULAR_PROPERTIES:
        validation["molecular_properties"] = MOLECULAR_PROPERTIES[molecule]

    # Estimate system size if possible
    if QISKIT_NATURE_AVAILABLE:
        try:
            mol_info = get_molecular_info_pyscf(molecule, geometry, basis)
            validation["estimated_qubits"] = mol_info["num_qubits"]

            # Check if system is large
            if mol_info["num_qubits"] > 12:
                validation["warnings"].append(
                    f"Large system: {mol_info['num_qubits']} qubits"
                )

                # Suggest active space
                if molecule in DEFAULT_ACTIVE_SPACES:
                    validation["recommended_active_space"] = DEFAULT_ACTIVE_SPACES[
                        molecule
                    ]
                    validation["suggestions"].append(
                        f"Consider active space: {validation['recommended_active_space']}"
                    )

        except Exception as e:
            validation["warnings"].append(f"Could not estimate system size: {e}")
    else:
        validation["warnings"].append(
            "PySCF not available - cannot estimate system size"
        )
        validation["suggestions"].append(
            "Install qiskit-nature and pyscf for molecular systems"
        )

    return validation


def estimate_hamiltonian_properties(hamiltonian: of.OperatorBase) -> Dict[str, Any]:
    """
    Estimate properties of a Hamiltonian operator.

    Args:
        `hamiltonian`: Hamiltonian operator

    Returns:
        Dictionary with Hamiltonian properties
    """
    properties = {
        "num_qubits": hamiltonian.num_qubits,
        "num_terms": len(hamiltonian.oplist) if hasattr(hamiltonian, "oplist") else 1,
        "is_hermitian": True,  # Molecular Hamiltonians are always Hermitian
        "operator_type": type(hamiltonian).__name__,
    }

    # Try to get coefficient statistics
    try:
        if hasattr(hamiltonian, "oplist") and hasattr(hamiltonian, "coeffs"):
            coeffs = np.array([complex(c) for c in hamiltonian.coeffs])
            real_coeffs = np.real(coeffs)

            properties.update(
                {
                    "max_coefficient": np.max(np.abs(real_coeffs)),
                    "min_coefficient": (
                        np.min(np.abs(real_coeffs[real_coeffs != 0]))
                        if np.any(real_coeffs != 0)
                        else 0
                    ),
                    "coefficient_range": np.max(real_coeffs) - np.min(real_coeffs),
                    "num_nonzero_terms": np.sum(real_coeffs != 0),
                }
            )
    except Exception as e:
        logger.debug(f"Could not extract coefficient statistics: {e}")

    return properties


# Convenience functions for common molecules
def h2_hamiltonian(
    geometry: str = "equilibrium", basis: str = "sto-3g", **kwargs
) -> of.OperatorBase:
    """Get H₂ Hamiltonian with default parameters."""
    return get_molecular_hamiltonian_pyscf("H2", geometry, basis, **kwargs)


def lih_hamiltonian(
    geometry: str = "equilibrium", basis: str = "sto-3g", **kwargs
) -> of.OperatorBase:
    """Get LiH Hamiltonian with default parameters."""
    return get_molecular_hamiltonian_pyscf("LiH", geometry, basis, **kwargs)


def h2o_hamiltonian(
    geometry: str = "equilibrium", basis: str = "sto-3g", **kwargs
) -> of.OperatorBase:
    """Get H₂O Hamiltonian with automatic active space."""
    if "active_space" not in kwargs:
        kwargs["active_space"] = (8, 6)  # Default active space for H2O
    return get_molecular_hamiltonian_pyscf("H2O", geometry, basis, **kwargs)


def beh2_hamiltonian(
    geometry: str = "equilibrium", basis: str = "sto-3g", **kwargs
) -> of.OperatorBase:
    """Get BeH₂ Hamiltonian with default parameters."""
    return get_molecular_hamiltonian_pyscf("BeH2", geometry, basis, **kwargs)


def n2_hamiltonian(
    geometry: str = "stretched", basis: str = "sto-3g", **kwargs
) -> of.OperatorBase:
    """Get N₂ Hamiltonian with default stretched geometry."""
    if "active_space" not in kwargs:
        kwargs["active_space"] = (6, 8)  # Default active space for N2
    return get_molecular_hamiltonian_pyscf("N2", geometry, basis, **kwargs)


if __name__ == "__main__":
    """Test the hamiltonian_builder module."""

    print("🧪 Testing Hamiltonian Builder Module")
    print("=" * 50)

    # Test availability
    print(f"Qiskit Nature available: {QISKIT_NATURE_AVAILABLE}")
    print(f"qubap available: {QUBAP_AVAILABLE}")

    # Test molecular system validation
    print(f"\n🔍 Testing molecular validation...")
    for molecule in ["H2", "LiH", "lih", "LIH", "INVALID"]:  # Test case sensitivity
        validation = validate_molecular_system(molecule)
        print(
            f"{molecule}: Valid={validation['valid']}, Qubits={validation['estimated_qubits']}"
        )
        if validation["warnings"]:
            print(f"  Warnings: {validation['warnings']}")

    # Test Hamiltonian generation
    if QISKIT_NATURE_AVAILABLE:
        print(f"\n⚛️  Testing molecular Hamiltonians...")
        try:
            # Test H2
            h2_ham = h2_hamiltonian()
            h2_props = estimate_hamiltonian_properties(h2_ham)
            print(f"H2: {h2_props['num_qubits']} qubits, {h2_props['num_terms']} terms")

            # Test LiH with various case inputs
            print(f"\n🔧 Testing case sensitivity fixes:")
            for lih_variant in ["LiH", "lih", "LIH"]:
                try:
                    lih_ham = lih_hamiltonian()  # This should work for all variants
                    lih_props = estimate_hamiltonian_properties(lih_ham)
                    print(
                        f"{lih_variant}: {lih_props['num_qubits']} qubits, {lih_props['num_terms']} terms"
                    )
                except Exception as e:
                    print(f"{lih_variant}: Failed - {e}")

            # Test H2O with active space
            h2o_ham = h2o_hamiltonian()
            h2o_props = estimate_hamiltonian_properties(h2o_ham)
            print(
                f"H2O: {h2o_props['num_qubits']} qubits, {h2o_props['num_terms']} terms"
            )

        except Exception as e:
            print(f"  Molecular Hamiltonian test failed: {e}")

    else:
        print(f"\n🔄 Testing fallback Hamiltonians...")
        try:
            test_ham = create_test_hamiltonian(6)
            test_props = estimate_hamiltonian_properties(test_ham)
            print(
                f"Test: {test_props['num_qubits']} qubits, {test_props['num_terms']} terms"
            )
        except Exception as e:
            print(f"  Test Hamiltonian failed: {e}")

    # Show available molecules
    print(f"\n📋 Available molecular systems:")
    available = get_available_molecules()
    for mol, geometries in available.items():
        print(f"  {mol}: {geometries}")

    print(f"\n✅ Hamiltonian builder module test complete!")
