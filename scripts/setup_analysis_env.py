#!/usr/bin/env python3
"""
Analysis Environment Setup Script
=================================

Setup script to verify installation and create proper directory structure
for barren plateau VQE analysis.

Usage:
    python3 scripts/setup_analysis_env.py
    python3 scripts/setup_analysis_env.py --check-only
    python3 scripts/setup_analysis_env.py --create-examples

"""

import argparse
import os
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ Python {version.major}.{version.minor} detected")
        print(f"   Required: Python 3.8 or higher")
        return False
    else:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True


def check_required_packages():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")

    required_packages = [
        ("numpy", "Scientific computing"),
        ("matplotlib", "Plotting and visualization"),
        ("pandas", "Data analysis and manipulation"),
        ("qiskit", "Quantum computing framework"),
        ("qiskit_aer", "Qiskit quantum simulator"),
    ]

    optional_packages = [
        ("qiskit_nature", "Quantum chemistry with Qiskit"),
        ("pyscf", "Quantum chemistry calculations"),
        ("seaborn", "Statistical visualization"),
        ("sklearn", "Machine learning (for PCA)"),
    ]

    results = {
        "required": [],
        "optional": [],
        "missing_required": [],
        "missing_optional": [],
    }

    # Check required packages
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
            results["required"].append(package)
        except ImportError:
            print(f"   ❌ {package} - {description} (MISSING)")
            results["missing_required"].append(package)

    # Check optional packages
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
            results["optional"].append(package)
        except ImportError:
            print(f"   ⚠️  {package} - {description} (OPTIONAL)")
            results["missing_optional"].append(package)

    return results


def check_package_structure():
    """Check if the barren_plateaus_vqe package structure is correct."""
    print("\n📁 Checking package structure...")

    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    package_dir = root_dir / "barren_plateaus_vqe"

    print(f"   Root directory: {root_dir}")
    print(f"   Package directory: {package_dir}")

    if not package_dir.exists():
        print(f"   ❌ Package directory not found: {package_dir}")
        print(f"   Create the directory and place your modules inside")
        return False

    # Check for __init__.py
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        print(f"   ⚠️  Creating missing __init__.py file...")
        init_file.touch()
        print(f"   ✅ __init__.py created")
    else:
        print(f"   ✅ __init__.py")

    required_files = [
        "hamiltonian_builder.py",
        "molecular_analyzer.py",
        "viz_landscape.py",
    ]

    missing_files = []

    for filename in required_files:
        filepath = package_dir / filename
        if filepath.exists():
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} (MISSING)")
            missing_files.append(filename)

    if missing_files:
        print(f"\n   ⚠️  Missing files: {missing_files}")
        print(f"   Make sure the barren_plateaus_vqe package is complete")
        print(f"   Copy your module files to: {package_dir}")
        return False

    return True


def test_package_import():
    """Test importing the barren_plateaus_vqe package."""
    print("\n🧪 Testing package imports...")

    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    sys.path.insert(0, str(root_dir))

    test_imports = [
        ("barren_plateaus_vqe.hamiltonian_builder", "Hamiltonian generation"),
        ("barren_plateaus_vqe.molecular_analyzer", "VQE analysis"),
        ("barren_plateaus_vqe.viz_landscape", "Visualization tools"),
    ]

    success_count = 0

    for module_name, description in test_imports:
        try:
            module = __import__(module_name, fromlist=[""])

            # Test key functions/classes exist
            if module_name.endswith("molecular_analyzer"):
                assert hasattr(
                    module, "create_molecular_analyzer"
                ), "create_molecular_analyzer not found"
                assert hasattr(
                    module, "MolecularVQEAnalyzer"
                ), "MolecularVQEAnalyzer class not found"
            elif module_name.endswith("viz_landscape"):
                assert hasattr(
                    module, "MolecularLandscapeVisualizer"
                ), "MolecularLandscapeVisualizer not found"
            elif module_name.endswith("hamiltonian_builder"):
                assert hasattr(
                    module, "get_available_molecules"
                ), "get_available_molecules not found"

            print(f"   ✅ {module_name} - {description}")
            success_count += 1
        except ImportError as e:
            print(f"   ❌ {module_name} - {description}")
            print(f"      ImportError: {e}")
        except AssertionError as e:
            print(f"   ❌ {module_name} - {description}")
            print(f"      Missing component: {e}")
        except Exception as e:
            print(f"   ❌ {module_name} - {description}")
            print(f"      Error: {e}")

    if success_count == len(test_imports):
        print(f"   🎉 All imports successful!")
        return True
    else:
        print(f"   ⚠️  {success_count}/{len(test_imports)} imports successful")
        return False


def create_directory_structure():
    """Create the required directory structure."""
    print("\n📂 Creating directory structure...")

    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    directories = [
        root_dir / "data",
        root_dir / "plots",
        root_dir / "configs",
        root_dir / "logs",
        root_dir / "barren_plateaus_vqe",  # Ensure package directory exists
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

    # Create __init__.py in package if it doesn't exist
    package_init = root_dir / "barren_plateaus_vqe" / "__init__.py"
    if not package_init.exists():
        package_init.touch()
        print(f"   ✅ {package_init}")

    # Create .gitignore for output directories
    gitignore_content = """# Analysis data and outputs
*.csv
*.json
*.pdf
*.png
*.jpg
*.log

# Keep directory structure
!.gitkeep
"""

    gitignore_paths = [
        root_dir / "data" / ".gitignore",
        root_dir / "plots" / ".gitignore",
        root_dir / "logs" / ".gitignore",
    ]

    for gitignore_path in gitignore_paths:
        if not gitignore_path.exists():
            with open(gitignore_path, "w") as f:
                f.write(gitignore_content)
            print(f"   ✅ {gitignore_path}")

    # Create a README in the package directory with instructions
    package_readme = root_dir / "barren_plateaus_vqe" / "README.md"
    if not package_readme.exists():
        readme_content = """# Barren Plateaus VQE Package

This directory should contain your barren plateau analysis modules:

## Required Files:
- `molecular_analyzer.py` - VQE analysis and molecular system handling
- `viz_landscape.py` - Visualization and plotting tools  
- `hamiltonian_builder.py` - Molecular Hamiltonian generation

## Usage:
Place your module files in this directory, then run the analysis scripts:

```bash
# Test the setup
python scripts/setup_analysis_env.py --check-only

# Run analysis
python scripts/molecular_bp_analysis.py --molecule "H2"
```
"""
        with open(package_readme, "w") as f:
            f.write(readme_content)
        print(f"   ✅ {package_readme}")


def create_example_configs():
    """Create example configuration files."""
    print("\n📄 Creating example configurations...")

    script_dir = Path(__file__).parent
    root_dir = script_dir.parent
    configs_dir = root_dir / "configs"

    # Example configuration for quick testing
    quick_test_config = {
        "description": "Quick test configuration",
        "molecules": ["H2"],
        "geometries": ["equilibrium"],
        "basis_sets": ["sto-3g"],
        "iterations": 200,
        "layer_scaling": False,
        "skip_landscapes": True,
    }

    # Example configuration for research
    research_config = {
        "description": "Research configuration",
        "molecules": ["H2", "LiH", "BeH2"],
        "geometries": ["equilibrium", "stretched"],
        "basis_sets": ["sto-3g", "6-31g"],
        "iterations": 1000,
        "layer_scaling": True,
        "max_layers": 6,
        "skip_landscapes": False,
    }

    configs = [
        ("quick_test.json", quick_test_config),
        ("research.json", research_config),
    ]

    import json

    for filename, config in configs:
        config_path = configs_dir / filename
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"   ✅ {config_path}")


def create_example_scripts():
    """Create example usage scripts."""
    print("\n📝 Creating example scripts...")

    script_dir = Path(__file__).parent

    # Example 1: Quick test script
    quick_test_script = """#!/bin/bash
# Quick test of the analysis system

echo "🧪 Running quick test..."

# Test H2 molecule with basic settings
python scripts/molecular_bp_analysis.py --molecule "H2" \\
    --geometry equilibrium \\
    --basis sto-3g \\
    --iterations 200 \\
    --skip-landscapes

echo "✅ Quick test complete! Check data/ directory for results."
"""

    # Example 2: Research script
    research_script = """#!/bin/bash
# Comprehensive research analysis

echo "🔬 Running comprehensive research analysis..."

# Run batch analysis with research preset
python scripts/batch_molecular_analysis.py \\
    --preset small_molecules \\
    --iterations 1000

echo "🎉 Research analysis complete! Check data/ directory for results."
"""

    scripts = [
        ("run_quick_test.sh", quick_test_script),
        ("run_research_analysis.sh", research_script),
    ]

    for filename, script_content in scripts:
        script_path = script_dir / filename
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make executable on Unix systems
        if os.name != "nt":  # Not Windows
            script_path.chmod(0o755)

        print(f"   ✅ {script_path}")


def print_installation_help(missing_packages):
    """Print installation help for missing packages."""
    if not missing_packages["missing_required"]:
        return

    print(f"\n💡 Installation Help:")
    print("=" * 50)

    # Basic pip install
    required = missing_packages["missing_required"]
    if required:
        print(f"\nInstall required packages:")
        print(f"pip install {' '.join(required)}")

    # Optional packages
    optional = missing_packages["missing_optional"]
    if optional:
        print(f"\nInstall optional packages for full functionality:")
        print(f"pip install {' '.join(optional)}")

    # Specific instructions for problematic packages
    if "qiskit_nature" in optional:
        print(f"\nFor molecular calculations:")
        print(f"pip install qiskit-nature pyscf")

    if "pyscf" in optional:
        print(f"\nNote: PySCF installation may require:")
        print(f"- C++ compiler")
        print(f"- cmake")
        print(f"- BLAS/LAPACK libraries")

    print(f"\nAlternatively, install from requirements file:")
    print(f"pip install -r requirements.txt")


def print_usage_examples():
    """Print usage examples."""
    print(f"\n🚀 Usage Examples:")
    print("=" * 50)

    examples = [
        (
            "Quick test",
            'python3 scripts/molecular_bp_analysis.py --molecule "H2" --iterations 200',
        ),
        (
            "Batch analysis",
            "python3 scripts/batch_molecular_analysis.py --preset quick_test",
        ),
        (
            "Layer scaling",
            'python3 scripts/molecular_bp_analysis.py --molecule "H2" --layer-scaling',
        ),
        (
            "Large molecule",
            'python3 scripts/molecular_bp_analysis.py --molecule "H2O" --active-space 8 6',
        ),
        (
            "Custom batch",
            "python3 scripts/batch_molecular_analysis.py --custom H2,LiH,BeH2",
        ),
    ]

    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup analysis environment")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check environment, don't create files",
    )
    parser.add_argument(
        "--create-examples",
        action="store_true",
        help="Create example configurations and scripts",
    )

    args = parser.parse_args()

    print("🧪 Barren Plateaus VQE Analysis Environment Setup")
    print("=" * 60)

    # Check system requirements
    python_ok = check_python_version()
    packages_info = check_required_packages()
    structure_ok = check_package_structure()

    if not python_ok:
        print(f"\n❌ Python version incompatible")
        return 1

    if packages_info["missing_required"]:
        print(f"\n❌ Missing required packages: {packages_info['missing_required']}")
        print_installation_help(packages_info)
        if not args.check_only:
            print(f"\nPlease install required packages before proceeding.")
            return 1

    if not structure_ok:
        print(f"\n❌ Package structure incomplete")
        return 1

    # Test imports
    import_ok = test_package_import()

    if not import_ok:
        print(f"\n❌ Package imports failed")
        if not args.check_only:
            return 1

    if args.check_only:
        print(f"\n✅ Environment check complete")
        if import_ok and not packages_info["missing_required"]:
            print(f"🎉 Ready for analysis!")
        return 0

    # Create directory structure
    create_directory_structure()

    # Create examples if requested
    if args.create_examples:
        create_example_configs()
        create_example_scripts()

    # Print final status
    print(f"\n🎉 Setup complete!")
    print("=" * 60)

    if packages_info["missing_optional"]:
        print(f"\n⚠️  Optional packages missing: {packages_info['missing_optional']}")
        print(f"Some features may be limited. Install for full functionality:")
        print(f"pip install {' '.join(packages_info['missing_optional'])}")

    print_usage_examples()

    print(f"\n📁 Directory structure created:")
    print(f"  data/ - Analysis results and data files")
    print(f"  plots/ - Generated plots and visualizations")
    print(f"  configs/ - Configuration files")
    print(f"  logs/ - Log files")
    print(f"  barren_plateaus_vqe/ - Package modules")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
