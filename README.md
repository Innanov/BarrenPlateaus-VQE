# Code for the paper: "Alleviating Barren Plateaus - But at What Cost to VQE Precision?"

This directory contains scripts for running comprehensive barren plateau analysis on molecular VQE systems.

**Note:** This codebase has been developed and tested on Linux systems. While it may work on other platforms, optimal performance and compatibility are ensured on Linux environments.

## Scripts Overview

### `molecular_bp_analysis.py`
Main analysis script for individual molecular systems. Provides complete barren plateau analysis with multiple VQE methods, gradient diagnostics, and visualization.

### `batch_molecular_analysis.py`
Batch processing helper for running multiple analyses with predefined configurations or custom molecule lists.

## Quick Start

### Single Molecule Analysis

```bash
# Basic H2 analysis
python3 scripts/molecular_bp_analysis.py --molecule "H2"

# H2 with specific parameters
python3 scripts/molecular_bp_analysis.py --molecule "H2" --basis sto-3g --geometry equilibrium --iterations 1000

# H2O with active space
python3 scripts/molecular_bp_analysis.py --molecule "H2O" --active-space 8 6 --iterations 800

# Layer scaling analysis
python3 scripts/molecular_bp_analysis.py --molecule "H2" --layer-scaling --max-layers 4

# LiH with all options
python3 scripts/molecular_bp_analysis.py --molecule "LiH" --geometry stretched --basis 6-31g --freeze-core --iterations 1200
```

### Batch Analysis

```bash
# List available presets
python3 scripts/batch_molecular_analysis.py --list-presets

# Quick test
python3 scripts/batch_molecular_analysis.py --preset quick_test

# Small molecules comparison
python3 scripts/batch_molecular_analysis.py --preset small_molecules

# Layer scaling study
python3 scripts/batch_molecular_analysis.py --preset layer_scaling

# Custom molecule list
python3 scripts/batch_molecular_analysis.py --custom H2,LiH,BeH2 --iterations 800

# Dry run (show commands without executing)
python3 scripts/batch_molecular_analysis.py --preset small_molecules --dry-run
```

## Command Line Options

### `molecular_bp_analysis.py`

| Option | Description | Default |
|--------|-------------|---------|
| `--molecule` | Molecule to analyze ("H2", "LiH", "BeH2", "H2O", "N2", "CO") | Required |
| `--geometry` | Molecular geometry (equilibrium, stretched, etc.) | equilibrium |
| `--basis` | Basis set (sto-3g, 6-31g, cc-pvdz, etc.) | sto-3g |
| `--iterations` | VQE iterations per method | 1000 |
| `--active-space` | Active space (electrons orbitals) | None |
| `--freeze-core` | Use frozen core approximation | False |
| `--layer-scaling` | Run layer scaling analysis | False |
| `--max-layers` | Maximum layers for scaling | 4 |
| `--landscape-grid` | Grid size for loss landscapes | 20 |
| `--skip-landscapes` | Skip loss landscape computation | False |
| `--test-hamiltonian` | Use test Hamiltonian instead | False |

### `batch_molecular_analysis.py`

| Option | Description |
|--------|-------------|
| `--preset` | Use predefined analysis preset |
| `--custom` | Custom molecule list (comma-separated) |
| `--iterations` | Override iterations for analysis |
| `--layer-scaling` | Enable layer scaling analysis |
| `--max-layers` | Maximum layers for scaling |
| `--basis` | Basis set for custom molecules |
| `--geometry` | Geometry for custom molecules |
| `--parallel` | Run analyses in parallel (experimental) |
| `--dry-run` | Show commands without executing |
| `--list-presets` | List available presets |

## Available Presets

### `quick_test`
- **Purpose:** Quick test with H2 molecule
- **Molecules:** H2 (equilibrium, sto-3g)
- **Iterations:** 200

### `small_molecules`
- **Purpose:** Analysis of small molecules
- **Molecules:** H2 (equilibrium/stretched), LiH, BeH2
- **Iterations:** 1000

### `layer_scaling`
- **Purpose:** Layer scaling analysis for H2
- **Molecules:** H2 (equilibrium, sto-3g)
- **Features:** Layer scaling up to 6 layers
- **Iterations:** 500

### `basis_comparison`
- **Purpose:** Basis set comparison for H2
- **Molecules:** H2 with sto-3g, 6-31g, cc-pvdz
- **Iterations:** 800

### `geometry_effects`
- **Purpose:** Geometry effects for multiple molecules
- **Molecules:** H2 and LiH (equilibrium/stretched)
- **Iterations:** 800

### `large_molecules`
- **Purpose:** Large molecules with active space
- **Molecules:** H2O (8e,6o), BeH2 (4e,6o)
- **Iterations:** 600

### `comprehensive`
- **Purpose:** Comprehensive analysis suite
- **Molecules:** H2, LiH, BeH2, H2O (with layer scaling)
- **Iterations:** 1000

## Output Structure

```
data/
├── [molecule]_[geometry]_[basis]/
│   ├── [timestamp]/
│   │   ├── basic_analysis_results.json
│   │   ├── method_comparison_summary.csv
│   │   ├── layer_scaling_data.csv          # if --layer-scaling
│   │   ├── layer_scaling_detailed.json     # if --layer-scaling
│   │   ├── analysis_config.json
│   │   └── analysis_report.md
└── batch_summary_[timestamp].json          # for batch runs

plots/
├── [molecule]_[geometry]_[basis]/
│   ├── [timestamp]/
│   │   ├── energy_convergence.pdf
│   │   ├── gradient_diagnostics.pdf
│   │   ├── performance_table.csv
│   │   ├── performance_table.tex
│   │   ├── loss_landscapes.pdf             # if not skipped
│   │   ├── layer_variance_scaling.pdf      # if --layer-scaling
│   │   ├── parameter_scaling.pdf           # if --layer-scaling
│   │   └── energy_error_scaling.pdf        # if --layer-scaling

configs/
├── quick_test.json
├── research.json
└── [custom_configs].json

logs/
└── [analysis_logs].log
```

## VQE Methods Analyzed

1. **Standard VQE** - Baseline implementation with `EfficientSU2` ansatz
2. **Local-Global VQE** - Two-stage optimization strategy
3. **Adiabatic VQE** - Gradual Hamiltonian transition approach
4. **State Efficient Ansatz (SEA)** - Reduced expressivity design
5. **Pretrained VQE** - MPS-based parameter initialization

## Generated Analysis

### Plots
- Energy convergence comparison across all methods
- Gradient diagnostics (variance, norms, distributions)
- Loss landscape visualization (2D/3D surfaces)
- Layer scaling plots (if enabled)
- Performance comparison tables

### Data
- Method comparison summaries
- Gradient variance statistics
- Energy convergence trajectories
- Layer scaling data (if enabled)
- Complete diagnostic results

### Reports
- Automated markdown reports
- LaTeX-formatted tables
- Configuration documentation

## System Requirements

- **Operating System:** Linux (developed and tested on Linux systems)
- **Python:** Python 3.8+ (use `python3` command)
- **Packages:**
  - Qiskit, Qiskit Nature
  - PySCF (for molecular Hamiltonians)
  - Matplotlib, Seaborn (for visualization)
  - Pandas, NumPy (for data processing)

## References

1. McClean, J. R., et al. "Barren plateaus in quantum neural network training landscapes." Nature Communications 9, 4812 (2018).
2. Cerezo, M., et al. "Cost function dependent barren plateaus in shallow parametrized quantum circuits." Nature Communications 12, 1791 (2021).
3. Larocca, M., et al. "A review of barren plateaus in variational quantum computing." arXiv:2405.00781 (2024).