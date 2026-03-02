# BarrenPlateausVQE.jl

A Julia package for studying barren plateau phenomena in VQE algorithms using molecular Hamiltonians from the PennyLane qchem collection. Built with Yao.jl.

> *"Investigating Different Barren Plateaus Mitigation Strategies in Variational Quantum Eigensolver"*
> -- Mostafa Atallah, Nouhaila Innan, Muhammad Kashif, Muhammad Shafique
> ([arXiv:2512.11171](https://arxiv.org/abs/2512.11171))

## Features

- **5 VQE Methods**: Standard, Local-Global, Adiabatic, SEA, Pretrained
- **35 Molecules** (4-32 qubits): Pre-fetched from PennyLane and cached as JSON
- **Analysis**: Gradient variance scaling, loss landscapes, convergence, barren plateau detection

## Setup

**Prerequisites**: Julia 1.8+, Python 3.10

```bash
git clone https://github.com/Innanov/BarrenPlateaus-VQE.git && cd BarrenPlateaus-VQE
julia setup.jl                        # or: julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

**Fetch Hamiltonians** (one-time, requires Python 3.10 venv):
```bash
py -3.10 -m venv .venv310 && .venv310\Scripts\Activate.ps1
pip install pennylane aiohttp fsspec h5py
python scripts/fetch_hamiltonians.py  # saves 85 JSON files to data/
```

## Usage

**Scripts** (recommended):
```bash
# VQE analysis -- all methods on a molecule
julia --project=. scripts/molecular_vqe_analysis.jl --molecule LiH --layers 2 --iterations 1000 --methods all --verbose

# Gradient variance vs circuit depth
julia --project=. scripts/gradient_variance_scaling.jl --molecule H2 --max-layers 15 --samples 100
```

Results are saved to `results/<molecule>/<analysis_type>/<timestamp>/`.

**Julia API**:
```julia
using BarrenPlateausVQE

analyzer = MolecularVQEAnalyzer("H2", n_layers=2)
results = run_complete_analysis(analyzer, num_iters=500)
create_comprehensive_visualization(analyzer, output_dir="./results/H2")
```

## VQE Methods

| Method | Description | Key Setting |
|--------|-------------|-------------|
| **Standard** | Hardware-efficient ansatz + SPSA | `rotation_gates=[:Rx,:Ry]`, circular entanglement |
| **Local-Global** | Local warmup then global refinement | 1/3 local + 2/3 global split |
| **Adiabatic** | Interpolate mixer to target Hamiltonian | 10 adiabatic steps |
| **SEA** | Full-connectivity CNOT ansatz | `depth=[1,1,1]` |
| **Pretrained** | MPS pretraining then full optimization | 1/4 pretrain + 3/4 full |

## Supported Molecules

35 molecules from PennyLane qchem (85 JSON files). Geometries: equilibrium, stretched, compressed where available.

| Qubits | Molecules |
|--------|-----------|
| 4 | H2, HeH+ |
| 6-10 | H3+, H4, He2, H5 |
| 12 | H6, HF, LiH, NeH+, OH- |
| 14 | BeH2, CH2, H2O, H7 |
| 16-18 | BH3, H8, NH3, CH4 |
| 20 | C2, CO, H10, Li2, N2, O2 |
| 22-24 | HCN, C2H2, CH2O, H2O2, N2H2 |
| 28-32 | C2H4, N2H4, CO2, O3, C2H6 |

## Project Structure

```
src/
  BarrenPlateausVQE.jl              # Main module
  core/
    hamiltonian_builder.jl          # Hamiltonians & energy evaluation
    ansatz_library.jl               # Ansatz circuits (SU2, MPS, SEA)
    dataset_loader.jl               # JSON cache loader
    molecular_analyzer.jl           # Analysis orchestration
    methods/                        # 5 VQE implementations
  utils/                            # Quantum utils, visualization, circuit plots
scripts/
  fetch_hamiltonians.py             # Python: fetch from PennyLane
  molecular_vqe_analysis.jl         # VQE comparison analysis
  gradient_variance_scaling.jl      # Gradient variance vs depth
  generate_circuit_plots.jl         # Circuit diagrams
data/                               # Cached Hamiltonian JSONs (git-ignored)
results/                            # Analysis output (git-ignored)
```

## Citation

```bibtex
@article{atallah2025investigating,
  title={Investigating Different Barren Plateaus Mitigation Strategies in Variational Quantum Eigensolver},
  author={Atallah, Mostafa and Innan, Nouhaila and Kashif, Muhammad and Shafique, Muhammad},
  journal={arXiv preprint arXiv:2512.11171},
  year={2025}
}
```