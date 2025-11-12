# BarrenPlateausVQE.jl

**Julia Implementation for VQE Barren Plateau Analysis**

A Julia package for studying barren plateau phenomena in Variational Quantum Eigensolver (VQE) algorithms using molecular Hamiltonians. Built with Yao.jl quantum computing framework.

> **Research Paper**: *"Investigating Different Barren Plateaus Mitigation Strategies in Variational Quantum Eigensolver"*
> Mostafa Atallah, Nouhaila Innan, Muhammed Kashif, Muhammed Shafique

## Key Features

- **5 VQE Methods**: Standard, Local-Global, Adiabatic, SEA, and Pretrained VQE with barren plateau mitigation
- **10 Molecular Systems**: H₂, LiH, BeH₂, H₂O, N₂, CO, NH₃, CH₄, HF, BH
- **Comprehensive Analysis**: Gradient diagnostics, loss landscapes, convergence tracking
- **Visualization**: Energy plots, 3D landscapes, circuit diagrams, LaTeX tables
- **Easy to Use**: Simple API with integrated analysis framework

## Installation

**Prerequisites**: Julia 1.8+ ([Download](https://julialang.org/downloads/))

```bash
# Clone repository
git clone https://github.com/Innanov/BarrenPlateaus-VQE.git
cd BarrenPlateaus-VQE

# Run automated setup
julia setup.jl
```

**Manual installation**:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start

```julia
# Load package
julia --project=.
include("src/BarrenPlateausVQE.jl")
using .BarrenPlateausVQE

# Run analysis on H₂ molecule
analyzer = MolecularVQEAnalyzer("H2", n_layers=2)
results = run_complete_analysis(analyzer, num_iters=500)

# Generate visualizations
create_comprehensive_visualization(analyzer, output_dir="./results/H2")
```

**Single method example**:
```julia
analyzer = MolecularVQEAnalyzer("H2", n_layers=2)
result = run_standard_vqe(analyzer, num_iters=300)

println("Final energy: $(result["method_result"]["vqe_result"].final_energy)")
println("Exact energy: $(analyzer.exact_energy)")
```

**Using scripts**:
```bash
julia scripts/molecular_vqe_analysis.jl --molecule H2 --layers 2 --iterations 1000
julia scripts/gradient_variance_scaling.jl --molecule LiH --max-qubits 12
julia scripts/generate_circuit_plots.jl --molecule H2 --layers 2
```

## Supported Molecules

| Molecule | Formula | Qubits | Electrons | Geometries |
|----------|---------|--------|-----------|------------|
| H₂ | H-H | 4 | 2 | equilibrium, stretched, compressed |
| LiH | Li-H | 8 | 4 | equilibrium, stretched, compressed |
| BeH₂ | Be-H₂ | 12 | 4 | equilibrium, stretched |
| H₂O | H-O-H | 12 | 8 | equilibrium, stretched |
| N₂ | N≡N | 16 | 10 | equilibrium, stretched |
| CO | C≡O | 16 | 10 | equilibrium, stretched |
| NH₃ | N-H₃ | 14 | 10 | equilibrium |
| CH₄ | C-H₄ | 16 | 10 | equilibrium |
| HF | H-F | 8 | 10 | equilibrium, stretched |
| BH | B-H | 8 | 6 | equilibrium, stretched |

```julia
# Create molecular systems
h2_system = create_molecular_hamiltonian("H2")
lih_stretched = create_molecular_hamiltonian("LiH", geometry="stretched")
h2_custom = create_molecular_hamiltonian("H2", bond_length=1.2)
```

## VQE Methods

### 1. Standard VQE
Hardware-efficient ansatz with SPSA optimization.

```julia
analyzer = MolecularVQEAnalyzer("H2", n_layers=2)
result = run_standard_vqe(analyzer, num_iters=300)
```

### 2. Local-Global VQE
Two-stage optimization: local warmup → global refinement.

```julia
result = run_local_global_vqe(analyzer, num_iters=300, shift_iter=100)
```

### 3. Adiabatic VQE
Gradual interpolation from simple to target Hamiltonian.

```julia
result = run_adiabatic_vqe(analyzer, num_iters=300)
```

### 4. State Efficient Ansatz (SEA) VQE
Optimized ansatz with reduced parameter count.

```julia
result = run_sea_vqe(analyzer, num_iters=300)
```

### 5. Pretrained VQE
Transfer learning from simpler Hamiltonians.

```julia
result = run_pretrained_vqe(analyzer, iters_vqe=300, iters_train=50)
```

## Analysis & Visualization

**Complete analysis**:
```julia
analyzer = MolecularVQEAnalyzer("LiH", n_layers=2)
results = run_complete_analysis(analyzer, num_iters=1000)

# Access results
for (method, data) in results
    vqe_result = data["method_result"]["vqe_result"]
    println("$(method): Energy = $(vqe_result.final_energy)")
end
```

**Visualizations**:
```julia
# Individual plots
plot_all_methods_energy_convergence(analyzer, save_path="convergence.pdf")
plot_all_methods_loss_landscape_3d(analyzer, save_path="landscape.pdf")

# Performance tables
create_performance_table_all_methods(analyzer, save_csv="performance.csv")
create_latex_table_all_methods(analyzer, save_path="table.tex")

# All visualizations
create_comprehensive_visualization(analyzer, output_dir="./analysis")
```

**Circuit visualization**:
```julia
include("src/circuit_visualization.jl")
using .CircuitVisualization

files = visualize_all_ansatz_circuits(analyzer, output_dir="./circuits")
analysis = analyze_circuit_architectures(analyzer)
```

## Package Structure

```
BarrenPlateaus-VQE/
├── src/
│   ├── BarrenPlateausVQE.jl        # Main module
│   ├── hamiltonian_builder.jl      # Molecular Hamiltonians
│   ├── quantum_utils.jl            # Quantum utilities
│   ├── molecular_analyzer.jl       # Analysis framework
│   ├── visualization.jl            # Plotting
│   ├── circuit_visualization.jl    # Circuit plots
│   └── methods/                    # VQE implementations
│       ├── standard_vqe.jl
│       ├── local_global_vqe.jl
│       ├── adiabatic_vqe.jl
│       ├── sea_vqe.jl
│       └── pretrained_vqe.jl
├── scripts/
│   ├── molecular_vqe_analysis.jl
│   ├── gradient_variance_scaling.jl
│   └── generate_circuit_plots.jl
└── test/
    └── basic_tests.jl
```

## Advanced Usage

**Custom Hamiltonians**:
```julia
terms = [("ZIII", 0.5), ("IZII", 0.5), ("ZZII", 0.2)]
hamiltonian = create_pauli_hamiltonian(4, terms)

vqe = StandardVQE(4, 2)
initial_params = random_initial_parameters(vqe.n_parameters)
result = run_vqe(vqe, hamiltonian, initial_params, 500)
```

**Batch analysis**:
```julia
for mol in ["H2", "LiH", "BeH2"], n_layers in [1, 2, 3]
    analyzer = MolecularVQEAnalyzer(mol, n_layers=n_layers)
    results = run_complete_analysis(analyzer, num_iters=500)

    output_dir = "./results/$(mol)_layers$(n_layers)"
    create_comprehensive_visualization(analyzer, output_dir=output_dir)
end
```

**Loss landscape**:
```julia
analyzer = MolecularVQEAnalyzer("H2", n_layers=2)
result = run_standard_vqe(analyzer, num_iters=200)

param_i, param_j, landscape = compute_loss_landscape_2d(
    analyzer.system.hamiltonian,
    result["method_result"]["vqe_result"].final_parameters,
    (1, 2),  # parameter indices
    50,      # resolution
    0.5,     # range
    result["ansatz"]
)
```

## Key Dependencies

- **Yao.jl** (v0.8) - Quantum computing framework
- **Optim.jl** - Optimization algorithms
- **Plots.jl** / **PlotlyJS.jl** - Plotting
- **DataFrames.jl** / **CSV.jl** - Data handling
- **YaoPlots.jl** - Circuit visualization

See [Project.toml](Project.toml) for complete list.

## Testing

```julia
# Run tests
julia --project=. test/basic_tests.jl

# Quick validation
include("src/BarrenPlateausVQE.jl")
using .BarrenPlateausVQE

analyzer = MolecularVQEAnalyzer("H2", use_test_hamiltonian=true)
result = run_standard_vqe(analyzer, num_iters=50)
```

## Troubleshooting

**Plotting (Windows)**:
```julia
using Pkg
Pkg.add("PlotlyJS")
using Plots; plotlyjs()
```

**Circuit visualization**:
```julia
Pkg.add("YaoPlots")
```

**Large molecules** (>12 qubits):
```julia
# Use lower resolution
plot_all_methods_loss_landscape_3d(analyzer, resolution=20)
```

## Citation

If you use this software in your research, please cite:

```bibtex
@article{atallah2025barren,
  title = {Investigating Different Barren Plateaus Mitigation Strategies in Variational Quantum Eigensolver},
  author = {Atallah, Mostafa and Innan, Nouhaila and Kashif, Muhammed and Shafique, Muhammed},
  year = {2025},
  journal = {[To be added]},
  note = {Preprint}
}
```
