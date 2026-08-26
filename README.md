# BarrenPlateaus-VQE

A PennyLane implementation for studying barren-plateau phenomena in VQE using
molecular Hamiltonians from the PennyLane qchem collection.

> *"Investigating Different Barren Plateaus Mitigation Strategies in Variational Quantum Eigensolver"*
> -- Mostafa Atallah, Nouhaila Innan, Muhammad Kashif, Muhammad Shafique
> ([arXiv:2512.11171](https://arxiv.org/abs/2512.11171))

## Features

- **5 VQE methods**: Standard (baseline) plus four mitigation methods
  (Local-Global, Adiabatic, SEA, Pretrained), all at a matched depth and iteration
  budget for a fair comparison.
- **Molecular systems** (4 to 32 qubits): pre-fetched from PennyLane and cached as
  JSON.
- **Metrics**: energy error, real state fidelity, and gradient-variance scaling
  (the barren-plateau metric), plus loss-landscape and convergence figures.
- **Three gradient optimizers**: Adam, QNG (quantum natural gradient), and Adagrad,
  compared iteration-for-iteration.

## Setup

**Prerequisites**: Python 3.10.

```bash
git clone https://github.com/Innanov/BarrenPlateaus-VQE.git && cd BarrenPlateaus-VQE
git checkout pennylane
py -3.10 -m venv .venv310 && .venv310\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
```

**Fetch Hamiltonians** (one-time, writes JSON files to `data/`):

```bash
python scripts/fetch_hamiltonians.py
```

### Simulator backend

QNodes run on **`lightning.qubit`** (the fast C++ statevector simulator, pulled in
by `pennylane-lightning`) by default, roughly 50x faster than `default.qubit` on
the 12 to 14 qubit systems. Override with the `BPVQE_DEVICE` environment variable.
The code falls back to `default.qubit` if the requested device is unavailable, and
records the resolved backend in each run's `run_parameters.json`.

> **GPU:** `lightning.gpu` (cuQuantum) is faster still but ships **Linux-only**.
> There is no Windows wheel, so it is not used on Windows. On Linux with an NVIDIA
> GPU: `pip install pennylane-lightning-gpu` then `BPVQE_DEVICE=lightning.gpu`.

The mitigation methods are ported to PennyLane in `src/core/methods`. The
published [qubap](https://github.com/jgidi/quantum-barren-plateaus) package, which
ships only Qiskit implementations, is the credited upstream reference. It is
**not** a runtime dependency.

## Usage

Computation and plotting are **separate**: `compute_*.py` runs the physics and
saves all data (CSV + NPZ) with no figures. `plot_*.py` rebuilds the figures from
that saved data, so restyling a plot never re-runs the expensive physics.

```bash
# COMPUTE: VQE analysis -- all five methods across all three optimizers by
# default. Saves the performance table, convergence_history.csv, and
# landscape_<method>.npz (grid + trajectories). No PDFs.
python scripts/compute_vqe_analysis.py --molecule LiH --depth 4 --iters 1000

#   ...a single optimizer, or skip landscapes, if desired:
python scripts/compute_vqe_analysis.py --molecule LiH --optimizers adam --no-landscape

# COMPUTE: gradient-variance scaling vs depth (the barren-plateau metric).
# Optimizer-independent -- samples gradients at random inits, no optimization.
python scripts/compute_gradient_scaling.py --molecules H2 LiH BeH2 --max-layers 50 --samples 100

# PLOT: rebuild figures from the saved data (re-run any time to restyle).
python scripts/plot_vqe_analysis.py --molecule LiH        # newest LiH VQE run
python scripts/plot_gradient_scaling.py --molecule H2     # newest H2 gradient run
python scripts/plot_vqe_analysis.py --dir results/LiH/vqe_analysis/<timestamp>

# REBUILD: re-sample the landscapes of a run from its saved optimizer paths,
# without re-running the VQE (e.g. after a landscape-method change).
python scripts/rebuild_landscapes.py --molecule LiH
```

Results are written to `results/<molecule>/<analysis_type>/<timestamp>/`, with a
`run_parameters.json` recording the run parameters, data path, and device for
provenance. The performance CSV carries an `optimizer` column so every optimizer's
metrics (energy, error, real fidelity) sit in one table. `n_params` and the
gradient-variance metric are optimizer-independent.

### Library API

```python
from src.core.backend import load_hamiltonian
from src.core.methods import run_method, MethodConfig
from src.core.analysis import metrics

system = load_hamiltonian("H2", "equilibrium")
cfg = MethodConfig(depth=4, optimizer="adam", max_iters=500)
result = run_method("standard", system, cfg)

print(result.final_energy, metrics.fidelity(result.ansatz, result.params, system))
```

## VQE methods

Every mitigation method generalizes a method from qubap. See each method module
for its own reference.

| Method | Description |
|--------|-------------|
| **Standard** | The unmitigated baseline: EfficientSU2 (Ry, Rz + circular CNOT) optimized directly against the full H. |
| **Local-Global** | Warm-start on a ground-state local cost (Cerezo et al.), then refine on the full H. |
| **Adiabatic** | Staged anneal `(1-s)*H_local + s*H_global`, each stage held fixed and warm-started from the previous. |
| **SEA** | Standard VQE with the State Efficient Ansatz (ported from qubap), which requires an even qubit count. |
| **Pretrained** | Two-stage MPS: train the diagonal MPS stage, transfer by prefix zero-pad into the full MPS stage, then refine. |

## Project structure

```
src/
  core/
    ansatze/      # EfficientSU2, MPS, SEA
    backend/      # devices, hamiltonians (JSON -> qml.Hamiltonian, sparse ground state), optimizers
    analysis/     # gradient-variance scaling, loss landscape, metrics
    methods/      # the 5 methods (one module each), plus base (config/result, cost, loop, registry)
  utils/
    helpers.py    # small shared helpers (statevector, tape_ops, ising_system, ...)
    io.py         # output dirs, CSV/JSON/NPZ writers + loaders, run-dir resolution, provenance
    plotting.py   # convergence, gradient scaling, 3D + contour landscapes (IEEE style)
    progress.py   # format_duration and the elapsed/ETA console logger
scripts/
  fetch_hamiltonians.py         # fetch from PennyLane -> data/*.json
  compute_vqe_analysis.py       # COMPUTE: VQE physics -> CSV + landscape NPZ (no plots)
  compute_gradient_scaling.py   # COMPUTE: gradient-variance sweep -> CSV (no plots)
  plot_vqe_analysis.py          # PLOT: rebuild convergence + landscape figures from saved data
  plot_gradient_scaling.py      # PLOT: rebuild the gradient-scaling figure from saved data
  rebuild_landscapes.py         # re-sample landscapes from saved optimizer paths (no VQE re-run)
data/                           # cached Hamiltonian JSONs (git-ignored)
results/                        # analysis output: data (CSV/NPZ) + figures (git-ignored)
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
