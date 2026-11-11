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
  (the barren-plateau metric), plus convergence figures, all as mean and standard
  deviation over seeded runs.
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

## Usage

Computation and plotting are **separate**: `compute_*.py` runs the physics and
saves all data (CSV) with no figures. `plot_*.py` rebuilds the figures from that
saved data, so restyling a plot never re-runs the expensive physics.

The VQE analysis runs one seed per call. The seed sets every method's initial
parameters, so repeating a run over several seeds shows how much a result depends
on where it starts. Three steps turn the seeds into tables and figures:

```bash
# 1. COMPUTE: one seed of all five methods. Run it once per seed with the same
#    other settings. Writes results/LiH/vqe_seeds/adam_depth4_iters1000/seed_<seed>/.
python scripts/compute_vqe_analysis.py --molecule LiH --optimizers adam --seed 0
python scripts/compute_vqe_analysis.py --molecule LiH --optimizers adam --seed 1

# 2. AGGREGATE: mean and sample std over the seeds (at least two), written as
#    performance_table_mean_std.csv and convergence_mean_std.csv.
python scripts/aggregate_vqe_seeds.py --molecule LiH --run-name adam_depth4_iters1000

# 3. PLOT: convergence of the seed mean with a shaded mean +- std band.
python scripts/plot_vqe_seeds.py --molecule LiH --run-name adam_depth4_iters1000

# Convergence of a single seed (or of a run made before seeding):
python scripts/plot_vqe_analysis.py --dir results/LiH/vqe_seeds/adam_depth4_iters1000/seed_0

# COMPUTE: gradient-variance scaling vs depth (the barren-plateau metric).
# Optimizer-independent, samples gradients at random inits, no optimization.
python scripts/compute_gradient_scaling.py --molecules H2 LiH BeH2 --max-layers 50 --samples 100

# PLOT: gradient variance vs depth, mean with a shaded mean +- std band.
python scripts/plot_gradient_scaling.py --molecule H2     # newest H2 gradient run
```

Seeded VQE runs are written to `results/<molecule>/vqe_seeds/<run_name>/seed_<seed>/`,
where `run_name` names the settings the seeds share (for example
`adam_depth4_iters1000`). Other analyses go to
`results/<molecule>/<analysis_type>/<timestamp>/`. Every folder has a
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

The four mitigation methods are based on the Qiskit code in
[qubap](https://github.com/jgidi/quantum-barren-plateaus). This project rewrites
each of them in PennyLane, in a more general form, in `src/core/methods`. You do
not need to install qubap. Each method module lists its own references.

| Method | Description |
| --- | --- |
| **Standard** | The unmitigated baseline: EfficientSU2 (Ry, Rz + circular CNOT) optimized directly against the full H. |
| **Local-Global** | Warm-start on a local cost (Cerezo et al.) whose minimum is the Hartree-Fock state, then refine on the full H. |
| **Adiabatic** | Staged anneal `(1-s)*H_0 + s*H_global` from a Hartree-Fock reference Hamiltonian `H_0`, each stage held fixed and warm-started from the previous. |
| **SEA** | Standard VQE with the State Efficient Ansatz (ported from qubap), which requires an even qubit count. |
| **Pretrained** | Two-stage MPS: train the diagonal MPS stage, transfer by prefix zero-pad into the full MPS stage, then refine. |

## Citation

```bibtex
@article{atallah2025investigating,
  title={Investigating Different Barren Plateaus Mitigation Strategies in Variational Quantum Eigensolver},
  author={Atallah, Mostafa and Innan, Nouhaila and Kashif, Muhammad and Shafique, Muhammad},
  journal={arXiv preprint arXiv:2512.11171},
  year={2025}
}
```
