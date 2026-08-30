"""Per-method minima census: run each method many times with different seeds,
collect the final energies, and see whether the method is stuck at one minimum
(structured init) or scatters across many (random init).

Uses each method's REAL init + optimization via run_method, so the distribution
reflects that method's own behavior, not a generic descent.
"""

import os
import sys

sys.path.insert(0, r"f:\QML-Barren Plateaus\code\BarrenPlateaus-VQE")

import numpy as np

from src.core.backend import load_hamiltonian
from src.core.methods import MethodConfig, run_method

MOL = "LiH"
N_SEEDS = 20
ITERS = 150
WARM = 60
DEPTH = 4

s = load_hamiltonian(MOL, "equilibrium")
fci = s.fci_energy
methods = ["standard", "local_global", "adiabatic", "sea", "pretrained"]

results = {m: [] for m in methods}
params = {m: [] for m in methods}
for m in methods:
    for seed in range(N_SEEDS):
        cfg = MethodConfig(
            depth=DEPTH,
            max_iters=ITERS,
            warm_iters=WARM,
            seed=seed,
            optimizer="adam",
            optimizer_kwargs={"stepsize": 0.05},
        )
        r = run_method(m, s, cfg)
        g = r.energy_history_global or r.energy_history
        e = float(g[-1])
        results[m].append(e)
        params[m].append(np.asarray(r.params, float))
        print(f"{m:14s} seed {seed:2d}  E={e:.4f}  err={e-fci:+.4f}", flush=True)

out = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "per_method_census.npz"
)
save = {"fci": fci}
for m in methods:
    save[f"E_{m}"] = np.array(results[m])
    save[f"P_{m}"] = np.array(params[m])
np.savez(out, **save)

print("---")
print(f"{MOL} per-method census, {N_SEEDS} seeds each, FCI={fci:.4f}")
for m in methods:
    E = np.array(results[m])
    spread = E.max() - E.min()
    # count distinct minima within 0.02 Ha
    order = np.sort(E)
    nmin = 1
    for i in range(1, len(order)):
        if order[i] - order[i - 1] > 0.02:
            nmin += 1
    verdict = "STUCK at 1 minimum" if nmin == 1 else f"scatters over {nmin} minima"
    print(f"  {m:14s} best={E.min():.4f} worst={E.max():.4f} spread={spread:.3f}  -> {verdict}")
print("DONE_PER_METHOD")
