"""Bond-length sweep run for the paper Figure 6(b) per method.

For each method, optimize once at every cached LiH geometry and store the minimum
parameter vector plus its energy and bond length. Saves everything to an npz so the
figure build does not need to re-run the optimizations.
"""

import glob
import os
import re
import sys

sys.path.insert(0, r"f:\QML-Barren Plateaus\code\BarrenPlateaus-VQE")

import numpy as np

from src.core.backend import load_hamiltonian
from src.core.methods import MethodConfig, run_method

DATA = r"f:\QML-Barren Plateaus\code\BarrenPlateaus-VQE\data"
METHODS = ["standard", "local_global", "adiabatic", "sea", "pretrained"]
DEPTH, ITERS, WARM = 4, 150, 60

# discover the r-labeled geometries we just cached
labels = []
for f in glob.glob(os.path.join(DATA, "LiH_r*.json")):
    m = re.search(r"LiH_(r[0-9.]+)\.json$", os.path.basename(f))
    if m:
        labels.append(m.group(1))
systems = {lab: load_hamiltonian("LiH", lab) for lab in labels}
labels = sorted(labels, key=lambda lab: systems[lab].bondlength)
rs = np.array([systems[lab].bondlength for lab in labels], float)
fcis = np.array([systems[lab].fci_energy for lab in labels], float)
print(f"{len(labels)} geometries, r in [{rs.min():.2f}, {rs.max():.2f}]", flush=True)

out = {"r": rs, "fci": fcis}
for m in METHODS:
    P = []  # param vectors ordered by r
    E = []  # final energies
    for lab in labels:
        s = systems[lab]
        cfg = MethodConfig(
            depth=DEPTH,
            max_iters=ITERS,
            warm_iters=WARM,
            seed=0,
            optimizer="adam",
            optimizer_kwargs={"stepsize": 0.05},
        )
        res = run_method(m, s, cfg)
        g = res.energy_history_global or res.energy_history
        e = float(g[-1])
        P.append(np.asarray(res.params, float))
        E.append(e)
        print(f"{m:14s} r={s.bondlength:.2f}  E={e:.4f}  err={e - s.fci_energy:+.4f}", flush=True)
    out[f"P_{m}"] = np.array(P)
    out[f"E_{m}"] = np.array(E)

dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sweep_LiH.npz")
np.savez(dst, **out)
print("wrote", dst)
print("DONE_SWEEP")
