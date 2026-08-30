"""Cache ~20 evenly spaced LiH geometries for the bond-length sweep figure.

Uses the repo's own fetch_molecule so the JSON schema matches the loader exactly.
Picks 20 bond lengths evenly across PennyLane's 42 available LiH values and saves
each as data/LiH_r<bond>.json (a geometry_label unique per bond length).
"""

import sys

sys.path.insert(0, r"f:\QML-Barren Plateaus\code\BarrenPlateaus-VQE")

import numpy as np
import pennylane as qml

sys.path.insert(0, r"f:\QML-Barren Plateaus\code\BarrenPlateaus-VQE\scripts")
from fetch_hamiltonians import fetch_molecule

BASIS = "STO-3G"
OUT = r"f:\QML-Barren Plateaus\code\BarrenPlateaus-VQE\data"
N_PICK = 20

avail = qml.data.list_datasets()["qchem"]["LiH"][BASIS]  # list of str bond lengths
avail_f = np.array([float(b) for b in avail])
order = np.argsort(avail_f)
avail = [avail[i] for i in order]
avail_f = avail_f[order]

# pick N_PICK evenly spaced by index across the sorted list
idx = np.unique(np.linspace(0, len(avail) - 1, N_PICK).round().astype(int))
picks = [avail[i] for i in idx]
print(f"picking {len(picks)} of {len(avail)} bond lengths: {picks}", flush=True)

ok = 0
for b in picks:
    label = f"r{b}"  # e.g. r0.9, r1.23
    good = fetch_molecule("LiH", b, BASIS, OUT, label, verbose=True, qml=qml)
    ok += int(bool(good))
print(f"cached {ok}/{len(picks)} geometries")
print("DONE_GEN")
