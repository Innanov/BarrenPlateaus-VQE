"""Paper Figure 6(b) 'Position of Minima', per method, SMOKE TEST on 3 cached r.

Faithful to the paper structure this time:
  - x axis of the physics is BOND LENGTH r, not random seed.
  - at each r we run the method once to its minimum and store the parameter vector.
  - PCA is fit on the collection of per-r minima (per ansatz group, dims must match).
  - each method is drawn as a PATH through PCA space, dots ordered by r and joined,
    colored by r (blue compressed to red stretched), like the paper's connected curve.
  - axes are labeled with the variance each PCA component captures.

Only 3 cached geometries here (r = 0.90, 0.93, 2.10), so each path is a 3 node
polyline. This is a smoke test to confirm the structure, the r coloring, and the
variance labels before committing to a dense r sweep.
"""

import os
import sys

sys.path.insert(0, r"f:\QML-Barren Plateaus\code\BarrenPlateaus-VQE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from orqviz.pca import get_pca

from src.core.backend import load_hamiltonian
from src.core.methods import MethodConfig, run_method

GEOMS = ["compressed", "equilibrium", "stretched"]
METHODS = ["standard", "local_global", "adiabatic", "sea", "pretrained"]
LABEL = {
    "standard": "standard",
    "local_global": "local-global",
    "adiabatic": "adiabatic",
    "sea": "SEA",
    "pretrained": "pretrained (MPS)",
}
DEPTH, ITERS, WARM = 4, 150, 60

# collect minima: minima[method] = list of (r, param_vector) ordered by r
systems = {g: load_hamiltonian("LiH", g) for g in GEOMS}
rs = np.array([systems[g].bondlength for g in GEOMS], float)
order = np.argsort(rs)

minima = {m: [] for m in METHODS}
for m in METHODS:
    for g in GEOMS:
        s = systems[g]
        cfg = MethodConfig(
            depth=DEPTH,
            max_iters=ITERS,
            warm_iters=WARM,
            seed=0,
            optimizer="adam",
            optimizer_kwargs={"stepsize": 0.05},
        )
        r = run_method(m, s, cfg)
        minima[m].append((s.bondlength, np.asarray(r.params, float)))
        print(f"{m:14s} r={s.bondlength:.2f} done", flush=True)

# group by ansatz (shared parameter space needed for a common PCA)
panels = [
    ("EfficientSU2 ansatz", ["standard", "local_global", "adiabatic"]),
    ("SEA ansatz", ["sea"]),
    ("MPS ansatz", ["pretrained"]),
]

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
cmap = plt.cm.coolwarm
rmin, rmax = rs.min(), rs.max()


def rnorm(r):
    return (r - rmin) / (rmax - rmin + 1e-12)


for ax, (title, methods) in zip(axes, panels, strict=False):
    # pool every method's minima in this ansatz group, fit one PCA
    pooled = np.vstack([np.vstack([p for _, p in minima[m]]) for m in methods])
    pca = get_pca(pooled)
    # orqviz wraps the sklearn PCA at pca.pca; the variance ratio lives there
    var = getattr(getattr(pca, "pca", None), "explained_variance_ratio_", None)
    for m in methods:
        seq = sorted(minima[m], key=lambda t: t[0])
        R = np.array([r for r, _ in seq])
        P = np.vstack([p for _, p in seq])
        xy = pca.get_transformed_points(P)
        # path colored by r
        pts = xy.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(rmin, rmax))
        lc.set_array((R[:-1] + R[1:]) / 2)
        lc.set_linewidth(2.0)
        ax.add_collection(lc)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=R,
            cmap=cmap,
            vmin=rmin,
            vmax=rmax,
            s=70,
            edgecolors="k",
            linewidths=0.5,
            zorder=3,
        )
        # label the method at its stretched-r end
        ax.annotate(LABEL[m], xy=xy[-1], fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    if var is not None and len(var) >= 2:
        ax.set_xlabel(f"axis contains {var[0]*100:.1f}% of variance")
        ax.set_ylabel(f"axis contains {var[1]*100:.1f}% of variance")
    else:
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
    ax.autoscale_view()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(rmin, rmax))
cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
cb.set_label("bond length r (Angstrom)")

fig.suptitle(
    "LiH Position of Minima per method: path of the minimum through PCA space "
    "as bond length sweeps (smoke test, 3 cached r)",
    fontsize=12,
    y=0.99,
)
out = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fig6b_path_smoke.png"
)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print("DONE_PATH_SMOKE")
