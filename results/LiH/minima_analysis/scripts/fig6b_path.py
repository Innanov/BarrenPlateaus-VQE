"""Paper Figure 6(b) 'Position of Minima', per method, from the dense r sweep.

Reads sweep_LiH.npz (one minimum per method per bond length) and draws, for each
ansatz group, the path of the minimum through PCA space as bond length sweeps:
  - dots ordered by r and joined into a curve, colored blue (compressed) to red
    (stretched), exactly the paper's connected-curve format,
  - PCA fit on that ansatz group's pooled minima, axes labeled by variance captured,
  - a red square at the reference bond length r = 1.56 Angstrom (the paper marks
    0.86 and 1.56; only 1.56 is in the LiH range here).
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

SC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = np.load(os.path.join(SC, "sweep_LiH.npz"))
rs = d["r"]
rmin, rmax = float(rs.min()), float(rs.max())

LABEL = {
    "standard": "standard",
    "local_global": "local-global",
    "adiabatic": "adiabatic",
    "sea": "SEA",
    "pretrained": "pretrained (MPS)",
}
panels = [
    ("EfficientSU2 ansatz", ["standard", "local_global", "adiabatic"]),
    ("SEA ansatz", ["sea"]),
    ("MPS ansatz", ["pretrained"]),
]
REF_R = 1.56  # paper reference bond length that falls inside the LiH range
EQ_R = 1.5  # near the LiH equilibrium minimum of the dissociation curve

# where to place each method label so they do not collide in the shared panel
LABEL_OFFSET = {
    "standard": (6, 2),
    "local_global": (6, 8),
    "adiabatic": (6, -12),
    "sea": (6, 2),
    "pretrained": (6, 2),
}

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
cmap = plt.cm.coolwarm
norm = plt.Normalize(rmin, rmax)


def draw_path(ax, xy, R):
    pts = xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm)
    lc.set_array((R[:-1] + R[1:]) / 2)
    lc.set_linewidth(2.2)
    ax.add_collection(lc)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=R,
        cmap=cmap,
        norm=norm,
        s=48,
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
    )
    # red square at the reference bond length
    j = int(np.argmin(np.abs(R - REF_R)))
    ax.scatter(
        xy[j, 0],
        xy[j, 1],
        marker="s",
        s=90,
        facecolors="none",
        edgecolors="red",
        linewidths=1.6,
        zorder=4,
    )


for ax, (title, methods) in zip(axes, panels, strict=False):
    pooled = np.vstack([d[f"P_{m}"] for m in methods])
    pca = get_pca(pooled)
    var = getattr(getattr(pca, "pca", None), "explained_variance_ratio_", None)
    for m in methods:
        P = np.asarray(d[f"P_{m}"], float)
        xy = pca.get_transformed_points(P)
        draw_path(ax, xy, rs)
        ax.annotate(
            LABEL[m],
            xy=xy[-1],
            fontsize=8,
            xytext=LABEL_OFFSET[m],
            textcoords="offset points",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )
    ax.set_title(title)
    if var is not None and len(var) >= 2:
        ax.set_xlabel(f"axis contains {var[0] * 100:.1f}% of variance")
        ax.set_ylabel(f"axis contains {var[1] * 100:.1f}% of variance")
    else:
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
    ax.autoscale_view()

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
cb.set_label("bond length r (Angstrom)")

fig.suptitle(
    "LiH Position of Minima per method: path of the minimum through PCA space "
    "as bond length sweeps (red square marks r = 1.56)",
    fontsize=12,
    y=0.99,
)
out = os.path.join(SC, "fig6b_path.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
print("DONE_PATH")
