# LiH minima analysis

Minima structure of the five VQE methods on LiH: how many minima each method lands
in, and how the minimum moves through parameter space as the bond length sweeps.
See results_summary.md in the repo root for the full reasoning trail.

## Data

- per_method_census.npz: minima census at equilibrium. 20 random inits per method,
  each run to its minimum. Arrays: fci (scalar), and E_<method> (20 final energies)
  and P_<method> (20 parameter vectors) for method in standard, local_global,
  adiabatic, sea, pretrained. Cluster E_<method> within 0.02 Ha to count distinct
  minima (gauge invariant, because it clusters on energy not parameters).
- sweep_LiH.npz: bond-length sweep. One optimization per method per geometry over 20
  LiH bond lengths (r = 0.90 to 2.10 Angstrom). Arrays: r (20 bond lengths), fci (20
  FCI energies), E_<method> (20 final energies), P_<method> (20 parameter vectors).

## Figure

- fig6b_path.png: paper Figure 6(b) Position of Minima, per method. For each ansatz
  group, PCA is fit on that group's per r minima and the minimum is drawn as a path
  through PCA space colored by bond length (blue compressed to red stretched), axes
  labeled by variance captured, red square at r = 1.56.

## Scripts (in scripts/)

Reproduce from the repo. Paths resolve relative to this folder, so the npz and png
land back here.

- per_method_census.py: runs the census, writes per_method_census.npz.
- gen_lih_sweep.py: caches 20 evenly spaced LiH geometries into data/ (needs
  PennyLane datasets). Run before sweep_run.py.
- sweep_run.py: runs the bond-length sweep, writes sweep_LiH.npz.
- fig6b_path.py: builds fig6b_path.png from sweep_LiH.npz.
- fig6b_path_smoke.py: 3 geometry smoke test of the path structure (validation only).

## Caveat

Raw parameter PCA position is gauge redundant (EfficientSU2 has redundant params),
so read color (bond length or energy) in any parameter space plot, not raw spatial
spread. Counting minima is the census's job, not the scans'.
