# 2026-06-04 Final Review Physics, Optuna, and Visualization Upgrade

## Scope

- Added `src/data/band_structure_dataset.py` with virtual-strain SSL augmentation.
- Integrated virtual strain into `scripts/train_ssl.py` and `scripts/run_full_pipeline.py`.
- Added `src/utils/physics_validator.py` for reusable physical-consistency scoring.
- Added `src/engine/auto_tuner.py` so Optuna trials are pruned when `physics_score < 0.95`.
- Added `src/utils/visualizer.py` and connected the supervised report overlay plot to `kpath_labels`.

## Contract

- Raw Materials Project downloads remain untouched.
- OOD split remains grouped by `spacegroup_number` with `train_size=0.8` and `random_state=42`.
- Gaussian-noise SSL augmentation is not used.
- Publication band overlays include high-symmetry vertical lines, LaTeX tick labels, and red Fermi-level reference.
