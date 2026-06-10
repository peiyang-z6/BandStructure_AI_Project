# Agent Log: OOD Tensor Data Pipeline

- Date: 2026-06-04
- Scope: data cleaning, interpolation, curvature feature construction, and OOD tensor split
- Files added:
  - `src/data/ood_tensor_builder.py`
  - `scripts/build_ood_tensors.py`
- Outputs:
  - `data_cache/ood_tensors/band_tensors_full.npz`
  - `data_cache/ood_tensors/band_tensors_ood_split.npz`
  - `data_cache/ood_tensors/ood_split_manifest.json`

## Constitution Boundary

- Did not use random sample-wise `train_test_split`.
- Used `GroupShuffleSplit` with `spacegroup_number` as the group key.
- Verified train/test spacegroup overlap is empty.
- Did not edit `src/data/spacegroup_splitter.py` or change its parameters.

## Pipeline

1. Load raw `data_cache/mp_bands.h5` and `data_cache/mp_metadata.json`.
2. Merge metadata from HDF5 attrs, HDF5 `metadata` subgroup, and JSON metadata.
3. Select a VBM-like and CBM-like band using `efermi` when available; fall back to median-energy band ordering.
4. Resample each selected band to `target_k_points=128` using `CubicSpline`; use linear interpolation for very short sequences.
5. Compute curvature with two `np.gradient` passes.
6. Build tensors with shape `(N, 2, 128, 2)`:
   - band axis: VBM-like, CBM-like
   - channel axis: energy, curvature
7. Save full tensors and GroupShuffleSplit OOD train/test tensors.

## Validation

- Input HDF5 count: 200 materials.
- Output tensor shape: `(200, 2, 128, 2)`.
- Output labels shape: `(200,)`.
- Spacegroups: 57.
- Train/test samples: 141 / 59.
- Train/test group overlap: `[]`.
- Skipped samples: 0.
- Numeric check: all tensor values finite.
