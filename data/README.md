# Data layout

## `raw/`

Downloaded source records only.

- `raw/aflow/aflow_bands.h5` + `json_cache/aflow/`: byte-identical v4 6,443 baseline retained for reproducibility.
- `raw/aflow/snapshots/aflow_30000_20260825/`: immutable v5 snapshot with 30,000 HDF5 groups, 30,000 raw JSON responses, metadata, exclusion/report files, aggregate hashes and a 30,000-entry per-JSON SHA-256 manifest.
- `raw/aflow/provenance/baselines/`: exact server-recovered v4 files used to repair the historical canonical path after expansion.
- `raw/aflow/supplemental/`: 20 unique early-smoke records outside both formal snapshots; intentionally isolated.
- `raw/aflow/provenance/`: legacy reports and content variants retained during semantic de-duplication.
- `raw/materials_project/mp_bands.h5`: 12 unique MP smoke records consolidated without HDF5/JSON conflicts.
- `raw/materials_project/provenance/`: original MP smoke download reports.

## `processed/`

Both scientifically valid immutable derived snapshots are retained:

- `processed/aflow/ood_tensors/`: v4, 6,441 samples, outer split 5,153/1,288, zero space-group overlap.
- `processed/aflow/ood_tensors_v5_30000_seed42/`: v5, 29,952 samples, tensor contract `(N,2,128,3)`, outer split 23,933/6,019, groups 161/45, zero overlap; includes `tensor_snapshot_audit.json` and full source/builder/NPZ hashes.

Do not mix supplemental records or append to either immutable snapshot. A future expansion must create a new raw and processed snapshot, new manifests and a new experiment ID.
