# Agent Log: Dual Materials Project Key Configuration

- Date: 2026-06-04
- Scope: API key configuration and download fallback behavior
- Files changed:
  - `configs/api_keys.env`
  - `src/data/mp_adapter.py`
  - `src/data/batch_download.py`

## Change Summary

- Added separate config slots:
  - `MP_API_KEY_NEW` for the 32-character current Materials Project API key.
  - `MP_API_KEY_LEGACY` for an old legacy key.
  - `MP_API_KEY` remains supported as a backward-compatible default.
- Updated key loading logic:
  - 32-character `MP_API_KEY` is treated as a new API key.
  - non-32-character `MP_API_KEY` is treated as a legacy fallback key.
  - explicit `MP_API_KEY_NEW` and `MP_API_KEY_LEGACY` override the inferred behavior.
- Updated downloader construction to pass both new and legacy keys into `MPAdapter`.
- New API failures in metadata or band-structure fetch paths retry with legacy when a legacy client is available.

## Constitution Boundary

- Did not change spacegroup split logic.
- Did not change OOD split ratios, thresholds, random seed, or grouping parameters.
- Download output still preserves `spacegroup_number` for downstream splitting.

## Validation

- `conda run -n bandstructure_ai_project python -m py_compile src/data/mp_adapter.py src/data/batch_download.py` passed.
- Current config detects a 32-character default key and enables the new API client.
- Since `MP_API_KEY_LEGACY` is currently empty, legacy fallback is configured but inactive until a legacy key is provided.
