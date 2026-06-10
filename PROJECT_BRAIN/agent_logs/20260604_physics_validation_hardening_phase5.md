# 2026-06-04 Physics Validation Hardening and Phase 5 Extensions

## Mandatory Fixes

- Hardened `src/engine/auto_tuner.py` so Optuna never validates against ground-truth tensors as a fallback.
- Supervised models with an SSL encoder now explicitly call `model.encoder.reconstruct(...)` for predicted bands.
- `scripts/finetune_supervised.py` now reports predicted-gap versus tensor-gap large residual rate as a core physical metric.
- `src/data/batch_download.py` catches unexpected per-material fetch exceptions and continues the batch.

## Phase 5 Extensions

- Added `src/utils/plot_to_tensor.py` for optional Plot-to-Physics image extraction.
- Added `src/models/inverse_generator.py` for optional inverse descriptor generation.
- Phase 5 modules are decoupled from the core four-step training pipeline.
