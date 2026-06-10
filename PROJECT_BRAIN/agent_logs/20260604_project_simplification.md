# 2026-06-04 Project Simplification

## Goal

Simplify the project around the active Materials Project -> OOD tensor -> SSL MBM -> supervised validation workflow, while preserving:

- `PROJECT_BRAIN/`
- `PROJECT_BRAIN/CONSTITUTION.md`
- current training data in `data_cache/`
- current training outputs in `models/`, `checkpoints/`, `logs/ssl_mbm`, and `reports/`

## Removed

Historical or inactive code and outputs were removed:

- legacy engine package: `src/engine/`
- old utility package: `src/utils/`
- old mock/legacy data modules:
  - `band_structure_dataset.py`
  - `base_adapter.py`
  - `data_checker.py`
  - `mock_generator.py`
  - `monitor_download.py`
  - `spacegroup_splitter.py`
- inactive model modules:
  - `losses.py`
  - `predictor.py`
  - `vision_net.py`
  - `generator.py`
- old scripts:
  - `scripts/finetune_gap.py`
  - `scripts/main_pipeline.py`
- old outputs:
  - `checkpoints/ssl/`
  - `checkpoints/finetune_test/`
  - legacy TensorBoard log folders
  - `pipeline_output/`
  - `reports/training_report.md`
  - `models/ssl_pretrained.keras`
- Python cache folders.

## Kept

Active workflow code:

- `src/data/mp_adapter.py`
- `src/data/batch_download.py`
- `src/data/ood_tensor_builder.py`
- `src/models/band_structure_encoder.py`
- `scripts/build_ood_tensors.py`
- `scripts/train_ssl.py`
- `scripts/finetune_supervised.py`

Protected artifacts:

- `data_cache/`
- `models/ssl_mbm_pretrained.keras`
- `models/ssl_mbm_final_epoch100.keras`
- `models/ssl_mbm_norm_stats.json`
- `models/finetuned_gap_predictor.weights.h5`
- `models/finetuned_gap_predictor_config.json`
- `checkpoints/ssl_mbm/`
- `checkpoints/finetune_supervised/`
- `reports/finetune_supervised/`
- `reports/ssl_mbm_pretraining_20260604.md`
- `reports/finetune_supervised_report_20260604.md`

## Updated

- `scripts/train_ssl.py` now only supports the active tensor MBM path.
- `src/data/__init__.py` and `src/models/__init__.py` now export only active modules.
- `README.md` rewritten with the simplified structure and commands.
- `requirements.txt` reduced to active dependencies.
- `PROJECT_BRAIN/CONSTITUTION.md` rewritten as a clean current-structure constitution.

## Verification

- Active scripts and modules pass `py_compile`.
- `scripts/train_ssl.py --help` works.
- `scripts/finetune_supervised.py --help` works.
- `python -m src.data.batch_download --help` works.
- No active source/script references remain to removed modules.
