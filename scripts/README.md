# Script entrypoints

The executable scripts remain flat to preserve stable commands and imports. They are classified by responsibility below; implementation modules remain under the existing `src/` packages rather than being duplicated.

## Data

- `build_ood_tensors.py` — raw HDF5 to group-disjoint 6D tensors.
- `extract_pdf_images.py`, `prepare_pdf_vision_dataset.py` — document/vision data preparation.
- `generate_synthetic_vision_data.py` — synthetic detector data.

## Training

- `train_ssl.py` — MBM self-supervised pretraining.
- `finetune_supervised.py` — gap/type supervised fine-tuning and evaluation.
- `train_vision_detector.py`, `train_from_human_annotations.py`, `human_in_the_loop_finetune.py` — vision feedback loop.

## Pipelines and applications

- `run_full_pipeline.py` — source-aware end-to-end pipeline.
- `gui_workbench.py` — Tkinter scientific workbench.
- `demo_vision_pipeline.py`, `literature_mining_pipeline.py` — vision/literature workflows.

## Environment

- `configure_wsl_conda_gpu.sh` — WSL2 GPU environment setup.

Run commands from the project root. Default paths use `data/` and `artifacts/`.
