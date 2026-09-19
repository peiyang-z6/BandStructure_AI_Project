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

## MCP-first delivery

- `demo_mcp_client.py` — real stdio SDK exercise, optional real PDF and AI-observation queue.
- `verify_mcp_paper_repair.py` — SHA-pinned ACEAMI paper regression via an isolated real stdio server: seven pages, PDF chunk integrity/restart expiry, composite/ROI routing, sparse v2 export and non-approving audit. Requires `--paper` and a new `--output-dir`; synthetic numeric controls remain explicitly separate from paper observations.
- `generate_mcp_client_configs.py` — generate a new hash-bound Claude/Cursor/VS Code/Codex config bundle without installing it.
- `mcp_human_review_workbench.py` — inspect pending candidates and write immutable non-approving pre-review packets; formal approval remains external.
- `harvest_europe_pmc_band_figures.py` — collect bounded CC-BY electronic-band figure candidates through Europe PMC's official OA REST APIs.
- `expand_oa_figure_dataset.py` — extend a frozen OA corpus into a new directory (`collect`), then inherit existing preannotations/visual batches unchanged and prepare only the added figures (`prepare`); never rewrite the parent snapshot.
- `preannotate_europe_pmc_band_figures.py` — add hash-bound CV panel candidates without creating human labels.
- `freeze_oa_figure_evaluation.py` — freeze the complete evaluation sampling frame before independent labels exist.
- `mcp_figure_candidate_workbench.py` — one-key rapid figure screening (`1/2/3`) with formal approval disabled.
- `mcp_figure_batch_workbench.py` — six original figures per screen; explicit per-figure visual notes and multiple percentage-coordinate boxes, submitted with Ctrl+Return.
- `export_oa_visual_review.py` — revalidate every batch/image binding and export CSV, AI-labelled COCO, dataset card and CV-versus-visual diagnostics.

## Environment

- `configure_wsl_conda_gpu.sh` — WSL2 GPU environment setup.

Run commands from the project root. Default paths use `data/` and `artifacts/`.
