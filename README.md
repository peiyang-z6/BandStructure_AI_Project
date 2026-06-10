# BandStructure AI Project

Physics-aware deep learning pipeline for crystal band-structure analysis,
with a native GUI workbench for human-in-the-loop Plot-to-Physics inference.

## Quick Start

```powershell
conda activate bandstructure_ai_project

# Launch the GUI workbench (human-in-the-loop annotation + Physics Brain)
python scripts\gui_workbench.py

# Run the full pipeline (download → OOD tensors → SSL → finetune)
python scripts\run_full_pipeline.py

# Train vision detector from human annotations
python scripts\train_from_human_annotations.py --epochs 50
```

## Project Layout

```text
BandStructure_AI_Project/
├── PROJECT_BRAIN/              Project memory and constitution. Do not delete.
│   ├── CONSTITUTION.md         Project rules and invariants.
│   ├── dev_context.md          Development history and current state.
│   └── agent_logs/             Session-by-session decision log.
├── configs/
│   └── api_keys.env            Materials Project API keys.
├── data_cache/                 Preserved raw and processed training data.
│   ├── mp_bands.h5             Downloaded MP band structures (HDF5).
│   ├── mp_metadata.json        MP material metadata (36K+ entries).
│   ├── json_cache/             Raw MP API response cache.
│   ├── ood_tensors/            Fixed-shape band tensors + OOD split manifest.
│   └── human_annotations/      GUI human-in-the-loop training labels.
│       └── training_manifest.jsonl
├── src/
│   ├── data/                   ── Core 4-step pipeline ──
│   │   ├── mp_adapter.py       Materials Project new/legacy API adapter.
│   │   ├── batch_download.py   Robust downloader with rate-limiting + resume.
│   │   ├── band_structure_dataset.py  SSL virtual-strain k-path augmentation.
│   │   └── ood_tensor_builder.py      Interpolation, curvature, OOD spacegroup split.
│   ├── engine/
│   │   ├── auto_tuner.py        Optuna objective with physics-score pruning.
│   │   ├── finetune_trainer.py  Fine-tuning strategy freeze + report helpers.
│   │   └── ssl_trainer.py       MBM trainer with adaptive physics-loss weights.
│   ├── models/
│   │   ├── band_structure_encoder.py  Transformer encoder + SSL projection/recon heads.
│   │   ├── losses.py                  Curvature-sign, curvature-consistency, Focal losses.
│   │   └── inverse_generator.py       Phase 5 optional inverse descriptor generator.
│   ├── utils/
│   │   ├── mc_dropout.py        MC Dropout epistemic uncertainty quantification.
│   │   ├── physics_validator.py Physics violation checking + scoring.
│   │   ├── visualizer.py        Band overlay grids, parity plots, t-SNE.
│   │   └── plot_to_tensor.py    Optional Plot-to-Physics image extraction.
│   └── vision/                  ── Phase 5 multimodal vision ──
│       ├── multi_format_parser.py     PDF/raster/video band-plot parser.
│       ├── synthetic_data_generator.py  Synthetic band-plot image generator.
│       ├── physics_reconstructor.py     Human-calibrated → 6D tensor reconstruction.
│       ├── brain_invoker.py            Physics Brain inference + application recommender.
│       └── material_classifier.py      Element/crystal-structure/crystal-type ID.
├── scripts/
│   ├── run_full_pipeline.py     Full automation (download → SSL → finetune → vision).
│   ├── build_ood_tensors.py     OOD tensor construction CLI.
│   ├── train_ssl.py             SSL Masked Band Modeling training.
│   ├── finetune_supervised.py   Supervised gap regression + classification.
│   ├── gui_workbench.py         Native Python GUI (tkinter): human-in-the-loop workbench.
│   ├── train_from_human_annotations.py  GUI annotations → YOLO dataset → fine-tune.
│   ├── train_vision_detector.py YOLOv8 pose detector training.
│   ├── generate_synthetic_vision_data.py Stage 1 synthetic data generation.
│   ├── prepare_pdf_vision_dataset.py    Stage 2 PDF pseudo-label dataset.
│   ├── extract_pdf_images.py    PyMuPDF band-image extraction from papers.
│   ├── demo_vision_pipeline.py  End-to-end vision pipeline demo.
│   ├── human_in_the_loop_finetune.py    Stage 2 few-shot Sim2Real bootstrapper.
│   └── literature_mining_pipeline.py   Stage 3 automated literature mining.
├── models/                     Trained model artifacts (do not delete).
│   ├── ssl_mbm_pretrained.keras
│   ├── ssl_mbm_final_epoch100.keras
│   ├── ssl_mbm_norm_stats.json
│   ├── finetuned_gap_predictor.weights.h5
│   ├── finetuned_gap_predictor_config.json
│   ├── physics_model_brain_manifest.json
│   └── vision_detector/
│       └── band_plot_yolov8_pose_best.pt
├── checkpoints/                Training checkpoints.
├── reports/                    Validation reports, figures, and GUI predictions.
└── tests/
    └── test_data_pipeline.py
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  CORE PIPELINE (4 steps)                                     │
│                                                              │
│  Download ──► OOD Tensors ──► SSL MBM ──► Supervised        │
│  (MP API)     (spacegroup      (physics    (gap+type)       │
│                split)           losses)                      │
└──────────────────────────────────┬──────────────────────────┘
                                   │ 6D Tensor Contract
                                   ▼
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5 — MULTIMODAL INFERENCE                              │
│                                                              │
│  Paper PDF/Image ──► CV Detection ──► Human Annotation      │
│                      (YOLOv8 pose)     (tkinter GUI)         │
│                                              │               │
│                          ┌───────────────────┘               │
│                          ▼                                   │
│                    PhysicsReconstructor                      │
│                          │                                   │
│                          ▼                                   │
│                    PhysicsBrainInvoker                       │
│                    (gap, type, eff. mass)                    │
│                          │                                   │
│                          ▼                                   │
│                    MaterialClassifier                        │
│                    (elements, crystal, type)                 │
└─────────────────────────────────────────────────────────────┘
```

## GUI Workbench (tkinter)

Native Python GUI — no browser, no Gradio, no HTML sanitization issues.

```
┌──────────────────────────────────────────────┐
│ Image: [_____________] [Browse] [Load]       │
├──────────────────────┬───────────────────────┤   ┌─────────────────────┐
│                      │ ▶ Recognize  💾 Save  │   │ [Panel][Fermi][VB]  │
│                      │ Axes: Y1/Y2/X1/X2     │   │ [CB][AutoVB][AutoCB]│
│   Drawing Canvas     │ Gap type indicator     │   │ [VBM][CBM][Erase]   │
│   (zoom/pan/draw)    │ ID: [____] Label:[___] │   │ [X][Y][Undo][Clear] │
│                      │ ─────────────────────  │   │ [−] 100% [+] [Fit]  │
│                      │ [Results|Training Log] │   └─────────────────────┘
│                      │ Output text area       │    ↑ Floating toolbar
└──────────────────────┴───────────────────────┘
```

Features:
- **Drawing tools**: Panel bbox, Fermi line, VB/CB freehand curves, VBM/CBM markers, axis endpoints
- **Auto-trace**: Click dark curves on light backgrounds to auto-trace band paths
- **Eraser**: Click any annotation to remove it
- **Zoom**: Mouse wheel + buttons (5%–1000%), middle-click pan
- **Recognition**: Submit → Physics Brain → gap + type + effective mass + recommendations
- **Material ID**: Elements, crystal structure (spacegroup → system + Bravais lattice), crystal type (metal/semiconductor/insulator)
- **Training**: Save annotations as JSON → `train_from_human_annotations.py` → YOLO fine-tune
- **Results**: Auto pop-out window with large readable text + curvature microscope + t-SNE

## Tensor Contract

```
Input:  (N, 128, 6) = [VBM_E, VBM_curv, VBM_k_dist,
                        CBM_E, CBM_curv, CBM_k_dist]

OOD split: Stratified-by-spacegroup, train/test = 80/20, seed=42
Current: 200 samples, 57 spacegroups, 144 train / 56 test
```

## Current Performance (2026-06-08)

| Metric | Train | OOD Test |
|--------|-------|----------|
| Gap MAE | 0.115 eV | 0.302 eV |
| R² | 0.979 | 0.885 |
| Type Accuracy | 1.000 | 0.982 |
| Direct Gap Recall | 1.000 | 1.000 |
| MC Uncertainty (median) | — | 0.151 eV |

## Non-Negotiables

- Keep `PROJECT_BRAIN/`, `data_cache/`, `models/`, `checkpoints/`, `reports/`.
- Keep `configs/api_keys.env`.
- OOD splitting must remain grouped by `spacegroup_number`.
- Phase 5 modules stay decoupled from the core 4-step pipeline.
