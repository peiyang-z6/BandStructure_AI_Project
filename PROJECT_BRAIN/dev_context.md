# BandStructure AI Project Dev Context

Last updated: 2026-06-10

## Current State (v5.1 — Native tkinter GUI)

**GUI**: Native Python tkinter application (`scripts/gui_workbench.py`). No browser.
Floating toolbar, zoom/pan, drawing tools, auto-trace, eraser. Always-visible
Recognize + Save buttons with axis calibration. Pop-out results window.

**MaterialClassifier**: `src/vision/material_classifier.py`. MP metadata lookup
(36K entries) for elements + crystal structure. Physics inference for crystal type
(metal/semiconductor/insulator). Integrated into GUI recognition results.

**Training Pipeline**: `scripts/train_from_human_annotations.py`. GUI annotations
→ YOLOv8 pose dataset → Sim2Real fine-tune. `--dry-run` for preview.

**Core Pipeline**: 200 samples, 144/56 train/test, 57 spacegroups, 0 overlap.
Best OOD metrics: gap MAE 0.302 eV, type acc 98.2%, direct gap recall 100%.

## Gradio History (archived — all superseded)

v3–v5.0: 6 failed Gradio fixes for canvas image display. Root cause: Gradio 6.0
HTML sanitization + JS selector bugs. Superseded by complete tkinter rewrite.

## Current Architecture

```
src/data/    — mp_adapter, batch_download, band_structure_dataset, ood_tensor_builder
src/engine/  — auto_tuner, finetune_trainer, ssl_trainer
src/models/  — band_structure_encoder, losses, inverse_generator
src/utils/   — mc_dropout, physics_validator, visualizer, plot_to_tensor
src/vision/  — multi_format_parser, synthetic_data_generator, physics_reconstructor,
               brain_invoker, material_classifier

scripts/
  gui_workbench.py (1224 lines, tkinter)
  train_from_human_annotations.py (340 lines)
  run_full_pipeline.py, build_ood_tensors.py, train_ssl.py, finetune_supervised.py
```

## Tensor Contract

```
X: (N, 2, 128, 3) → flatten → (N, 128, 6)
  [VBM_E, VBM_curv, VBM_k_dist, CBM_E, CBM_curv, CBM_k_dist]
OOD: Stratified group by spacegroup, 80/20, seed=42
```

## Model Brain

- SSL encoder: `models/ssl_mbm_pretrained.keras`
- Normalization: `models/ssl_mbm_norm_stats.json`
- Fine-tuned: `models/finetuned_gap_predictor.weights.h5`
- Vision detector: `models/vision_detector/band_plot_yolov8_pose_best.pt`

## Key Metrics (2026-06-08)

| Metric | Train | OOD Test |
|--------|-------|----------|
| Gap MAE | 0.115 eV | 0.302 eV |
| R² | 0.979 | 0.885 |
| Type Accuracy | 1.000 | 0.982 |
| Direct Gap Recall | 1.000 | 1.000 |

## Optimization Changelog (archived highlights)

- 2026-06-08: Per-layer FFN, cosine LR warmup, MC Dropout, local polyfit curvature,
  stratified group OOD split, virtual strain augmentation
- 2026-06-05: Probability sharpening, extremum gap head, topology injection
- 2026-06-04: Line-mode gap rescue, SSL MBM full run, supervised fine-tuning

## Phase 5 Literature Mining Strategy

3-stage Sim2Real:
1. Synthetic pretraining (600 images, mAP50=0.995)
2. PDF Sim2Real (108 pages, box mAP50 0.443→0.487)
3. Human-in-the-loop GUI annotations → `train_from_human_annotations.py`

## Protected Assets

- `PROJECT_BRAIN/`, `data_cache/`, `models/`, `checkpoints/`, `reports/`
- `configs/api_keys.env`
