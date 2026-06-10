# BandStructure AI Project Constitution

Version: 4.0
Updated: 2026-06-10
Status: active project rules

## 1. Mission

This project builds a physics-aware AI pipeline for crystal band-structure data,
with a native GUI workbench for human-in-the-loop Plot-to-Physics inference.

Core 4-step pipeline:
1. Download band structures from Materials Project.
2. Convert variable-length band data into fixed OOD tensors.
3. Pretrain an encoder with Masked Band Modeling (physics losses).
4. Fine-tune for band-gap regression and direct/indirect/metal classification.

Phase 5 multimodal extensions:
5. Vision: CV detection (YOLOv8), human annotation (tkinter GUI), 6D reconstruction.
6. Inference: Physics Brain gap/type/effective-mass prediction.
7. Classification: Element, crystal-structure, and crystal-type identification.
8. Training: Human annotations → YOLO dataset → Sim2Real fine-tuning.

## 2. Protected Assets

Never delete these unless the user explicitly asks for that exact operation:

- `PROJECT_BRAIN/` and `PROJECT_BRAIN/CONSTITUTION.md`
- `configs/api_keys.env`
- `data_cache/` — all raw and processed data
- `models/` — all trained model artifacts
- `checkpoints/` — all training checkpoints
- `reports/` — all validation reports and figures

## 3. Data Leakage Rules

- OOD splitting must be grouped by `spacegroup_number`.
- Do not replace the OOD split with sample-wise random `train_test_split`.
- Train and test spacegroups must not overlap.
- Current split parameters are part of the experimental contract:
  - `train_size=0.8`, `random_state=42`, `group_key=spacegroup_number`

## 4. Source Structure

```text
src/
  data/       — Core pipeline data processing
  engine/     — Training loops and hyperparameter tuning
  models/     — Transformer encoder, SSL heads, physics losses
  utils/      — MC Dropout, physics validation, visualization
  vision/     — Phase 5: CV parser, physics reconstructor, brain invoker, classifier

scripts/
  gui_workbench.py                 — Native tkinter GUI (human-in-the-loop workbench)
  train_from_human_annotations.py  — Human labels → YOLO dataset → fine-tune
  run_full_pipeline.py             — Full automation script
  build_ood_tensors.py / train_ssl.py / finetune_supervised.py

tests/
  test_data_pipeline.py
```

## 5. GUI Rules

- The GUI is a **native Python tkinter application** (`scripts/gui_workbench.py`).
- Do NOT reintroduce Gradio/web-based GUI approaches.
- All drawing annotations are stored in **image pixel coordinates** (zoom/pan invariant).
- Floating toolbar must always be on top and not obscure the canvas.
- Recognition results must pop out in a separate large window.
- Training annotations must include `source_image_path`, `canvas_width`, `canvas_height`.

## 6. Tensor Contract

```
X: (N, 2, seq_len, 3)
  band axis:    0=VBM-like, 1=CBM-like
  channel axis: 0=energy, 1=curvature, 2=extremum k-distance

Model input (flattened):
  (N, seq_len, 6) = [VBM_E, VBM_curv, VBM_k_dist, CBM_E, CBM_curv, CBM_k_dist]
```

## 7. Training Rules (unchanged from v3.0)

SSL pretraining: MBM with virtual-strain augmentation, mask ratio 15-30%,
curvature-sign + curvature-consistency losses, TensorBoard logging.

Optuna tuning: PhysicsValidator with `physics_score >= 0.95` pruning.

Supervised fine-tuning: freeze early layers, layered LRs (head 1e-3, encoder 1e-5),
extremum expected-value gap head (learned temperature τ ∈ [0.01, 0.5]),
topology-injected type head, auxiliary losses (topology, entropy, extremum).

## 8. Phase 5 Rules

- Phase 5 modules (`src/vision/`) must stay decoupled from the core pipeline.
- Physics Reconstructor accepts ONLY human-calibrated GUI inputs for real figures.
- MaterialClassifier queries MP metadata cache first, falls back to physics inference.
- Vision detector training uses synthetic data (Stage 1) then human labels (Stage 2).
- Human annotations are exported via `data_cache/human_annotations/training_manifest.jsonl`.

## 9. Documentation Rules

Any structural change must update:
- `README.md`
- `PROJECT_BRAIN/dev_context.md`
- `PROJECT_BRAIN/CONSTITUTION.md` (this file)
- a dated file under `PROJECT_BRAIN/agent_logs/`
