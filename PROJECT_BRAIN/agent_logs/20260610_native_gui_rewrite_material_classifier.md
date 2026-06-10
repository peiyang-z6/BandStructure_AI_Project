# 2026-06-10 — GUI Complete Rewrite + Material Classification + Training Pipeline

## Context

After 6 failed Gradio fixes (v3→v4.3→v5.0), the GUI was completely rewritten
as a native Python tkinter application. All Gradio-related code removed.
Three major feature additions followed: floating toolbar, zoom/pan, auto-trace,
eraser, material classifier, and training pipeline.

## Changes

### GUI — Native tkinter rewrite (`scripts/gui_workbench.py`)

- Replaced Gradio web GUI with native tkinter application (~1200 lines)
- Zero browser dependency: no HTML sanitization, no JS injection, no DOM issues
- Architecture: PanedWindow (canvas | right panel), always-visible action bar
- Floating tool palette: always-on-top, draggable, independent of canvas
- Zoom: mouse wheel (5%–1000%), +/-/Fit/1:1 buttons, middle-click pan
- Drawing tools: Panel, Fermi, VB, CB, VBM, CBM, X-axis, Y-axis
- Auto-trace: click dark curves on light bg → bidirectional path tracing
- Eraser: click any annotation to delete it
- All annotations stored in image pixel coordinates (zoom-invariant)

### MaterialClassifier (`src/vision/material_classifier.py`)

- Elements: queries MP metadata (36K entries) by material_id
- Crystal structure: 230 spacegroups → system + Bravais lattice + point group
- Crystal type: gap + effective mass → metal/semiconductor/insulator + bonding
- Fallback: parses chemical formulas from material_id (e.g., "CsPbI3")
- Integrated into GUI recognition results pop-out window

### Training Pipeline (`scripts/train_from_human_annotations.py`)

- Reads GUI annotation JSONs → YOLOv8 pose dataset → fine-tunes detector
- Annotation records include source_image_path, canvas dimensions
- Supports dry-run, custom epochs/batch/LR, validation split

### Documentation

- README.md: complete rewrite with current architecture and GUI docs
- CONSTITUTION.md v4.0: added Phase 5, GUI, and training rules
- requirements.txt: added PyMuPDF
- dev_context.md: consolidated all history

## Files Modified
- `scripts/gui_workbench.py` — complete rewrite (Gradio → tkinter)
- `src/vision/material_classifier.py` — new (520 lines)
- `scripts/train_from_human_annotations.py` — new (340 lines)
- `README.md` — complete rewrite
- `PROJECT_BRAIN/CONSTITUTION.md` — v3.0 → v4.0
- `PROJECT_BRAIN/dev_context.md` — updated with v5.1 history
- `requirements.txt` — added PyMuPDF

## Key Design Decisions

1. tkinter over Gradio: Eliminated browser/HTML/JS dependency. Image display
   is native PIL→PhotoImage, mouse events are direct Python callbacks.
2. Image-coordinate annotations: Zoom/pan never corrupts saved positions.
3. Floating toolbar: Doesn't compete for canvas space.
4. MP metadata as primary source for element ID: 36K entries cover most materials.
5. Physics-only fallback for crystal type: Always available even without MP data.
