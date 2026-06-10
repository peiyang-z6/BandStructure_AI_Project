"""Stage-2 few-shot human-in-the-loop Sim2Real correction.

The script bootstraps predictions on 10-20 real paper screenshots, stores human
corrections, and optionally launches a Gradio UI. Detector fine-tuning is kept
optional because Gradio/YOLO stacks are not core project dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.multi_format_parser import MultiFormatParser
from src.vision.physics_reconstructor import PhysicsReconstructor


def collect_candidates(image_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parser = MultiFormatParser(image_height_ev=8.0)
    reconstructor = PhysicsReconstructor()
    candidates: List[Dict[str, Any]] = []
    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        try:
            parsed = parser.parse(str(path))
            recon = reconstructor.reconstruct(parsed)
            item = {
                "image": str(path.resolve()),
                "status": "needs_review",
                "auto_vbm_index": recon.vbm_index,
                "auto_cbm_index": recon.cbm_index,
                "auto_tensor_gap_ev": recon.tensor_gap,
                "human_vbm_xy": None,
                "human_cbm_xy": None,
            }
        except Exception as exc:
            item = {"image": str(path.resolve()), "status": "parse_failed", "error": str(exc)}
        candidates.append(item)
    (output_dir / "hitl_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    return candidates


def launch_gradio(candidates_path: Path) -> None:
    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Gradio UI requires optional dependency. Install when needed: pip install gradio"
        ) from exc

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    state = {"index": 0, "items": candidates}

    def load_current():
        item = state["items"][state["index"]]
        return item["image"], json.dumps(item, indent=2)

    def save_points(vbm_x, vbm_y, cbm_x, cbm_y):
        item = state["items"][state["index"]]
        item["human_vbm_xy"] = [float(vbm_x), float(vbm_y)]
        item["human_cbm_xy"] = [float(cbm_x), float(cbm_y)]
        item["status"] = "corrected"
        candidates_path.write_text(json.dumps(state["items"], indent=2), encoding="utf-8")
        return json.dumps(item, indent=2)

    def next_item():
        state["index"] = min(state["index"] + 1, len(state["items"]) - 1)
        return load_current()

    with gr.Blocks() as demo:
        gr.Markdown("# Few-shot Band Plot Correction")
        image = gr.Image(type="filepath")
        meta = gr.Textbox(lines=12)
        with gr.Row():
            vbm_x = gr.Number(label="VBM x")
            vbm_y = gr.Number(label="VBM y")
            cbm_x = gr.Number(label="CBM x")
            cbm_y = gr.Number(label="CBM y")
        save = gr.Button("Save correction")
        nxt = gr.Button("Next")
        save.click(save_points, [vbm_x, vbm_y, cbm_x, cbm_y], meta)
        nxt.click(next_item, outputs=[image, meta])
        demo.load(load_current, outputs=[image, meta])
    demo.launch()


def write_finetune_stub(output_dir: Path) -> None:
    text = """# Few-shot Detector Fine-tuning

Use `hitl_candidates.json` as the real-paper correction source. Convert corrected
points to your detector's preferred format, then fine-tune the Stage-1 YOLO/RTMDet
checkpoint for a small number of epochs with a low learning rate.

Suggested optional stack:

```powershell
pip install ultralytics gradio
```

The core project keeps these dependencies optional so the physics pipeline remains
lightweight and reproducible.
"""
    (output_dir / "few_shot_finetune_instructions.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human-in-the-loop few-shot correction for real paper band plots")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="data_cache/vision_fewshot")
    parser.add_argument("--launch-ui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    candidates = collect_candidates(Path(args.image_dir), output_dir)
    write_finetune_stub(output_dir)
    print(json.dumps({"candidates": len(candidates), "output_dir": str(output_dir)}, indent=2))
    if args.launch_ui:
        launch_gradio(output_dir / "hitl_candidates.json")


if __name__ == "__main__":
    main()
