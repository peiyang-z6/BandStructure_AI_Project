"""Rapidly screen the 300 Europe PMC figure candidates without granting approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.figure_review_queue import (
    load_figure_review_queue, make_screening_packet, save_screening_packet)


class FigureCandidateWorkbench:
    def __init__(self, queue: dict, output_dir: Path, operator_mode: str,
                 start_index: int = 0, auto_advance: bool = True) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.records = queue["records"]
        self.output_dir = output_dir
        self.operator_mode = operator_mode
        self.auto_advance = auto_advance
        self.index = max(0, min(start_index, len(self.records) - 1))
        self.photo = None

        root = tk.Tk()
        self.root = root
        root.title("BandStructure MCP — 300 OA Figure Candidates")
        root.geometry("1500x920")
        root.minsize(1150, 720)
        ttk.Label(
            root,
            text=("OA FIGURE SCREENING — AI ASSISTED; FORMAL HUMAN APPROVAL BLOCKED "
                  "(external authenticated registry required)"),
            foreground="#9b1c1c", font=("Segoe UI", 12, "bold"),
        ).pack(fill="x", padx=12, pady=(10, 6))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=12, pady=6)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self.image_label = ttk.Label(body)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        panel = ttk.Frame(body)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        self.counter = ttk.Label(panel, font=("Segoe UI", 11, "bold"))
        self.counter.grid(row=0, column=0, sticky="w")
        self.summary = tk.Text(panel, wrap="word", state="disabled", font=("Consolas", 9))
        self.summary.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        ttk.Label(panel, text="Screening notes (not human approval):").grid(row=2, column=0, sticky="w")
        self.notes = tk.Text(panel, height=5, wrap="word", font=("Segoe UI", 10))
        self.notes.grid(row=3, column=0, sticky="ew", pady=(4, 8))

        buttons = ttk.Frame(panel)
        buttons.grid(row=4, column=0, sticky="ew")
        ttk.Button(buttons, text="Previous", command=self.previous).pack(side="left")
        ttk.Button(buttons, text="Next", command=self.next).pack(side="left", padx=5)
        ttk.Button(buttons, text="AI screen positive",
                   command=lambda: self.record("screen_positive")).pack(side="left", padx=5)
        ttk.Button(buttons, text="Needs panel box",
                   command=lambda: self.record("needs_panel_box")).pack(side="left", padx=5)
        ttk.Button(buttons, text="Exclude non-target",
                   command=lambda: self.record("exclude_non_target")).pack(side="left")
        ttk.Button(panel, text="Formal Approve (external registry only)",
                   state="disabled").grid(row=5, column=0, sticky="ew", pady=(10, 4))
        self.status = ttk.Label(panel, text="No screening packet written.")
        self.status.grid(row=6, column=0, sticky="w")
        root.bind("<Left>", lambda _event: self.previous())
        root.bind("<Right>", lambda _event: self.next())
        root.bind("1", lambda _event: self.record("screen_positive"))
        root.bind("2", lambda _event: self.record("needs_panel_box"))
        root.bind("3", lambda _event: self.record("exclude_non_target"))
        self.show_record(self.index)

    def show_record(self, index: int) -> None:
        from PIL import Image, ImageDraw, ImageTk

        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
        bbox = record["preannotation"].get("panel_bbox_xyxy")
        if bbox:
            ImageDraw.Draw(image).rectangle(tuple(bbox), outline="#ef4444",
                                            width=max(3, image.width // 250))
        image.thumbnail((850, 800), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        source = record["source"]
        pre = record["preannotation"]
        lines = [
            f"Candidate {index + 1} / {len(self.records)}",
            f"ID: {record['candidate_id']}",
            f"PMCID/group: {record['group_id']}",
            f"Figure: {source.get('figure_label')}",
            f"License: {source.get('license')}",
            f"Year: {source.get('publication_year')}",
            f"CV status: {pre.get('status')}",
            f"Panel detection: {pre.get('panel_detection')}",
            f"Skeleton points: {pre.get('skeleton_point_count')}",
            f"Panel bbox: {bbox}",
            "",
            "Caption:", str(source.get("caption") or ""),
            "", "Formal approval supported here: NO",
        ]
        self.counter.configure(text=f"Candidate {index + 1} / {len(self.records)}")
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(lines))
        self.summary.configure(state="disabled")
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", "AI-assisted screening of CC-BY source figure and CV panel candidate.")
        self.status.configure(text="No screening packet written for this candidate.", foreground="#444")

    def previous(self) -> None:
        self.index = (self.index - 1) % len(self.records)
        self.show_record(self.index)

    def next(self) -> None:
        self.index = (self.index + 1) % len(self.records)
        self.show_record(self.index)

    def record(self, decision: str) -> None:
        from tkinter import messagebox

        try:
            packet = make_screening_packet(
                self.records[self.index], decision, self.operator_mode,
                self.notes.get("1.0", "end").strip())
            saved = save_screening_packet(packet, self.output_dir)
            message = f"Saved {Path(saved['path']).name} — NOT HUMAN APPROVED"
            if self.auto_advance and self.index + 1 < len(self.records):
                self.index += 1
                self.show_record(self.index)
            self.status.configure(
                text=message,
                foreground="#14532d")
        except FileExistsError:
            self.status.configure(text="Packet already exists; no overwrite performed.",
                                  foreground="#92400e")
        except ValueError as exc:
            messagebox.showerror("Screening refused", str(exc))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--preannotation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operator-mode", choices=("ai_assisted", "human"),
                        default="ai_assisted")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--no-auto-advance", action="store_true")
    parser.add_argument("--headless-check", action="store_true")
    args = parser.parse_args()
    queue = load_figure_review_queue(args.dataset_manifest, args.preannotation_manifest)
    if args.headless_check:
        counts: dict[str, int] = {}
        for record in queue["records"]:
            status = record["preannotation"]["status"]
            counts[status] = counts.get(status, 0) + 1
        print(json.dumps({
            "status": queue["status"], "record_count": queue["record_count"],
            "preannotation_status_counts": counts, "formal_approval_supported": False,
            "dataset_manifest_sha256": queue["dataset_manifest_sha256"],
            "preannotation_manifest_sha256": queue["preannotation_manifest_sha256"],
        }, ensure_ascii=False))
        return
    FigureCandidateWorkbench(queue, args.output_dir, args.operator_mode,
                             args.start_index, not args.no_auto_advance).run()


if __name__ == "__main__":
    main()
