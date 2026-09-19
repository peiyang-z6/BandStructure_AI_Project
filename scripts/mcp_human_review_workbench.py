"""Local review helper for MCP pending-human candidates; never grants approval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.review_queue import load_pending_queue, make_review_packet, save_review_packet


def validate_inputs(queue_path: Path, image_path: Path) -> tuple[dict, tuple[int, int]]:
    from PIL import Image

    queue = load_pending_queue(queue_path)
    if not image_path.is_file() or image_path.is_symlink() or image_path.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("invalid review image path")
    with Image.open(image_path) as image:
        image.verify()
    with Image.open(image_path) as image:
        size = image.size
    if size[0] < 100 or size[1] < 100 or size[0] * size[1] > 30_000_000:
        raise ValueError("review image dimensions out of bounds")
    return queue, size


class ReviewWorkbench:
    def __init__(self, queue: dict, image_path: Path, output_dir: Path,
                 operator_mode: str) -> None:
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk

        self.tk = tk
        self.ttk = ttk
        self.queue = queue
        self.records = queue["records"]
        self.output_dir = output_dir
        self.operator_mode = operator_mode
        self.index = 0

        root = tk.Tk()
        self.root = root
        root.title("BandStructure MCP — Pending Human Review")
        root.geometry("1420x900")
        root.minsize(1100, 720)

        banner = ttk.Label(
            root,
            text=("AI-assisted pre-review — FORMAL HUMAN APPROVAL BLOCKED "
                  "(external authenticated registry required)"),
            foreground="#9b1c1c",
            font=("Segoe UI", 12, "bold"),
        )
        banner.pack(fill="x", padx=12, pady=(10, 6))

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=12, pady=6)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((760, 790), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        image_label = ttk.Label(body, image=self.photo)
        image_label.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        panel = ttk.Frame(body)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)

        self.counter = ttk.Label(panel, font=("Segoe UI", 11, "bold"))
        self.counter.grid(row=0, column=0, sticky="w")
        self.summary = tk.Text(panel, height=13, wrap="word", state="disabled",
                               font=("Consolas", 10))
        self.summary.grid(row=1, column=0, sticky="ew", pady=(8, 8))

        ttk.Label(panel, text="Pre-review notes (not human approval):").grid(
            row=2, column=0, sticky="nw")
        self.notes = tk.Text(panel, height=8, wrap="word", font=("Segoe UI", 10))
        self.notes.grid(row=3, column=0, sticky="nsew", pady=(4, 8))

        buttons = ttk.Frame(panel)
        buttons.grid(row=4, column=0, sticky="ew")
        ttk.Button(buttons, text="Previous", command=self.previous).pack(side="left")
        ttk.Button(buttons, text="Next", command=self.next).pack(side="left", padx=6)
        ttk.Button(buttons, text="Record needs more evidence",
                   command=lambda: self.record("needs_more_evidence")).pack(side="left", padx=6)
        ttk.Button(buttons, text="Reject candidate",
                   command=lambda: self.record("rejected")).pack(side="left")

        self.approve = ttk.Button(
            panel, text="Formal Approve (external registry only)", state="disabled")
        self.approve.grid(row=5, column=0, sticky="ew", pady=(10, 4))
        self.status = ttk.Label(panel, text="No review packet written.", foreground="#444")
        self.status.grid(row=6, column=0, sticky="w")
        self.show_record(0)

    def show_record(self, index: int) -> None:
        record = self.records[index]
        source = record["source"]
        state = record["review_state"]
        self.counter.configure(text=f"Candidate {index + 1} / {len(self.records)}")
        lines = [
            f"Record: {record['record_id']}",
            f"Figure: {source.get('figure_label')}",
            f"Panel: {source.get('panel_label')}",
            f"Material: {source.get('material_label')}",
            f"Page: {source.get('page_number')}",
            "",
            "Missing evidence:",
            *[f"- {item}" for item in state["missing_evidence"]],
            "",
            "Formal approval supported here: NO",
        ]
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(lines))
        self.summary.configure(state="disabled")
        self.notes.delete("1.0", "end")
        self.notes.insert(
            "1.0",
            "AI-assisted visual pre-review: energy axis, ±2 eV ticks and E_F are visible; "
            "continuous k sampling, segment IDs and band identities remain unresolved.",
        )
        self.status.configure(text="No review packet written for this candidate.")

    def previous(self) -> None:
        self.index = (self.index - 1) % len(self.records)
        self.show_record(self.index)

    def next(self) -> None:
        self.index = (self.index + 1) % len(self.records)
        self.show_record(self.index)

    def record(self, decision: str) -> None:
        from tkinter import messagebox

        notes = self.notes.get("1.0", "end").strip()
        try:
            packet = make_review_packet(
                self.records[self.index], decision, self.operator_mode, notes)
            saved = save_review_packet(packet, self.output_dir)
            self.status.configure(
                text=f"Saved {Path(saved['path']).name} — NOT HUMAN APPROVED",
                foreground="#14532d",
            )
        except FileExistsError:
            self.status.configure(text="Packet already exists; no overwrite performed.",
                                  foreground="#92400e")
        except ValueError as exc:
            messagebox.showerror("Review refused", str(exc))

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operator-mode", choices=("ai_assisted", "human"),
                        default="ai_assisted")
    parser.add_argument("--headless-check", action="store_true")
    args = parser.parse_args()
    queue, image_size = validate_inputs(args.queue, args.image)
    if args.headless_check:
        print(json.dumps({
            "status": "ready_for_manual_review",
            "record_count": queue["record_count"],
            "approval_ready_count": sum(
                int(record["review_state"]["approval_ready"])
                for record in queue["records"]),
            "formal_approval_supported": False,
            "image_size": list(image_size),
            "queue_sha256": queue["source_sha256"],
        }, ensure_ascii=False))
        return
    ReviewWorkbench(queue, args.image, args.output_dir, args.operator_mode).run()


if __name__ == "__main__":
    main()
