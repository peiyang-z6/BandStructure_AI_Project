"""Six-figure screen for explicit Computer Use visual review and multi-panel labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mcp_server.figure_review_queue import load_figure_review_queue
from mcp_server.visual_batch_review import save_visual_batch, finish_visual_review


class BatchWorkbench:
    def __init__(self, queue, output, start=0):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.queue, self.output, self.start = tk, queue, output, start
        self.root = tk.Tk()
        self.root.title(f'BandStructure — Batch Visual Annotation ({queue["record_count"]})')
        self.root.geometry('1500x920')
        self.root.state('zoomed')
        self.title = ttk.Label(self.root, text='AI visual labels | box coordinates: percent of full image | Ctrl+Return saves batch', font=('Segoe UI', 11))
        self.title.pack(fill='x', padx=6, pady=4)
        self.grid = ttk.Frame(self.root)
        self.grid.pack(fill='both', expand=True)
        self.canvases = []
        for row in range(2):
            self.grid.rowconfigure(row, weight=1)
            for col in range(3):
                self.grid.columnconfigure(col, weight=1)
                canvas = tk.Canvas(self.grid, background='#f4f6f8', highlightthickness=1, highlightbackground='#aab')
                canvas.grid(row=row, column=col, sticky='nsew', padx=2, pady=2)
                self.canvases.append(canvas)
        self.entry = tk.Text(self.root, height=3, wrap='word', font=('Consolas', 9))
        self.entry.pack(fill='x', padx=6, pady=3)
        self.status = ttk.Label(self.root, text='Enter [{"i":1,"target":"yes","boxes":[[10,10,90,90]],"note":"observed details"}, ...]')
        self.status.pack(fill='x', padx=6)
        self.entry.bind('<Control-Return>', self.save)
        self.root.bind('<Configure>', self.schedule_render)
        self.pending = None
        self.root.after(200, self.render)

    def schedule_render(self, event):
        if event.widget is self.root:
            if self.pending:
                self.root.after_cancel(self.pending)
            self.pending = self.root.after(180, self.render)

    def render(self):
        from PIL import Image, ImageTk
        self.pending = None
        if self.start >= self.queue['record_count']:
            return
        self.photos = []
        self.title.configure(text=f'Figures {self.start+1}–{min(self.start+6,self.queue["record_count"])} / {self.queue["record_count"]} | AI visual review | boxes in full-image % | Ctrl+Return: save & next')
        for slot, canvas in enumerate(self.canvases):
            canvas.delete('all')
            idx = self.start + slot
            if idx >= len(self.queue['records']):
                continue
            record = self.queue['records'][idx]
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            canvas.create_text(6, 3, anchor='nw', text=f'#{idx+1}  {record["group_id"]}  {record["source"]["figure_label"]}', font=('Segoe UI', 10, 'bold'))
            with Image.open(record['image_path']) as src:
                pic = src.convert('RGB')
            pic.thumbnail((max(60,cw-44), max(60,ch-48)), Image.Resampling.LANCZOS)
            x, y = (cw-pic.width)//2, 32+(ch-48-pic.height)//2
            photo = ImageTk.PhotoImage(pic)
            self.photos.append(photo)
            canvas.create_image(x, y, image=photo, anchor='nw')
            canvas.create_rectangle(x, y, x+pic.width, y+pic.height, outline='#8fa0b0')
            for pct in (0,25,50,75,100):
                canvas.create_text(x+pic.width*pct/100, y-7, text=str(pct), font=('Consolas',7), fill='#6b7280')
                canvas.create_text(x-4, y+pic.height*pct/100, anchor='e', text=str(pct), font=('Consolas',7), fill='#6b7280')
        self.entry.focus_set()

    def save(self, event=None):
        try:
            submissions = json.loads(self.entry.get('1.0','end'))
            path = save_visual_batch(self.queue, self.start, submissions, self.output)
            self.start += len(submissions)
            self.entry.delete('1.0','end')
            if self.start >= len(self.queue['records']):
                result = finish_visual_review(self.queue, self.output)
                self.status.configure(text=f'ALL {result["record_count"]} REVIEWED — {result["target_counts"]}, {result["panel_count"]} panel boxes; AI labels')
                self.title.configure(text=f'{result["record_count"]} / {result["record_count"]} AI visual annotations saved')
                self.entry.configure(state='disabled')
            else:
                self.status.configure(text=f'Saved {path.name}; individual AI observations only.')
                self.render()
        except Exception as exc:
            self.status.configure(text=f'NOT SAVED: {type(exc).__name__}: {exc}')
        return 'break'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', required=True, type=Path)
    parser.add_argument('--preannotation-manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--start-index', type=int, default=0)
    args=parser.parse_args()
    queue=load_figure_review_queue(args.dataset_manifest,args.preannotation_manifest)
    app=BatchWorkbench(queue,args.output_dir,args.start_index)
    app.root.mainloop()

if __name__=='__main__':
    main()
