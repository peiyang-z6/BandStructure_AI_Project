#!/usr/bin/env python3
"""
BandStructure AI — Native Python GUI Workbench v2
==================================================
v2 enhancements:
  - Floating tool palette (doesn't obscure canvas)
  - Zoom in/out with mouse wheel, buttons, and fit-to-window
  - Auto pop-out results window after Physics Brain recognition
  - Scrollable canvas for large images
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageTk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Paths ──
OUTPUT_DIR = ROOT / "artifacts" / "reports" / "gui_workbench"
TRAINING_DATA_DIR = ROOT / "data" / "annotations" / "human"
TSNE_IMAGE = ROOT / "artifacts" / "reports" / "aflow_noleak_v5_30k_seed42" / "latent_tsne_spacegroups.png"
DETECTOR_PATH = ROOT / "artifacts" / "models" / "vision_detector" / "band_plot_yolov8_pose_best.pt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors ──
BG, FG = "#0d1117", "#e6edf3"
CANVAS_BG = "#1a1a2e"
BORDER = "#30363d"
ACCENT = "#1f6feb"
TOOLBAR_BG, BTN_BG, BTN_HOVER, BTN_ACTIVE = "#161b22", "#21262d", "#30363d", "#1f6feb"
PANEL_COLOR, FERMI_COLOR = "#ffd600", "#00e676"
VB_COLOR, CB_COLOR = "#ff9100", "#448aff"
VBM_COLOR, CBM_COLOR = "#ef5350", "#42a5f5"
AXIS_X_COLOR, AXIS_Y_COLOR = "#00bcd4", "#ffc107"
DIRECT_COLOR, INDIRECT_COLOR = "#66bb6a", "#ab47bc"
TEXT_DIM = "#8b949e"

CANVAS_W, CANVAS_H = 900, 600
MAX_IMG_DIM = 1200  # increased for zoom capability

TOOL_PANEL, TOOL_FERMI, TOOL_VB, TOOL_CB = "panel", "fermi", "vb", "cb"
TOOL_VBM, TOOL_CBM, TOOL_XAXIS, TOOL_YAXIS = "vbm", "cbm", "xaxis", "yaxis"
TOOL_ERASER, TOOL_AUTO_VB, TOOL_AUTO_CB = "eraser", "auto_vb", "auto_cb"

_BRAIN = None

def _get_brain():
    global _BRAIN
    if _BRAIN is None:
        from src.vision.brain_invoker import PhysicsBrainInvoker
        _BRAIN = PhysicsBrainInvoker()
    return _BRAIN


# ═════════════════════════════════════════════════════════════════════════════
#  Zoomable Drawing Canvas
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
#  Color-based auto curve tracer
# ═════════════════════════════════════════════════════════════════════════════

class CurveTracer:
    """Trace dark curves on light backgrounds by clicking on them."""

    @staticmethod
    def trace(gray_img: "Image.Image", click_x: int, click_y: int,
              threshold: int = 128, search_radius: int = 40):
        gray = np.array(gray_img.convert("L"), dtype=np.uint8)
        h, w = gray.shape
        mask = gray < threshold
        best_pt, best_d = None, 999
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                nx, ny = click_x + dx, click_y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx]:
                    d = abs(dx) + abs(dy)
                    if d < best_d:
                        best_d, best_pt = d, (nx, ny)
        if not best_pt:
            return []
        all_points = set()
        def follow(start, init_dir):
            cx, cy = start
            prev_dx, prev_dy = init_dir
            seg = [(cx, cy)]
            for _ in range(4000):
                best_nx, best_ny = -1, -1
                best_score = -99
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0: continue
                        nx, ny = cx + dx, cy + dy
                        if nx < 0 or nx >= w or ny < 0 or ny >= h: continue
                        if not mask[ny, nx] or (nx, ny) in all_points: continue
                        s = dx * prev_dx + dy * prev_dy
                        if s > best_score:
                            best_score, best_nx, best_ny = s, nx, ny
                if best_nx < 0: break
                prev_dx, prev_dy = best_nx - cx, best_ny - cy
                cx, cy = best_nx, best_ny
                all_points.add((cx, cy))
                seg.append((cx, cy))
            return seg
        all_points.add(best_pt)
        fwd = follow(best_pt, (1, 0))
        bwd = follow(best_pt, (-1, 0))
        bwd.reverse()
        path = bwd + fwd[1:]
        path.sort(key=lambda p: p[0])
        return path


class DrawingCanvas(tk.Canvas):
    """Interactive canvas with zoom, scroll, and band-structure annotation tools."""

    def __init__(self, parent, on_status: callable = None, **kw):
        kw.setdefault("width", CANVAS_W)
        kw.setdefault("height", CANVAS_H)
        kw.setdefault("bg", CANVAS_BG)
        kw.setdefault("highlightthickness", 0)
        kw.setdefault("cursor", "crosshair")
        super().__init__(parent, **kw)

        self.on_status = on_status or (lambda s: None)

        # Image state
        self._pil_original: Optional[Image.Image] = None  # always the unresized original
        self._pil_display: Optional[Image.Image] = None    # current zoom level
        self._tk_image: Optional[ImageTk.PhotoImage] = None

        # Zoom state
        self._zoom = 1.0
        self._offset_x, self._offset_y = 0, 0
        self._pan_start_x, self._pan_start_y = 0, 0
        self._panning = False

        # Tool state
        self._tool = TOOL_PANEL
        self._drawing = False
        self._start_x = self._start_y = 0
        self._current_stroke: List[Tuple[float, float]] = []

        # Annotations (in IMAGE pixel coordinates — invariant under zoom)
        self.annotations = {
            "panel": None, "fermi_y": None,
            "vb_strokes": [], "cb_strokes": [],
            "xaxis_pts": [], "yaxis_pts": [],
            "vbm": None, "cbm": None, "gap_type": None,
        }

        # Bind events
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<MouseWheel>", self._on_wheel)        # Windows
        self.bind("<Button-4>", self._on_wheel_up)        # Linux scroll up
        self.bind("<Button-5>", self._on_wheel_down)      # Linux scroll down
        self.bind("<ButtonPress-2>", self._on_pan_start)  # middle-click pan
        self.bind("<B2-Motion>", self._on_pan_drag)
        self.bind("<ButtonRelease-2>", self._on_pan_end)

        self._redraw_grid()
        self._draw_hint()

    # ── Coordinate transforms ────────────────────────────────────────────
    def _img_to_canvas(self, ix: float, iy: float) -> Tuple[float, float]:
        """Image coords → canvas coords."""
        return ix * self._zoom + self._offset_x, iy * self._zoom + self._offset_y

    def _canvas_to_img(self, cx: float, cy: float) -> Tuple[float, float]:
        """Canvas coords → image coords."""
        return (cx - self._offset_x) / max(self._zoom, 0.01), \
               (cy - self._offset_y) / max(self._zoom, 0.01)

    # ── Zoom ─────────────────────────────────────────────────────────────
    def zoom_in(self, factor: float = 1.2):
        self._apply_zoom(self._zoom * factor)

    def zoom_out(self, factor: float = 1.2):
        self._apply_zoom(self._zoom / factor)

    def zoom_fit(self):
        """Fit image to canvas."""
        if not self._pil_original:
            return
        cw = int(self.master.winfo_width() or CANVAS_W)
        ch = int(self.master.winfo_height() or CANVAS_H)
        ow, oh = self._pil_original.size
        fit = min(cw / max(ow, 1), ch / max(oh, 1), 1.0)
        self._zoom = fit
        self._offset_x = max(0, (cw - ow * self._zoom) / 2)
        self._offset_y = max(0, (ch - oh * self._zoom) / 2)
        self._update_display()

    def zoom_100(self):
        self._apply_zoom(1.0)

    def _on_wheel(self, event):
        """Windows mouse wheel."""
        if event.delta > 0:
            self.zoom_in(1.15)
        else:
            self.zoom_out(1.15)

    def _on_wheel_up(self, event):
        self.zoom_in(1.15)

    def _on_wheel_down(self, event):
        self.zoom_out(1.15)

    def _apply_zoom(self, new_zoom: float):
        if not self._pil_original:
            return
        new_zoom = max(0.05, min(10.0, new_zoom))
        if abs(new_zoom - self._zoom) < 0.001:
            return
        # Keep the center point stable
        cw = int(self.master.winfo_width() or CANVAS_W)
        ch = int(self.master.winfo_height() or CANVAS_H)
        cx, cy = cw / 2, ch / 2
        ix, iy = self._canvas_to_img(cx, cy)
        self._zoom = new_zoom
        self._offset_x = cx - ix * self._zoom
        self._offset_y = cy - iy * self._zoom
        self._update_display()

    def _update_display(self):
        """Rebuild display image at current zoom."""
        if not self._pil_original:
            return
        ow, oh = self._pil_original.size
        nw = max(1, int(ow * self._zoom))
        nh = max(1, int(oh * self._zoom))
        self._pil_display = self._pil_original.resize((nw, nh), Image.LANCZOS)
        self._tk_image = ImageTk.PhotoImage(self._pil_display)
        self.config(width=nw + max(0, int(self._offset_x)),
                    height=nh + max(0, int(self._offset_y)))
        self._redraw_all()

    # ── Pan ──────────────────────────────────────────────────────────────
    def _on_pan_start(self, event):
        self._panning = True
        self._pan_start_x, self._pan_start_y = event.x, event.y
        self.config(cursor="fleur")

    def _on_pan_drag(self, event):
        if not self._panning:
            return
        dx, dy = event.x - self._pan_start_x, event.y - self._pan_start_y
        self._offset_x += dx
        self._offset_y += dy
        self._pan_start_x, self._pan_start_y = event.x, event.y
        self._update_display()

    def _on_pan_end(self, event):
        self._panning = False
        self.config(cursor="crosshair")

    # ── Tool management ──────────────────────────────────────────────────
    def set_tool(self, tool: str):
        self._tool = tool
        self.config(cursor="crosshair" if tool != TOOL_ERASER else "dotbox")
        labels = {
            TOOL_PANEL: "Drag to draw yellow box around band-plot panel",
            TOOL_FERMI: "Click to place green Fermi level line (E=0 eV)",
            TOOL_VB: "Draw strokes along valence band curves (below Fermi)",
            TOOL_CB: "Draw strokes along conduction band curves (above Fermi)",
            TOOL_XAXIS: "Click two X-axis endpoints (X1=0.0, X2=1.0 for normalized k-path)",
            TOOL_YAXIS: "Click two Y-axis endpoints, then enter eV-values above",
            TOOL_VBM: "Click the valence band MAXIMUM point",
            TOOL_CBM: "Click the conduction band MINIMUM point",
            TOOL_ERASER: "Click near an annotation to erase it",
            TOOL_AUTO_VB: "Click a dark curve on light bg to auto-trace as VB",
            TOOL_AUTO_CB: "Click a dark curve on light bg to auto-trace as CB",
        }
        self.on_status(labels.get(tool, ""))

    def undo(self):
        tool = self._tool
        if tool in (TOOL_VB, TOOL_AUTO_VB) and self.annotations["vb_strokes"]:
            self.annotations["vb_strokes"].pop()
        elif tool in (TOOL_CB, TOOL_AUTO_CB) and self.annotations["cb_strokes"]:
            self.annotations["cb_strokes"].pop()
        elif tool == TOOL_VBM:
            self.annotations["vbm"] = None
        elif tool == TOOL_CBM:
            self.annotations["cbm"] = None
        elif tool in (TOOL_XAXIS, TOOL_YAXIS):
            key = "xaxis_pts" if tool == TOOL_XAXIS else "yaxis_pts"
            if self.annotations[key]:
                self.annotations[key].pop()
        elif tool == TOOL_ERASER:
            pass  # eraser undo not supported
        else:
            self.annotations["panel"] = None
            self.annotations["fermi_y"] = None
        self._redraw_all()

    def clear_all(self):
        self.annotations = {
            "panel": None, "fermi_y": None, "vb_strokes": [], "cb_strokes": [],
            "xaxis_pts": [], "yaxis_pts": [], "vbm": None, "cbm": None,
            "gap_type": None,
        }
        self._redraw_all()

    def get_annotation_json(self) -> str:
        ann = {}
        for k, v in self.annotations.items():
            if k in ("panel", "fermi_y", "gap_type"):
                ann[k] = v
            elif k in ("vbm", "cbm"):
                ann[k] = list(v) if v else None
            elif k in ("vb_strokes", "cb_strokes"):
                ann[k] = [[list(p) for p in stroke] for stroke in v]
            elif k in ("xaxis_pts", "yaxis_pts"):
                ann[k] = [list(p) for p in v]
        return json.dumps(ann, ensure_ascii=True)

    # ── Image loading ────────────────────────────────────────────────────
    def load_image(self, pil_img: Image.Image):
        """Load a PIL image onto the canvas (keeps original for zoom)."""
        self._pil_original = pil_img.convert("RGB")
        self._zoom = 1.0
        # Fit to window initially
        self.zoom_fit()
        self._redraw_all()

    def get_canvas_dims(self) -> Tuple[int, int]:
        if self._pil_original:
            return self._pil_original.size
        return int(self["width"]), int(self["height"])

    # ── Drawing ──────────────────────────────────────────────────────────
    def _redraw_all(self):
        self.delete("all")
        if self._tk_image:
            self.create_image(int(self._offset_x), int(self._offset_y),
                              anchor="nw", image=self._tk_image)
        else:
            self._redraw_grid()
            self._draw_hint()
        self._draw_annotations()

    def _redraw_grid(self):
        w, h = int(self["width"]), int(self["height"])
        for i in range(0, w, 40):
            self.create_line(i, 0, i, h, fill="#2a2a4a", width=1)
        for j in range(0, h, 40):
            self.create_line(0, j, w, j, fill="#2a2a4a", width=1)

    def _draw_hint(self):
        w, h = int(self["width"]), int(self["height"])
        self.create_text(w//2, h//2, text="Upload a band-structure image\n"
                         "Mouse-wheel to zoom  |  Middle-click to pan\n"
                         "Right-click toolbar for tools",
                         fill="#c9d1d9", font=("Segoe UI", 14), justify="center")

    def _draw_annotations(self):
        ann = self.annotations
        # Convert image coords to canvas coords
        def tx(ix, iy):
            return self._img_to_canvas(ix, iy)

        # Panel bbox
        if ann["panel"]:
            x, y, pw, ph = ann["panel"]
            cx, cy = tx(x, y)
            cpw, cph = pw * self._zoom, ph * self._zoom
            self.create_rectangle(cx, cy, cx+cpw, cy+cph,
                                  outline=PANEL_COLOR, width=2)
            self.create_rectangle(cx, cy, cx+cpw, cy+cph,
                                  fill="", stipple="gray25", outline="")

        # Fermi line
        if ann["fermi_y"] is not None:
            _, cy = tx(0, ann["fermi_y"])
            cw = int(self["width"])
            self.create_line(0, cy, cw, cy, fill=FERMI_COLOR, width=2,
                             dash=(8, 4))
            self.create_text(8, cy-6, text="EF=0", anchor="w",
                             fill=FERMI_COLOR, font=("Consolas", 10, "bold"))

        # VB strokes
        for stroke in ann["vb_strokes"]:
            if len(stroke) >= 2:
                pts = []
                for ix, iy in stroke:
                    cx, cy = tx(ix, iy)
                    pts.extend([cx, cy])
                self.create_line(*pts, fill=VB_COLOR, width=2,
                                 capstyle="round", joinstyle="round")

        # CB strokes
        for stroke in ann["cb_strokes"]:
            if len(stroke) >= 2:
                pts = []
                for ix, iy in stroke:
                    cx, cy = tx(ix, iy)
                    pts.extend([cx, cy])
                self.create_line(*pts, fill=CB_COLOR, width=2,
                                 capstyle="round", joinstyle="round")

        # Axis points
        for i, (ix, iy) in enumerate(ann["xaxis_pts"]):
            cx, cy = tx(ix, iy)
            self._draw_point(cx, cy, AXIS_X_COLOR, f"X{i+1}")
        for i, (ix, iy) in enumerate(ann["yaxis_pts"]):
            cx, cy = tx(ix, iy)
            self._draw_point(cx, cy, AXIS_Y_COLOR, f"Y{i+1}")

        # VBM diamond
        if ann["vbm"]:
            cx, cy = tx(*ann["vbm"])
            s = 7
            self.create_polygon(cx, cy-s, cx+s, cy, cx, cy+s, cx-s, cy,
                                fill=VBM_COLOR, outline="white", width=1.5)
            self.create_text(cx+12, cy-8, text="VBM", anchor="w",
                             fill="white", font=("Consolas", 10, "bold"))

        # CBM triangle
        if ann["cbm"]:
            cx, cy = tx(*ann["cbm"])
            s = 7
            self.create_polygon(cx, cy-s, cx+s, cy+s, cx-s, cy+s,
                                fill=CBM_COLOR, outline="white", width=1.5)
            self.create_text(cx+12, cy-8, text="CBM", anchor="w",
                             fill="white", font=("Consolas", 10, "bold"))

        # VBM-CBM connection
        if ann["vbm"] and ann["cbm"]:
            cvx, cvy = tx(*ann["vbm"])
            ccx, ccy = tx(*ann["cbm"])
            dx = abs(ann["vbm"][0] - ann["cbm"][0])
            is_direct = dx <= 5
            ann["gap_type"] = "direct" if is_direct else "indirect"
            color = DIRECT_COLOR if is_direct else INDIRECT_COLOR
            self.create_line(cvx, cvy, ccx, ccy, fill=color, width=2.5, dash=(5, 3))
            mx, my = (cvx+ccx)//2, (cvy+ccy)//2
            self.create_text(mx+8, my-6, text="DIRECT" if is_direct else "INDIRECT",
                             anchor="w", fill=color, font=("Consolas", 11, "bold"))

    def _draw_point(self, x, y, color, label):
        self.create_oval(x-5, y-5, x+5, y+5, fill=color, outline="white", width=1.5)
        self.create_text(x+10, y-6, text=label, anchor="w",
                         fill="white", font=("Consolas", 9, "bold"))

    # ── Mouse events (annotations stored in IMAGE coordinates) ───────────
    def _on_press(self, event):
        if self._panning:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        self._drawing = True
        self._start_x, self._start_y = ix, iy
        self._current_stroke = [(ix, iy)]

        tool = self._tool
        if tool == TOOL_FERMI:
            self.annotations["fermi_y"] = iy
            self._drawing = False
            self._redraw_all()
        elif tool == TOOL_VBM:
            self.annotations["vbm"] = (ix, iy)
            self._drawing = False
            self._redraw_all()
        elif tool == TOOL_CBM:
            self.annotations["cbm"] = (ix, iy)
            self._drawing = False
            self._redraw_all()
        elif tool == TOOL_ERASER:
            self._erase_near(ix, iy)
            self._drawing = False
            self._redraw_all()
        elif tool in (TOOL_AUTO_VB, TOOL_AUTO_CB):
            self._auto_trace(ix, iy, tool)
            self._drawing = False
        elif tool in (TOOL_XAXIS, TOOL_YAXIS):
            pts = self.annotations["xaxis_pts" if tool == TOOL_XAXIS else "yaxis_pts"]
            pts.append((ix, iy))
            if len(pts) > 2:
                pts.pop(0)
            self._drawing = False
            self._redraw_all()

    def _on_drag(self, event):
        if not self._drawing or self._panning:
            return
        ix, iy = self._canvas_to_img(event.x, event.y)
        tool = self._tool
        if tool == TOOL_PANEL:
            x0, y0 = self._start_x, self._start_y
            self.annotations["panel"] = (min(x0, ix), min(y0, iy),
                                         abs(ix - x0), abs(iy - y0))
            self._redraw_all()
        elif tool in (TOOL_VB, TOOL_CB):
            self._current_stroke.append((ix, iy))
            strokes = self.annotations["vb_strokes" if tool == TOOL_VB else "cb_strokes"]
            if not strokes:
                strokes.append([])
            strokes[-1] = self._current_stroke[:]
            self._redraw_all()

    def _on_release(self, event):
        if not self._drawing:
            return
        self._drawing = False
        tool = self._tool
        if tool in (TOOL_VB, TOOL_CB):
            strokes = self.annotations["vb_strokes" if tool == TOOL_VB else "cb_strokes"]
            if len(self._current_stroke) > 1:
                strokes.append(self._current_stroke[:])
            elif strokes and len(strokes[-1]) < 2:
                strokes.pop()
            self._current_stroke = []
            self._redraw_all()

    # ── Eraser ──────────────────────────────────────────────────────────
    def _erase_near(self, ix, iy, radius=15):
        ann = self.annotations
        best_key, best_idx, best_dist = None, None, radius
        def chk(k, val, idx=None):
            nonlocal best_key, best_idx, best_dist
            if val is None: return
            if k == "panel":
                d = ((ix-(val[0]+val[2]/2))**2 + (iy-(val[1]+val[3]/2))**2)**0.5
            elif k == "fermi_y":
                d = abs(iy - val)
            elif k in ("vbm","cbm"):
                d = ((ix-val[0])**2 + (iy-val[1])**2)**0.5
            else: return
            if d < best_dist: best_key, best_idx, best_dist = k, idx, d
        chk("panel", ann["panel"]); chk("fermi_y", ann["fermi_y"])
        chk("vbm", ann["vbm"]); chk("cbm", ann["cbm"])
        for i,pt in enumerate(ann["xaxis_pts"]):
            d = ((ix-pt[0])**2+(iy-pt[1])**2)**0.5
            if d<best_dist: best_key,best_idx,best_dist="xaxis_pts",i,d
        for i,pt in enumerate(ann["yaxis_pts"]):
            d = ((ix-pt[0])**2+(iy-pt[1])**2)**0.5
            if d<best_dist: best_key,best_idx,best_dist="yaxis_pts",i,d
        for si,s in enumerate(ann["vb_strokes"]):
            for pt in s:
                d = ((ix-pt[0])**2+(iy-pt[1])**2)**0.5
                if d<best_dist: best_key,best_idx,best_dist="vb_strokes",si,d
        for si,s in enumerate(ann["cb_strokes"]):
            for pt in s:
                d = ((ix-pt[0])**2+(iy-pt[1])**2)**0.5
                if d<best_dist: best_key,best_idx,best_dist="cb_strokes",si,d
        if best_key:
            if best_key in ("panel","fermi_y","vbm","cbm"): ann[best_key]=None
            elif best_key in ("xaxis_pts","yaxis_pts"):
                if best_idx is not None: ann[best_key].pop(best_idx)
            elif best_key in ("vb_strokes","cb_strokes"):
                if best_idx is not None: ann[best_key].pop(best_idx)
            self.on_status(f"Erased: {best_key}")

    # ── Auto-trace ──────────────────────────────────────────────────────
    def _auto_trace(self, ix, iy, tool):
        if not self._pil_original:
            self.on_status("Auto-trace: no image loaded"); return
        path = CurveTracer.trace(self._pil_original, int(ix), int(iy))
        if len(path) < 3:
            self.on_status(f"Auto-trace: no curve found near click"); return
        target = self.annotations["vb_strokes" if tool == TOOL_AUTO_VB else "cb_strokes"]
        target.append(path)
        band = "VB" if tool == TOOL_AUTO_VB else "CB"
        self.on_status(f"Auto-trace: {band} curve ({len(path)} pts)")
        self._redraw_all()


# ═════════════════════════════════════════════════════════════════════════════
#  Floating Tool Palette
# ═════════════════════════════════════════════════════════════════════════════

class FloatingToolbar(tk.Toplevel):
    """Always-on-top floating tool palette."""

    def __init__(self, parent, on_tool_select: callable, on_undo: callable,
                 on_clear: callable, on_zoom_in: callable, on_zoom_out: callable,
                 on_zoom_fit: callable, on_zoom_100: callable):
        super().__init__(parent)
        self.title("Tools")
        self.configure(bg=TOOLBAR_BG)
        self.overrideredirect(True)  # no title bar
        self.attributes("-topmost", True)

        self._tool_buttons: Dict[str, tk.Button] = {}
        self._on_tool_select = on_tool_select

        # Make window draggable
        self.bind("<ButtonPress-1>", self._start_move)
        self.bind("<B1-Motion>", self._on_move)

        # Row 1: Annotation tools
        tools = [
            ("[ ] Panel", TOOL_PANEL, "#ffd600"),
            ("── Fermi", TOOL_FERMI, "#00e676"),
            ("  VB  ", TOOL_VB, "#ff9100"),
            ("  CB  ", TOOL_CB, "#448aff"),
            ("AutoVB", TOOL_AUTO_VB, "#ff9100"),
            ("AutoCB", TOOL_AUTO_CB, "#448aff"),
        ]
        for i, (label, tid, color) in enumerate(tools):
            btn = self._mk_btn(label, lambda t=tid: self._select(t), color)
            btn.grid(row=0, column=i, padx=1, pady=2)
            self._tool_buttons[tid] = btn

        # Row 2: Extrema + Eraser
        tools2 = [
            (" VBM ", TOOL_VBM, "#ef5350"),
            (" CBM ", TOOL_CBM, "#42a5f5"),
            ("Erase", TOOL_ERASER, "#ff5252"),
        ]
        for i, (label, tid, color) in enumerate(tools2):
            col = len(tools) + i
            btn = self._mk_btn(label, lambda t=tid: self._select(t), color)
            btn.grid(row=1, column=col, padx=1, pady=2)
            self._tool_buttons[tid] = btn

        # Axis tools (same row as extrema+eraser)
        axis_tools = [
            ("X-axis", TOOL_XAXIS, "#00bcd4"),
            ("Y-axis", TOOL_YAXIS, "#ffc107"),
        ]
        for i, (label, tid, color) in enumerate(axis_tools):
            btn = self._mk_btn(label, lambda t=tid: self._select(t), color)
            btn.grid(row=1, column=col + 1 + i, padx=1, pady=2)

        # Separator
        sep = tk.Frame(self, bg=BORDER, width=1, height=22)
        sep.grid(row=1, column=col + 1 + len(axis_tools), padx=4)

        # Row 2 cont: Edit actions
        edit_col = 3
        self._mk_btn(" Undo ", on_undo, "#8b949e").grid(row=1, column=edit_col, padx=1, pady=2)
        self._mk_btn("Clear", on_clear, "#c62828").grid(row=1, column=edit_col+1, padx=1, pady=2)

        # Row 3: Zoom controls
        zoom_frame = tk.Frame(self, bg=TOOLBAR_BG)
        zoom_frame.grid(row=2, column=0, columnspan=8, pady=(3, 2))
        self._mk_small(zoom_frame, "−", on_zoom_out).pack(side="left", padx=1)
        self._zoom_label = tk.Label(zoom_frame, text="100%", bg=TOOLBAR_BG, fg=TEXT_DIM,
                                    font=("Consolas", 9), width=5)
        self._zoom_label.pack(side="left", padx=2)
        self._mk_small(zoom_frame, "+", on_zoom_in).pack(side="left", padx=1)
        self._mk_small(zoom_frame, "Fit", on_zoom_fit).pack(side="left", padx=3)
        self._mk_small(zoom_frame, "1:1", on_zoom_100).pack(side="left", padx=1)

        self._select(TOOL_PANEL)

        # Position near top-right of parent
        self.update_idletasks()
        px, py = parent.winfo_x(), parent.winfo_y()
        self.geometry(f"+{px+parent.winfo_width()-self.winfo_width()-20}+{py+10}")

    def _mk_btn(self, text, command, color):
        btn = tk.Button(self, text=text, command=command,
                        bg=BTN_BG, fg=color, activebackground=BTN_ACTIVE,
                        activeforeground="white", relief="flat",
                        font=("Segoe UI", 9, "bold"), padx=6, pady=3,
                        cursor="hand2", borderwidth=0)
        return btn

    def _mk_small(self, parent, text, command):
        return tk.Button(parent, text=text, command=command,
                         bg=BTN_BG, fg=FG, activebackground=BTN_HOVER,
                         relief="flat", font=("Segoe UI", 9), padx=5, pady=1,
                         cursor="hand2", borderwidth=0)

    def _select(self, tool: str):
        self._on_tool_select(tool)
        for tid, btn in self._tool_buttons.items():
            if tid == tool:
                btn.configure(bg=BTN_ACTIVE, fg="white")
            else:
                btn.configure(bg=BTN_BG)
        colors = {TOOL_PANEL: "#ffd600", TOOL_FERMI: "#00e676", TOOL_VB: "#ff9100",
                  TOOL_CB: "#448aff", TOOL_VBM: "#ef5350", TOOL_CBM: "#42a5f5",
                  TOOL_XAXIS: "#00bcd4", TOOL_YAXIS: "#ffc107",
                  TOOL_ERASER: "#ff5252", TOOL_AUTO_VB: "#ff9100", TOOL_AUTO_CB: "#448aff"}
        for tid, btn in self._tool_buttons.items():
            if tid != tool and tid in colors:
                btn.configure(fg=colors[tid])

    def update_zoom_label(self, zoom: float):
        self._zoom_label.config(text=f"{int(zoom*100)}%")

    def _start_move(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_move(self, event):
        x = self.winfo_x() + event.x - self._drag_x
        y = self.winfo_y() + event.y - self._drag_y
        self.geometry(f"+{x}+{y}")


# ═════════════════════════════════════════════════════════════════════════════
#  Results Pop-Out Window
# ═════════════════════════════════════════════════════════════════════════════

class ResultsWindow(tk.Toplevel):
    """Large pop-out window for Physics Brain results."""

    def __init__(self, parent, title: str, text: str, curv_fig: Optional[Figure] = None,
                 tsne_fig: Optional[Figure] = None):
        super().__init__(parent)
        self.title(f"Physics Brain — {title}")
        self.configure(bg=BG)
        self.geometry("700x750")

        # Results text (large font)
        text_frame = tk.Frame(self, bg=BG)
        text_frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        result_text = tk.Text(text_frame, bg="#161b22", fg=FG, wrap="word",
                              relief="flat", font=("Consolas", 13),
                              padx=12, pady=12)
        result_text.pack(fill="both", expand=True)
        result_text.insert("1.0", text)
        result_text.config(state="disabled")  # read-only

        # Figures area
        if curv_fig or tsne_fig:
            fig_frame = tk.Frame(self, bg=BG)
            fig_frame.pack(fill="x", padx=10, pady=5)

            if curv_fig:
                curv_canvas = FigureCanvasTkAgg(curv_fig, fig_frame)
                curv_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

            if tsne_fig:
                tsne_canvas = FigureCanvasTkAgg(tsne_fig, fig_frame)
                tsne_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        # Close button
        tk.Button(self, text="Close", command=self.destroy,
                  bg=ACCENT, fg="white", activebackground=BTN_HOVER,
                  relief="flat", font=("Segoe UI", 11, "bold"),
                  padx=30, pady=8, cursor="hand2").pack(pady=(5, 10))

        self.transient(parent)
        self.grab_set()
        self.focus_force()


# ═════════════════════════════════════════════════════════════════════════════
#  Main Application
# ═════════════════════════════════════════════════════════════════════════════

class BandStructureWorkbench:
    """Main tkinter application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BandStructure AI — Workbench")
        self.root.geometry("1300x850")
        self.root.configure(bg=BG)
        self.root.minsize(900, 600)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 11, "bold"),
                        padding=[18, 6], background=BTN_BG, foreground=FG)
        style.map("TNotebook.Tab", background=[("selected", BTN_ACTIVE)],
                  foreground=[("selected", "white")])

        self._current_file_path = ""
        self._canvas_w, self._canvas_h = CANVAS_W, CANVAS_H
        self._toolbar: Optional[FloatingToolbar] = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Show toolbar after main window is mapped
        self.root.after(200, self._show_toolbar)

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top: file bar
        top = tk.Frame(self.root, bg=TOOLBAR_BG, height=44)
        top.pack(fill="x")
        ttk.Label(top, text="Image:", background=TOOLBAR_BG,
                  font=("Segoe UI", 10)).pack(side="left", padx=(10, 5), pady=10)
        self._file_var = tk.StringVar()
        tk.Entry(top, textvariable=self._file_var, width=55, bg="#161b22", fg=FG,
                 insertbackground=FG, relief="flat",
                 font=("Segoe UI", 10)).pack(side="left", padx=4, pady=10)
        self._mk_btn(top, "Browse", self._browse_file, bg=BTN_BG).pack(side="left", padx=2)
        self._mk_btn(top, "Load", self._load_file, bg=ACCENT, fg="white").pack(side="left", padx=2)

        ttk.Separator(self.root, orient="horizontal").pack(fill="x")

        # Main: resizable horizontal split (PanedWindow)
        main = tk.PanedWindow(self.root, orient="horizontal", bg=BORDER, sashwidth=4,
                              sashrelief="raised")
        main.pack(fill="both", expand=True, padx=5, pady=5)

        # Left: Canvas area
        canvas_area = tk.Frame(main, bg=BG)
        canvas_frame = tk.Frame(canvas_area, bg=BORDER)
        canvas_frame.pack(fill="both", expand=True)
        self._canvas = DrawingCanvas(canvas_frame, on_status=self._set_status)
        self._canvas.pack(fill="both", expand=True)
        main.add(canvas_area, minsize=400, stretch="always")

        # Right: Notebook + Action buttons
        right_container = tk.Frame(main, bg=BG)
        right_container.pack_propagate(False)

        # ── Always-visible controls (no tab switching needed) ──

        # Action buttons
        action_bar = tk.Frame(right_container, bg=TOOLBAR_BG)
        action_bar.pack(fill="x", pady=(0, 3))
        self._mk_btn(action_bar, "▶  Recognize (Physics Brain)", self._submit_recognition,
                     bg="#1f6feb", fg="white", font_size=12, pady=8).pack(
                         side="left", fill="x", expand=True, padx=(2,1))
        self._mk_btn(action_bar, "💾  Save Training Data", self._save_training,
                     bg="#238636", fg="white", font_size=12, pady=8).pack(
                         side="left", fill="x", expand=True, padx=(1,2))

        # Axis calibration (always visible — needed by both modes)
        cal_frame = tk.Frame(right_container, bg=BG)
        cal_frame.pack(fill="x", pady=(0, 4))
        fields = [
            ("Y pt 1 (eV):", "y1", 0.0), ("Y pt 2 (eV):", "y2", 5.0),
            ("X pt 1 (k):",  "x1", 0.0), ("X pt 2 (k):",  "x2", 1.0),
        ]
        for i, (label, key, default) in enumerate(fields):
            row, col = i // 2, i % 2
            sub = tk.Frame(cal_frame, bg=BG)
            sub.grid(row=row, column=col, sticky="w", padx=(0, 8), pady=1)
            tk.Label(sub, text=label, bg=BG, fg=TEXT_DIM,
                     font=("Segoe UI", 9)).pack(side="left", padx=(0, 3))
            var = tk.DoubleVar(value=default)
            setattr(self, f"_{key}_var", var)
            tk.Entry(sub, textvariable=var, width=7, bg="#161b22", fg=FG,
                     relief="flat", font=("Segoe UI", 9)).pack(side="left")

        # Gap badge
        self._gap_var = tk.StringVar(value="Gap type: draw VBM+CBM")
        tk.Label(right_container, textvariable=self._gap_var, bg="#1a1a2e", fg="#8b949e",
                 font=("Segoe UI", 10, "bold"), wraplength=340, pady=5).pack(
                     fill="x", pady=(0, 4))

        # Training info (always visible, compact)
        train_info_row = tk.Frame(right_container, bg=BG)
        train_info_row.pack(fill="x", pady=(0, 4))
        tk.Label(train_info_row, text="ID:", bg=BG, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 3))
        self._mat_id_var = tk.StringVar()
        tk.Entry(train_info_row, textvariable=self._mat_id_var, width=14, bg="#161b22", fg=FG,
                 relief="flat", font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))
        tk.Label(train_info_row, text="Label:", bg=BG, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 3))
        self._gap_label_var = tk.StringVar()
        tk.Entry(train_info_row, textvariable=self._gap_label_var, width=14, bg="#161b22", fg=FG,
                 relief="flat", font=("Segoe UI", 9)).pack(side="left")

        ttk.Separator(right_container, orient="horizontal").pack(fill="x", pady=4)

        # Notebook for supplementary controls (results, etc.)
        self._notebook = ttk.Notebook(right_container)
        self._notebook.pack(fill="both", expand=True)
        self._build_recognition_tab()
        self._build_training_tab()
        main.add(right_container, minsize=350, stretch="never")

        # Status bar
        self._status_var = tk.StringVar(value="Ready — Open an image to begin.")
        tk.Label(self.root, textvariable=self._status_var, anchor="w",
                 bg=TOOLBAR_BG, fg=TEXT_DIM, font=("Segoe UI", 9),
                 padx=10, pady=3).pack(fill="x", side="bottom")

    def _show_toolbar(self):
        if self._toolbar is None:
            self._toolbar = FloatingToolbar(
                self.root,
                on_tool_select=self._canvas.set_tool,
                on_undo=self._canvas.undo,
                on_clear=self._canvas.clear_all,
                on_zoom_in=lambda: self._canvas.zoom_in() or self._sync_zoom_label(),
                on_zoom_out=lambda: self._canvas.zoom_out() or self._sync_zoom_label(),
                on_zoom_fit=lambda: self._canvas.zoom_fit() or self._sync_zoom_label(),
                on_zoom_100=lambda: self._canvas.zoom_100() or self._sync_zoom_label(),
            )

    def _sync_zoom_label(self):
        if self._toolbar:
            self._toolbar.update_zoom_label(self._canvas._zoom)

    def _build_recognition_tab(self):
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Results  ")
        right = tk.Frame(tab, bg=BG)
        right.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(right, text="Physics Brain output (click Recognize above):",
                  background=BG, foreground=TEXT_DIM,
                  font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 3))

        self._recog_result = tk.Text(right, bg="#161b22", fg=FG, wrap="word",
                                     relief="flat", font=("Consolas", 9),
                                     padx=8, pady=8)
        self._recog_result.pack(fill="both", expand=True)
        self._recog_result.insert("1.0", "After recognition, a pop-up window will\n"
                                  "show results in large, readable text.\n\n"
                                  "This panel shows a quick preview.")

        self._update_gap_badge_loop()

    def _build_training_tab(self):
        tab = ttk.Frame(self._notebook)
        self._notebook.add(tab, text="  Training Log  ")
        right = tk.Frame(tab, bg=BG)
        right.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(right, text="Training export log (click Save above):",
                  background=BG, foreground=TEXT_DIM,
                  font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 3))

        self._train_result = tk.Text(right, bg="#161b22", fg=FG, wrap="word",
                                     relief="flat", font=("Consolas", 9),
                                     padx=8, pady=8)
        self._train_result.pack(fill="both", expand=True)
        self._train_result.insert("1.0", "Fill in Material ID and Gap Label above,\n"
                                  "then click 'Save Training Data'.\n\n"
                                  "Annotations are stored in IMAGE coordinates\n"
                                  "-- zoom/pan doesn't affect saved positions.")

    # ── Button helper ──
    def _mk_btn(self, parent, text, command, bg=BTN_BG, fg=FG, padx=12, pady=5,
                font_size=10):
        btn = tk.Button(parent, text=text, command=command,
                        bg=bg, fg=fg, activebackground=BTN_HOVER, activeforeground="white",
                        relief="flat", font=("Segoe UI", font_size),
                        padx=padx, pady=pady, cursor="hand2", borderwidth=0)
        btn.bind("<Enter>", lambda e, b=btn: b.configure(bg=BTN_HOVER))
        btn.bind("<Leave>", lambda e, b=btn, orig=bg: b.configure(bg=orig))
        return btn

    # ── File ops ──
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Open band-structure image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.pdf *.gif"),
                       ("All", "*.*")])
        if path:
            self._file_var.set(path)

    def _load_file(self):
        path = self._file_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Not found", "Select a valid image file.")
            return
        self._set_status(f"Loading: {os.path.basename(path)} ...")
        self.root.update_idletasks()
        try:
            if path.lower().endswith(".pdf"):
                self._load_pdf(path)
            else:
                pil_img = Image.open(path)
                self._canvas.load_image(pil_img)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self._current_file_path = path
        self._canvas_w, self._canvas_h = self._canvas.get_canvas_dims()
        self._sync_zoom_label()
        self._set_status(f"Loaded: {os.path.basename(path)} "
                         f"({self._canvas_w}x{self._canvas_h}) — "
                         f"Mouse-wheel=zoom, Middle-click=pan")
        self.root.after(100, lambda: self._run_cv(path))

    def _load_pdf(self, path: str):
        try:
            import fitz
            doc = fitz.open(path)
            pix = doc[0].get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            doc.close()
        except ImportError:
            img = Image.open(path).convert("RGB")
        self._canvas.load_image(img)

    def _run_cv(self, path: str):
        try:
            from src.vision.multi_format_parser import MultiFormatParser
            det = str(DETECTOR_PATH) if DETECTOR_PATH.exists() else None
            parsed = MultiFormatParser(detector_path=det).parse(path)
            n = 0
            if parsed and hasattr(parsed, "metadata"):
                pv = getattr(parsed.metadata, "get", lambda _: None)(
                    "vision_detector_panel_count", None)
                n = int(pv) if isinstance(pv, (int, float)) else 0
            self._set_status(f"CV: {n} panel(s) detected — annotate above.")
        except Exception:
            self._set_status("CV skipped — use manual annotation.")

    # ── Recognition ──
    def _submit_recognition(self):
        ann_json = self._canvas.get_annotation_json()
        ann_data = json.loads(ann_json)

        if not ann_data.get("panel"):
            self._set_recog("Missing: draw the yellow panel box.")
            return
        if ann_data.get("fermi_y") is None:
            self._set_recog("Missing: click the Fermi level line.")
            return
        if len(ann_data.get("xaxis_pts", [])) < 2:
            self._set_recog("Missing: click two X-axis endpoints.")
            return
        if len(ann_data.get("yaxis_pts", [])) < 2:
            self._set_recog("Missing: click two Y-axis endpoints.")
            return
        if not ann_data.get("vbm") or not ann_data.get("cbm"):
            self._set_recog("Missing: click VBM and CBM extrema.")
            return

        try:
            from src.vision.physics_reconstructor import PhysicsReconstructor
            rec = PhysicsReconstructor()

            panel = ann_data["panel"]
            panel_bbox = [panel[0], panel[1], panel[0]+panel[2], panel[1]+panel[3]]

            y_calib = [
                {"y": float(ann_data["yaxis_pts"][0][1]), "value": self._y1_var.get()},
                {"y": float(ann_data["yaxis_pts"][1][1]), "value": self._y2_var.get()},
            ]
            x_calib = [
                {"x": float(ann_data["xaxis_pts"][0][0]), "value": self._x1_var.get()},
                {"x": float(ann_data["xaxis_pts"][1][0]), "value": self._x2_var.get()},
            ]

            vb_points = [{"x": p[0], "y": p[1]} for s in ann_data.get("vb_strokes", []) for p in s]
            cb_points = [{"x": p[0], "y": p[1]} for s in ann_data.get("cb_strokes", []) for p in s]

            tensor = rec.reconstruct_from_manual(
                source_path=self._current_file_path,
                panel_bbox=panel_bbox,
                y_calibration=y_calib, x_calibration=x_calib,
                vbm_pixel={"x": ann_data["vbm"][0], "y": ann_data["vbm"][1]},
                cbm_pixel={"x": ann_data["cbm"][0], "y": ann_data["cbm"][1]},
                valence_points=vb_points, conduction_points=cb_points,
                fermi_y_pixel=float(ann_data["fermi_y"]),
            )

            brain = _get_brain()
            result = brain.predict(tensor)

            gap_ev = result.line_mode_gap_ev
            gap_type = result.predicted_type
            dp = result.type_probabilities.get("direct", 0.0)
            vbm_m = result.vbm_effective_mass_proxy
            cbm_m = result.cbm_effective_mass_proxy
            recs = result.application_recommendations

            # ── Material Classification (elements, crystal structure, type) ──
            from src.vision.material_classifier import MaterialClassifier
            classifier = MaterialClassifier()
            is_direct = gap_type == "direct"
            mat_class = classifier.classify(
                gap_ev=gap_ev, is_direct=is_direct,
                vbm_mass=vbm_m, cbm_mass=cbm_m,
                material_id=self._mat_id_var.get().strip(),
                file_path=self._current_file_path,
            )
            class_text = mat_class.to_text()

            canvas_type = ann_data.get("gap_type", "?")
            badge = "DIRECT" if canvas_type == "direct" else "INDIRECT"

            # Build result text
            lines = [
                "╔══════════════════════════════╗",
                "║   PHYSICS BRAIN RESULT       ║",
                "╚══════════════════════════════╝",
                "",
                f"  Band gap:        {gap_ev:.3f} eV",
                f"  Gap type:        {badge} (annotated), {gap_type} (model)",
                f"  Direct prob:     {dp:.1%}",
            ]
            if vbm_m is not None and abs(vbm_m) < 100:
                lines.append(f"  VB curvature proxy: {vbm_m:.4f} (relative)")
            if cbm_m is not None and abs(cbm_m) < 100:
                lines.append(f"  CB curvature proxy: {cbm_m:.4f} (relative)")
            if recs:
                lines.append("")
                lines.append("  Recommendations:")
                for r in recs[:5]:
                    lines.append(f"    • {r}")
            result_text = "\n".join(lines)

            # Append Material Classification
            result_text += "\n\n" + class_text

            # Update inline preview
            self._set_recog(result_text)

            # Build figures
            curv_fig = self._make_curvature_figure(
                tensor.flat_tensor[0].numpy() if hasattr(tensor.flat_tensor, 'numpy')
                else np.asarray(tensor.flat_tensor[0]))
            tsne_fig = self._make_tsne_figure()

            # Pop out results window
            ResultsWindow(self.root, f"Gap={gap_ev:.3f}eV {badge}",
                          result_text, curv_fig, tsne_fig)

            # Save
            ts = time.strftime("%Y%m%d_%H%M%S")
            (OUTPUT_DIR / f"prediction_{ts}.json").write_text(
                json.dumps({"timestamp": ts, "annotations": ann_data,
                            "prediction": {"gap_ev": gap_ev, "gap_type": gap_type,
                                           "direct_prob": dp, "vbm_mass": vbm_m,
                                           "cbm_mass": cbm_m, "recs": recs},
                            "material_classification": {
                                "formula": mat_class.formula,
                                "elements": mat_class.elements,
                                "crystal_system": mat_class.crystal_system,
                                "bravais_lattice": mat_class.bravais_lattice,
                                "spacegroup_number": mat_class.spacegroup_number,
                                "crystal_type": mat_class.crystal_type,
                                "confidence": mat_class.confidence,
                            }},
                           indent=2, ensure_ascii=True))
            self._set_status(f"Done: {gap_ev:.3f} eV {badge} — results window opened.")

        except Exception as e:
            import traceback
            self._set_recog(f"Error:\n{traceback.format_exc()[-500:]}")

    def _make_curvature_figure(self, flat: np.ndarray) -> Figure:
        fig = Figure(figsize=(5, 3.5), dpi=90, facecolor=BG)
        names = ["VBM_E", "VBM_curv", "VBM_k_dist", "CBM_E", "CBM_curv", "CBM_k_dist"]
        for i, n in enumerate(names):
            ax = fig.add_subplot(2, 3, i+1)
            ax.plot(flat[:, i], color="steelblue", linewidth=1)
            ax.set_title(n, fontsize=7, color=FG)
            ax.tick_params(labelsize=5, colors=TEXT_DIM)
            ax.set_facecolor("#161b22")
            for s in ax.spines.values():
                s.set_color(BORDER)
        fig.tight_layout(pad=1.5)
        return fig

    def _make_tsne_figure(self) -> Optional[Figure]:
        if not TSNE_IMAGE.exists():
            return None
        fig = Figure(figsize=(5, 3.5), dpi=90, facecolor=BG)
        ax = fig.add_subplot(111)
        ax.imshow(Image.open(TSNE_IMAGE))
        ax.axis("off")
        ax.set_title("t-SNE by spacegroup", fontsize=8, color=FG)
        fig.tight_layout()
        return fig

    # ── Training ──
    def _save_training(self):
        ann_json = self._canvas.get_annotation_json()
        ann_data = json.loads(ann_json)
        if not ann_json or ann_json == "{}":
            self._set_train("No annotations to save.")
            return
        gap_type = ann_data.get("gap_type", "")
        if ann_data.get("vbm") and ann_data.get("cbm"):
            gap_type = "direct" if abs(ann_data["vbm"][0] - ann_data["cbm"][0]) <= 5 else "indirect"

        ts = time.strftime("%Y%m%d_%H%M%S")
        mid = self._mat_id_var.get().strip().replace(" ", "_") or "unknown"
        rid = f"{mid}_{ts}"

        cw, ch = self._canvas_w, self._canvas_h
        record = {
            "record_id": rid, "timestamp": ts,
            "material_id": self._mat_id_var.get().strip(),
            "gap_label": self._gap_label_var.get().strip(),
            "gap_type": gap_type,
            "source_image_path": self._current_file_path,
            "canvas_width": cw, "canvas_height": ch,
            "annotations": ann_data,
            "calibration": {"y_values": [self._y1_var.get(), self._y2_var.get()],
                            "x_values": [self._x1_var.get(), self._x2_var.get()]},
        }
        (TRAINING_DATA_DIR / f"{rid}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=True))
        with open(TRAINING_DATA_DIR / "training_manifest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

        n = len(list(TRAINING_DATA_DIR.glob("*.json")))
        self._set_train(f"Saved! {n} annotations total.\n"
                        f"File: {rid}.json\n"
                        f"Gap type: {'DIRECT' if gap_type=='direct' else 'INDIRECT'}\n\n"
                        f"Run training: python scripts/train_from_human_annotations.py")
        self._set_status(f"Saved: {rid}.json ({n} total)")

    # ── Helpers ──
    def _set_status(self, msg): self._status_var.set(msg)
    def _set_recog(self, text): self._recog_result.delete("1.0", "end"); self._recog_result.insert("1.0", text)
    def _set_train(self, text): self._train_result.delete("1.0", "end"); self._train_result.insert("1.0", text)

    def _update_gap_badge_loop(self):
        ann = self._canvas.annotations
        if ann["vbm"] and ann["cbm"]:
            dx = abs(ann["vbm"][0] - ann["cbm"][0])
            self._gap_var.set(f"{'DIRECT' if dx<=5 else 'INDIRECT'} "
                              f"({'VBM/CBM aligned' if dx<=5 else f'offset {dx:.1f}px'})")
        self.root.after(500, self._update_gap_badge_loop)

    def _on_close(self):
        if self._toolbar:
            self._toolbar.destroy()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = BandStructureWorkbench()
    app.run()

if __name__ == "__main__":
    main()
