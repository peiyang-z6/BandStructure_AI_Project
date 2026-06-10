"""Synthetic Sim2Real band-plot data generator.

Stage 1 uses the existing Materials Project cache to render degraded band
figures with perfect geometry labels. It exports both YOLO-style labels and a
COCO keypoint JSON so a detector can learn plot panels and extrema before any
real-paper fine-tuning.

Optional detector training is intentionally external. Install a detector stack
only when needed, for example:

    pip install ultralytics
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np


@dataclass
class SyntheticConfig:
    h5_path: str = "data_cache/mp_bands.h5"
    output_dir: str = "data_cache/vision_synthetic"
    count: int = 5000
    image_size: Tuple[int, int] = (768, 768)
    random_state: int = 42
    jpeg_quality_min: int = 10
    jpeg_quality_max: int = 50
    val_fraction: float = 0.2


class SyntheticBandPlotGenerator:
    def __init__(self, config: SyntheticConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_state)
        random.seed(config.random_state)

    def generate(self) -> Dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc

        out = Path(self.config.output_dir)
        img_train_dir = out / "images" / "train"
        img_val_dir = out / "images" / "val"
        label_train_dir = out / "labels" / "train"
        label_val_dir = out / "labels" / "val"
        for folder in (img_train_dir, img_val_dir, label_train_dir, label_val_dir):
            folder.mkdir(parents=True, exist_ok=True)

        material_ids = self._material_ids()
        coco = self._empty_coco()
        manifest: List[Dict[str, Any]] = []
        for idx in range(self.config.count):
            mid = material_ids[idx % len(material_ids)]
            bands, meta = self._load_bands(mid)
            image, ann = self._render_sample(bands, meta)
            image, ann = self._degrade(image, ann)

            stem = f"synthetic_{idx:05d}_{mid.replace('-', '_')}"
            is_val = (idx % max(int(round(1.0 / max(self.config.val_fraction, 1e-6))), 2)) == 0
            img_dir = img_val_dir if is_val else img_train_dir
            label_dir = label_val_dir if is_val else label_train_dir
            image_path = img_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            quality = int(self.rng.integers(self.config.jpeg_quality_min, self.config.jpeg_quality_max + 1))
            cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            label_path.write_text(self._yolo_label(ann), encoding="utf-8")
            self._append_coco(coco, idx, image_path, image.shape, ann)
            manifest.append({"material_id": mid, "image": str(image_path), "label": str(label_path), **ann["physics"]})

        (out / "annotations_coco_keypoints.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (out / "dataset.yaml").write_text(self._dataset_yaml(label_train_dir, label_val_dir), encoding="utf-8")
        return {"output_dir": str(out), "count": self.config.count, "coco": str(out / "annotations_coco_keypoints.json")}

    def _material_ids(self) -> List[str]:
        with h5py.File(self.config.h5_path, "r") as h5:
            ids = list(h5.keys())
        if not ids:
            raise ValueError(f"No materials found in {self.config.h5_path}")
        self.rng.shuffle(ids)
        return ids

    def _load_bands(self, material_id: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        with h5py.File(self.config.h5_path, "r") as h5:
            grp = h5[material_id]
            energies = np.asarray(grp["energies"], dtype=np.float32)
            attrs = dict(grp.attrs)
        return energies, attrs

    def _render_sample(self, energies: np.ndarray, meta: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc

        h, w = self.config.image_size
        image = np.ones((h, w, 3), dtype=np.uint8) * int(self.rng.integers(230, 256))
        objects: List[Dict[str, Any]] = []

        plots = self._sample_plot_layout(w, h)
        self._draw_page_distractors(image, plots)
        for plot in plots:
            objects.append(self._draw_band_panel(image, plot, energies, meta))

        primary = objects[0]
        return image, {
            "bbox_xyxy": primary["bbox_xyxy"],
            "keypoints": primary["keypoints"],
            "physics": primary["physics"],
            "objects": objects,
        }

    def _sample_plot_layout(self, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Create one or more band-plot panels inside a paper-like page.

        Multi-panel layouts are deliberately common so the detector learns that
        one source image may contain several independent band-structure plots.
        """

        layouts: List[List[Tuple[float, float, float, float]]] = [
            [(0.08, 0.08, 0.92, 0.90)],
            [(0.07, 0.12, 0.48, 0.56), (0.54, 0.12, 0.94, 0.56)],
            [(0.08, 0.08, 0.46, 0.45), (0.54, 0.08, 0.92, 0.45), (0.08, 0.55, 0.46, 0.92), (0.54, 0.55, 0.92, 0.92)],
            [(0.47, 0.06, 0.78, 0.35), (0.47, 0.38, 0.78, 0.67)],
            [(0.52, 0.08, 0.93, 0.30), (0.52, 0.34, 0.93, 0.56), (0.52, 0.60, 0.93, 0.84)],
        ]
        weights = np.array([0.30, 0.22, 0.20, 0.14, 0.14], dtype=float)
        idx = int(self.rng.choice(len(layouts), p=weights / weights.sum()))
        jitter = float(self.rng.uniform(0.0, 0.025))
        plots: List[Tuple[int, int, int, int]] = []
        for x0, y0, x1, y1 in layouts[idx]:
            dx0, dy0, dx1, dy1 = self.rng.normal(0, jitter, size=4)
            px0 = int(np.clip((x0 + dx0) * w, 12, w - 40))
            py0 = int(np.clip((y0 + dy0) * h, 12, h - 40))
            px1 = int(np.clip((x1 + dx1) * w, px0 + 80, w - 12))
            py1 = int(np.clip((y1 + dy1) * h, py0 + 80, h - 12))
            plots.append((px0, py0, px1, py1))
        return plots

    def _draw_band_panel(
        self,
        image: np.ndarray,
        plot: Tuple[int, int, int, int],
        energies: np.ndarray,
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc

        x0, y0, x1, y1 = plot
        cv2.rectangle(image, (x0, y0), (x1, y1), (20, 20, 20), 2)

        bands = self._select_visible_bands(energies)
        e_min, e_max = float(np.nanmin(bands)), float(np.nanmax(bands))
        if e_max - e_min < 1e-3:
            e_min -= 1.0
            e_max += 1.0

        if self.rng.random() < 0.75:
            self._draw_grid(image, plot)

        for band in bands:
            pts = self._band_to_pixels(band, plot, e_min, e_max)
            if self.rng.random() < 0.30:
                palette = [(210, 110, 10), (40, 120, 210), (190, 50, 65), (60, 150, 85)]
                color = palette[int(self.rng.integers(0, len(palette)))]
            else:
                tone = int(self.rng.integers(0, 95))
                color = (tone, tone, tone)
            thickness = int(self.rng.integers(1, 4))
            line_type = cv2.LINE_AA
            if self.rng.random() < 0.35:
                self._draw_dashed_polyline(image, pts, color, thickness)
            else:
                cv2.polylines(image, [pts.astype(np.int32)], False, color, thickness, line_type)

        vbm_band = bands[np.argmax(np.max(bands, axis=1))]
        cbm_band = bands[np.argmin(np.min(bands, axis=1))]
        vbm_idx = int(np.argmax(vbm_band))
        cbm_idx = int(np.argmin(cbm_band))
        vbm_xy = self._band_to_pixels(vbm_band, plot, e_min, e_max)[vbm_idx]
        cbm_xy = self._band_to_pixels(cbm_band, plot, e_min, e_max)[cbm_idx]
        keypoints = {
            "vbm": (float(vbm_xy[0]), float(vbm_xy[1])),
            "cbm": (float(cbm_xy[0]), float(cbm_xy[1])),
        }
        if self.rng.random() < 0.65:
            self._draw_axis_text(image, plot)

        return {
            "bbox_xyxy": [float(x0), float(y0), float(x1), float(y1)],
            "keypoints": keypoints,
            "physics": {
                "band_gap": float(np.min(cbm_band) - np.max(vbm_band)),
                "spacegroup_number": int(meta.get("spacegroup_number", -1)),
            },
        }

    def _draw_page_distractors(self, image: np.ndarray, plots: List[Tuple[int, int, int, int]]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc

        h, w = image.shape[:2]
        occupied = [tuple(map(int, p)) for p in plots]
        if self.rng.random() < 0.65:
            # DOS-like panels: axes plus noisy filled/colored traces, intentionally not labeled.
            for _ in range(int(self.rng.integers(1, 4))):
                x0 = int(self.rng.integers(15, max(16, w - 170)))
                y0 = int(self.rng.integers(15, max(16, h - 180)))
                x1 = min(w - 12, x0 + int(self.rng.integers(80, 150)))
                y1 = min(h - 12, y0 + int(self.rng.integers(110, 220)))
                if self._overlaps_any((x0, y0, x1, y1), occupied, pad=18):
                    continue
                cv2.rectangle(image, (x0, y0), (x1, y1), (80, 80, 80), 1)
                xs = np.linspace(x0 + 4, x1 - 4, 90)
                for c in [(80, 80, 80), (60, 130, 60), (80, 80, 180)]:
                    noise = self.rng.normal(0, 0.22, size=90).cumsum()
                    noise = (noise - noise.min()) / max(noise.ptp(), 1e-6)
                    ys = y1 - noise * (y1 - y0 - 8) - 4
                    cv2.polylines(image, [np.stack([xs, ys], axis=1).astype(np.int32)], False, c, 1, cv2.LINE_AA)
        if self.rng.random() < 0.75:
            for _ in range(int(self.rng.integers(2, 8))):
                x = int(self.rng.integers(5, w - 90))
                y = int(self.rng.integers(18, h - 8))
                if self._overlaps_any((x, y - 18, x + 90, y + 4), occupied, pad=8):
                    continue
                cv2.putText(image, random.choice(["DOS", "(a)", "Energy", "VBM", "CBM", "GGA+SOC", "Total"]),
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, float(self.rng.uniform(0.35, 0.7)),
                            (40, 40, 40), int(self.rng.integers(1, 3)), cv2.LINE_AA)
        if self.rng.random() < 0.35:
            # Structure-like colored blobs that should be ignored by the detector.
            for _ in range(int(self.rng.integers(8, 22))):
                x = int(self.rng.integers(10, w - 10))
                y = int(self.rng.integers(10, h - 10))
                if self._overlaps_any((x - 12, y - 12, x + 12, y + 12), occupied, pad=5):
                    continue
                color = tuple(int(v) for v in self.rng.integers(30, 220, size=3))
                cv2.circle(image, (x, y), int(self.rng.integers(4, 11)), color, -1, cv2.LINE_AA)

    def _draw_axis_text(self, image: np.ndarray, plot: Tuple[int, int, int, int]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc

        x0, y0, x1, y1 = plot
        labels = ["G", "X", "M", "R", "Gamma", "K"]
        for i, x in enumerate(np.linspace(x0, x1, int(self.rng.integers(3, 7)))):
            cv2.putText(image, labels[i % len(labels)], (int(x) - 8, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        if x0 > 35:
            cv2.putText(image, "Energy (eV)", (max(2, x0 - 35), max(18, y0 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)

    def _overlaps_any(
        self,
        box: Tuple[int, int, int, int],
        others: List[Tuple[int, int, int, int]],
        pad: int = 0,
    ) -> bool:
        x0, y0, x1, y1 = box
        for ox0, oy0, ox1, oy1 in others:
            if x0 < ox1 + pad and x1 > ox0 - pad and y0 < oy1 + pad and y1 > oy0 - pad:
                return True
        return False

    def _select_visible_bands(self, energies: np.ndarray) -> np.ndarray:
        if energies.ndim != 2:
            energies = energies.reshape(energies.shape[0], -1)
        n = min(int(self.rng.integers(6, 16)), energies.shape[0])
        order = np.argsort(np.nanmean(energies, axis=1))
        center = len(order) // 2
        start = max(0, center - n // 2)
        return energies[order[start : start + n]]

    def _band_to_pixels(self, band: np.ndarray, plot: Tuple[int, int, int, int], e_min: float, e_max: float) -> np.ndarray:
        x0, y0, x1, y1 = plot
        xs = np.linspace(x0, x1, len(band), dtype=np.float32)
        ys = y1 - ((band - e_min) / max(e_max - e_min, 1e-6)) * (y1 - y0)
        return np.stack([xs, ys], axis=1)

    def _draw_grid(self, image: np.ndarray, plot: Tuple[int, int, int, int]) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc

        x0, y0, x1, y1 = plot
        color = int(self.rng.integers(120, 220))
        for x in np.linspace(x0, x1, int(self.rng.integers(4, 9))):
            cv2.line(image, (int(x), y0), (int(x), y1), (color, color, color), 1)
        for y in np.linspace(y0, y1, int(self.rng.integers(4, 9))):
            cv2.line(image, (x0, int(y)), (x1, int(y)), (color, color, color), 1)

    def _draw_dashed_polyline(self, image: np.ndarray, pts: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc

        for i in range(len(pts) - 1):
            if i % 4 in (0, 1):
                cv2.line(image, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), color, thickness, cv2.LINE_AA)

    def _degrade(self, image: np.ndarray, ann: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc

        h, w = image.shape[:2]
        if self.rng.random() < 0.72:
            jitter = self.rng.normal(0, 0.025, size=(4, 2)).astype(np.float32)
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = src + jitter * np.float32([w, h])
            m = cv2.getPerspectiveTransform(src, dst)
            image = cv2.warpPerspective(image, m, (w, h), borderValue=(255, 255, 255))
            ann = self._transform_annotation(ann, m, w, h)
        if self.rng.random() < 0.85:
            noise = self.rng.normal(0, self.rng.uniform(4, 20), image.shape).astype(np.float32)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if self.rng.random() < 0.65:
            for _ in range(int(self.rng.integers(1, 5))):
                x0 = int(self.rng.integers(0, w - 20))
                y0 = int(self.rng.integers(0, h - 20))
                x1 = min(w, x0 + int(self.rng.integers(20, w // 4)))
                y1 = min(h, y0 + int(self.rng.integers(12, h // 5)))
                cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), -1)
        return image, ann

    def _transform_annotation(self, ann: Dict[str, Any], matrix: np.ndarray, w: int, h: int) -> Dict[str, Any]:
        transformed_objects = []
        for obj in ann.get("objects", [ann]):
            x0, y0, x1, y1 = obj["bbox_xyxy"]
            corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
            warped = self._perspective_points(corners, matrix).reshape(-1, 2)
            nx0 = float(np.clip(warped[:, 0].min(), 0, w - 1))
            ny0 = float(np.clip(warped[:, 1].min(), 0, h - 1))
            nx1 = float(np.clip(warped[:, 0].max(), 0, w - 1))
            ny1 = float(np.clip(warped[:, 1].max(), 0, h - 1))
            keypoints: Dict[str, Tuple[float, float]] = {}
            for name, (kx, ky) in obj["keypoints"].items():
                pt = np.float32([[kx, ky]]).reshape(-1, 1, 2)
                wk = self._perspective_points(pt, matrix).reshape(-1, 2)[0]
                keypoints[name] = (float(np.clip(wk[0], 0, w - 1)), float(np.clip(wk[1], 0, h - 1)))
            transformed = {**obj, "bbox_xyxy": [nx0, ny0, nx1, ny1], "keypoints": keypoints}
            transformed_objects.append(transformed)
        primary = transformed_objects[0]
        return {
            "bbox_xyxy": primary["bbox_xyxy"],
            "keypoints": primary["keypoints"],
            "physics": primary["physics"],
            "objects": transformed_objects,
        }

    def _perspective_points(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for synthetic vision data generation") from exc
        return cv2.perspectiveTransform(points, matrix)

    def _yolo_label(self, ann: Dict[str, Any]) -> str:
        h, w = self.config.image_size
        lines = []
        for obj in ann.get("objects", [ann]):
            x0, y0, x1, y1 = obj["bbox_xyxy"]
            xc = ((x0 + x1) * 0.5) / w
            yc = ((y0 + y1) * 0.5) / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h
            kp = obj["keypoints"]
            values = [0, xc, yc, bw, bh]
            for name in ("vbm", "cbm"):
                x, y = kp[name]
                values.extend([x / w, y / h, 2])
            lines.append(" ".join(f"{v:.8f}" if isinstance(v, float) else str(v) for v in values))
        return "\n".join(lines) + "\n"

    def _empty_coco(self) -> Dict[str, Any]:
        return {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "band_plot", "keypoints": ["vbm", "cbm"], "skeleton": []}
            ],
        }

    def _append_coco(self, coco: Dict[str, Any], idx: int, image_path: Path, shape: Tuple[int, ...], ann: Dict[str, Any]) -> None:
        h, w = shape[:2]
        coco["images"].append({"id": idx, "file_name": image_path.name, "width": w, "height": h})
        for obj_idx, obj in enumerate(ann.get("objects", [ann])):
            x0, y0, x1, y1 = obj["bbox_xyxy"]
            kps: List[float] = []
            for name in ("vbm", "cbm"):
                x, y = obj["keypoints"][name]
                kps.extend([x, y, 2])
            coco["annotations"].append(
                {
                    "id": idx * 100 + obj_idx,
                    "image_id": idx,
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                    "num_keypoints": 2,
                    "keypoints": kps,
                }
            )

    def _dataset_yaml(self, label_train_dir: Path, label_val_dir: Path) -> str:
        return f"""path: {Path(self.config.output_dir).resolve().as_posix()}
train: images/train
val: images/val
names:
  0: band_plot
kpt_shape: [2, 3]
# YOLO labels: {label_train_dir.resolve().as_posix()} and {label_val_dir.resolve().as_posix()}
"""


def train_yolov8_pose(data_yaml: str, model: str = "yolov8n-pose.pt", epochs: int = 50) -> None:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YOLOv8 training requires ultralytics. Run: pip install ultralytics") from exc
    YOLO(model).train(data=data_yaml, epochs=epochs, task="pose")
