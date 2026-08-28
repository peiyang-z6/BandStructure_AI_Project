"""Multimodal band-plot parsers for Phase 5 Plot-to-Physics.

The parser extracts the visual plot panel and curve skeleton from figures. It
intentionally does not infer Fermi level, VBM/CBM, or direct/indirect physics.
Those physical decisions are made by human calibration in the GUI and then by
the Physics Brain.

PDF/EPS support is optional. Install PyMuPDF manually if vector parsing is
needed:

    pip install PyMuPDF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".gif"}
VECTOR_SUFFIXES = {".pdf", ".eps"}


@dataclass
class ParsedBandData:
    """Geometry extracted from a band-structure carrier."""

    source_path: str
    source_type: str
    k: np.ndarray
    energy: np.ndarray
    band_hint: np.ndarray
    fermi_energy: float = 0.0
    kpath_labels: List[Dict[str, Any]] = field(default_factory=list)
    skeleton_pixels: Optional[np.ndarray] = None
    overlay_image: Optional[np.ndarray] = None
    frame_gaps: Optional[np.ndarray] = None
    frame_numbers: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_points(self) -> np.ndarray:
        return np.stack([self.k, self.energy], axis=1).astype(np.float32)


class MultiFormatParser:
    """Extract `(k, E)` band traces from PDF/EPS, raster images, and videos."""

    def __init__(
        self,
        image_height_ev: float = 16.0,
        target_k_points: int = 128,
        max_video_frames: int = 48,
        threshold: int = 190,
        detector_path: Optional[str] = "artifacts/models/vision_detector/band_plot_yolov8_pose_best.pt",
        detector_conf: float = 0.05,
    ) -> None:
        self.image_height_ev = float(image_height_ev)
        self.target_k_points = int(target_k_points)
        self.max_video_frames = int(max_video_frames)
        self.threshold = int(threshold)
        self.detector_path = str(detector_path) if detector_path else ""
        self.detector_conf = float(detector_conf)
        self._detector = None

    def parse(self, path: str) -> ParsedBandData:
        suffix = Path(path).suffix.lower()
        if suffix in VECTOR_SUFFIXES:
            return self.parse_vector(path)
        if suffix in IMAGE_SUFFIXES:
            return self.parse_image(path)
        if suffix in VIDEO_SUFFIXES:
            return self.parse_video(path)
        raise ValueError(f"Unsupported input format: {suffix}")

    def parse_vector(self, path: str) -> ParsedBandData:
        """Parse PDF path/line drawings when PyMuPDF is available."""

        suffix = Path(path).suffix.lower()
        if suffix == ".eps":
            raise RuntimeError("EPS vector extraction requires converting EPS to PDF before parsing.")

        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PDF vector parsing requires PyMuPDF. Run: pip install PyMuPDF") from exc

        doc = fitz.open(path)
        if doc.page_count == 0:
            raise ValueError(f"No pages found in {path}")

        page = doc[0]
        drawings = page.get_drawings()
        points: List[List[float]] = []
        for drawing in drawings:
            for item in drawing.get("items", []):
                op = item[0]
                if op == "l":
                    p0, p1 = item[1], item[2]
                    points.extend([[float(p0.x), float(p0.y)], [float(p1.x), float(p1.y)]])
                elif op == "re":
                    rect = item[1]
                    points.extend(
                        [
                            [float(rect.x0), float(rect.y0)],
                            [float(rect.x1), float(rect.y0)],
                            [float(rect.x1), float(rect.y1)],
                            [float(rect.x0), float(rect.y1)],
                        ]
                    )
                elif op == "c":
                    for p in item[1:]:
                        points.append([float(p.x), float(p.y)])

        if len(points) < 4:
            raise ValueError("No usable vector path points were detected in the PDF")

        xy = np.asarray(points, dtype=np.float32)
        return self._coords_to_band_data(
            path,
            "vector",
            xy,
            image_shape=(float(page.rect.height), float(page.rect.width)),
            metadata={"vector_points": int(len(xy)), "parser": "PyMuPDF"},
        )

    def parse_image(self, path: str) -> ParsedBandData:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image Plot-to-Physics parsing") from exc

        image = self._imread_unicode(path)
        if image is None:
            raise FileNotFoundError(path)

        panel, offset, panel_meta = self._select_physical_panel(image)
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        binary, mask_mode = self._auto_band_mask(panel)
        skeleton = self._skeletonize(binary)
        ys, xs = np.where(skeleton > 0)
        if len(xs) < 8:
            raise ValueError("No band-like skeleton was detected")

        overlay = image.copy()
        x0, y0 = offset
        self._draw_detection_overlay(overlay, panel_meta)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        parsed = self._coords_to_band_data(
            path,
            "image",
            coords,
            image_shape=panel.shape[:2],
            skeleton_pixels=coords,
            overlay_image=overlay,
            panel_image=panel,
            metadata={"skeleton_pixels": int(len(coords)), "mask_mode": mask_mode, **panel_meta},
        )
        self._draw_skeleton_overlay(overlay, panel.shape[:2], offset, skeleton)
        parsed.overlay_image = overlay
        return parsed

    def _select_physical_panel(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], Dict[str, Any]]:
        """Prefer the complete coordinate-frame panel over a local detector crop."""

        geometry_panel = self._extract_band_panel(image)
        detected = self._detect_with_vision_model(image)
        if detected is None:
            return geometry_panel

        g_panel, g_offset, g_meta = geometry_panel
        d_panel, d_offset, d_meta = detected
        g_area = int(g_panel.shape[0] * g_panel.shape[1])
        d_area = int(d_panel.shape[0] * d_panel.shape[1])
        g_frame = g_meta.get("frame_crop") not in (None, "none") or "frame" in str(g_meta.get("panel_detection", ""))

        # Detector boxes can latch onto inset labels or small subplots. If the
        # geometry detector found a much larger axis-bounded frame, that is the
        # physically valid coordinate system for Plot-to-Physics.
        if g_frame and g_area >= max(int(d_area * 1.35), d_area + 5000):
            merged = {
                **g_meta,
                "panel_detection": f"{g_meta.get('panel_detection', 'geometry_frame')}_preferred_over_yolo_crop",
                "vision_detector_all_panels": d_meta.get("vision_detector_all_panels", []),
                "vision_detector_panel_count": d_meta.get("vision_detector_panel_count", 0),
                "rejected_yolo_panel_bbox": d_meta.get("panel_bbox"),
                "rejected_yolo_reason": "geometry frame is larger and axis-bounded",
            }
            return g_panel, g_offset, merged
        return detected

    def _detect_with_vision_model(self, image: np.ndarray) -> Optional[tuple[np.ndarray, tuple[int, int], Dict[str, Any]]]:
        """Use the trained pose model as a panel locator, not as the physics brain."""

        detector_path = Path(self.detector_path) if self.detector_path else None
        if detector_path is None or not detector_path.exists():
            return None

        try:
            detector = self._load_detector(detector_path)
            if detector is None:
                return None
            results = detector.predict(image, conf=self.detector_conf, verbose=False)
        except Exception:
            return None

        if not results:
            return None
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy.cpu().numpy().astype(float)
        conf = boxes.conf.cpu().numpy().astype(float) if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy))
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        h, w = image.shape[:2]
        order = np.argsort(-(conf * np.sqrt(np.maximum(areas, 1.0))))
        candidates: List[Dict[str, Any]] = []
        best_payload = None
        pad = max(4, int(0.01 * max(w, h)))

        for idx in order:
            raw_x0, raw_y0, raw_x1, raw_y1 = xyxy[int(idx)]
            x0 = max(int(round(raw_x0)) - pad, 0)
            y0 = max(int(round(raw_y0)) - pad, 0)
            x1 = min(int(round(raw_x1)) + pad, w)
            y1 = min(int(round(raw_y1)) + pad, h)

            box_w = x1 - x0
            box_h = y1 - y0
            if box_w < max(70, int(0.08 * w)) or box_h < max(60, int(0.08 * h)):
                continue
            if box_w * box_h > 0.96 * w * h:
                continue

            panel = image[y0:y1, x0:x1].copy()
            panel, inner_offset, frame_meta = self._crop_to_plot_frame(panel)
            px0 = x0 + inner_offset[0]
            py0 = y0 + inner_offset[1]
            px1 = px0 + panel.shape[1]
            py1 = py0 + panel.shape[0]
            keypoints = self._extract_detection_keypoints(result, int(idx), px0, py0, panel.shape)
            if keypoints and not all(bool(item.get("inside_panel", False)) for item in keypoints.values()):
                keypoints = {}
            candidate_score = self._score_detected_panel(panel, float(conf[int(idx)]), frame_meta)

            candidate = {
                "panel_bbox": [int(px0), int(py0), int(px1), int(py1)],
                "confidence": float(conf[int(idx)]),
                "candidate_score": float(candidate_score),
                "raw_bbox": [float(raw_x0), float(raw_y0), float(raw_x1), float(raw_y1)],
                "keypoints": keypoints,
                **frame_meta,
            }
            candidates.append(candidate)
            if best_payload is None or candidate_score > best_payload[2].get("candidate_score", -1e9):
                best_payload = (panel, (int(px0), int(py0)), candidate)

        if best_payload is None:
            return None

        panel, offset, selected = best_payload
        meta = {
            "panel_bbox": selected["panel_bbox"],
            "panel_detection": "vision_detector_yolov8_pose",
            "vision_detector_path": str(detector_path),
            "vision_detector_confidence": float(selected["confidence"]),
            "vision_detector_candidate_score": float(selected.get("candidate_score", selected["confidence"])),
            "vision_detector_keypoints": selected.get("keypoints", {}),
            "vision_detector_all_panels": candidates,
            "vision_detector_panel_count": len(candidates),
            **{k: v for k, v in selected.items() if k not in {"panel_bbox", "confidence", "raw_bbox", "keypoints"}},
        }
        return panel, offset, meta

    def _score_detected_panel(self, panel: np.ndarray, confidence: float, frame_meta: Dict[str, Any]) -> float:
        """Rank detector boxes by detector confidence and band-plot geometry.

        DOS panels and text boxes can receive high detector confidence on real
        paper composites. A true band plot usually has a recoverable rectangular
        k-path frame and curve pixels spread across many x positions.
        """

        score = float(confidence)
        if frame_meta.get("frame_crop") not in (None, "none"):
            score += 0.18
        try:
            mask, _ = self._auto_band_mask(panel)
            height, width = mask.shape[:2]
            if height > 0 and width > 0:
                col_occ = float(((mask > 0).sum(axis=0) > 0).mean())
                row_occ = float(((mask > 0).sum(axis=1) > 0).mean())
                density = float((mask > 0).sum() / max(height * width, 1))
                score += 0.22 * col_occ + 0.08 * row_occ
                if density < 0.002:
                    score -= 0.15
                if col_occ < 0.18:
                    score -= 0.18
        except Exception:
            pass
        return score

    def _load_detector(self, detector_path: Path) -> Any:
        if self._detector is not None:
            return self._detector
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            return None
        self._detector = YOLO(str(detector_path))
        return self._detector

    def _extract_detection_keypoints(
        self,
        result: Any,
        index: int,
        x0: int,
        y0: int,
        panel_shape: Optional[tuple[int, int, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        keypoints_obj = getattr(result, "keypoints", None)
        if keypoints_obj is None or getattr(keypoints_obj, "xy", None) is None:
            return {}
        xy = keypoints_obj.xy.cpu().numpy()
        conf_arr = None
        if getattr(keypoints_obj, "conf", None) is not None:
            conf_arr = keypoints_obj.conf.cpu().numpy()
        if index >= len(xy) or xy[index].shape[0] < 2:
            return {}

        names = ("vbm", "cbm")
        out: Dict[str, Dict[str, float]] = {}
        for kp_idx, name in enumerate(names):
            px, py = xy[index][kp_idx]
            item = {"x": float(px - x0), "y": float(py - y0)}
            if conf_arr is not None and index < len(conf_arr) and kp_idx < conf_arr[index].shape[0]:
                item["confidence"] = float(conf_arr[index][kp_idx])
            if panel_shape is not None:
                height, width = panel_shape[:2]
                item["inside_panel"] = bool(0.0 <= item["x"] < width and 0.0 <= item["y"] < height)
            out[name] = item
        return out

    def _draw_detection_overlay(self, overlay: np.ndarray, panel_meta: Dict[str, Any]) -> None:
        try:
            import cv2
        except ImportError:
            return

        all_panels = panel_meta.get("vision_detector_all_panels") or []
        for candidate in all_panels:
            bx0, by0, bx1, by1 = [int(v) for v in candidate.get("panel_bbox", [0, 0, 0, 0])]
            cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0, 180, 255), 1)
            label = f"{float(candidate.get('confidence', 0.0)):.2f}"
            cv2.putText(overlay, label, (bx0, max(12, by0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 120, 255), 1)

        x0, y0, x1, y1 = [int(v) for v in panel_meta.get("panel_bbox", [0, 0, 0, 0])]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 180, 255), 3)
        keypoints = panel_meta.get("vision_detector_keypoints", {})
        colors = {"vbm": (255, 0, 255), "cbm": (0, 255, 255)}
        for name, point in keypoints.items():
            px = int(round(x0 + float(point.get("x", 0.0))))
            py = int(round(y0 + float(point.get("y", 0.0))))
            cv2.circle(overlay, (px, py), 5, colors.get(name, (0, 255, 255)), -1)

    def _draw_skeleton_overlay(
        self,
        overlay: np.ndarray,
        panel_shape: tuple[int, int],
        offset: tuple[int, int],
        skeleton: np.ndarray,
    ) -> None:
        """Draw only CV-owned outputs: plot boundary and raw skeleton pixels.

        This intentionally does not draw E_F, VBM/CBM, band gaps, or
        valence/conduction labels. Those require human calibration in the GUI.
        """

        try:
            import cv2
        except ImportError:
            return

        x0, y0 = offset
        height, width = panel_shape[:2]
        x1, y1 = x0 + width, y0 + height
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 140, 0), 3)

        ys, xs = np.where(skeleton > 0)
        for px, py in zip(xs, ys):
            cv2.circle(overlay, (int(x0 + px), int(y0 + py)), 1, (0, 220, 255), -1)

    def parse_video(self, path: str) -> ParsedBandData:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for video Plot-to-Physics parsing") from exc

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.max_video_frames
        stride = max(frame_count // self.max_video_frames, 1)
        parsed_frames: List[ParsedBandData] = []
        frame_numbers: List[int] = []
        prev_gray = None
        flow_magnitudes: List[float] = []

        idx = 0
        while len(parsed_frames) < self.max_video_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue

            tmp = self._parse_frame_array(path, frame)
            parsed_frames.append(tmp)
            frame_numbers.append(idx)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_magnitudes.append(float(np.linalg.norm(flow, axis=2).mean()))
            prev_gray = gray
            idx += 1

        cap.release()
        if not parsed_frames:
            raise ValueError("No parseable frames were found in the video")

        frame_gaps = np.asarray([self._rough_gap_from_points(p) for p in parsed_frames], dtype=np.float32)
        base = parsed_frames[len(parsed_frames) // 2]
        base.frame_gaps = frame_gaps - frame_gaps[0]
        base.frame_numbers = np.asarray(frame_numbers, dtype=np.int32)
        base.source_type = "video"
        base.metadata.update(
            {
                "frames_used": len(parsed_frames),
                "delta_gap_ev": base.frame_gaps.tolist(),
                "mean_optical_flow": float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0,
            }
        )
        return base

    def _parse_frame_array(self, source_path: str, frame: np.ndarray) -> ParsedBandData:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for video Plot-to-Physics parsing") from exc

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        binary = self._band_binary_mask(gray)
        skeleton = self._skeletonize(binary)
        ys, xs = np.where(skeleton > 0)
        if len(xs) < 8:
            raise ValueError("Frame did not contain a usable band skeleton")
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        overlay = frame.copy()
        overlay[skeleton > 0] = (0, 0, 255)
        return self._coords_to_band_data(
            source_path,
            "video_frame",
            coords,
            image_shape=frame.shape[:2],
            skeleton_pixels=coords,
            overlay_image=overlay,
        )

    def _band_binary_mask(self, gray: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, dark = cv2.threshold(blurred, self.threshold, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(blurred, 60, 160)
        mask = cv2.bitwise_or(dark, edges)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _extract_band_panel(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], Dict[str, Any]]:
        """Find the main band-structure panel in screenshots or scanned plots."""
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        line_panel = self._extract_line_plot_panel(image)
        if line_panel is not None:
            return line_panel

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray_bg = (
            (hsv[:, :, 1] < 45)
            & (hsv[:, :, 2] > 170)
            & (hsv[:, :, 2] < 245)
        ).astype(np.uint8) * 255
        upper = np.zeros_like(gray_bg)
        upper[: int(image.shape[0] * 0.62), :] = gray_bg[: int(image.shape[0] * 0.62), :]
        kernel = np.ones((13, 13), np.uint8)
        mask = cv2.morphologyEx(upper, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        candidates = []
        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            if area < 0.02 * image.shape[0] * image.shape[1]:
                continue
            if w < 150 or h < 120:
                continue
            aspect = w / max(h, 1)
            if aspect < 1.1:
                continue
            candidates.append((area, x, y, w, h))

        if not candidates:
            dark_crop = self._extract_dark_plot_panel(image)
            if dark_crop is not None:
                return dark_crop
            return image, (0, 0), {"panel_bbox": [0, 0, int(image.shape[1]), int(image.shape[0])], "panel_detection": "fallback_full_image"}

        # The band plot is normally the largest wide gray panel. DOS panels are
        # narrower and text blocks are outside the gray plotting background.
        _area, x, y, w, h = max(candidates, key=lambda item: (item[0], item[3]))
        pad = 3
        x0 = max(int(x) - pad, 0)
        y0 = max(int(y) - pad, 0)
        x1 = min(int(x + w) + pad, image.shape[1])
        y1 = min(int(y + h) + pad, image.shape[0])
        panel, inner_offset, frame_meta = self._crop_to_plot_frame(image[y0:y1, x0:x1].copy())
        x0 += inner_offset[0]
        y0 += inner_offset[1]
        x1 = x0 + panel.shape[1]
        y1 = y0 + panel.shape[0]
        return panel, (x0, y0), {"panel_bbox": [x0, y0, x1, y1], "panel_detection": "largest_gray_band_panel", **frame_meta}

    def _extract_dark_plot_panel(self, image: np.ndarray) -> Optional[tuple[np.ndarray, tuple[int, int], Dict[str, Any]]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dark = (gray < 95).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < image.shape[1] * 0.45 or h < image.shape[0] * 0.45:
                continue
            if w * h > image.shape[0] * image.shape[1] * 0.95:
                continue
            candidates.append((w * h, x, y, w, h))
        if not candidates:
            return None

        _area, x, y, w, h = max(candidates, key=lambda item: item[0])
        pad = 4
        x0 = max(int(x) - pad, 0)
        y0 = max(int(y) - pad, 0)
        x1 = min(int(x + w) + pad, image.shape[1])
        y1 = min(int(y + h) + pad, image.shape[0])
        panel, inner_offset, frame_meta = self._crop_to_plot_frame(image[y0:y1, x0:x1].copy())
        x0 += inner_offset[0]
        y0 += inner_offset[1]
        x1 = x0 + panel.shape[1]
        y1 = y0 + panel.shape[0]
        return (
            panel,
            (x0, y0),
            {"panel_bbox": [x0, y0, x1, y1], "panel_detection": "largest_dark_plot_frame", **frame_meta},
        )

    def _extract_line_plot_panel(self, image: np.ndarray) -> Optional[tuple[np.ndarray, tuple[int, int], Dict[str, Any]]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, dark = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        height, width = gray.shape[:2]

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 4)))
        v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
        col_counts = (v_lines > 0).sum(axis=0)
        cols = np.where(col_counts >= max(int(height * 0.45), 80))[0]
        groups = self._contiguous_groups(cols)
        centers = [int(round((g[0] + g[-1]) / 2.0)) for g in groups]

        usable = []
        for c in centers:
            ys = np.where(v_lines[:, c] > 0)[0]
            if len(ys) < max(int(height * 0.45), 80):
                continue
            usable.append((c, int(ys[0]), int(ys[-1]), int(len(ys))))
        if len(usable) < 2:
            return None
        min_top = min(item[1] for item in usable)
        max_bottom = max(item[2] for item in usable)
        top_tol = max(6, int(height * 0.03))
        bottom_tol = max(8, int(height * 0.06))
        frame_like = [
            item
            for item in usable
            if item[1] <= min_top + top_tol and item[2] >= max_bottom - bottom_tol
        ]
        if len(frame_like) >= 2:
            usable = frame_like

        x_left = min(item[0] for item in usable)
        x_right = max(item[0] for item in usable)
        y_top = int(np.median([item[1] for item in usable]))
        y_bottom = int(np.median([item[2] for item in usable]))
        if x_right - x_left < width * 0.35 or y_bottom - y_top < height * 0.40:
            return None

        x0 = max(x_left, 0)
        y0 = max(y_top, 0)
        x1 = min(x_right + 1, width)
        y1 = min(y_bottom + 1, height)
        if x1 <= x0 or y1 <= y0:
            return None
        return (
            image[y0:y1, x0:x1].copy(),
            (x0, y0),
            {
                "panel_bbox": [x0, y0, x1, y1],
                "panel_detection": "line_plot_frame",
                "frame_crop": "global_long_line_plot_frame",
                "frame_line_columns": [int(item[0]) for item in usable],
            },
        )

    def _crop_to_plot_frame(self, panel: np.ndarray) -> tuple[np.ndarray, tuple[int, int], Dict[str, Any]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        robust = self._crop_to_long_line_frame(panel, gray)
        if robust is not None:
            return robust

        dark = (gray < 120).astype(np.uint8)
        row_counts = dark.sum(axis=1)
        min_run = max(int(panel.shape[1] * 0.45), 80)
        rows = np.where(row_counts >= min_run)[0]
        if len(rows) < 2:
            return panel, (0, 0), {"frame_crop": "none"}

        y_top = int(rows[0])
        y_bottom = int(rows[-1])
        if y_bottom - y_top < panel.shape[0] * 0.35:
            return panel, (0, 0), {"frame_crop": "none"}

        row_band = dark[max(0, y_top - 2) : min(panel.shape[0], y_top + 3), :].max(axis=0)
        if row_band.sum() < min_run:
            row_band = dark[max(0, y_bottom - 2) : min(panel.shape[0], y_bottom + 3), :].max(axis=0)
        xs = np.where(row_band > 0)[0]
        if len(xs) < 2:
            return panel, (0, 0), {"frame_crop": "none"}
        x_left = int(xs[0])
        x_right = int(xs[-1])
        if x_right - x_left < panel.shape[1] * 0.35:
            return panel, (0, 0), {"frame_crop": "none"}

        pad = 2
        x0 = max(x_left + pad, 0)
        y0 = max(y_top + pad, 0)
        x1 = min(x_right - pad, panel.shape[1])
        y1 = min(y_bottom - pad, panel.shape[0])
        if x1 <= x0 or y1 <= y0:
            return panel, (0, 0), {"frame_crop": "none"}
        return panel[y0:y1, x0:x1].copy(), (x0, y0), {"frame_crop": "plot_frame"}

    def _crop_to_long_line_frame(self, panel: np.ndarray, gray: np.ndarray) -> Optional[tuple[np.ndarray, tuple[int, int], Dict[str, Any]]]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        height, width = gray.shape[:2]
        _, dark = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY_INV)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, height // 4)))
        v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
        col_counts = (v_lines > 0).sum(axis=0)
        cols = np.where(col_counts >= max(int(height * 0.45), 80))[0]
        groups = self._contiguous_groups(cols)
        if len(groups) < 2:
            return None

        centers = [int(round((g[0] + g[-1]) / 2.0)) for g in groups]
        line_heights = [int(col_counts[c]) for c in centers]
        usable = [(c, h) for c, h in zip(centers, line_heights) if h >= max(int(height * 0.55), 120)]
        if len(usable) < 2:
            return None

        x_left = min(c for c, _h in usable)
        x_right = max(c for c, _h in usable)
        if x_right - x_left < width * 0.35:
            return None

        y_samples = []
        for c, _h in usable:
            ys = np.where(v_lines[:, c] > 0)[0]
            if len(ys):
                y_samples.append((int(ys[0]), int(ys[-1])))
        if len(y_samples) < 2:
            return None
        y_top = int(np.median([y0 for y0, _y1 in y_samples]))
        y_bottom = int(np.median([y1 for _y0, y1 in y_samples]))
        if y_bottom - y_top < height * 0.35:
            return None

        pad = 3
        x0 = max(x_left + pad, 0)
        y0 = max(y_top + pad, 0)
        x1 = min(x_right - pad, width)
        y1 = min(y_bottom - pad, height)
        if x1 <= x0 or y1 <= y0:
            return None
        return (
            panel[y0:y1, x0:x1].copy(),
            (x0, y0),
            {
                "frame_crop": "long_line_plot_frame",
                "frame_line_columns": [int(c) for c, _h in usable],
            },
        )

    def _contiguous_groups(self, values: np.ndarray) -> List[np.ndarray]:
        if len(values) == 0:
            return []
        breaks = np.where(np.diff(values) > 1)[0] + 1
        return [group for group in np.split(values, breaks) if len(group) > 0]

    def _auto_band_mask(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        # Color-agnostic skeleton extraction. CV must not decide whether a
        # curve is valence/conduction from color; human GUI calibration assigns
        # physical meaning later.
        gray = self._remove_ruling_lines(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        mask = self._dark_curve_mask(gray)
        mask = self._remove_reference_line_rows(mask)
        mask = self._remove_reference_line_columns(mask)
        # Filter out small text-like blobs so letters aren't misidentified as
        # band-structure curves (connected components that are too compact or
        # too small relative to the image).
        mask = self._filter_text_blobs(mask)
        return mask, "color_agnostic_skeleton"

    def _filter_text_blobs(self, binary: np.ndarray) -> np.ndarray:
        """Remove connected components that look like text/letters/axis-labels.

        Text characters appear as compact dark blobs (e.g. axis labels,
        inset text, tick numbers). Band curves are long, thin curvilinear
        structures with very low aspect ratio (width >> height or vice versa).
        We filter aggressively:
          - area < 80 px: noise/punctuation
          - aspect > 0.20 and area < 800: too square, likely a letter/number
        """
        try:
            import cv2
            n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
        except ImportError:
            return binary

        height, width = binary.shape[:2]
        cleaned = binary.copy()
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            # Skip the background label (i=0)
            if area < 80:
                # Tiny speck -- noise, punctuation, thin tick marks
                cleaned[labels == i] = 0
                continue
            aspect = min(w, h) / max(w, max(h, 1))
            if aspect > 0.20 and area < 800:
                # Nearly square or moderately elongated blob, with bounded area.
                # Band curves are very thin: aspect << 0.1 for long lines.
                # Letters/numbers/axis labels have aspect > 0.20.
                cleaned[labels == i] = 0
                continue
        return cleaned

    def _dark_curve_mask(self, gray: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def _suppress_annotation_colors(self, image: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        red = (((hue <= 12) | (hue >= 168)) & (sat > 35) & (val > 45))
        cleaned = image.copy()
        cleaned[red] = (255, 255, 255)
        return cleaned

    def _remove_reference_line_rows(self, binary: np.ndarray) -> np.ndarray:
        cleaned = binary.copy()
        height, width = cleaned.shape[:2]
        row_counts = (cleaned > 0).sum(axis=1)
        rows_idx = np.arange(height)
        interior = (rows_idx > int(height * 0.08)) & (rows_idx < int(height * 0.92))
        threshold = max(int(width * 0.20), 80)
        candidates = np.where((row_counts >= threshold) & interior)[0]
        for group in self._contiguous_groups(candidates):
            if len(group) > max(3, int(height * 0.02)):
                continue
            y0 = max(int(group[0]) - 2, 0)
            y1 = min(int(group[-1]) + 3, height)
            cleaned[y0:y1, :] = 0
        return cleaned

    def _remove_reference_line_columns(self, binary: np.ndarray) -> np.ndarray:
        cleaned = binary.copy()
        height, width = cleaned.shape[:2]
        col_counts = (cleaned > 0).sum(axis=0)
        cols_idx = np.arange(width)
        interior = (cols_idx > int(width * 0.02)) & (cols_idx < int(width * 0.98))
        threshold = max(int(height * 0.35), 140)
        candidates = np.where((col_counts >= threshold) & interior)[0]
        for group in self._contiguous_groups(candidates):
            if len(group) > max(8, int(width * 0.03)):
                continue
            x0 = max(int(group[0]) - 2, 0)
            x1 = min(int(group[-1]) + 3, width)
            cleaned[:, x0:x1] = 0
        return cleaned

    def _remove_ruling_lines(self, gray: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        _, dark = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, gray.shape[0] // 9)))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, gray.shape[1] // 8), 1))
        vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel)
        horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel)
        ruling = cv2.bitwise_or(vertical, horizontal)
        cleaned_dark = cv2.bitwise_and(dark, cv2.bitwise_not(ruling))
        return cv2.bitwise_not(cleaned_dark)

    def _blue_band_mask(self, image: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image parsing") from exc

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        b, g, r = cv2.split(image)
        blue_hsv = (hue >= 85) & (hue <= 115) & (sat > 45) & (val > 45)
        blue_bgr = (b.astype(np.int16) > r.astype(np.int16) + 20) & (b.astype(np.int16) > g.astype(np.int16) + 5)
        mask = (blue_hsv | blue_bgr).astype(np.uint8) * 255
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def _imread_unicode(self, path: str) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for image Plot-to-Physics parsing") from exc

        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            raise FileNotFoundError(path)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        return image

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for skeletonization") from exc

        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(binary)

        img = (binary > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, opened)
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    def _coords_to_band_data(
        self,
        path: str,
        source_type: str,
        coords_xy: np.ndarray,
        image_shape: Any,
        skeleton_pixels: Optional[np.ndarray] = None,
        overlay_image: Optional[np.ndarray] = None,
        panel_image: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsedBandData:
        height = float(image_shape[0])
        width = float(image_shape[1])
        x = coords_xy[:, 0]
        y = coords_xy[:, 1]

        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        x_span = max(x_max - x_min, 1.0)
        y_span = max(height, y_max - y_min, 1.0)

        k = np.clip((x - x_min) / x_span, 0.0, 1.0)
        # CV-owned output only: normalized image coordinates. These are not
        # physical energies and must not be used for final physics inference.
        energy = (1.0 - np.clip(y / max(height, 1.0), 0.0, 1.0)).astype(np.float32)
        band_hint = np.zeros_like(energy, dtype=np.int8)

        kpath_labels = self._estimate_kpath_labels(coords_xy, width)
        meta = metadata or {}
        meta.update(
            {
                "image_shape": [height, width],
                "requires_human_calibration": True,
                "cv_physics_inference": "disabled",
            }
        )
        return ParsedBandData(
            source_path=str(path),
            source_type=source_type,
            k=k.astype(np.float32),
            energy=energy.astype(np.float32),
            band_hint=band_hint,
            fermi_energy=0.0,
            kpath_labels=kpath_labels,
            skeleton_pixels=skeleton_pixels,
            overlay_image=overlay_image,
            metadata=meta,
        )

    # Fermi-level and band-type inference deliberately removed from CV parser.
    # Human calibration in scripts/gui_workbench.py owns physical axes, VBM/CBM,
    # and curve assignment before PhysicsBrainInvoker performs topology fusion.

    def _estimate_kpath_labels(self, coords_xy: np.ndarray, width: float) -> List[Dict[str, Any]]:
        x = coords_xy[:, 0]
        hist, bins = np.histogram(x, bins=32)
        threshold = max(float(hist.max()) * 0.65, 8.0)
        peaks = np.where(hist >= threshold)[0]
        ticks: List[Dict[str, Any]] = []
        labels = ["$\\Gamma$", "X", "M", "$\\Gamma$"]
        for i, peak in enumerate(peaks[:4]):
            x_center = 0.5 * (bins[peak] + bins[peak + 1])
            ticks.append({"index": int(round((x_center / max(width, 1.0)) * (self.target_k_points - 1))), "label": labels[i]})
        if not ticks:
            ticks = [
                {"index": 0, "label": "$\\Gamma$"},
                {"index": self.target_k_points - 1, "label": "X"},
            ]
        return ticks

    def _rough_gap_from_points(self, parsed: ParsedBandData) -> float:
        vb = parsed.energy[parsed.energy <= 0.0]
        cb = parsed.energy[parsed.energy >= 0.0]
        if len(vb) == 0 or len(cb) == 0:
            return 0.0
        return float(np.min(cb) - np.max(vb))
