"""Shared image/calibration contracts; no retrieval or learned-model dependencies."""

import numpy as np

MANUAL_EXTREMUM_PIXEL_TOLERANCE = 2.0
MANUAL_EXTREMUM_ENERGY_TOLERANCE_EV = 0.05


def decode_raster_bytes(payload):
    """Strict full RGB raster from exactly the bytes hashed/displayed by P2.

    Pillow verify() alone accepts truncated JPEGs. Never reopen the source path
    or toggle Pillow's process-global recovery flag (unsafe for other threads).
    """
    import io
    from PIL import Image, ImageFile

    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ValueError("strict raster decoding required; truncated image recovery is enabled")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(
            "P2 needs a completely decodable raster image; export PDF panels as PNG"
        ) from exc


def validate_manual_calibration(annotations, calibration, *, image_size):
    """Validate original-pixel, single-segment manual inputs before reconstruction.

    This validates data shape/geometry, not the truth of a person's annotation.
    External reviewed images are a separate audit gate.
    """
    ann, cal = annotations, calibration
    if not isinstance(ann, dict) or not isinstance(cal, dict):
        raise ValueError("manual calibration must be mappings")
    if ann.get("input_space", "raw6d") != "raw6d":
        raise ValueError("manual query accepts raw pixels only, never normalized tensors")
    if ann.get("segment_boundaries"):
        raise ValueError("multiple k segments unsupported; calibrate one continuous segment")
    try:
        size = np.asarray(image_size, dtype=float)
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 2):
            raise ValueError("decoded raster image dimensions required")
        panel = np.asarray(ann["panel"], dtype=float)
        if panel.shape != (4,) or not np.isfinite(panel).all() or np.any(panel[2:] <= 0):
            raise ValueError("invalid panel calibration")
        x, y, width, height = panel
        # Pixel centers are [0, W-1] x [0, H-1]; no synthetic canvas size.
        if x < 0 or y < 0 or x + width > size[0] - 1 or y + height > size[1] - 1:
            raise ValueError("manual panel outside decoded raster image")
        fermi = float(ann["fermi_y"])
        if not np.isfinite(fermi) or not y <= fermi <= y + height:
            raise ValueError("invalid manual Fermi calibration")
        for field in ("xaxis_pts", "yaxis_pts", "vbm", "cbm"):
            points = np.asarray(ann[field], dtype=float)
            shape = (2, 2) if field.endswith("_pts") else (2,)
            if points.shape != shape or not np.isfinite(points).all():
                raise ValueError("invalid manual calibration points: " + field)
        k_min, k_max = sorted(p[0] for p in ann["xaxis_pts"])
        for field in ("xaxis_pts", "yaxis_pts", "vbm", "cbm", "vb_strokes", "cb_strokes"):
            if field.endswith("strokes"):
                points = np.asarray([p for stroke in ann[field] for p in stroke], dtype=float)
                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or len(points) < 2
                    or len(np.unique(points[:, 0])) < 2
                ):
                    raise ValueError("manual curve needs two distinct k points")
            elif field.endswith("_pts"):
                points = np.asarray(ann[field], dtype=float)
            else:
                points = np.asarray([ann[field]], dtype=float)
            if (
                not np.isfinite(points).all()
                or np.any(points[:, 0] < x)
                or np.any(points[:, 0] > x + width)
                or np.any(points[:, 1] < y)
                or np.any(points[:, 1] > y + height)
            ):
                raise ValueError("manual curve/extremum points must be finite and inside panel")
            if np.any(points[:, 0] < k_min) or np.any(points[:, 0] > k_max):
                raise ValueError("manual points outside calibrated k interval; clipping forbidden")
        xv, yv = np.asarray(cal["x_values"], dtype=float), np.asarray(cal["y_values"], dtype=float)
        if (
            xv.shape != (2,)
            or yv.shape != (2,)
            or not np.isfinite(yv).all()
            or not np.array_equal(np.sort(xv), [0.0, 1.0])
            or yv[0] == yv[1]
            or ann["xaxis_pts"][0][0] == ann["xaxis_pts"][1][0]
            or ann["yaxis_pts"][0][1] == ann["yaxis_pts"][1][1]
        ):
            raise ValueError(
                "calibration needs distinct axes, eV values and normalized k endpoints 0/1"
            )
        validate_manual_extrema(ann, cal)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("missing/invalid manual calibration") from exc


def validate_manual_extrema(annotations, calibration, *, reconstructed=None):
    """P2-only check against the unaugmented complete trace, not inserted labels.

    Fixed engineering tolerances: 2 original pixels in k/energy, with vertical
    tolerance additionally capped at 0.05 eV. Neither is fitted to evaluation
    data or a statement of human accuracy. All source knots are included so a
    narrow extremum between the 128 tensor samples cannot disappear. Reuse the
    existing shape-preserving resampler; never change the supervised core.
    """
    from src.vision.physics_reconstructor import PhysicsReconstructor

    ann, cal = annotations, calibration
    x0, x1 = np.asarray(ann["xaxis_pts"], dtype=float)[:, 0]
    y0, y1 = np.asarray(ann["yaxis_pts"], dtype=float)[:, 1]
    x_scale = (cal["x_values"][1] - cal["x_values"][0]) / (x1 - x0)
    y_scale = (cal["y_values"][1] - cal["y_values"][0]) / (y1 - y0)
    e_tol = min(MANUAL_EXTREMUM_ENERGY_TOLERANCE_EV, MANUAL_EXTREMUM_PIXEL_TOLERANCE * abs(y_scale))
    k_tol = MANUAL_EXTREMUM_PIXEL_TOLERANCE * abs(x_scale)
    if reconstructed is not None:
        reconstructed = np.asarray(reconstructed)
        if reconstructed.shape != (1, 128, 6) or not np.isfinite(reconstructed).all():
            raise ValueError("invalid reconstructed extremum channels")
    reconstructor = PhysicsReconstructor()
    for field, strokes, mode, extreme in [
        ("vbm", "vb_strokes", "valence", np.max),
        ("cbm", "cb_strokes", "conduction", np.min),
    ]:
        points = np.asarray([p for s in ann[strokes] for p in s], dtype=float)
        k = cal["x_values"][0] + (points[:, 0] - x0) * x_scale
        energy = (points[:, 1] - ann["fermi_y"]) * y_scale
        manual_k = cal["x_values"][0] + (ann[field][0] - x0) * x_scale
        manual_e = (ann[field][1] - ann["fermi_y"]) * y_scale
        grid = np.unique(np.r_[np.linspace(0.0, 1.0, 128), k, manual_k]).astype(np.float32)
        values = reconstructor._resample_manual_trace(list(zip(k, energy)), grid, mode)
        at_marker = float(values[np.argmin(abs(grid - manual_k))])
        global_e = float(extreme(values))
        # PCHIP has no interval overshoot; source knots cover all extrema.
        ties = grid[np.abs(values - global_e) <= 1.0e-6]
        if (
            abs(at_marker - manual_e) > e_tol + 1.0e-6
            or abs(global_e - manual_e) > e_tol + 1.0e-6
            or np.min(abs(ties - manual_k)) > k_tol + 1.0e-6
        ):
            raise ValueError(
                field + " manual extremum inconsistent with complete curve (2 px / 0.05 eV cap)"
            )
        if reconstructed is not None:
            target_k = np.linspace(0.0, 1.0, 128, dtype=np.float32)
            index = int(np.argmin(abs(target_k - manual_k)))
            channel = 0 if field == "vbm" else 3
            actual_e = reconstructed[0, :, channel]
            actual_d = reconstructed[0, :, channel + 2]
            expected_d = (np.arange(128, dtype=np.float32) - index) / 127.0
            # Bind EVERY consumed energy sample to the unaugmented trace at
            # the same k, using the same min(2 px, 0.05 eV) vertical tolerance.
            # An in-tolerance inserted marker can still change interpolation
            # far from that marker (e.g. three linear knots -> four PCHIP knots).
            expected_e = reconstructor._resample_manual_trace(list(zip(k, energy)), target_k, mode)
            if np.any(np.abs(actual_e - expected_e) > e_tol + 1.0e-6):
                raise ValueError(
                    field
                    + " reconstructed energy channel inconsistent with complete curve (2 px / 0.05 eV cap)"
                )
            # Check the consumed channels, not argmax identity: true flat or
            # tied extrema may use ANY manually selected extremal position.
            if (
                abs(float(extreme(actual_e)) - float(actual_e[index])) > e_tol + 1.0e-6
                or abs(float(actual_e[index]) - manual_e) > e_tol + 1.0e-6
                or not np.allclose(actual_d, expected_d, rtol=0.0, atol=1.0e-6)
            ):
                raise ValueError(field + " reconstructed energy/distance extremum channel conflict")
