"""Create bounded CV panel preannotations for a frozen Europe PMC snapshot."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


_CANDIDATE_ID = re.compile(r"epmc-band-[0-9a-f]{24}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_RECORD_BYTES = 256 * 1024
MAX_METADATA_BYTES = 32 * 1024 * 1024


def _bounded_read(path: Path, limit: int, label: str) -> bytes:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise ValueError(label + " exceeds size budget or is not a regular file")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(label + " exceeds size budget")
    return payload


def _hash_image(path: Path) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds size budget or is not a regular file")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds size budget")
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: Any, prefix: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid snapshot relative path")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("invalid snapshot relative path")
    if not value.replace("\\", "/").startswith(prefix + "/"):
        raise ValueError("snapshot path is outside declared area")
    return path.as_posix()


def _load_snapshot(manifest_path: Path) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Path]]]:
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_size > 5 * 1024 * 1024
    ):
        raise ValueError("invalid candidate manifest path")
    root = manifest_path.parent.resolve()
    manifest_payload = _bounded_read(manifest_path, 5 * 1024 * 1024, "manifest")
    manifest = json.loads(manifest_payload)
    if not isinstance(manifest, dict):
        raise ValueError("candidate manifest must be an object")
    if (
        manifest.get("status") != "complete_ai_assisted_figure_candidate_snapshot"
        or manifest.get("eligible_for_scientific_acceptance") is not False
        or manifest.get("human_audited_count") != 0
        or manifest.get("formal_approved_count") != 0
    ):
        raise ValueError("candidate manifest approval boundary is invalid")
    entries = manifest.get("records")
    if not isinstance(entries, list) or len(entries) != manifest.get("record_count"):
        raise ValueError("candidate manifest record count mismatch")
    if not 1 <= len(entries) <= 5000:
        raise ValueError("candidate manifest count outside bounds")
    loaded = []
    seen = set()
    metadata_bytes = 0
    for entry in entries:
        candidate_id = str(entry.get("candidate_id") or "")
        if not _CANDIDATE_ID.fullmatch(candidate_id) or candidate_id in seen:
            raise ValueError("invalid or duplicate candidate ID")
        seen.add(candidate_id)
        record_rel = _safe_relative(entry.get("record_path"), "records")
        image_rel = _safe_relative(entry.get("image_path"), "images")
        record_path = root / Path(record_rel)
        image_path = root / Path(image_rel)
        if (
            record_path.is_symlink()
            or image_path.is_symlink()
            or not record_path.resolve().is_relative_to(root)
            or not image_path.resolve().is_relative_to(root)
        ):
            raise ValueError("snapshot symlink refused")
        record_payload = _bounded_read(record_path, MAX_RECORD_BYTES, "record")
        metadata_bytes += len(record_payload)
        if metadata_bytes > MAX_METADATA_BYTES:
            raise ValueError("snapshot metadata exceeds budget")
        record = json.loads(record_payload)
        if not isinstance(record, dict):
            raise ValueError("candidate record must be an object")
        image_sha = _hash_image(image_path)
        if (
            record.get("candidate_id") != candidate_id
            or record.get("image", {}).get("path") != image_rel
            or record.get("image", {}).get("sha256") != image_sha
            or entry.get("image_sha256") != image_sha
        ):
            raise ValueError("snapshot image SHA or record binding mismatch")
        if (
            record.get("human_audited") is not False
            or record.get("formal_approval") is not None
            or record.get("eligible_for_scientific_acceptance") is not False
        ):
            raise ValueError("snapshot contains an approval claim")
        # Only bounded metadata and a locator survive admission; image bytes are
        # loaded in at most `workers` active tasks, and their digest is rechecked.
        loaded.append((record, image_path))
    return {**manifest, "manifest_sha256": _sha256(manifest_payload)}, loaded


def _preannotate(item: tuple[dict[str, Any], Path | bytes]) -> dict[str, Any]:
    from src.vision.multi_format_parser import extract_band_image

    record, payload = item
    if isinstance(payload, Path):
        payload = _bounded_read(payload, MAX_IMAGE_BYTES, "image")
    if not isinstance(payload, bytes) or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds size budget")
    if _sha256(payload) != record["image"]["sha256"]:
        raise ValueError("snapshot image SHA changed before consumption")
    base = {
        "schema_version": 1,
        "candidate_id": record["candidate_id"],
        "group_id": record["group_id"],
        "image_sha256": record["image"]["sha256"],
        "class": "electronic_band_structure_figure",
        "label_origin": "automated_cv_ai_assisted",
        "human_audited": False,
        "formal_approval": None,
        "operator_authentication": None,
        "eligible_for_scientific_acceptance": False,
    }
    try:
        result = extract_band_image(payload)
        bbox = result.get("panel_bbox")
        detection = (result.get("cv_quality") or {}).get("components", {}).get("panel_detection")
        usable_bbox = (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(type(value) in {int, float} for value in bbox)
            and bbox[2] > bbox[0]
            and bbox[3] > bbox[1]
            and detection != "fallback_full_image"
        )
        return {
            **base,
            "status": "ai_assisted_panel_candidate" if usable_bbox else "needs_human_panel_box",
            "panel_bbox_xyxy": bbox if usable_bbox else None,
            "panel_detection": detection,
            "skeleton_point_count": result.get("skeleton_point_count"),
            "cv_quality": result.get("cv_quality"),
            "physical_calibration": None,
            "curve_level_annotation": None,
        }
    except Exception as exc:
        return {
            **base,
            "status": "needs_human_panel_box",
            "panel_bbox_xyxy": None,
            "panel_detection": None,
            "skeleton_point_count": None,
            "cv_quality": None,
            "physical_calibration": None,
            "curve_level_annotation": None,
            "failure_type": type(exc).__name__,
        }


def preannotate_snapshot(
    manifest_path: str | Path, output_dir: str | Path, workers: int = 4
) -> dict[str, Any]:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be 1-8")
    source_path = Path(manifest_path)
    manifest, loaded = _load_snapshot(source_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(str(output))
    output.mkdir(parents=True)
    records_dir = output / "records"
    records_dir.mkdir()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        annotations = list(pool.map(_preannotate, loaded))
    for annotation in annotations:
        path = records_dir / f"{annotation['candidate_id']}.json"
        with path.open("xb") as handle:
            handle.write(
                json.dumps(annotation, indent=2, ensure_ascii=False, allow_nan=False).encode(
                    "utf-8"
                )
            )
    status_counts: dict[str, int] = {}
    for annotation in annotations:
        status_counts[annotation["status"]] = status_counts.get(annotation["status"], 0) + 1
    output_manifest = {
        "schema_version": 1,
        "status": "complete_ai_assisted_panel_preannotations",
        "source_manifest_path": str(source_path.resolve()),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "record_count": len(annotations),
        "status_counts": status_counts,
        "human_audited_count": 0,
        "formal_approved_count": 0,
        "eligible_for_scientific_acceptance": False,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "records": [
            {
                "candidate_id": item["candidate_id"],
                "group_id": item["group_id"],
                "image_sha256": item["image_sha256"],
                "status": item["status"],
                "record_path": f"records/{item['candidate_id']}.json",
            }
            for item in annotations
        ],
    }
    manifest_payload = json.dumps(
        output_manifest, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    output_manifest_path = output / "manifest.json"
    with output_manifest_path.open("xb") as handle:
        handle.write(manifest_payload)
    return {
        "status": output_manifest["status"],
        "record_count": len(annotations),
        "status_counts": status_counts,
        "manifest_path": str(output_manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_payload),
    }
