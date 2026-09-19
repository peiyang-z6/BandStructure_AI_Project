"""Strict review helpers for harvested figure candidates; never grant approval."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


_CANDIDATE_ID = re.compile(r"epmc-band-[0-9a-f]{24}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DECISIONS = {"screen_positive", "needs_panel_box", "exclude_non_target"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _load_json(path: Path, max_bytes: int = 5 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError("invalid review input path")
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("review input exceeds JSON budget")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid review input JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("review input must be an object")
    return value, payload


def _relative(value: Any, prefix: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid review relative path")
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or not path.as_posix().startswith(prefix + "/")
    ):
        raise ValueError("invalid review relative path")
    return path.as_posix()


def load_figure_review_queue(
    dataset_manifest_path: str | Path, preannotation_manifest_path: str | Path
) -> dict[str, Any]:
    dataset_path = Path(dataset_manifest_path)
    pre_path = Path(preannotation_manifest_path)
    dataset, dataset_payload = _load_json(dataset_path)
    pre, pre_payload = _load_json(pre_path)
    if (
        dataset.get("status") != "complete_ai_assisted_figure_candidate_snapshot"
        or dataset.get("human_audited_count") != 0
        or dataset.get("formal_approved_count") != 0
        or dataset.get("eligible_for_scientific_acceptance") is not False
    ):
        raise ValueError("invalid figure candidate approval boundary")
    dataset_sha = _sha256(dataset_payload)
    if (
        pre.get("status") != "complete_ai_assisted_panel_preannotations"
        or pre.get("source_manifest_sha256") != dataset_sha
        or pre.get("human_audited_count") != 0
        or pre.get("formal_approved_count") != 0
        or pre.get("eligible_for_scientific_acceptance") is not False
    ):
        raise ValueError("preannotation manifest is not bound to candidate snapshot")
    dataset_entries = dataset.get("records")
    pre_entries = pre.get("records")
    if (
        not isinstance(dataset_entries, list)
        or not isinstance(pre_entries, list)
        or len(dataset_entries) != dataset.get("record_count")
        or len(pre_entries) != pre.get("record_count")
        or len(dataset_entries) != len(pre_entries)
    ):
        raise ValueError("figure review manifest count mismatch")
    if not 1 <= len(dataset_entries) <= 5000:
        raise ValueError("figure review count outside bounds")
    pre_by_id = {}
    for entry in pre_entries:
        candidate_id = str(entry.get("candidate_id") or "")
        if not _CANDIDATE_ID.fullmatch(candidate_id) or candidate_id in pre_by_id:
            raise ValueError("invalid preannotation candidate ID")
        pre_by_id[candidate_id] = entry
    dataset_root = dataset_path.parent.resolve()
    pre_root = pre_path.parent.resolve()
    records = []
    seen = set()
    metadata_bytes = 0
    for entry in dataset_entries:
        candidate_id = str(entry.get("candidate_id") or "")
        if not _CANDIDATE_ID.fullmatch(candidate_id) or candidate_id in seen:
            raise ValueError("invalid candidate ID")
        seen.add(candidate_id)
        paired = pre_by_id.get(candidate_id)
        if paired is None:
            raise ValueError("missing paired preannotation")
        record_rel = _relative(entry.get("record_path"), "records")
        image_rel = _relative(entry.get("image_path"), "images")
        pre_rel = _relative(paired.get("record_path"), "records")
        record, record_bytes = _load_json(dataset_root / Path(record_rel), 512 * 1024)
        annotation, annotation_bytes = _load_json(pre_root / Path(pre_rel), 512 * 1024)
        metadata_bytes += len(record_bytes) + len(annotation_bytes)
        if metadata_bytes > 32 * 1024 * 1024:
            raise ValueError("review metadata exceeds aggregate budget")
        image_path = dataset_root / Path(image_rel)
        if not image_path.is_file() or image_path.is_symlink():
            raise ValueError("invalid review image path")
        from src.vision.figure_candidate_preannotator import _hash_image

        image_sha = _hash_image(image_path)
        if (
            record.get("candidate_id") != candidate_id
            or annotation.get("candidate_id") != candidate_id
            or image_sha != entry.get("image_sha256")
            or image_sha != paired.get("image_sha256")
            or image_sha != record.get("image", {}).get("sha256")
            or image_sha != annotation.get("image_sha256")
        ):
            raise ValueError("figure review source binding mismatch")
        for value in (record, annotation):
            if (
                value.get("human_audited") is not False
                or value.get("formal_approval") is not None
                or value.get("eligible_for_scientific_acceptance") is not False
            ):
                raise ValueError("review source contains an approval claim")
        records.append(
            {
                "candidate_id": candidate_id,
                "group_id": record["group_id"],
                "source": record["source"],
                "image": record["image"],
                "image_path": str(image_path.resolve()),
                "preannotation": annotation,
                "preannotation_sha256": _sha256(_canonical(annotation)),
            }
        )
    return {
        "schema_version": 1,
        "status": "ready_for_figure_screening",
        "record_count": len(records),
        "formal_approval_supported": False,
        "dataset_manifest_sha256": dataset_sha,
        "preannotation_manifest_sha256": _sha256(pre_payload),
        "records": records,
    }


def make_screening_packet(
    record: dict[str, Any], decision: str, reviewer_kind: str, notes: str
) -> dict[str, Any]:
    if decision == "approved":
        raise ValueError("formal approval requires an external authenticated registry")
    if decision not in _DECISIONS:
        raise ValueError("unsupported screening decision")
    if reviewer_kind not in {"ai_assisted", "human"}:
        raise ValueError("unsupported reviewer kind")
    if not isinstance(notes, str) or not notes.strip() or len(notes) > 5000:
        raise ValueError("screening notes must contain 1-5000 characters")
    candidate_id = str(record.get("candidate_id") or "")
    image_sha = str(record.get("image", {}).get("sha256") or "")
    pre_sha = str(record.get("preannotation_sha256") or "")
    if (
        not _CANDIDATE_ID.fullmatch(candidate_id)
        or not _SHA256.fullmatch(image_sha)
        or not _SHA256.fullmatch(pre_sha)
    ):
        raise ValueError("invalid screening source binding")
    return {
        "schema_version": 1,
        "status": (
            "ai_assisted_figure_screening"
            if reviewer_kind == "ai_assisted"
            else "human_figure_screening_pending_operator_authentication"
        ),
        "decision": decision,
        "reviewer_kind": reviewer_kind,
        "screened_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "group_id": record["group_id"],
        "image_sha256": image_sha,
        "preannotation_sha256": pre_sha,
        "source": record["source"],
        "notes": notes.strip(),
        "human_audited": False,
        "formal_approval": None,
        "operator_authentication": None,
        "eligible_for_scientific_acceptance": False,
    }


def _validate_packet(packet: Any) -> None:
    if not isinstance(packet, dict) or packet.get("schema_version") != 1:
        raise ValueError("invalid screening packet")
    expected = {
        "ai_assisted": "ai_assisted_figure_screening",
        "human": "human_figure_screening_pending_operator_authentication",
    }.get(packet.get("reviewer_kind"))
    if expected is None or packet.get("status") != expected:
        raise ValueError("invalid screening status")
    if packet.get("decision") not in _DECISIONS:
        raise ValueError("screening packet cannot grant approval")
    if not _CANDIDATE_ID.fullmatch(str(packet.get("candidate_id") or "")):
        raise ValueError("invalid screening candidate ID")
    if not _SHA256.fullmatch(str(packet.get("image_sha256") or "")):
        raise ValueError("invalid screening image SHA")
    if (
        packet.get("human_audited") is not False
        or packet.get("formal_approval") is not None
        or packet.get("operator_authentication") is not None
        or packet.get("eligible_for_scientific_acceptance") is not False
    ):
        raise ValueError("screening packet cannot contain approval credentials")


def save_screening_packet(packet: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    _validate_packet(packet)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("invalid screening output directory")
    filename = f"{packet['candidate_id']}__{packet['reviewer_kind']}__{packet['decision']}.json"
    path = output / filename
    payload = json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(payload),
        "eligible_for_scientific_acceptance": False,
    }


def freeze_evaluation_allocation(
    queue: dict[str, Any], output_dir: str | Path, benchmark_id: str
) -> dict[str, Any]:
    """Freeze the full candidate frame before labels exist to prevent cherry-picking."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,79}", benchmark_id):
        raise ValueError("invalid benchmark ID")
    if (
        not isinstance(queue, dict)
        or queue.get("status") != "ready_for_figure_screening"
        or queue.get("formal_approval_supported") is not False
    ):
        raise ValueError("invalid figure review queue")
    records = queue.get("records")
    if not isinstance(records, list) or len(records) != queue.get("record_count"):
        raise ValueError("invalid figure review queue count")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(str(output))
    output.mkdir(parents=True)
    allocation = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "status": "frozen_pending_independent_human_labels",
        "task": "figure_contains_electronic_band_plot_and_panel_localization",
        "dataset_manifest_sha256": queue["dataset_manifest_sha256"],
        "preannotation_manifest_sha256": queue["preannotation_manifest_sha256"],
        "record_count": len(records),
        "document_group_count": len({record["group_id"] for record in records}),
        "frozen_before_human_review": True,
        "split_policy": "all records reserved for evaluation; group key is PMCID",
        "reference_label_status": "missing_independent_human_labels",
        "human_audited_count": 0,
        "formal_approved_count": 0,
        "eligible_for_scientific_acceptance": False,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "records": [
            {
                "candidate_id": record["candidate_id"],
                "group_id": record["group_id"],
                "image_sha256": record["image"]["sha256"],
                "preannotation_sha256": record["preannotation_sha256"],
                "intended_split": "evaluation",
                "reference_label": None,
                "reference_panel_bbox_xyxy": None,
                "independent_human_review": None,
            }
            for record in records
        ],
    }
    allocation_payload = json.dumps(
        allocation, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    allocation_path = output / "evaluation_allocation.json"
    with allocation_path.open("xb") as handle:
        handle.write(allocation_payload)
    data_card = (
        f"# {benchmark_id}\n\n"
        "This is a frozen CC-BY Europe PMC figure candidate frame, not a gold dataset.\n\n"
        f"- Records: {len(records)}\n"
        f"- Document groups (PMCID): {allocation['document_group_count']}\n"
        "- Intended split: evaluation only\n"
        "- AI/CV preannotations: available but not reference labels\n"
        "- Independent human labels: missing\n"
        "- Formal approvals: 0\n\n"
        "Before reporting benchmark metrics, an independent reviewer must inspect every original "
        "figure, assign target/non-target and panel boxes, and bind authenticated decisions to "
        "the hashes in evaluation_allocation.json.\n"
    ).encode("utf-8")
    card_path = output / "DATA_CARD.md"
    with card_path.open("xb") as handle:
        handle.write(data_card)
    return {
        "path": str(allocation_path.resolve()),
        "sha256": _sha256(allocation_payload),
        "data_card_path": str(card_path.resolve()),
        "record_count": len(records),
        "eligible_for_scientific_acceptance": False,
    }
