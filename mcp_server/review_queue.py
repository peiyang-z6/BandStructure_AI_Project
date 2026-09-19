"""Strict pending-review queue helpers; formal approval stays out of process."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _validate_record(record: Any) -> dict[str, Any]:
    from mcp_server.server import _ai_observation_review

    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise ValueError("invalid pending review record")
    if record.get("intended_split") not in {"train", "validation", "evaluation"}:
        raise ValueError("invalid pending review split")
    observations = record.get("observations")
    if not isinstance(observations, dict):
        raise ValueError("missing pending observations")
    digest = hashlib.sha256(_canonical(observations)).hexdigest()
    if record.get("observation_sha256") != digest:
        raise ValueError("pending observation SHA mismatch")
    if record.get("record_id") != f"mcp-audit-{digest[:20]}":
        raise ValueError("pending record ID mismatch")
    if record.get("source") != observations.get("source"):
        raise ValueError("pending source binding mismatch")
    if record.get("independent_human_review") is not None:
        raise ValueError("pending queue cannot contain human approval")
    if record.get("operator_authentication") is not None:
        raise ValueError("pending queue cannot contain operator authentication")
    if record.get("ai_prefilled") is not True:
        raise ValueError("pending queue must identify AI-prefilled records")
    if not re.fullmatch(r"mcp-audit-[0-9a-f]{20}", record["record_id"]):
        raise ValueError("invalid pending record ID")
    review = _ai_observation_review(observations)
    if review["invalid_fields"]:
        raise ValueError("pending observations contain invalid fields")
    return {
        "missing_evidence": review["missing_evidence"],
        "next_questions": review["next_questions"],
        "annotation_complete": not review["missing_evidence"],
        "approval_ready": False,
        "formal_approval_supported": False,
    }


def _validate_review_packet(packet: Any) -> None:
    if not isinstance(packet, dict) or packet.get("schema_version") != 1:
        raise ValueError("invalid review packet")
    reviewer_kind = packet.get("reviewer_kind")
    expected_status = {
        "ai_assisted": "ai_assisted_pre_review",
        "human": "human_review_packet_pending_operator_authentication",
    }.get(reviewer_kind)
    if expected_status is None or packet.get("status") != expected_status:
        raise ValueError("invalid non-approving review status")
    if packet.get("decision") not in {"needs_more_evidence", "rejected"}:
        raise ValueError("review packet cannot grant approval")
    if not re.fullmatch(r"mcp-audit-[0-9a-f]{20}", str(packet.get("record_id", ""))):
        raise ValueError("invalid review packet record ID")
    if not re.fullmatch(r"[0-9a-f]{64}", str(packet.get("observation_sha256", ""))):
        raise ValueError("invalid review packet observation SHA")
    if (
        packet.get("human_audited") is not False
        or packet.get("operator_authentication") is not None
        or packet.get("eligible_for_scientific_acceptance") is not False
    ):
        raise ValueError("review packet cannot contain approval credentials")
    if not isinstance(packet.get("source"), dict):
        raise ValueError("review packet source is required")
    notes = packet.get("notes")
    if not isinstance(notes, str) or not notes.strip() or len(notes) > 5000:
        raise ValueError("invalid review packet notes")


def load_pending_queue(path: str | Path) -> dict[str, Any]:
    """Load and verify one AI-prefilled queue without upgrading its status."""
    source = Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("invalid pending queue path")
    try:
        payload = json.loads(source.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid pending queue JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("status") != "pending_human"
        or payload.get("eligible_for_scientific_acceptance") is not False
    ):
        raise ValueError("invalid pending queue envelope")
    records = payload.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 500:
        raise ValueError("pending queue must contain 1-500 records")
    seen = set()
    reviewed_records = []
    for record in records:
        state = _validate_record(record)
        if record["record_id"] in seen:
            raise ValueError("duplicate pending record ID")
        seen.add(record["record_id"])
        reviewed_records.append({**record, "review_state": state})
    return {
        "schema_version": 1,
        "status": "pending_human",
        "eligible_for_scientific_acceptance": False,
        "formal_approval_supported": False,
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "record_count": len(reviewed_records),
        "records": reviewed_records,
    }


def make_review_packet(
    record: dict[str, Any],
    decision: str,
    reviewer_kind: str,
    notes: str,
) -> dict[str, Any]:
    """Create a non-approving pre-review packet bound to the candidate SHA."""
    from datetime import datetime, timezone

    if decision == "approved":
        raise ValueError("formal approval requires an external authenticated registry")
    if decision not in {"needs_more_evidence", "rejected"}:
        raise ValueError("unsupported review decision")
    if reviewer_kind not in {"ai_assisted", "human"}:
        raise ValueError("unsupported reviewer kind")
    if not isinstance(notes, str) or not notes.strip() or len(notes) > 5000:
        raise ValueError("review notes must contain 1-5000 characters")
    state = _validate_record(record)
    return {
        "schema_version": 1,
        "status": (
            "ai_assisted_pre_review"
            if reviewer_kind == "ai_assisted"
            else "human_review_packet_pending_operator_authentication"
        ),
        "decision": decision,
        "reviewer_kind": reviewer_kind,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "record_id": record["record_id"],
        "observation_sha256": record["observation_sha256"],
        "source": record["source"],
        "missing_evidence": state["missing_evidence"],
        "notes": notes.strip(),
        "human_audited": False,
        "operator_authentication": None,
        "eligible_for_scientific_acceptance": False,
    }


def save_review_packet(packet: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Save one immutable pre-review packet without overwriting prior evidence."""
    _validate_review_packet(packet)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("invalid review packet output directory")
    filename = f"{packet['record_id']}__{packet['reviewer_kind']}__{packet['decision']}.json"
    path = directory / filename
    payload = json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "eligible_for_scientific_acceptance": False,
    }
