"""Opaque, content-verified local artifacts shared with an operator-side adapter.

This is a single-user local cache, not a multi-tenant authorization boundary.
No MCP argument can select a filesystem path.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
from contextlib import contextmanager

_LOCK = threading.RLock()
MAX_OBJECT_BYTES = 32 * 1024 * 1024
MAX_STORE_BYTES = 512 * 1024 * 1024


@contextmanager
def _process_lock(root):
    """Serialize quota checks and publication across local client processes."""
    lock_path = root / ".store.lock"
    if lock_path.is_symlink():
        raise ValueError("INVALID_ARTIFACT_PATH")
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def store_directory(*, create=True) -> Path:
    configured = os.environ.get("BAND_MCP_STORE_DIR")
    base = (
        Path(configured)
        if configured
        else Path(
            os.environ.get(
                "LOCALAPPDATA",
                str(Path.home() / ("AppData/Local" if os.name == "nt" else ".local/share")),
            )
        )
        / "BandStructureMCP"
        / "artifacts"
    )
    if not base.is_absolute():
        raise ValueError("Artifact store must be an operator-configured absolute directory")
    for parent in [base, *base.parents]:
        if parent.is_symlink() or (
            parent.exists() and getattr(parent.stat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("Artifact store cannot contain symlinks or reparse points")
    if create:
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return base.resolve()


def _paths(identifier: str):
    if not isinstance(identifier, str) or not re.fullmatch(r"(?:att|res)_[0-9a-f]{32}", identifier):
        raise ValueError("INVALID_ARTIFACT_ID")
    root = store_directory(create=False)
    payload, metadata = root / f"{identifier}.bin", root / f"{identifier}.json"
    if payload.is_symlink() or metadata.is_symlink():
        raise ValueError("INVALID_ARTIFACT_PATH")
    return payload, metadata


def diagnostics():
    """Compare client/adapter stores without exposing local paths or attachment IDs."""
    try:
        root = store_directory(create=False)
        location = os.path.normcase(str(root)).encode("utf-8")
        return {
            "configuration": "explicit"
            if os.environ.get("BAND_MCP_STORE_DIR")
            else "platform_default",
            "location_sha256": hashlib.sha256(location).hexdigest(),
            "registered_attachment_count": sum(1 for _ in root.glob("att_*.json")),
        }
    except (ValueError, OSError):
        return {"configuration": "unavailable"}


def put(
    payload: bytes,
    *,
    kind: str,
    metadata: dict | None = None,
    ttl_seconds: int | None = 86400,
    attachment: bool = False,
    deduplicate: bool = False,
) -> dict:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_OBJECT_BYTES:
        raise ValueError("ARTIFACT_TOO_LARGE")
    root = store_directory()
    digest = hashlib.sha256(payload).hexdigest()
    with _LOCK, _process_lock(root):
        if deduplicate:
            prefix = "att_" if attachment else "res_"
            for existing in root.glob(prefix + "*.json"):
                try:
                    if existing.stat().st_size > 65536 or existing.is_symlink():
                        continue
                    candidate = json.loads(existing.read_bytes())
                    if (
                        candidate.get("sha256") == digest
                        and candidate.get("kind") == kind
                        and candidate.get("metadata") == (metadata or {})
                    ):
                        _, verified = get(existing.stem, expected_kind=kind)
                        return verified
                except (ValueError, OSError, TypeError):
                    continue
        files = list(root.glob("*.bin"))
        if (
            len(files) >= 256
            or sum(p.stat().st_size for p in files) + len(payload) > MAX_STORE_BYTES
        ):
            raise ValueError("ARTIFACT_STORE_FULL; operator must prune expired results")
        identifier = ("att_" if attachment else "res_") + secrets.token_hex(16)
        data_path, meta_path = _paths(identifier)
        now = time.time()
        record = {
            "artifact_id": identifier,
            "kind": kind,
            "sha256": digest,
            "bytes": len(payload),
            "created_at_unix": now,
            "expires_at_unix": now + ttl_seconds if ttl_seconds is not None else None,
            "metadata": metadata or {},
        }
        encoded = json.dumps(record, ensure_ascii=True, allow_nan=False).encode("ascii")
        if len(encoded) > 65536:
            raise ValueError("ARTIFACT_METADATA_TOO_LARGE")
        with data_path.open("xb") as stream:
            stream.write(payload)
        with meta_path.open("xb") as stream:
            stream.write(encoded)
    return record


def get(identifier: str, *, expected_kind: str | None = None) -> tuple[bytes, dict]:
    data_path, meta_path = _paths(identifier)
    try:
        if meta_path.stat().st_size > 65536 or data_path.stat().st_size > MAX_OBJECT_BYTES:
            raise ValueError("ARTIFACT_INTEGRITY_ERROR")
        record = json.loads(meta_path.read_bytes())
        if record["artifact_id"] != identifier or (
            expected_kind and record["kind"] != expected_kind
        ):
            raise ValueError("ARTIFACT_KIND_MISMATCH")
        expiry = record.get("expires_at_unix")
        if expiry is not None and time.time() >= expiry:
            raise ValueError("ARTIFACT_EXPIRED")
        payload = data_path.read_bytes()
        if (
            len(payload) != record["bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError("ARTIFACT_INTEGRITY_ERROR")
        return payload, record
    except (FileNotFoundError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("ARTIFACT_UNAVAILABLE:" + type(exc).__name__) from exc


def read_chunk(
    identifier: str, offset: int = 0, limit: int = 100000, expected_sha256: str | None = None
) -> dict:
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200000:
        raise ValueError("INVALID_ARTIFACT_CURSOR")
    payload, record = get(identifier)
    if not identifier.startswith("res_"):
        raise ValueError("Only result artifacts can be read as text chunks")
    if expected_sha256 is not None and expected_sha256 != record["sha256"]:
        raise ValueError("ARTIFACT_VERSION_CHANGED")
    if offset and expected_sha256 is None:
        raise ValueError("Continuation requires the complete result SHA")
    text = payload.decode("utf-8")
    if offset >= len(text):
        raise ValueError("INVALID_ARTIFACT_CURSOR")
    chunk = text[offset : offset + limit]
    following = offset + len(chunk)
    return {
        "status": "paged_result",
        "result_id": identifier,
        "payload_sha256": record["sha256"],
        "encoding": "utf-8",
        "offset_unit": "unicode_character",
        "chunk_offset": offset,
        "payload_chunk": chunk,
        "next_offset": following if following < len(text) else None,
        "total_characters": len(text),
        "metadata": record["metadata"],
        "confidence": None,
        "ood_flag": None,
    }


def prune_expired(*, apply=False):
    """Operator-only: report/delete expired result pairs, never attachments or unknown files."""
    root = store_directory()
    selected = []
    with _LOCK, _process_lock(root):
        for path in root.glob("res_*.json"):
            if not re.fullmatch(r"res_[0-9a-f]{32}\.json", path.name):
                continue
            data, meta = _paths(path.stem)
            try:
                record = json.loads(meta.read_bytes())
                expiry = record.get("expires_at_unix")
                if (
                    record.get("artifact_id") != path.stem
                    or type(expiry) not in (int, float)
                    or expiry > time.time()
                ):
                    continue
                if data.resolve().parent != root or meta.resolve().parent != root:
                    raise ValueError("INVALID_ARTIFACT_PATH")
                selected.append(path.stem)
                if apply:
                    data.unlink(missing_ok=True)
                    meta.unlink()
            except (OSError, ValueError, TypeError):
                continue
    return {"applied": apply, "expired_result_ids": selected, "attachments_removed": 0}
