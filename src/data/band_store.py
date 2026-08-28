"""Shared, source-agnostic persistence helpers for band-structure records."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping

import h5py
import numpy as np


_SEMANTIC_PROVENANCE_KEYS = {
    "material_id",
    "source",
    "source_efermi_absolute",
    "source_sha256",
    "source_url",
    "download_url",
}
_INTERNAL_ATTR_KEYS = {"_semantic_sha256", "spacegroup"}


@dataclass(frozen=True)
class BandRecordSaveResult:
    """Outcome of a canonical/variant-aware band-record commit."""

    status: str
    material_id: str
    semantic_sha256: str
    path: str


class BandStoreLockError(RuntimeError):
    """Raised when another process already owns a cache writer lock."""


@contextmanager
def exclusive_file_lock(lock_path: str) -> Iterator[None]:
    """Acquire a fail-fast cross-process advisory lock for one cache writer."""
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    handle = open(lock_path, "a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BandStoreLockError(
                    f"writer lock is already held: {lock_path}"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise BandStoreLockError(
                    f"writer lock is already held: {lock_path}"
                ) from exc
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def to_jsonable(value: Any) -> Any:
    """Convert numpy-heavy records to objects accepted by ``json``."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _stored_scalar(value: Any) -> Any:
    serial = to_jsonable(value)
    if isinstance(serial, (dict, list)):
        return json.dumps(
            serial,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return serial


def _update_semantic_digest(digest: Any, key: str, value: Any) -> None:
    digest.update(key.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest.update(b"array\0")
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(contiguous.tobytes(order="C"))
    else:
        digest.update(b"scalar\0")
        digest.update(
            json.dumps(
                _stored_scalar(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        )
    digest.update(b"\0")


def record_semantic_sha256(record: Mapping[str, Any]) -> str:
    """Hash physical record content while excluding mutable metadata/provenance."""
    digest = hashlib.sha256()
    for key in sorted(record):
        if (
            key == "metadata"
            or key in _SEMANTIC_PROVENANCE_KEYS
            or record[key] is None
        ):
            continue
        _update_semantic_digest(digest, key, record[key])
    return digest.hexdigest()


def _semantic_attr_value(value: Any) -> Any:
    serial = to_jsonable(value)
    if isinstance(serial, bytes):
        serial = serial.decode("utf-8", errors="replace")
    if isinstance(serial, str):
        stripped = serial.strip()
        if stripped.startswith(("[", "{")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
    return serial


def _group_semantic_sha256(group: h5py.Group) -> str:
    metadata_keys = set()
    if "metadata" in group and isinstance(group["metadata"], h5py.Group):
        metadata_keys.update(str(key) for key in group["metadata"].attrs)

    semantic_values: Dict[str, Any] = {}
    for key in sorted(group.keys()):
        value = group[key]
        if key == "metadata" or not isinstance(value, h5py.Dataset):
            continue
        semantic_values[key] = value[...]
    for key in sorted(group.attrs):
        if (
            key in metadata_keys
            or key in _INTERNAL_ATTR_KEYS
            or key in _SEMANTIC_PROVENANCE_KEYS
        ):
            continue
        semantic_values[key] = _semantic_attr_value(group.attrs[key])

    digest = hashlib.sha256()
    for key in sorted(semantic_values):
        _update_semantic_digest(digest, key, semantic_values[key])
    return digest.hexdigest()


def read_material_ids(h5_path: str) -> set[str]:
    if not os.path.exists(h5_path):
        return set()
    with h5py.File(h5_path, "r") as handle:
        return {key for key in handle.keys() if not key.startswith("__tmp__")}


def _write_group(group: h5py.Group, record: Mapping[str, Any]) -> None:
    for key, value in record.items():
        if key == "metadata":
            metadata_group = group.create_group("metadata")
            for meta_key, meta_value in value.items():
                if meta_value is None:
                    continue
                serial = to_jsonable(meta_value)
                if isinstance(serial, (dict, list)):
                    serial = json.dumps(serial, ensure_ascii=False)
                metadata_group.attrs[meta_key] = serial
                group.attrs[meta_key] = serial
            if value.get("spacegroup_number") is not None:
                group.attrs["spacegroup"] = value["spacegroup_number"]
            continue

        if isinstance(value, np.ndarray):
            group.create_dataset(key, data=value, compression="gzip", compression_opts=4)
        elif isinstance(value, (list, tuple, dict)):
            group.attrs[key] = json.dumps(to_jsonable(value), ensure_ascii=False)
        elif value is not None:
            group.attrs[key] = value


def _variant_h5_path(h5_path: str) -> str:
    canonical = os.path.abspath(h5_path)
    stem = os.path.splitext(os.path.basename(canonical))[0]
    return os.path.join(
        os.path.dirname(canonical),
        "provenance",
        "h5_variants",
        f"{stem}_variants.h5",
    )


def _save_variant_record(
    h5_path: str,
    record: Mapping[str, Any],
    semantic_sha256: str,
    canonical_semantic_sha256: str,
) -> BandRecordSaveResult:
    material_id = str(record["material_id"])
    variant_path = _variant_h5_path(h5_path)
    os.makedirs(os.path.dirname(variant_path), exist_ok=True)
    variant_name = f"{material_id}__{semantic_sha256}"
    temp_name = f"__tmp__{uuid.uuid4().hex}"
    with h5py.File(variant_path, "a") as handle:
        if variant_name in handle:
            return BandRecordSaveResult(
                status="duplicate_variant",
                material_id=material_id,
                semantic_sha256=semantic_sha256,
                path=variant_path,
            )
        temp_group = handle.create_group(temp_name)
        try:
            _write_group(temp_group, record)
            temp_group.attrs["_semantic_sha256"] = semantic_sha256
            temp_group.attrs["canonical_material_id"] = material_id
            temp_group.attrs[
                "canonical_semantic_sha256"
            ] = canonical_semantic_sha256
            temp_group.attrs["variant_reason"] = "same material_id, different physical content"
            handle.flush()
            handle.move(temp_name, variant_name)
            handle.flush()
        except Exception:
            if temp_name in handle:
                del handle[temp_name]
            raise
    return BandRecordSaveResult(
        status="variant",
        material_id=material_id,
        semantic_sha256=semantic_sha256,
        path=variant_path,
    )


def save_band_record(
    h5_path: str,
    record: Mapping[str, Any],
) -> BandRecordSaveResult:
    """Commit a new canonical material without rewriting identical content."""
    material_id = str(record["material_id"])
    if "/" in material_id:
        raise ValueError(f"material_id cannot contain '/': {material_id}")
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    semantic_sha256 = record_semantic_sha256(record)
    temp_name = f"__tmp__{uuid.uuid4().hex}"
    with exclusive_file_lock(h5_path + ".lock"):
        with h5py.File(h5_path, "a") as handle:
            if material_id in handle:
                canonical_semantic_sha256 = _group_semantic_sha256(
                    handle[material_id]
                )
                if canonical_semantic_sha256 == semantic_sha256:
                    return BandRecordSaveResult(
                        status="duplicate",
                        material_id=material_id,
                        semantic_sha256=semantic_sha256,
                        path=h5_path,
                    )
                return _save_variant_record(
                    h5_path=h5_path,
                    record=record,
                    semantic_sha256=semantic_sha256,
                    canonical_semantic_sha256=canonical_semantic_sha256,
                )
            temp_group = handle.create_group(temp_name)
            try:
                _write_group(temp_group, record)
                temp_group.attrs["_semantic_sha256"] = semantic_sha256
                handle.flush()
                handle.move(temp_name, material_id)
                handle.flush()
            except Exception:
                if temp_name in handle:
                    del handle[temp_name]
                raise
    return BandRecordSaveResult(
        status="canonical",
        material_id=material_id,
        semantic_sha256=semantic_sha256,
        path=h5_path,
    )


def atomic_write_json(path: str, payload: Any) -> None:
    """Atomically replace one JSON sidecar without exposing partial content."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(os.path.dirname(os.path.abspath(path)))
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _fsync_directory(directory: str) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_remove(path: str) -> None:
    if not os.path.exists(path):
        return
    os.remove(path)
    _fsync_directory(os.path.dirname(os.path.abspath(path)))


def _metadata_value_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _metadata_record_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()


def _metadata_variant_path(path: str) -> str:
    absolute = os.path.abspath(path)
    stem = os.path.splitext(os.path.basename(absolute))[0]
    return os.path.join(
        os.path.dirname(absolute),
        "provenance",
        "metadata_variants",
        f"{stem}_variants.json",
    )


def _metadata_transaction_path(path: str) -> str:
    absolute = os.path.abspath(path)
    stem = os.path.splitext(os.path.basename(absolute))[0]
    return os.path.join(
        os.path.dirname(absolute),
        "provenance",
        "metadata_transactions",
        f"{stem}_pending.json",
    )


def _recover_metadata_transaction(path: str) -> None:
    pending_path = _metadata_transaction_path(path)
    if not os.path.exists(pending_path):
        return
    with open(pending_path, "r", encoding="utf-8") as handle:
        transaction = json.load(handle)
    if not isinstance(transaction, dict):
        raise ValueError(f"invalid metadata transaction journal: {pending_path}")
    canonical_records = transaction.get("canonical_records")
    variant_records = transaction.get("variant_records")
    if not isinstance(canonical_records, list) or not isinstance(
        variant_records, list
    ):
        raise ValueError(f"invalid metadata transaction payload: {pending_path}")
    if variant_records:
        atomic_write_json(_metadata_variant_path(path), variant_records)
    atomic_write_json(path, canonical_records)
    _durable_remove(pending_path)


def _metadata_conflict_sha256(
    material_id: str,
    conflicts: Mapping[str, Any],
) -> str:
    return _metadata_record_sha256(
        {
            "material_id": material_id,
            "conflicts": to_jsonable(conflicts),
        }
    )


def _metadata_audit_signature(item: Mapping[str, Any]) -> tuple[str, str, str]:
    material_id = str(item.get("material_id") or "")
    conflicts = item.get("conflicts")
    conflict_sha256 = str(item.get("conflict_sha256") or "")
    if not conflict_sha256 and isinstance(conflicts, Mapping):
        conflict_sha256 = _metadata_conflict_sha256(material_id, conflicts)
    return (
        material_id,
        str(item.get("incoming_sha256") or ""),
        conflict_sha256,
    )


def _merge_metadata_record(
    canonical: Mapping[str, Any],
    incoming: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    merged = dict(to_jsonable(canonical))
    conflicts: Dict[str, Dict[str, Any]] = {}
    for key, raw_value in to_jsonable(incoming).items():
        if key == "material_id":
            continue
        if _metadata_value_is_empty(raw_value):
            continue
        canonical_value = merged.get(key)
        if _metadata_value_is_empty(canonical_value):
            merged[key] = raw_value
            continue
        if _canonical_json(canonical_value) == _canonical_json(raw_value):
            continue
        conflicts[key] = {
            "canonical": canonical_value,
            "incoming": raw_value,
        }
    return merged, conflicts


def _build_metadata_conflict_audit(
    material_id: str,
    canonical_before: Mapping[str, Any],
    incoming: Mapping[str, Any],
    conflicts: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "material_id": material_id,
        "canonical_sha256": _metadata_record_sha256(canonical_before),
        "incoming_sha256": _metadata_record_sha256(incoming),
        "conflict_sha256": _metadata_conflict_sha256(material_id, conflicts),
        "conflicts": to_jsonable(conflicts),
        "incoming_record": to_jsonable(incoming),
    }


def _fold_metadata_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], list[Dict[str, Any]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    conflict_audits: list[Dict[str, Any]] = []
    for item in records:
        material_id = str(item.get("material_id") or "")
        if not material_id:
            continue
        incoming = to_jsonable(item)
        if material_id not in by_id:
            by_id[material_id] = incoming
            continue
        canonical_before = dict(by_id[material_id])
        merged, conflicts = _merge_metadata_record(canonical_before, incoming)
        by_id[material_id] = merged
        if conflicts:
            conflict_audits.append(
                _build_metadata_conflict_audit(
                    material_id,
                    canonical_before,
                    incoming,
                    conflicts,
                )
            )
    return by_id, conflict_audits


def _commit_metadata_state(
    path: str,
    by_id: Mapping[str, Mapping[str, Any]],
    conflict_audits: Iterable[Mapping[str, Any]],
) -> None:
    audits = [to_jsonable(item) for item in conflict_audits]
    variant_path = _metadata_variant_path(path)
    previous: list[Dict[str, Any]] = []
    if os.path.exists(variant_path):
        with open(variant_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(
                f"metadata variant sidecar must contain a JSON list: {variant_path}"
            )
        previous = [item for item in loaded if isinstance(item, dict)]
    variants_changed = False
    if audits:
        signatures = {_metadata_audit_signature(item) for item in previous}
        for audit in audits:
            signature = _metadata_audit_signature(audit)
            if signature not in signatures:
                previous.append(audit)
                signatures.add(signature)
                variants_changed = True
    canonical_records = [
        to_jsonable(by_id[material_id]) for material_id in sorted(by_id)
    ]
    if variants_changed:
        pending_path = _metadata_transaction_path(path)
        atomic_write_json(
            pending_path,
            {
                "schema_version": 1,
                "canonical_records": canonical_records,
                "variant_records": previous,
            },
        )
        atomic_write_json(variant_path, previous)
        atomic_write_json(path, canonical_records)
        _durable_remove(pending_path)
    else:
        atomic_write_json(path, canonical_records)


def merge_metadata_json(path: str, records: Iterable[Dict[str, Any]]) -> None:
    _recover_metadata_transaction(path)
    existing: list[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(f"metadata sidecar must contain a JSON list: {path}")
        existing = [item for item in loaded if isinstance(item, dict)]
    by_id, conflict_audits = _fold_metadata_records([*existing, *records])
    _commit_metadata_state(path, by_id, conflict_audits)


def _jsonable_h5_attr(value: Any) -> Any:
    return _semantic_attr_value(value)


def _synchronize_metadata_with_hdf5_unlocked(
    h5_path: str,
    metadata_path: str,
) -> Dict[str, int]:
    """Make canonical metadata IDs exactly match persisted canonical HDF5 IDs."""
    existing: list[Dict[str, Any]] = []
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(
                f"metadata sidecar must contain a JSON list: {metadata_path}"
            )
        existing = [item for item in loaded if isinstance(item, dict)]

    h5_ids = read_material_ids(h5_path)
    existing_by_id, conflict_audits = _fold_metadata_records(existing)
    existing_ids = set(existing_by_id)
    if (
        existing_ids == h5_ids
        and len(existing) == len(existing_ids)
        and not conflict_audits
    ):
        return {
            "h5_ids": len(h5_ids),
            "metadata_ids": len(existing_ids),
            "backfilled": 0,
            "removed": 0,
        }

    missing_ids = sorted(h5_ids - existing_ids)
    embedded_records: list[Dict[str, Any]] = []
    if missing_ids:
        with h5py.File(h5_path, "r") as handle:
            for material_id in missing_ids:
                group = handle[material_id]
                record: Dict[str, Any] = {"material_id": material_id}
                if "metadata" in group and isinstance(group["metadata"], h5py.Group):
                    for key, value in group["metadata"].attrs.items():
                        record[str(key)] = _jsonable_h5_attr(value)
                embedded_records.append(record)

    retained_and_embedded = [
        existing_by_id[material_id]
        for material_id in sorted(h5_ids & existing_ids)
    ] + embedded_records
    reconciled_by_id, embedded_conflicts = _fold_metadata_records(
        retained_and_embedded
    )
    _commit_metadata_state(
        metadata_path,
        reconciled_by_id,
        [*conflict_audits, *embedded_conflicts],
    )

    return {
        "h5_ids": len(h5_ids),
        "metadata_ids": len(h5_ids),
        "backfilled": len(missing_ids),
        "removed": len(existing_ids - h5_ids),
    }


def synchronize_metadata_with_hdf5(
    h5_path: str,
    metadata_path: str,
) -> Dict[str, int]:
    """Reconcile metadata while excluding concurrent canonical writers."""
    with exclusive_file_lock(h5_path + ".lock"):
        _recover_metadata_transaction(metadata_path)
        return _synchronize_metadata_with_hdf5_unlocked(h5_path, metadata_path)
