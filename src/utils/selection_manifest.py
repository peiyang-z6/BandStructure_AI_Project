"""Selection-manifest validation shared by the pipeline and the finetune CLI.

This module intentionally has no heavy dependencies (no TensorFlow, no
matplotlib). The full pipeline imports it directly so that a `scripts`
package name collision elsewhere on the import path cannot break the final
artifact content gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_state_path(manifest: Dict[str, object], label: str) -> Path:
    states = manifest.get("states")
    if not isinstance(states, dict) or not isinstance(states.get(label), dict):
        raise RuntimeError(f"Inner selection manifest missing {label} state")
    path = Path(str(states[label].get("path", "")))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def validate_inner_selection_manifest(output_dir: str) -> Dict[str, object]:
    """Fail closed before outer evaluation unless all frozen states still match."""
    manifest_path = Path(output_dir) / "inner_selection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing inner selection manifest before outer evaluation: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("outer_test_accessed") is not False:
        raise RuntimeError("Inner selection manifest is not blind to outer test")
    if manifest.get("monitor") != "val_loss":
        raise RuntimeError("Inner selection manifest does not use aggregate val_loss")
    states = manifest.get("states")
    if not isinstance(states, dict):
        raise RuntimeError("Inner selection manifest has no frozen states")
    resolved_state_paths = []
    for label in ("best", "last", "accepted"):
        item = states.get(label)
        if not isinstance(item, dict):
            raise RuntimeError(f"Inner selection manifest missing {label} state")
        path = manifest_state_path(manifest, label)
        resolved_state_paths.append(path)
        if not path.is_file():
            raise RuntimeError(f"Frozen {label} state is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != item.get("sha256"):
            raise RuntimeError(
                f"Frozen {label} state hash mismatch: {actual_hash} != {item.get('sha256')}"
            )
        if path.stat().st_size != int(item.get("bytes", -1)):
            raise RuntimeError(f"Frozen {label} state size mismatch: {path}")
    if len(set(resolved_state_paths)) != 3:
        raise RuntimeError(
            "Frozen supervised best, last, and accepted paths are not distinct"
        )
    return manifest
