"""Generate a hash-bound, non-installed bundle of local MCP client configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.client_configs import build_client_configs


def _bytes(value) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path,
                        help="absolute Python executable for the isolated MCP environment")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    bundle = build_client_configs(args.python, args.project_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    hashes = {}
    for name, value in bundle.items():
        if name == "manifest.json":
            continue
        payload = _bytes(value)
        (args.output_dir / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    manifest = dict(bundle["manifest.json"], files_sha256=hashes)
    manifest_payload = _bytes(manifest)
    (args.output_dir / "manifest.json").write_bytes(manifest_payload)
    print(json.dumps({
        "status": "generated_not_installed",
        "output_dir": str(args.output_dir),
        "files": len(hashes) + 1,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
