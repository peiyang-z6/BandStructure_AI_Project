"""Private bounded worker for native PDF/image parsing; not an MCP server."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from mcp_server.resource_limits import enforce_memory_limit

    try:
        limits = enforce_memory_limit()
    except (OSError, ValueError, ImportError):
        raise SystemExit(5)
    payload = sys.stdin.buffer.read(20 * 1024 * 1024 + 1)
    if not payload or len(payload) > 20 * 1024 * 1024:
        raise SystemExit(2)
    try:
        request = json.loads(payload)
        tool = request["tool"]
        arguments = request["arguments"]
        if not isinstance(arguments, dict):
            raise ValueError("arguments")
        from mcp_server import server

        if tool == "extract_document":
            result = server.extract_document(**arguments)
        elif tool == "extract_band_from_image":
            result = server.extract_band_from_image(**arguments)
        elif tool == "extract_pdf_payload":
            result = {
                "status": "ok",
                "payload_json": server._complete_pdf_payload(
                    server._decode_upload(arguments["payload_base64"]),
                    arguments["page_number"],
                    arguments.get("panel_selection"),
                ),
            }
        else:
            raise ValueError("tool")
        result["resource_limits"] = limits
        output = json.dumps(
            result, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
        if len(output) > 34 * 1024 * 1024:
            raise ValueError("result")
    except MemoryError:
        raise SystemExit(4)
    except (
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
        json.JSONDecodeError,
        RecursionError,
    ):
        raise SystemExit(3)
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
