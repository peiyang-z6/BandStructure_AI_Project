"""Operator-only guarded file adapter. This is not registered as an MCP tool."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from mcp_server.resource_limits import enforce_memory_limit

    try:
        limits = enforce_memory_limit()
    except (ValueError, OSError, ImportError):
        raise SystemExit(5)
    try:
        raw = sys.stdin.buffer.read(65537)
        if len(raw) > 65536:
            raise ValueError("ADAPTER_REQUEST_BUDGET")
        request = json.loads(raw)
        from mcp_server.attachment_adapter import _prepare_inline

        path = Path(request["path"])
        if request["operation"] == "prepare":
            result = _prepare_inline(
                path,
                page_number=request.get("page_number"),
                max_dimension=request.get("max_dimension", 3000),
            )
        elif request["operation"] == "count_pdf_pages":
            if not path.is_file() or path.is_symlink() or path.stat().st_size > 256 * 1024 * 1024:
                raise ValueError("INVALID_PDF_SOURCE")
            import pymupdf

            with pymupdf.open(path) as doc:
                result = {"page_count": len(doc)}
        else:
            raise ValueError("INVALID_ADAPTER_OPERATION")
        result["resource_limits"] = limits
        reply = {"ok": True, "result": result}
    except MemoryError:
        raise SystemExit(4)
    except (ValueError, OSError, ImportError, KeyError, IndexError, TypeError) as exc:
        reply = {"ok": False, "error": str(exc)[:300]}
    sys.stdout.write(json.dumps(reply, ensure_ascii=True, allow_nan=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
