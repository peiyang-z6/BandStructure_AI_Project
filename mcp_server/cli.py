"""Local operator entry points. File paths are never exposed as MCP tool arguments."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def serve():
    """Start stdio or authenticated Streamable HTTP with bounded parsing."""
    parser = argparse.ArgumentParser(description=serve.__doc__)
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    os.environ["BAND_MCP_TRANSPORT"] = args.transport
    os.environ.setdefault("BAND_MCP_UPLOAD_ISOLATION", "1")
    if os.environ["BAND_MCP_UPLOAD_ISOLATION"] != "1":
        raise ValueError(
            "MCP launch requires isolated parsing; use low-level libraries only for trusted in-process development"
        )
    os.environ.pop("BAND_MCP_WORKER", None)
    import numpy  # noqa: F401 -- prewarm before Windows stdio reader threads.
    from scipy.interpolate import PchipInterpolator  # noqa: F401
    from mcp_server.server import mcp

    if args.transport == "http":
        from mcp_server.http_server import serve as serve_http
        serve_http(args.host, args.port)
    else:
        mcp.run(transport="stdio")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    attach = sub.add_parser(
        "attach", help="Prepare a chosen local attachment and print its opaque ID"
    )
    attach.add_argument("path", type=Path)
    pages = attach.add_mutually_exclusive_group()
    pages.add_argument("--page", type=int)
    pages.add_argument("--all-pages", action="store_true", help="Render all PDF pages, at most 50")
    attach.add_argument("--max-dimension", type=int, default=3000)
    sub.add_parser(
        "doctor", help="Print version and installed dependencies without starting an LLM"
    )
    prune = sub.add_parser(
        "prune-expired", help="List expired result IDs; never removes registered attachments"
    )
    prune.add_argument(
        "--apply", action="store_true", help="Delete only the listed expired result pairs"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "attach":
            from mcp_server.attachment_adapter import prepare

            if args.all_pages:
                if args.path.suffix.lower() != ".pdf":
                    raise ValueError("--all-pages requires a PDF")
                from mcp_server.attachment_adapter import count_pdf_pages

                count = count_pdf_pages(args.path)
                if not 1 <= count <= 50:
                    raise ValueError("Use explicit --page for PDFs exceeding 50 pages")
                # Emit each committed ID immediately so partial failures do not orphan successful work.
                for page in range(1, count + 1):
                    print(
                        json.dumps(
                            prepare(args.path, page_number=page, max_dimension=args.max_dimension)
                        )
                    )
                return
            result = prepare(args.path, page_number=args.page, max_dimension=args.max_dimension)
        elif args.command == "doctor":
            from mcp_server.server import get_service_status

            dependencies = {}
            for name in [
                "mcp",
                "numpy",
                "scipy",
                "Pillow",
                "jsonschema",
                "PyMuPDF",
                "opencv-python",
                "rapidocr-onnxruntime",
            ]:
                try:
                    dependencies[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    dependencies[name] = None
            result = {
                "service": get_service_status(),
                "installed_dependencies": dependencies,
                "scientific_acceptance": False,
                "os_sandboxed": False,
            }
        else:
            from mcp_server.artifacts import prune_expired

            result = prune_expired(apply=args.apply)
        print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    except (ValueError, OSError, ImportError, KeyError, IndexError, TypeError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


def attach_main():
    import sys

    main(["attach", *sys.argv[1:]])


if __name__ == "__main__":
    main()
