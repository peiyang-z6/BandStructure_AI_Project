"""Private six-paper integration receipts. Documents are data, never instructions.

The operator adapter transfers bytes. The MCP extracts bounded evidence. Host AI
interpretation and independent scientific verification are separate later steps.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPERS = [
    "1-PRXEn-2025-Tc.pdf",
    "2-ACSEL-2022-Cells.pdf",
    "3-JMCA-2022-SSE.pdf",
    "4-ACEAMI-2025-Semiconductor.pdf",
    "5-EES-2020-Solar.pdf",
    "6-JMCA-2025-ZT.pdf",
]


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


async def run(args):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp_server.attachment_adapter import count_pdf_pages, prepare

    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["BAND_MCP_STORE_DIR"] = str(args.store.resolve())
    summary = {
        "scope": "six_supplied_papers_integration_not_blind_evaluation",
        "status": "running",
        "calls": [],
        "papers": [],
        "scientific_acceptance": False,
    }
    for name in PAPERS:
        source = args.sources / name
        planning = {"pages": count_pdf_pages(source)}
        registered = prepare(source)
        directory = args.output / name.split("-")[0]
        directory.mkdir()
        (directory / "pages").mkdir()
        paper = {
            "filename": name,
            "planning": planning,
            "attachment": registered,
            "pages_complete": 0,
            "errors": [],
        }
        summary["papers"].append(paper)
        save_json(directory / "source.json", paper)
    save_json(args.output / "summary.json", summary)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-B", str(ROOT / "mcp_server/server.py")],
        env={
            **os.environ,
            "BAND_MCP_UPLOAD_ISOLATION": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    with (args.output / "server.stderr.txt").open("w", encoding="utf-8") as log:
        async with stdio_client(params, errlog=log) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                save_json(
                    args.output / "tools.json",
                    (await session.list_tools()).model_dump(mode="json", by_alias=True),
                )

                async def call(name, arguments, receipt):
                    started = time.monotonic()
                    response = await asyncio.wait_for(session.call_tool(name, arguments), 125)
                    wire = response.model_dump(mode="json", by_alias=True)
                    record = {
                        "tool": name,
                        "arguments": arguments,
                        "elapsed_seconds": time.monotonic() - started,
                        "response": wire,
                    }
                    save_json(receipt, record)
                    result = wire.get("structuredContent", {})
                    summary["calls"].append(
                        {
                            "receipt": receipt.relative_to(args.output).as_posix(),
                            "tool": name,
                            "status": result.get("status"),
                            "isError": wire.get("isError"),
                            "elapsed_seconds": record["elapsed_seconds"],
                        }
                    )
                    save_json(args.output / "summary.json", summary)
                    return result

                summary["service"] = await call(
                    "get_service_status", {}, args.output / "service.json"
                )
                for index, paper in enumerate(summary["papers"], 1):
                    directory = args.output / str(index)
                    all_text = []
                    await call(
                        "inspect_attachment",
                        {"attachment_id": paper["attachment"]["attachment_id"]},
                        directory / "inspect.json",
                    )
                    for page in range(1, paper["planning"]["pages"] + 1):
                        result = await call(
                            "analyze_attachment",
                            {
                                "attachment_id": paper["attachment"]["attachment_id"],
                                "operation": "document",
                                "page_number": page,
                            },
                            directory / "pages" / f"{page:02}.json",
                        )
                        if result.get("status") == "ok":
                            text = result.get("text", "")
                            all_text.append(f"\n\n===== PDF PAGE {page} =====\n{text}")
                            (directory / "pages" / f"{page:02}.txt").write_text(
                                text, encoding="utf-8"
                            )
                            paper["pages_complete"] += 1
                        else:
                            paper["errors"].append(
                                {
                                    "page": page,
                                    "status": result.get("status"),
                                    "error_code": result.get("error_code"),
                                }
                            )
                    (directory / "full_text.txt").write_text("".join(all_text), encoding="utf-8")
                    print(
                        json.dumps(
                            {
                                "paper": paper["filename"],
                                "pages_complete": paper["pages_complete"],
                                "expected_pages": paper["planning"]["pages"],
                                "errors": paper["errors"],
                            },
                            ensure_ascii=True,
                        ),
                        flush=True,
                    )
                    save_json(directory / "source.json", paper)
                    save_json(args.output / "summary.json", summary)
    summary["status"] = (
        "passed"
        if not any(c["isError"] for c in summary["calls"]) and all(
            not p["errors"] and p["pages_complete"] == p["planning"]["pages"]
            for p in summary["papers"]
        )
        else "issues_found"
    )
    save_json(args.output / "summary.json", summary)
    print(
        json.dumps({"status": summary["status"], "tool_calls": len(summary["calls"])}), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
