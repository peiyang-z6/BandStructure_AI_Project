"""Verify operator-selected real attachments through stdio, preserving private receipts.

No files are downloaded, and no result is promoted to independent scientific truth.
"""

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


async def verify(paths, output_dir):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp_server.attachment_adapter import prepare

    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "integration_not_scientific_acceptance",
        "status": "incomplete",
        "attachments": [],
        "calls": [],
    }

    def save():
        (output_dir / "receipts.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )

    try:
        for path in paths:
            report["attachments"].append(prepare(path))
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
        with (output_dir / "stderr.txt").open("w", encoding="utf-8") as errors:
            async with stdio_client(params, errlog=errors) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()

                    async def call(name, args):
                        started = time.monotonic()
                        reply = await asyncio.wait_for(session.call_tool(name, args), 120)
                        wire = reply.model_dump(mode="json", by_alias=True)
                        report["calls"].append(
                            {
                                "tool": name,
                                "arguments": args,
                                "seconds": time.monotonic() - started,
                                "reply": wire,
                            }
                        )
                        save()
                        assert not wire.get("isError"), name
                        result = wire["structuredContent"]
                        assert result["status"] not in {"refused", "unavailable"}, result
                        return result

                    report["service"] = await call("get_service_status", {})
                    for item in report["attachments"]:
                        identifier = item["attachment_id"]
                        kind = item["kind"]
                        await call("inspect_attachment", {"attachment_id": identifier})
                        operation = (
                            "scientific"
                            if kind in {"structure", "electronic_data"}
                            else "document"
                            if kind == "pdf"
                            else "image_band"
                        )
                        result = await call(
                            "analyze_attachment",
                            {"attachment_id": identifier, "operation": operation},
                        )
                        item["analysis_summary"] = {
                            k: v
                            for k, v in result.items()
                            if k
                            in {
                                "status",
                                "band_count",
                                "k_point_count",
                                "sampled_mesh_gap_eV",
                                "sampled_direct_gap_eV",
                                "panel_candidates_total",
                                "source_sha256",
                            }
                        }
                        if operation == "scientific":
                            exported = await call(
                                "export_attachment", {"attachment_id": identifier}
                            )
                            offset = 0
                            parts = []
                            while True:
                                chunk = await call(
                                    "read_result_chunk",
                                    {
                                        "result_id": exported["result_id"],
                                        "offset": offset,
                                        "expected_sha256": exported["payload_sha256"],
                                        "limit": 80000,
                                    },
                                )
                                parts.append(chunk["payload_chunk"])
                                offset = chunk["next_offset"]
                                if offset is None:
                                    break
                            payload = "".join(parts).encode("utf-8")
                            assert hashlib.sha256(payload).hexdigest() == item["source_sha256"]
                            original = json.loads(payload)
                            if kind == "electronic_data":
                                assert len(original["energies_eV"]) == result["band_count"]
                            item["export_verified"] = True
        report["status"] = "passed"
    finally:
        save()
    return {
        "status": report["status"],
        "call_count": len(report["calls"]),
        "attachments": report["attachments"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(verify(args.paths, args.output_dir)), ensure_ascii=True))


if __name__ == "__main__":
    main()
