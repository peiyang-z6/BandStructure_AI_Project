"""Exercise real Streamable HTTP. Receipts never include authentication tokens."""

import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]


async def verify(url, config_path, output):
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    config = json.loads(config_path.read_bytes())
    headers = config["servers"]["bandstructure"]["headers"]
    report = {"status": "running", "url": url, "calls": [], "scientific_acceptance": False}
    async with httpx.AsyncClient(timeout=120, trust_env=False) as bare:
        assert (await bare.get(url.replace("/mcp", "/healthz"))).status_code == 200
        assert (await bare.post(url, json={})).status_code == 401
        assert (
            await bare.post(url, json={}, headers={"Authorization": "Bearer invalid"})
        ).status_code == 401
        assert (
            await bare.post(
                url, json={}, headers={**headers, "Origin": "https://untrusted.example"}
            )
        ).status_code == 403
        report["negative_controls"] = ["missing_token_401", "wrong_token_401", "foreign_origin_403"]
    async with httpx.AsyncClient(headers=headers, timeout=120, trust_env=False) as transport:
        async with streamable_http_client(url, http_client=transport) as streams:
            async with ClientSession(
                streams[0], streams[1], read_timeout_seconds=timedelta(seconds=120)
            ) as session:
                init = await session.initialize()
                report["server"] = init.serverInfo.model_dump(mode="json")
                assert len((await session.list_tools()).tools) == 17
                assert len((await session.list_resources()).resources) == 5
                for name, args, expected in [
                    ("get_service_status", {}, "ok"),
                    (
                        "analyze_electronic_data",
                        {
                            "data": {
                                "data_kind": "line_mode",
                                "energies_eV": [[-1, 1], [2, 3]],
                                "k_distance": [0, 1],
                                "segment_ids": [0, 0],
                                "fermi_eV": 0,
                                "k_unit": "relative",
                            }
                        },
                        "ok",
                    ),
                    ("inspect_attachment", {"attachment_id": "../../not-an-attachment"}, "refused"),
                    (
                        "extract_band_from_image",
                        {
                            "payload_base64": base64.b64encode(
                                (ROOT / "mcp_server/examples/synthetic_band.png").read_bytes()
                            ).decode()
                        },
                        "requires_calibration",
                    ),
                ]:
                    wire = (await session.call_tool(name, args)).model_dump(
                        mode="json", by_alias=True
                    )
                    result = wire.get("structuredContent", {})
                    report["calls"].append(
                        {
                            "tool": name,
                            "status": result.get("status"),
                            "isError": wire.get("isError"),
                            "result": result,
                        }
                    )
                    assert not wire.get("isError") and result.get("status") == expected, (
                        name,
                        result,
                    )
                    if name == "get_service_status":
                        assert (
                            result["delivery_version"] == "2.0.0" and result["transport"] == "http"
                        )
                    if name == "analyze_electronic_data":
                        assert result["band_count"] == 2 and result["line_mode_topology"] == "metal"
                    if name == "extract_band_from_image":
                        assert (
                            result["worker_isolated"]
                            and result["resource_limits"]["memory_limit_enforced"]
                        )
    report["status"] = "passed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "passed",
                "tool_calls": len(report["calls"]),
                "negative_controls": report["negative_controls"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument(
        "--config", type=Path, required=True, help="Private generated vscode.mcp.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    asyncio.run(verify(a.url, a.config, a.output))
