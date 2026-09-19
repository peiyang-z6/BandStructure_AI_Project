"""Exercise the real local MCP protocol. Fixtures are explicit, never trained/model outputs."""

from __future__ import annotations

import argparse
import asyncio
import base64
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
TOOLS = {
    "get_service_status",
    "analyze_band_structure",
    "validate_physics",
    "extract_document",
    "extract_band_from_image",
    "get_material_knowledge",
    "plan_band_image_analysis",
    "request_missing_evidence",
    "analyze_visual_observations",
    "prepare_human_audit_candidate",
}
TOOLS |= {
    "analyze_electronic_data",
    "inspect_attachment",
    "analyze_attachment",
    "export_attachment",
    "read_result_chunk",
    "validate_axis_calibration",
    "review_material_evidence",
}


async def run_demo(
    server_python: str,
    output_dir: Path,
    paper_scan: Path | None = None,
    ai_observations_path: Path | None = None,
) -> dict:
    """Run all exposed tools plus resources/prompts, preserving success or failed receipts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cases": {},
        "called_tools": [],
        "source_sha256": {},
        "scientific_acceptance": "NOT_ESTABLISHED",
        "fixture_kind": "synthetic_and_public_reference_only",
    }
    for relative in [
        "mcp_server/server.py",
        "src/utils/physics_validator.py",
        "src/vision/multi_format_parser.py",
        "scripts/demo_mcp_client.py",
    ]:
        report["source_sha256"][relative] = hashlib.sha256(
            (ROOT / relative).read_bytes()
        ).hexdigest()

    def save():
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )

    save()
    examples = ROOT / "mcp_server/examples"
    image = base64.b64encode((examples / "synthetic_band.png").read_bytes()).decode("ascii")
    manual = json.loads((examples / "synthetic_band_calibration.json").read_text(encoding="utf-8"))
    document = paper_scan or examples / "synthetic_text.png"
    document_bytes = document.read_bytes()
    report["document"] = {
        "path": str(document),
        "sha256": hashlib.sha256(document_bytes).hexdigest(),
        "kind": "real_public_scan" if paper_scan else "synthetic_text",
    }
    external_observations: list[dict] = []
    if ai_observations_path is not None:
        observation_bytes = Path(ai_observations_path).read_bytes()
        if len(observation_bytes) > 1024 * 1024:
            raise ValueError("AI observation file exceeds 1 MiB client limit")
        loaded_observations = json.loads(observation_bytes)
        if isinstance(loaded_observations, dict):
            external_observations = [loaded_observations]
        elif (
            isinstance(loaded_observations, list)
            and 1 <= len(loaded_observations) <= 500
            and all(isinstance(item, dict) for item in loaded_observations)
        ):
            external_observations = loaded_observations
        else:
            raise ValueError("AI observation file must contain one object or 1-500 objects")
        report["external_ai_observations"] = {
            "path": str(ai_observations_path),
            "sha256": hashlib.sha256(observation_bytes).hexdigest(),
            "record_count": len(external_observations),
        }
    numeric = {
        "energies_eV": [[-2.0, -1.0, -2.0], [2.0, 1.0, 2.0]],
        "k_distance": [0.0, 0.5, 1.0],
        "segment_ids": [0, 0, 0],
        "fermi_eV": 0.0,
        "k_unit": "relative",
    }
    import os
    from mcp_server.artifacts import put

    store = str((output_dir / "artifact_store").resolve())
    previous = os.environ.get("BAND_MCP_STORE_DIR")
    os.environ["BAND_MCP_STORE_DIR"] = store
    try:
        image_record = put(
            (examples / "synthetic_band.png").read_bytes(),
            kind="image",
            attachment=True,
            ttl_seconds=None,
        )
        scientific_record = put(
            json.dumps({"data_kind": "line_mode", **numeric}).encode(),
            kind="electronic_data",
            attachment=True,
            ttl_seconds=None,
        )
    finally:
        if previous is None:
            os.environ.pop("BAND_MCP_STORE_DIR", None)
        else:
            os.environ["BAND_MCP_STORE_DIR"] = previous
    params = StdioServerParameters(
        command=server_python,
        args=["-B", str(ROOT / "mcp_server/server.py")],
        env={
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "BAND_MCP_UPLOAD_ISOLATION": "1",
            "BAND_MCP_STORE_DIR": store,
        },
    )
    try:
        with (output_dir / "server.stderr.txt").open("w", encoding="utf-8") as errlog:
            async with stdio_client(params, errlog=errlog) as streams:
                async with ClientSession(*streams) as session:
                    report["initialize"] = (await session.initialize()).model_dump(
                        mode="json", by_alias=True
                    )
                    tools = await session.list_tools()
                    report["discovered_tools"] = sorted(t.name for t in tools.tools)
                    assert set(report["discovered_tools"]) == TOOLS
                    report["schemas"] = tools.model_dump(mode="json", by_alias=True)

                    async def call(case, name, arguments, expected_status):
                        start = time.monotonic()
                        reply = await asyncio.wait_for(
                            session.call_tool(name, arguments), timeout=60
                        )
                        # SDK1 uses camelCase attributes; SDK2 uses snake_case.
                        # Read the stable MCP wire aliases, not Python attribute names.
                        wire = reply.model_dump(mode="json", by_alias=True)
                        structured = wire.get("structuredContent")
                        report["called_tools"].append(name)
                        report["cases"][case] = structured
                        report.setdefault("protocol_replies", {})[case] = wire
                        report.setdefault("seconds", {})[case] = time.monotonic() - start
                        save()
                        assert not wire["isError"], (case, wire)
                        assert isinstance(structured, dict), case
                        allowed = (
                            {expected_status}
                            if isinstance(expected_status, str)
                            else set(expected_status)
                        )
                        assert structured["status"] in allowed, case
                        assert structured["confidence"] is None
                        assert structured["ood_flag"] is None
                        return structured

                    await call("status", "get_service_status", {}, "ok")
                    await call(
                        "electronic_line",
                        "analyze_electronic_data",
                        {"data": {"data_kind": "line_mode", **numeric}},
                        "ok",
                    )
                    await call(
                        "attachment_inspection",
                        "inspect_attachment",
                        {"attachment_id": image_record["artifact_id"]},
                        "ok",
                    )
                    await call(
                        "attachment_analysis",
                        "analyze_attachment",
                        {
                            "attachment_id": scientific_record["artifact_id"],
                            "operation": "scientific",
                        },
                        "ok",
                    )
                    exported = await call(
                        "attachment_export",
                        "export_attachment",
                        {"attachment_id": scientific_record["artifact_id"]},
                        "paged_export",
                    )
                    chunk = await call(
                        "read_export",
                        "read_result_chunk",
                        {"result_id": exported["result_id"]},
                        "paged_result",
                    )
                    assert (
                        json.loads(chunk["payload_chunk"])["energies_eV"] == numeric["energies_eV"]
                    )
                    await call(
                        "axis_anchors",
                        "validate_axis_calibration",
                        {
                            "attachment_id": image_record["artifact_id"],
                            "energy_ticks": [
                                {"pixel": 10, "value": 2, "raw_text": "2"},
                                {"pixel": 20, "value": 1, "raw_text": "1"},
                            ],
                            "k_ticks": [
                                {"pixel": 10, "value": 0, "raw_text": "0"},
                                {"pixel": 20, "value": 1, "raw_text": "1"},
                            ],
                        },
                        "axis_geometry_consistent_unverified",
                    )
                    await call(
                        "material_evidence",
                        "review_material_evidence",
                        {
                            "report": {
                                "document_kind": "computational",
                                "claims": [
                                    {
                                        "kind": "experimental_synthesis",
                                        "basis": "not_reported",
                                        "value": None,
                                    }
                                ],
                            }
                        },
                        "evidence_contract_valid_unverified",
                    )
                    await call("ai_analysis_plan", "plan_band_image_analysis", {}, "ok")
                    await call(
                        "ai_missing_evidence",
                        "request_missing_evidence",
                        {
                            "observations": {
                                "source": {
                                    "document_id": "demo:synthetic",
                                    "page_number": 1,
                                    "panel_label": "(a)",
                                },
                                "energy_unit": "eV",
                                "ambiguities": ["Fermi marker not yet observed"],
                            }
                        },
                        "needs_more_evidence",
                    )
                    ai_observations = json.loads(
                        (examples / "ai_native_observations.json").read_text(encoding="utf-8")
                    )
                    ai_observations["source"]["document_id"] = "demo:synthetic"
                    observed = await call(
                        "ai_visual_observations",
                        "analyze_visual_observations",
                        {"observations": ai_observations},
                        "ai_observation_unverified",
                    )
                    assert observed["line_mode_gap_eV"] == 2.0
                    await call(
                        "ai_audit_candidate",
                        "prepare_human_audit_candidate",
                        {"observations": ai_observations, "intended_split": "evaluation"},
                        "pending_human",
                    )
                    for index, external_observation in enumerate(external_observations, start=1):
                        suffix = "" if len(external_observations) == 1 else f"_{index}"
                        external_review = await call(
                            f"external_ai_missing_evidence{suffix}",
                            "request_missing_evidence",
                            {"observations": external_observation},
                            {"needs_more_evidence", "ready_for_limited_measurement"},
                        )
                        await call(
                            f"external_ai_audit_candidate{suffix}",
                            "prepare_human_audit_candidate",
                            {"observations": external_observation, "intended_split": "evaluation"},
                            "pending_human",
                        )
                        if external_review["status"] == "ready_for_limited_measurement":
                            await call(
                                f"external_ai_analysis{suffix}",
                                "analyze_visual_observations",
                                {"observations": external_observation},
                                {"ai_observation_unverified", "needs_more_evidence"},
                            )
                    measured = await call(
                        "numerical",
                        "analyze_band_structure",
                        {"input_type": "numerical_data", "data": numeric},
                        "ok",
                    )
                    assert measured["line_mode_gap_eV"] == 2.0
                    public = await call(
                        "public_material",
                        "analyze_band_structure",
                        {"input_type": "material_id", "data": "pymatgen-Cu2O_361"},
                        "ok",
                    )
                    assert abs(public["line_mode_gap_eV"] - 0.5047) <= 1e-5
                    await call("validation", "validate_physics", {"band_data": numeric}, "ok")
                    await call(
                        "ocr",
                        "extract_document",
                        {
                            "payload_base64": base64.b64encode(document_bytes).decode("ascii"),
                            "kind": "pdf" if paper_scan else "image",
                            "max_pages": 1,
                        },
                        "ok",
                    )
                    assert report["cases"]["ocr"]["text"].strip()
                    await call(
                        "uncalibrated_image",
                        "extract_band_from_image",
                        {"payload_base64": image},
                        "requires_calibration",
                    )
                    reconstructed = await call(
                        "calibrated_image",
                        "analyze_band_structure",
                        {
                            "input_type": "image",
                            "data": image,
                            "annotations": manual["annotations"],
                            "calibration": manual["calibration"],
                        },
                        "caller_calibrated_unverified",
                    )
                    assert abs(reconstructed["line_mode_gap_eV"] - 1.5) <= 1e-5
                    await call(
                        "knowledge", "get_material_knowledge", {"material_system": "general"}, "ok"
                    )
                    await call(
                        "invalid_upload",
                        "extract_document",
                        {"payload_base64": "%%%", "kind": "pdf"},
                        "refused",
                    )
                    await call(
                        "invalid_json",
                        "analyze_band_structure",
                        {"input_type": "numerical_data", "data": '{"a":0,"a":1}'},
                        "refused",
                    )
                    await call(
                        "unknown_material",
                        "analyze_band_structure",
                        {"input_type": "material_id", "data": "../../data/cache"},
                        "unavailable",
                    )
                    resources = await session.list_resources()
                    report["resources"] = {}
                    for resource in resources.resources:
                        response = await session.read_resource(resource.uri)
                        report["resources"][str(resource.uri)] = response.model_dump(
                            mode="json", by_alias=True
                        )
                    prompts = await session.list_prompts()
                    report["prompts"] = {}
                    for prompt in prompts.prompts:
                        report["prompts"][prompt.name] = (
                            await session.get_prompt(prompt.name)
                        ).model_dump(mode="json", by_alias=True)
                    report["prompt"] = report["prompts"]["band_analysis_guide"]
                    assert len(report["resources"]) == 5 and len(report["prompts"]) == 2
                    assert all(item["messages"] for item in report["prompts"].values())
        assert set(report["called_tools"]) == TOOLS
        if external_observations:
            candidates = [
                value["candidate_record"]
                for name, value in report["cases"].items()
                if name.startswith("external_ai_audit_candidate")
            ]
            queue = {
                "schema_version": 1,
                "status": "pending_human",
                "eligible_for_scientific_acceptance": False,
                "records": candidates,
            }
            queue_bytes = json.dumps(queue, indent=2, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            queue_path = output_dir / "pending_human_candidates.json"
            queue_path.write_bytes(queue_bytes)
            report["pending_human_queue"] = {
                "path": str(queue_path),
                "sha256": hashlib.sha256(queue_bytes).hexdigest(),
                "record_count": len(candidates),
                "eligible_for_scientific_acceptance": False,
            }
        report["status"] = "passed"
        report["tool_calls"] = len(report["called_tools"])
        report["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        return report
    except BaseException as exc:
        report.update(status="failed", exception_type=type(exc).__name__, error=str(exc))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paper-scan", type=Path)
    parser.add_argument(
        "--ai-observations",
        type=Path,
        help="client-side JSON observations produced by native AI document vision/OCR",
    )
    args = parser.parse_args()
    result = asyncio.run(
        asyncio.wait_for(
            run_demo(args.server_python, args.output_dir, args.paper_scan, args.ai_observations),
            180,
        )
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "tool_calls": result["tool_calls"],
                "public_gap_eV": result["cases"]["public_material"]["line_mode_gap_eV"],
                "report": str(args.output_dir / "report.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
