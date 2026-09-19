"""AI-native document-vision contracts for the MCP-first direction."""

from __future__ import annotations

import copy
import asyncio
import json


def complete_observations() -> dict:
    return {
        "source": {
            "document_id": "sha256:synthetic-ai-observation",
            "page_number": 1,
            "panel_label": "(a)",
            "figure_label": "Figure 1(a)",
            "material_label": "synthetic",
        },
        "energy_unit": "eV",
        "fermi_eV": 0.0,
        "k_unit": "relative",
        "k_distance": [0.0, 0.5, 1.0],
        "segment_ids": [0, 0, 0],
        "bands": [
            {"role": "valence", "energies_eV": [-2.0, -1.0, -2.0]},
            {"role": "conduction", "energies_eV": [2.0, 1.0, 2.0]},
        ],
        "calibration_evidence": {
            "energy_tick_count": 3,
            "k_anchor_count": 3,
            "fermi_reference_observed": True,
        },
        "ambiguities": [],
        "perception_confidence": 0.93,
        "human_confirmed": False,
    }


def test_plan_makes_ai_native_vision_primary_and_local_ocr_fallback():
    from mcp_server.server import plan_band_image_analysis

    result = plan_band_image_analysis()
    assert result["status"] == "ok"
    assert result["primary_perception_path"] == "ai_client_native_document_vision_ocr"
    assert result["local_fallback_tools"] == ["extract_document", "extract_band_from_image"]
    assert result["confidence"] is None and result["ood_flag"] is None
    assert "calibration_evidence" in result["observation_contract"]["required"]


def test_partial_observations_return_actionable_missing_evidence_without_physics():
    from mcp_server.server import analyze_visual_observations, request_missing_evidence

    partial = {
        "source": {"document_id": "attachment:1", "page_number": 7},
        "energy_unit": "eV",
        "ambiguities": ["Fermi line may be the zero tick"],
    }
    result = analyze_visual_observations(partial)
    assert result["status"] == "needs_more_evidence"
    assert "resolve_ambiguities" in result["missing_evidence"]
    assert "bands" in result["missing_evidence"]
    assert result.get("line_mode_gap_eV") is None
    assert result["next_questions"]
    assert result["confidence"] is None and result["ood_flag"] is None

    requested = request_missing_evidence(partial)
    assert requested["status"] == "needs_more_evidence"
    assert requested["missing_evidence"] == result["missing_evidence"]


def test_complete_ai_observations_are_measured_but_never_promoted_to_audit():
    from mcp_server.server import analyze_visual_observations, request_missing_evidence

    observations = complete_observations()
    result = analyze_visual_observations(observations)
    assert result["status"] == "ai_observation_unverified"
    assert result["analysis_kind"] == "ai_observation_analytic_measurement"
    assert result["line_mode_gap_eV"] == 2.0
    assert result["line_mode_topology"] == "direct"
    assert result["reported_perception_confidence"] == 0.93
    assert result["confidence"] is None and result["ood_flag"] is None
    assert result["human_audited"] is False
    assert result["calibration_status"] == "ai_reported_complete_unverified"
    assert result["source"] == observations["source"]

    ready = request_missing_evidence(observations)
    assert ready["status"] == "ready_for_limited_measurement"
    assert ready["missing_evidence"] == []
    assert ready["next_questions"] == []


def test_ai_reported_ambiguity_blocks_quantitative_measurement():
    from mcp_server.server import analyze_visual_observations

    observations = complete_observations()
    observations["ambiguities"] = ["one trace may be a grid line"]
    result = analyze_visual_observations(observations)
    assert result["status"] == "needs_more_evidence"
    assert "resolve_ambiguities" in result["missing_evidence"]
    assert result.get("line_mode_gap_eV") is None


def test_invalid_ai_observation_types_are_refused_not_coerced():
    from mcp_server.server import analyze_visual_observations

    mutations = []
    value = complete_observations()
    value["source"]["page_number"] = True
    mutations.append(value)
    value = complete_observations()
    value["perception_confidence"] = 1.2
    mutations.append(value)
    value = complete_observations()
    value["bands"][0]["energies_eV"][1] = "-1"
    mutations.append(value)
    value = complete_observations()
    value["calibration_evidence"]["fermi_reference_observed"] = 1
    mutations.append(value)
    value = complete_observations()
    value["human_confirmed"] = "false"
    mutations.append(value)
    value = complete_observations()
    value["ocr_text"] = "x" * 200001
    mutations.append(value)
    value = complete_observations()
    value["k_cartesian_invA"] = [[0.0, 0.0, 0.0]]
    mutations.append(value)
    value = complete_observations()
    value["unbounded_context"] = {"ignored": True}
    mutations.append(value)
    value = complete_observations()
    value["source"]["raw_document"] = "must-not-be-carried"
    mutations.append(value)

    for observations in mutations:
        result = analyze_visual_observations(observations)
        assert result["status"] == "refused"
        assert result["error_code"] == "INVALID_AI_OBSERVATIONS"
        assert result["confidence"] is None and result["ood_flag"] is None


def test_ai_observation_contract_rejects_inconsistent_curve_lengths():
    from mcp_server.server import analyze_visual_observations

    observations = complete_observations()
    observations["bands"][1]["energies_eV"] = [2.0, 1.0]
    result = analyze_visual_observations(observations)
    assert result["status"] == "refused"
    assert "bands[1].energies_eV" in result["invalid_fields"]


def test_caller_human_claim_does_not_become_independent_human_audit():
    from mcp_server.server import analyze_visual_observations

    observations = complete_observations()
    observations["human_confirmed"] = True
    result = analyze_visual_observations(observations)
    assert result["caller_claimed_human_confirmation"] is True
    assert result["human_audited"] is False


def test_ai_observation_input_is_not_mutated():
    from mcp_server.server import analyze_visual_observations

    observations = complete_observations()
    before = copy.deepcopy(observations)
    analyze_visual_observations(observations)
    assert observations == before


def test_audit_candidate_is_deterministic_pending_and_never_self_approves():
    from mcp_server.server import prepare_human_audit_candidate

    observations = complete_observations()
    observations["human_confirmed"] = True
    first = prepare_human_audit_candidate(observations, "evaluation")
    second = prepare_human_audit_candidate(copy.deepcopy(observations), "evaluation")
    assert first["status"] == "pending_human"
    assert first["candidate_record"] == second["candidate_record"]
    assert first["candidate_record"]["record_id"].startswith("mcp-audit-")
    assert len(first["candidate_record"]["observation_sha256"]) == 64
    assert first["candidate_record"]["independent_human_review"] is None
    assert first["eligible_for_scientific_acceptance"] is False
    assert first["human_audited"] is False
    assert first["confidence"] is None and first["ood_flag"] is None


def test_partial_real_observation_can_enter_queue_but_not_scientific_gate():
    from mcp_server.server import prepare_human_audit_candidate

    partial = {
        "source": {
            "document_id": "sha256:real-paper",
            "page_number": 7,
            "panel_label": "(a)",
        },
        "energy_unit": "eV",
        "fermi_eV": 0.0,
        "k_unit": "relative",
        "calibration_evidence": {
            "energy_tick_count": 2,
            "k_anchor_count": 8,
            "fermi_reference_observed": True,
        },
        "ambiguities": ["band identities are not traced continuously"],
    }
    result = prepare_human_audit_candidate(partial, "evaluation")
    assert result["status"] == "pending_human"
    assert "bands" in result["missing_evidence"]
    assert "resolve_ambiguities" in result["missing_evidence"]
    assert result["candidate_record"]["intended_split"] == "evaluation"
    assert result["eligible_for_scientific_acceptance"] is False


def test_invalid_audit_candidate_is_refused_and_split_is_strict():
    from mcp_server.server import prepare_human_audit_candidate

    invalid = complete_observations()
    invalid["source"]["page_number"] = False
    assert prepare_human_audit_candidate(invalid)["status"] == "refused"
    for split in ("test", "Evaluation", 1, None):
        result = prepare_human_audit_candidate(complete_observations(), split)
        assert result["status"] == "refused"
        assert result["error_code"] == "INVALID_AUDIT_SPLIT"


def test_human_audit_resource_and_prompt_forbid_ai_self_approval():
    from mcp_server.server import mcp

    async def probe():
        resources = await mcp.list_resources()
        assert "band://human_audit_protocol" in {str(item.uri) for item in resources}
        parts = await mcp.read_resource("band://human_audit_protocol")
        protocol = json.loads(list(parts)[0].content)
        assert protocol["ai_candidate_status"] == "pending_human"
        assert protocol["formal_evaluation_required_min"] == 200
        prompts = await mcp.list_prompts()
        assert "band_human_audit_guide" in {item.name for item in prompts}
        prompt = await mcp.get_prompt("band_human_audit_guide")
        assert "never" in prompt.messages[0].content.text.lower()

    asyncio.run(probe())
