"""Geometry checks and field-level evidence review, never human authentication."""

import hashlib
import io
import json
import math
import re

import numpy as np

from mcp_server.artifacts import get, put


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e6
    except OverflowError:
        return False


def _fit(ticks, coordinate, maximum, *, allow_symbolic=False):
    if not isinstance(ticks, list) or not 2 <= len(ticks) <= 32:
        raise ValueError("TWO_OR_MORE_AXIS_ANCHORS_REQUIRED")
    coords = []
    values = []
    corrected = False
    for tick in ticks:
        if not isinstance(tick, dict) or set(tick) - {
            "pixel",
            "value",
            "raw_text",
            "corrected_text",
            "correction_evidence",
            "value_basis",
        }:
            raise ValueError("INVALID_AXIS_ANCHOR")
        if (
            not _number(tick.get("pixel"))
            or not 0 <= tick["pixel"] < maximum
            or not _number(tick.get("value"))
        ):
            raise ValueError("INVALID_AXIS_ANCHOR")
        raw = tick.get("raw_text")
        if not isinstance(raw, str) or not 0 < len(raw) <= 128:
            raise ValueError("RAW_AXIS_LABEL_REQUIRED")
        text = raw
        symbolic = tick.get("value_basis") == "relative_plot_position"
        if "value_basis" in tick and not symbolic:
            raise ValueError("INVALID_ANCHOR_VALUE_BASIS")
        if symbolic:
            if not allow_symbolic or not 0 <= tick["value"] <= 1 or "corrected_text" in tick:
                raise ValueError("SYMBOLIC_LABEL_REQUIRES_RELATIVE_K_COORDINATES")
        if "corrected_text" in tick:
            if (
                not isinstance(tick.get("correction_evidence"), str)
                or not 1 <= len(tick["correction_evidence"]) <= 1000
            ):
                raise ValueError("AXIS_CORRECTION_EVIDENCE_REQUIRED")
            text = tick["corrected_text"]
            corrected = True
        if not symbolic and (
            not isinstance(text, str)
            or not re.fullmatch(r"[+\-−]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", text.strip())
        ):
            raise ValueError("AXIS_LABEL_REQUIRES_VISUAL_CORRECTION")
        if not symbolic and float(text.replace("−", "-")) != tick["value"]:
            raise ValueError("AXIS_LABEL_VALUE_MISMATCH")
        coords.append(tick["pixel"])
        values.append(tick["value"])
    order = np.argsort(coords)
    x = np.array(coords)[order]
    y = np.array(values)[order]
    if any(t.get("value_basis") == "relative_plot_position" for t in ticks):
        if not all(t.get("value_basis") == "relative_plot_position" for t in ticks):
            raise ValueError("MIXED_AXIS_VALUE_BASIS")
        if not np.allclose(y, (x - x[0]) / (x[-1] - x[0]), atol=1e-8, rtol=0):
            raise ValueError("RELATIVE_K_MUST_MATCH_PLOT_POSITION")
    if np.any(np.diff(x) <= 0) or not (np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0)):
        raise ValueError("AXIS_ANCHORS_NOT_STRICTLY_MONOTONIC")
    slope, intercept = np.polyfit(x, y, 1)
    error = float(np.max(np.abs(slope * x + intercept - y)))
    tolerance = max(1e-6, abs(float(slope)) * 0.5)
    if error > tolerance:
        raise ValueError("AXIS_NOT_LINEAR_WITHIN_HALF_PIXEL")
    return {
        "coordinate": coordinate,
        "slope": float(slope),
        "intercept": float(intercept),
        "max_residual": error,
        "declared_corrections_used": corrected,
        "anchors": ticks,
        "tolerance": tolerance,
    }


def validate_axes(attachment_id, energy_ticks, k_ticks, k_unit="relative"):
    from PIL import Image

    if not isinstance(k_unit, str) or k_unit not in {"relative", "angstrom^-1"}:
        raise ValueError("INVALID_K_UNIT")
    payload, record = get(attachment_id, expected_kind="image")
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
    if width * height > 4096 * 4096:
        raise ValueError("IMAGE_PIXEL_BUDGET")
    energy = _fit(energy_ticks, "y", height)
    k = _fit(k_ticks, "x", width, allow_symbolic=k_unit == "relative")
    result = {
        "status": "axis_geometry_consistent_unverified",
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "attachment_id": attachment_id,
        "image_size": [width, height],
        "energy_unit": "eV",
        "k_unit": k_unit,
        "energy_transform": energy,
        "k_transform": k,
        "human_audited": False,
        "eligible_for_scientific_acceptance": False,
        "confidence": None,
        "ood_flag": None,
        "limits": [
            "Verifies registered bytes and declared anchor geometry, not OCR truth or physical branch identity."
        ],
    }
    saved = put(json.dumps(result, allow_nan=False).encode(), kind="axis_calibration")
    return {**result, "calibration_id": saved["artifact_id"]}


def review_calibrated_samples(observation):
    """Check every supplied coordinate against the registered calibration record."""
    identifier = observation.get("calibration_id")
    if identifier is None:
        return []
    try:
        payload, _ = get(identifier, expected_kind="axis_calibration")
        cal = json.loads(payload)
        source = observation.get("source", {})
        if (
            source.get("attachment_id") != cal["attachment_id"]
            or source.get("source_sha256") != cal["source_sha256"]
        ):
            return ["calibration.source_binding"]
        if (
            observation.get("energy_unit") != cal["energy_unit"]
            or observation.get("k_unit") != cal["k_unit"]
        ):
            return ["calibration.units"]
        y, x = cal["energy_transform"], cal["k_transform"]
        width, height = cal["image_size"]
        for band in observation.get("bands", []):
            points = band.get("pixel_points")
            energies = band.get("energies_eV", [])
            if not isinstance(points, list) or len(points) != len(energies):
                return ["calibration.pixel_points"]
            for i, (point, energy) in enumerate(zip(points, energies)):
                if energy is None:
                    if point is not None:
                        return ["calibration.missing_sample"]
                    continue
                if (
                    not isinstance(point, list)
                    or len(point) != 2
                    or not all(_number(v) for v in point)
                    or not 0 <= point[0] < width
                    or not 0 <= point[1] < height
                ):
                    return ["calibration.pixel_bounds"]
                if abs(y["slope"] * point[1] + y["intercept"] - energy) > y["tolerance"]:
                    return ["calibration.energy_mismatch"]
                if (
                    abs(x["slope"] * point[0] + x["intercept"] - observation["k_distance"][i])
                    > x["tolerance"]
                ):
                    return ["calibration.k_mismatch"]
        return []
    except (ValueError, OSError, TypeError, KeyError, IndexError):
        return ["calibration.unavailable_or_invalid"]


def review_material_report(report):
    result = {
        "status": "evidence_contract_valid_unverified",
        "issues": [],
        "confidence": None,
        "ood_flag": None,
        "human_audited": False,
        "eligible_for_scientific_acceptance": False,
    }
    try:
        if len(json.dumps(report, allow_nan=False)) > 256000:
            raise ValueError("report budget")
    except (ValueError, TypeError, RecursionError):
        return {**result, "status": "refused", "issues": ["invalid_or_oversized_report"]}
    if (
        not isinstance(report, dict)
        or set(report) - {"document_kind", "claims"}
        or not isinstance(report.get("document_kind"), str)
        or report.get("document_kind") not in {"computational", "experimental", "mixed", "review"}
    ):
        return {**result, "status": "refused", "issues": ["document_kind"]}
    claims = report.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 128:
        return {**result, "status": "refused", "issues": ["claims"]}
    kinds = {
        "host_composition",
        "tested_dopants",
        "crystal_structure",
        "lattice_parameters",
        "experimental_synthesis",
        "cited_synthesis",
        "property",
    }
    for i, claim in enumerate(claims):
        prefix = f"claims[{i}]"
        if not isinstance(claim, dict) or set(claim) - {
            "kind",
            "value",
            "basis",
            "evidence",
            "reason",
        }:
            result["issues"].append(prefix + ".fields")
            continue
        if (
            not isinstance(claim.get("kind"), str)
            or not isinstance(claim.get("basis"), str)
            or claim.get("kind") not in kinds
            or claim.get("basis") not in {"reported", "inferred", "not_reported"}
        ):
            result["issues"].append(prefix + ".kind_or_basis")
            continue
        if claim["basis"] == "not_reported":
            if claim.get("value") is not None:
                result["issues"].append(prefix + ".not_reported_value")
            continue
        if claim["basis"] == "inferred" and (
            not isinstance(claim.get("reason"), str) or not claim["reason"].strip()
        ):
            result["issues"].append(prefix + ".inference_reason")
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 16:
            result["issues"].append(prefix + ".evidence")
            continue
        for item in evidence:
            if not isinstance(item, dict) or not {
                "document_id",
                "page_number",
                "excerpt",
                "origin",
            } <= set(item):
                result["issues"].append(prefix + ".source")
                continue
            if (
                not isinstance(item["document_id"], str)
                or not 1 <= len(item["document_id"]) <= 512
                or type(item["page_number"]) is not int
                or item["page_number"] < 1
                or not isinstance(item["excerpt"], str)
                or not 1 <= len(item["excerpt"]) <= 2000
                or not isinstance(item["origin"], str)
                or item["origin"] not in {"current_paper", "cited_work", "author_data"}
            ):
                result["issues"].append(prefix + ".source")
            if claim["kind"] == "cited_synthesis" and item["origin"] != "cited_work":
                result["issues"].append(prefix + ".cited_origin")
        if (
            claim["kind"] == "experimental_synthesis"
            and report["document_kind"] == "computational"
            and claim["basis"] == "reported"
        ):
            result["issues"].append(prefix + ".computational_paper_is_not_an_experimental_sop")
    if result["issues"]:
        result["status"] = "needs_more_evidence"
    result["claims"] = claims
    result["verification_scope"] = (
        "Field-level attribution and consistency, not independent verification of quoted source contents."
    )
    return result
