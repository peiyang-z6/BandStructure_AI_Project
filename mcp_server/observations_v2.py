"""Reference-aware sparse multi-band observations; missing evidence stays missing."""

import csv
import io
import math
import re
from html import escape
from src.utils.csv_safety import spreadsheet_text


def finite(x):
    try:
        return type(x) in (int, float) and math.isfinite(float(x))
    except (ValueError, OverflowError):
        return False


def review(o):
    invalid = []
    missing = []
    allowed = {
        "schema_version",
        "source",
        "energy_unit",
        "energy_reference",
        "fermi_eV",
        "k_unit",
        "k_distance",
        "segment_ids",
        "bands",
        "calibration_evidence",
        "ambiguities",
        "ocr_text",
        "ocr_corrections",
        "human_confirmed",
        "perception_confidence",
        "path_segments",
        "k_cartesian_invA",
        "physical_metadata",
        "calibration_id",
    }
    invalid.extend("unexpected_field:" + k for k in o if k not in allowed)
    if type(o.get("schema_version")) is not int or o["schema_version"] != 2:
        invalid.append("schema_version")
    source = o.get("source")
    if not isinstance(source, dict):
        missing.append("source")
    else:
        invalid.extend(
            "source." + k
            for k in source
            if k
            not in {
                "document_id",
                "page_number",
                "panel_label",
                "figure_label",
                "material_label",
                "source_sha256",
                "attachment_id",
            }
        )
        for k, limit in [("document_id", 512), ("panel_label", 128)]:
            if (
                not isinstance(source.get(k), str)
                or not source[k].strip()
                or len(source[k]) > limit
            ):
                invalid.append("source." + k)
        if type(source.get("page_number")) is not int or not 1 <= source["page_number"] <= 100000:
            invalid.append("source.page_number")
        for k in ["figure_label", "material_label"]:
            if k in source and (
                not isinstance(source[k], str) or not source[k].strip() or len(source[k]) > 256
            ):
                invalid.append("source." + k)
        if "source_sha256" in source and (
            not isinstance(source["source_sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", source["source_sha256"])
        ):
            invalid.append("source.source_sha256")
        if "attachment_id" in source:
            try:
                from mcp_server.artifacts import get

                _, record = get(source["attachment_id"])
                if not source["attachment_id"].startswith("att_") or record["sha256"] != source.get(
                    "source_sha256"
                ):
                    invalid.append("source.attachment_binding")
            except (ValueError, OSError, TypeError):
                invalid.append("source.attachment_binding")
    if o.get("energy_unit") != "eV":
        invalid.append("energy_unit")
    ref = o.get("energy_reference")
    if not isinstance(ref, dict):
        missing.append("energy_reference")
    else:
        if set(ref) != {"kind", "value_eV", "observed", "evidence"}:
            invalid.append("energy_reference.fields")
        if not isinstance(ref.get("kind"), str) or ref["kind"] not in {"fermi", "vbm", "arbitrary"}:
            invalid.append("energy_reference.kind")
        if not finite(ref.get("value_eV")):
            invalid.append("energy_reference.value_eV")
        if type(ref.get("observed")) is not bool:
            invalid.append("energy_reference.observed")
        elif not ref["observed"]:
            missing.append("energy_reference.observed")
        if (
            not isinstance(ref.get("evidence"), str)
            or not ref["evidence"].strip()
            or len(ref["evidence"]) > 1000
        ):
            invalid.append("energy_reference.evidence")
        if ref.get("kind") == "fermi":
            if not finite(o.get("fermi_eV")):
                missing.append("fermi_eV")
            elif o["fermi_eV"] != ref.get("value_eV"):
                invalid.append("fermi_eV.reference_mismatch")
        elif o.get("fermi_eV") is not None:
            invalid.append("fermi_eV.requires_fermi_reference")
    k = o.get("k_distance")
    s = o.get("segment_ids")
    n = None
    if not isinstance(o.get("k_unit"), str) or o["k_unit"] not in {"relative", "angstrom^-1"}:
        invalid.append("k_unit")
    if not isinstance(k, list) or not 2 <= len(k) <= 8192 or not all(finite(x) for x in k):
        invalid.append("k_distance")
    else:
        n = len(k)
        if not finite(max(k) - min(k)) or max(k) <= min(k):
            invalid.append("k_distance.range")
    if (
        not isinstance(s, list)
        or n is None
        or len(s) != n
        or any(type(x) is not int or x < 0 for x in s)
    ):
        invalid.append("segment_ids")
    else:
        seen = set()
        last = None
        for i, seg in enumerate(s):
            if seg != last:
                if seg in seen:
                    invalid.append("segment_ids.reentered")
                seen.add(seg)
                last = seg
            elif not k[i] > k[i - 1]:
                invalid.append("k_distance.nonmonotonic")
    if "k_cartesian_invA" in o:
        cart = o["k_cartesian_invA"]
        if (
            not isinstance(cart, list)
            or len(cart) != n
            or any(
                not isinstance(row, list)
                or len(row) != 3
                or not all(finite(v) and abs(v) <= 1e6 for v in row)
                for row in cart
            )
        ):
            invalid.append("k_cartesian_invA")
    if "physical_metadata" in o:
        from mcp_server.scientific_data import validate_metadata

        invalid.extend(validate_metadata(o["physical_metadata"]))
    if "path_segments" in o:
        paths = o["path_segments"]
        path_ids = []
        if not isinstance(paths, list) or not 1 <= len(paths) <= 8192:
            invalid.append("path_segments")
        else:
            for path in paths:
                if not isinstance(path, dict) or set(path) != {
                    "segment_id",
                    "start_label",
                    "end_label",
                }:
                    invalid.append("path_segments.fields")
                    continue
                if type(path["segment_id"]) is not int or path["segment_id"] < 0:
                    invalid.append("path_segments.segment_id")
                else:
                    path_ids.append(path["segment_id"])
                if any(
                    not isinstance(path[x], str) or not path[x].strip() or len(path[x]) > 128
                    for x in ["start_label", "end_label"]
                ):
                    invalid.append("path_segments.labels")
            if (
                isinstance(s, list)
                and all(type(x) is int for x in s)
                and path_ids != list(dict.fromkeys(s))
            ):
                invalid.append("path_segments.coverage_or_order")
    bands = o.get("bands")
    ids = set()
    energies = []
    if not isinstance(bands, list) or not 1 <= len(bands) <= 4096:
        invalid.append("bands")
    else:
        if n is not None and len(bands) * n > 262144:
            invalid.append("bands.budget")
        for i, b in enumerate(bands):
            pre = f"bands[{i}]"
            if (
                not isinstance(b, dict)
                or not {"band_id", "role", "energies_eV"} <= set(b)
                or set(b) - {"band_id", "role", "energies_eV", "pixel_points"}
            ):
                invalid.append(pre)
                continue
            name = b["band_id"]
            if not isinstance(name, str) or not name.strip() or len(name) > 128 or name in ids:
                invalid.append(pre + ".band_id")
            elif isinstance(name, str):
                ids.add(name)
            if not isinstance(b["role"], str) or b["role"] not in {
                "valence",
                "conduction",
                "unknown",
            }:
                invalid.append(pre + ".role")
            e = b["energies_eV"]
            if (
                not isinstance(e, list)
                or len(e) != n
                or any(v is not None and not finite(v) for v in e)
            ):
                invalid.append(pre + ".energies_eV")
            else:
                energies.extend(v for v in e if v is not None)
        if not energies:
            missing.append("visible_energy_samples")
        elif not finite(max(energies) - min(energies)):
            invalid.append("bands.energy_range_overflow")
    cal = o.get("calibration_evidence")
    if not isinstance(cal, dict):
        missing.append("calibration_evidence")
    else:
        invalid.extend(
            "calibration_evidence." + x
            for x in cal
            if x not in {"energy_tick_count", "k_anchor_count", "fermi_reference_observed"}
        )
        for x in ["energy_tick_count", "k_anchor_count"]:
            if type(cal.get(x)) is not int or not 0 <= cal[x] <= 10000:
                invalid.append("calibration_evidence." + x)
            elif cal[x] < 2:
                missing.append("calibration_evidence." + x)
        if "fermi_reference_observed" in cal and type(cal["fermi_reference_observed"]) is not bool:
            invalid.append("calibration_evidence.fermi_reference_observed")
        if (
            isinstance(ref, dict)
            and ref.get("kind") == "fermi"
            and cal.get("fermi_reference_observed") is not True
        ):
            missing.append("calibration_evidence.fermi_reference_observed")
    amb = o.get("ambiguities")
    if (
        not isinstance(amb, list)
        or len(amb) > 64
        or any(not isinstance(v, str) or not v.strip() or len(v) > 1000 for v in amb)
    ):
        invalid.append("ambiguities")
    elif amb:
        missing.append("resolve_ambiguities")
    if isinstance(bands, list):
        if any(
            isinstance(b, dict)
            and isinstance(b.get("energies_eV"), list)
            and None in b["energies_eV"]
            for b in bands
        ):
            missing.append("missing_band_samples")
        if any(isinstance(b, dict) and b.get("role") == "unknown" for b in bands):
            missing.append("band_roles")
        roles = [b.get("role") for b in bands if isinstance(b, dict)]
        if not invalid and not _has_fermi_crossing(o):
            if "valence" not in roles:
                missing.append("valence_band_evidence")
            if "conduction" not in roles:
                missing.append("conduction_band_evidence")
    if "human_confirmed" in o and type(o["human_confirmed"]) is not bool:
        invalid.append("human_confirmed")
    if o.get("perception_confidence") is not None and (
        not finite(o["perception_confidence"]) or not 0 <= o["perception_confidence"] <= 1
    ):
        invalid.append("perception_confidence")
    if "ocr_text" in o and (not isinstance(o["ocr_text"], str) or len(o["ocr_text"]) > 200000):
        invalid.append("ocr_text")
    corrections = o.get("ocr_corrections", [])
    if not isinstance(corrections, list) or len(corrections) > 128:
        invalid.append("ocr_corrections")
    else:
        for item in corrections:
            if not isinstance(item, dict) or set(item) != {
                "raw_text",
                "corrected_text",
                "bbox",
                "evidence",
                "reviewer_kind",
                "source_sha256",
                "coordinate_unit",
            }:
                invalid.append("ocr_corrections.fields")
                continue
            if (
                not isinstance(source, dict)
                or not source.get("source_sha256")
                or item["source_sha256"] != source["source_sha256"]
            ):
                invalid.append("ocr_corrections.source_binding")
            if not isinstance(item["coordinate_unit"], str) or item["coordinate_unit"] not in {
                "pixel",
                "pdf_point",
            }:
                invalid.append("ocr_corrections.coordinate_unit")
            if any(
                not isinstance(item[x], str) or not item[x].strip() or len(item[x]) > 1000
                for x in ["raw_text", "corrected_text", "evidence"]
            ):
                invalid.append("ocr_corrections.text")
            if item["reviewer_kind"] != "ai_client":
                invalid.append("ocr_corrections.reviewer_kind")
            box = item["bbox"]
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(finite(x) for x in box)
                or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])
            ):
                invalid.append("ocr_corrections.bbox")
    if not invalid and "calibration_id" in o:
        from mcp_server.evidence import review_calibrated_samples

        invalid.extend(review_calibrated_samples(o))
    return {
        "invalid_fields": list(dict.fromkeys(invalid)),
        "missing_evidence": list(dict.fromkeys(missing)),
        "next_questions": [
            "Provide or correct " + x + " without guessing." for x in dict.fromkeys(missing)
        ],
    }


def _has_fermi_crossing(o):
    """Only observed contiguous samples can establish a crossing; nulls break runs."""
    if (
        not isinstance(o.get("energy_reference"), dict)
        or o["energy_reference"].get("kind") != "fermi"
        or not finite(o.get("fermi_eV"))
    ):
        return False
    segments = o.get("segment_ids", [])
    for band in o.get("bands", []):
        previous = None
        last_segment = None
        for e, s in zip(band.get("energies_eV", []), segments):
            if not finite(e) or s != last_segment:
                previous = None
            last_segment = s
            if not finite(e):
                continue
            delta = e - o["fermi_eV"]
            if delta == 0:
                continue
            if previous is not None and (previous < 0) != (delta < 0):
                return True
            previous = delta
    return False


def _csv_samples(o):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["band_id", "role", "segment_id", "sample_index", "k", "energy_eV", "sample_state"]
    )
    for band in o["bands"]:
        for i, e in enumerate(band["energies_eV"]):
            writer.writerow(
                [
                    spreadsheet_text(band["band_id"]),
                    band["role"],
                    o["segment_ids"][i],
                    i,
                    o["k_distance"][i],
                    "" if e is None else e,
                    "missing" if e is None else "observed",
                ]
            )
    return buf.getvalue()


def export_samples(o):
    k, s, bands = o["k_distance"], o["segment_ids"], o["bands"]
    if len(k) * len(bands) > 4096:
        from mcp_server.artifacts import put

        try:
            record = put(
                _csv_samples(o).encode("utf-8"),
                kind="band_samples_csv",
                metadata={
                    "total_rows": len(k) * len(bands),
                    "source": o["source"],
                    "energy_reference": o["energy_reference"],
                },
            )
        except (ValueError, OSError) as exc:
            return {
                "status": "export_unavailable",
                "csv_text": None,
                "svg": None,
                "error_code": str(exc),
            }
        return {
            "status": "paged_export",
            "csv_text": None,
            "svg": None,
            "result_id": record["artifact_id"],
            "payload_sha256": record["sha256"],
            "total_rows": len(k) * len(bands),
            "message": "Read every UTF-8 CSV chunk with read_result_chunk and verify the complete SHA; no rows were dropped.",
        }
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["band_id", "role", "segment_id", "sample_index", "k", "energy_eV", "sample_state"])
    known = [v for b in bands for v in b["energies_eV"] if v is not None]
    low, high = min(known), max(known)
    energy_range = high - low
    # Give each contiguous segment its own interval; never connect disconnected paths.
    unique = list(dict.fromkeys(s))
    spans = []
    for seg in unique:
        idx = [i for i, x in enumerate(s) if x == seg]
        spans.append((seg, idx, k[idx[0]], k[idx[-1]]))
    total = 600.0
    gap = 12.0
    available = total - gap * (len(spans) - 1)
    if available <= 2 * len(spans):
        return {"status": "csv_only_many_segments", "csv_text": _csv_samples(o), "svg": None}
    lengths = [z - a for _, _, a, z in spans]
    fallback = max(lengths + [1.0]) * 0.01
    lengths = [length if length > 0 else fallback for length in lengths]
    widths = [available * length / sum(lengths) for length in lengths]
    starts = [70 + sum(widths[:i]) + i * gap for i in range(len(spans))]
    positions = {}
    for j, (seg, idx, a, z) in enumerate(spans):
        for i in idx:
            positions[i] = starts[j] + (k[i] - a) / (z - a or 1) * widths[j]
    # Normalize before multiplication; padding raw 1e308 values loses precision.
    sy = lambda e: 45 + 420 * (0.04 + (high - e) / energy_range) / 1.08 if energy_range else 255.0
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 730 535">',
        '<rect width="730" height="535" fill="white"/>',
        '<text x="70" y="23" font-family="sans-serif" font-size="14">Unverified supplied multi-band samples; missing points are not interpolated</text>',
    ]
    for b in bands:
        color = {"valence": "#126da5", "conduction": "#ce7513", "unknown": "#666666"}[b["role"]]
        path = []
        previous = None
        for i, e in enumerate(b["energies_eV"]):
            w.writerow(
                [
                    spreadsheet_text(b["band_id"]),
                    b["role"],
                    s[i],
                    i,
                    k[i],
                    "" if e is None else e,
                    "missing" if e is None else "observed",
                ]
            )
            if e is None:
                previous = None
                continue
            op = "L" if previous is not None and s[previous] == s[i] else "M"
            path.append(f"{op}{positions[i]:.2f},{sy(e):.2f}")
            previous = i
            if op == "M":
                parts.append(
                    f'<circle cx="{positions[i]:.2f}" cy="{sy(e):.2f}" r="1.5" fill="{color}"/>'
                )
        parts.append(
            f'<path d="{" ".join(path)}" fill="none" stroke="{color}" stroke-width="1.1"><title>{escape(b["band_id"])}</title></path>'
        )
    for j, (seg, idx, a, z) in enumerate(spans):
        x = starts[j]
        width = widths[j]
        labels = next((p for p in o.get("path_segments", []) if p["segment_id"] == seg), None)
        label = (
            escape(labels["start_label"] + "–" + labels["end_label"])
            if labels
            else f"segment {seg}"
        )
        parts.append(
            f'<rect x="{x}" y="45" width="{width}" height="420" fill="none" stroke="#aaaaaa"/><text x="{x}" y="487" font-size="11">{label}: {a:.4g} to {z:.4g}</text>'
        )
    for j in range(6 if energy_range else 1):
        e = low + energy_range * (j / 5)
        parts.append(f'<text x="12" y="{sy(e):.2f}" font-size="11">{e:.3g} eV</text>')
    parts.append(
        f'<text x="70" y="518" font-size="12">k: {escape(o["k_unit"])}; energy reference: {escape(o["energy_reference"]["kind"])}; not independent ground truth</text></svg>'
    )
    return {
        "status": "unverified_sample_export",
        "csv_text": buf.getvalue(),
        "svg": "".join(parts),
        "x_axis_scaling": "segment widths proportional to supplied k span; gaps are not physical distances",
    }


def analyze(o):
    checked = review(o)
    base = {
        "confidence": None,
        "ood_flag": None,
        "human_audited": False,
        "eligible_for_scientific_acceptance": False,
        "fermi_eV": None,
        "line_mode_gap_eV": None,
        "line_mode_topology": "unknown",
        "effective_mass_electron_m0": None,
        "effective_mass_hole_m0": None,
        "full_numerical_band_reconstruction": False,
        **checked,
    }
    if checked["invalid_fields"]:
        return {**base, "status": "refused", "error_code": "INVALID_AI_OBSERVATIONS"}
    blocking = set(checked["missing_evidence"]) - {
        "resolve_ambiguities",
        "missing_band_samples",
        "band_roles",
        "valence_band_evidence",
        "conduction_band_evidence",
    }
    if blocking:
        return {**base, "status": "needs_more_evidence"}
    bands = o["bands"]
    missing = sum(v is None for b in bands for v in b["energies_eV"])
    val = [v for b in bands if b["role"] == "valence" for v in b["energies_eV"] if v is not None]
    cond = [
        v for b in bands if b["role"] == "conduction" for v in b["energies_eV"] if v is not None
    ]
    unknown = any(b["role"] == "unknown" for b in bands)
    separation = min(cond) - max(val) if cond and val else None
    ambiguity = bool(o["ambiguities"])
    partial = missing > 0 or unknown or ambiguity or separation is None or separation < 0
    measurement = {}
    if not missing and not ambiguity:
        from src.utils.physics_validator import analyze_numerical_band_data

        raw = {
            "energies_eV": [b["energies_eV"] for b in bands],
            "k_distance": o["k_distance"],
            "segment_ids": o["segment_ids"],
            "k_unit": o["k_unit"],
        }
        if "k_cartesian_invA" in o:
            raw["k_cartesian_invA"] = o["k_cartesian_invA"]
        if not unknown:
            raw["band_roles"] = [b["role"] for b in bands]
        if o["energy_reference"]["kind"] == "fermi":
            raw["fermi_eV"] = o["fermi_eV"]
            measurement = analyze_numerical_band_data(raw)
        elif not unknown:
            raw["reference_eV"] = o["energy_reference"]["value_eV"]
            measurement = analyze_numerical_band_data(raw, reference_from_roles=True)
        if measurement.get("line_mode_topology") == "metal":
            partial = False
    # Unknown physical k identity, occupancies or missing bands cannot establish topology.
    return {
        **base,
        "status": "partial_observations" if partial else "ai_observation_unverified",
        "analysis_kind": "reference_aware_supplied_samples",
        "source": o["source"],
        "energy_reference": o["energy_reference"],
        "source_binding_status": "registered_byte_digest_verified"
        if o["source"].get("attachment_id")
        else "caller_declared_unverified",
        "path_segments": o.get("path_segments"),
        "fermi_eV": o.get("fermi_eV") if o["energy_reference"]["kind"] == "fermi" else None,
        "line_mode_gap_eV": measurement.get("line_mode_gap_eV") if not partial else None,
        "line_mode_topology": measurement.get("line_mode_topology", "unknown")
        if not partial
        else "unknown",
        "effective_mass_electron_m0": measurement.get("effective_mass_electron_m0")
        if not partial
        else None,
        "effective_mass_hole_m0": measurement.get("effective_mass_hole_m0")
        if not partial
        else None,
        "diagnostics": measurement.get("diagnostics"),
        "sampled_edge_separation_eV": separation if not ambiguity else None,
        "gap_scope": "supplied_role_labelled_samples_only; not full band coverage or global BZ gap",
        "observation_summary": {"band_count": len(bands), "k_point_count": len(o["k_distance"])},
        "coverage": {
            "missing_samples": missing,
            "observed_samples": len(bands) * len(o["k_distance"]) - missing,
            "input_complete": missing == 0,
            "physical_band_set_complete": None,
        },
        "reported_perception_confidence": o.get("perception_confidence"),
        "caller_claimed_human_confirmation": o.get("human_confirmed", False),
        "ambiguities": o["ambiguities"],
        "ocr_corrections": o.get("ocr_corrections", []),
        "physical_metadata": o.get("physical_metadata"),
        "exports": export_samples(o),
        "limitations": [
            "Supplied band roles and reference evidence are AI/caller observations, not human authentication.",
            "Null samples and path discontinuities are preserved; no physical band identities are invented.",
            "Effective mass and directness require independent physical k identity evidence.",
        ],
    }
