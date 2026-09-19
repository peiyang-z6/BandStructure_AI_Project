"""AFLOW AFLUX adapter for public line-mode band-structure downloads.

AFLOW is used as a second first-class E(k) source. The adapter maps its public
``*_bandsdata.json.xz`` files into the same record contract as MPAdapter while
retaining source provenance and the real cumulative k-path coordinate.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import os
import random
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np

from .band_store import merge_metadata_json, save_band_record, to_jsonable


SOURCE_QA_RULE_VERSION = "aflow-source-qa-v1"
SOURCE_FINGERPRINT_VERSION = "aflow-k-energy-f32-v1"
# Explicit, bounded forensic evidence, NOT a general index/occupancy detector.
# Full E(k)+k content at the historical float32 storage precision is the key;
# material_id, amplitude, rank inversions and provider labels are NOT keys.
SOURCE_QUARANTINE = {
    "4744452fa657cfd15b2fa82cd7720fe3d01e188d35a28f0462c98a6dc84bbb1e": {
        "evidence_version": "published-response-quarantine-20260909-v2",
        "material_id_observed": "aflow-46374655e4730ce2",
        "compressed_sha256": "fdd89582c412d6e5a4b9502781e5400e97715075ae8591842a9b5c3d1c67b4aa",
        "source_url": "https://aflowlib.duke.edu/AFLOWDATA/LIB3_WEB/In_dPTe/ICSD_60260.ABC/In_dPTe.ICSD_60260.ABC_bandsdata.json.xz",
        "evidence_path": "artifacts/reports/remediation_20260909/cycle1_reviewed_evidence/independent_source/review_report.json",
        "evidence_report_sha256": "cf8448d579a486c2752fec191dc105271f7734f9c82cb52cbe849bae1f927750",
        "evidence_json_pointer": "/qa_fingerprint_binding_table/0",
        "reason": "source_quarantine_published_response_v1",
        "review_status": "verified_invalid_published_energy",
        "scope": "exact full k/energy content; not a population screening threshold",
    },
    "556c3f291ff693d3f8c641e3f002b9ae7fec51c6db277efb113cdb3da7cdfd7a": {
        "evidence_version": "published-response-quarantine-20260909-v2",
        "material_id_observed": "aflow-540aee2190a21eec",
        "compressed_sha256": "d4a9e3730dbc079e44895acaf86a374ae1adc9ccc171fc89a06541605222f8fe",
        "source_url": "https://aflowlib.duke.edu/AFLOWDATA/LIB1_WEB/P:PAW_PBE_KIN:SCAN:06Sep2000/ICSD_68326.A/P:PAW_PBE_KIN:SCAN:06Sep2000.ICSD_68326.A_bandsdata.json.xz",
        "evidence_path": "artifacts/reports/remediation_20260909/cycle1_reviewed_evidence/independent_source/review_report.json",
        "evidence_report_sha256": "cf8448d579a486c2752fec191dc105271f7734f9c82cb52cbe849bae1f927750",
        "evidence_json_pointer": "/qa_fingerprint_binding_table/1",
        "reason": "source_quarantine_published_response_v1",
        "review_status": "verified_invalid_published_energy",
        "scope": "exact full k/energy content; not a population screening threshold",
    },
    "fef2f64003f4d4b4914ec5034bb9036a4e93040da1b1c5b056ebb0e5fcca77ff": {
        "evidence_version": "published-response-quarantine-20260909-v2",
        "material_id_observed": "aflow-7c7ee61f60207c27",
        "compressed_sha256": "030ffa6092bdd2cf3cc51dfd0927016998113a166e284419c0659a89367ef993",
        "source_url": "https://aflowlib.duke.edu/AFLOWDATA/LIB3_WEB/B_hBrMo_pv/ICSD_82245.ABC/B_hBrMo_pv.ICSD_82245.ABC_bandsdata.json.xz",
        "evidence_path": "artifacts/reports/remediation_20260909/cycle1_reviewed_evidence/independent_source/review_report.json",
        "evidence_report_sha256": "cf8448d579a486c2752fec191dc105271f7734f9c82cb52cbe849bae1f927750",
        "evidence_json_pointer": "/qa_fingerprint_binding_table/2",
        "reason": "source_quarantine_published_response_v1",
        "review_status": "verified_invalid_published_energy",
        "scope": "exact full k/energy content; not a population screening threshold",
    },
    "601833f3bbb336b1e941e73718a442a64ccfe1243f4bbcc1ae1675d6dce8b22c": {
        "evidence_version": "published-response-quarantine-20260909-v2",
        "material_id_observed": "aflow-8bee7b2755ee75ce",
        "compressed_sha256": "7ef850f2f50bb8fa3bf86a3625b6d7c8cff91571a5ecdf2731ff6d22d22cfd92",
        "source_url": "https://aflowlib.duke.edu/AFLOWDATA/LIB3_WEB/B_hPTl_d/ICSD_402064.ABC/B_hPTl_d.ICSD_402064.ABC_bandsdata.json.xz",
        "evidence_path": "artifacts/reports/remediation_20260909/cycle1_reviewed_evidence/independent_source/review_report.json",
        "evidence_report_sha256": "cf8448d579a486c2752fec191dc105271f7734f9c82cb52cbe849bae1f927750",
        "evidence_json_pointer": "/qa_fingerprint_binding_table/3",
        "reason": "source_quarantine_published_response_v1",
        "review_status": "verified_invalid_published_energy",
        "scope": "exact full k/energy content; not a population screening threshold",
    },
    "0c919996547105922493e32f8063497491b54657e8b68b363310027584939a15": {
        "evidence_version": "published-response-quarantine-20260909-v2",
        "material_id_observed": "aflow-c3dae23bece6500c",
        "compressed_sha256": "9bd6e3fc1a418fea786476018ac81243a304ec0022edcdcc58d1155c2b6cf851",
        "source_url": "https://aflowlib.duke.edu/AFLOWDATA/LIB3_WEB/CdNa_svP/T0008.ABC:LDAU2/CdNa_svP.T0008.ABC:LDAU2_bandsdata.json.xz",
        "evidence_path": "artifacts/reports/remediation_20260909/cycle1_reviewed_evidence/independent_source/review_report.json",
        "evidence_report_sha256": "cf8448d579a486c2752fec191dc105271f7734f9c82cb52cbe849bae1f927750",
        "evidence_json_pointer": "/qa_fingerprint_binding_table/4",
        "reason": "source_quarantine_published_response_v1",
        "review_status": "verified_invalid_published_energy",
        "scope": "exact full k/energy content; not a population screening threshold",
    },
}


def source_content_fingerprint(energies, k_distances):
    """SHA of shape-as-Python-tuple ASCII + k-first table, C-order little f32.

    Canonicalization matches the historical parser/storage, without sorting,
    shifting, clipping or interpreting spin. This is not a raw-response SHA.
    """
    energies = np.asarray(energies)
    flat = energies.reshape(-1, energies.shape[-1])
    table = np.asarray(np.column_stack((k_distances, flat.T)), dtype="<f4")
    return hashlib.sha256(str(table.shape).encode("ascii") + table.tobytes()).hexdigest()


class SourceQualityError(ValueError):
    """Rejected input; carries the recomputed QA, never a cached assertion."""

    def __init__(self, quality):
        self.quality = quality
        super().__init__(quality["reason"])


def _numeric_array(value):
    # Do not coerce strings/bools/complex values into apparent valid physics.
    if isinstance(value, np.ndarray) and value.dtype.kind in "iuf":
        return np.asarray(value, dtype=float)
    objects = np.asarray(value, dtype=object)
    if any(isinstance(v, (bool, np.bool_)) or
           not isinstance(v, (int, float, np.integer, np.floating)) for v in objects.flat):
        raise ValueError("invalid_numeric_type")
    return np.asarray(value, dtype=float)


def assess_source_quality(record):
    """Pure source-only QA for parsed records and read-only HDF5 consumers.

    verified_valid means finite/shape-consistent data with a recognized explicit
    importer contract and declared spin/layout evidence, NOT independent DFT
    truth. The historical AFLOW parser's hardcoded False is never evidence.
    Normal deep valence, EF-flat bands and rank inversions are not vetoes.

    Recognized normalized importer declarations (not native AFLOW field claims):
    source_schema=aflow_bandsdata_v1 or mp_pymatgen_v1; band_layout=bands_k
    (2D single channel), spin_major_flat (2D equally sized spin blocks), or
    spin_band_k (3D explicit spin axis); spin_channels=1/2; nonempty text
    source_semantics_evidence explaining the declaration. AFLOW raw mapping
    copies band_schema to source_schema. Never backfill these just to pass QA.
    Missing/unsupported declarations remain ambiguous even if numerically sane.
    """
    if not isinstance(record, dict):
        return {"rule_version": SOURCE_QA_RULE_VERSION, "status": "invalid",
                "reason": "invalid_record_mapping", "reasons": ["invalid_record_mapping"],
                "content_sha256": None}
    declarations = {key: record.get(key) if isinstance(record.get(key), str) else None
                    for key in ("source_schema", "band_layout", "source_semantics_evidence")}
    quality = {
        "rule_version": SOURCE_QA_RULE_VERSION,
        "fingerprint_version": SOURCE_FINGERPRINT_VERSION,
        "content_sha256": None, "status": "ambiguous", "reasons": [],
        "schema": declarations["source_schema"] or "unknown",
        "layout": declarations["band_layout"] or "unknown",
        "spin_channels": None, "spin_status": "unknown",
        "semantics_evidence": declarations["source_semantics_evidence"],
    }
    try:
        nested = record.get("metadata")
        if isinstance(nested, dict):
            for key in ("source", "efermi", "source_efermi_absolute", "energy_reference",
                        "source_schema", "band_layout", "spin_channels", "source_semantics_evidence",
                        "n_bands", "n_kpoints", "num_bands", "num_kpoints"):
                if key in nested and key in record and not np.array_equal(nested[key], record[key]):
                    raise ValueError("conflicting_source_metadata")
        energies = _numeric_array(record.get("energies"))
        k = _numeric_array(record.get("k_distances"))
        if energies.ndim not in (2, 3) or not energies.size or energies.shape[-1] < 2:
            raise ValueError("band data unavailable: invalid_energy_shape")
        if not np.isfinite(energies).all() or np.any(np.abs(energies) > np.finfo(np.float32).max):
            raise ValueError("nonfinite_energy_or_float32_overflow")
        for key, observed in (("n_bands", energies.shape[-2]), ("num_bands", energies.shape[-2]),
                              ("n_kpoints", energies.shape[-1]), ("num_kpoints", energies.shape[-1])):
            if key in record:
                declared = record[key]
                if (isinstance(declared, (bool, np.bool_))
                        or not isinstance(declared, (int, np.integer)) or declared != observed):
                    raise ValueError(f"declared_{key}_mismatch_or_illegal")
        if (k.ndim != 1 or len(k) != energies.shape[-1] or not np.isfinite(k).all()
                or np.any(k < 0) or np.any(np.diff(k) < 0) or k[-1] <= k[0]
                or np.any(k > np.finfo(np.float32).max)):
            raise ValueError("invalid_k_distances")
        stored_k = k.astype(np.float32)
        if stored_k[-1] <= stored_k[0]:
            raise ValueError("invalid_k_distances: float32 storage loses positive span")
        if "kpoints" in record:
            points = _numeric_array(record["kpoints"])
            if (points.ndim != 2 or points.shape[0] != len(k) or points.shape[1] not in (1, 3)
                    or not np.isfinite(points).all() or np.any(np.abs(points) > np.finfo(np.float32).max)):
                raise ValueError("invalid_kpoints")
        for key in ("efermi", "source_efermi_absolute"):
            if key not in record and key == "source_efermi_absolute":
                continue
            value = record.get(key)
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)):
                raise ValueError("invalid_source_efermi" if key == "source_efermi_absolute" else "invalid_efermi")
        if record.get("source") == "aflow" and float(record["efermi"]) != 0:
            raise ValueError("invalid_aflow_energy_reference")
        if record.get("energy_reference") == "fermi_shifted_zero" and float(record["efermi"]) != 0:
            raise ValueError("inconsistent_energy_reference")
        fingerprint = source_content_fingerprint(energies, k)
        quality["content_sha256"] = fingerprint
        quality["observed_shape"] = list(energies.shape)
        if record.get("source") == "aflow" and fingerprint in SOURCE_QUARANTINE:
            quality["quarantine_evidence"] = dict(SOURCE_QUARANTINE[fingerprint])
            raise ValueError(SOURCE_QUARANTINE[fingerprint]["reason"])
        channels = record.get("spin_channels")
        layout = quality["layout"]
        if channels is not None and channels != "unknown":
            if (isinstance(channels, (bool, np.bool_))
                    or not isinstance(channels, (int, np.integer)) or channels < 1):
                raise ValueError("invalid_spin_channels")
            if ((layout == "bands_k" and (channels != 1 or energies.ndim != 2))
                    or (layout == "spin_major_flat" and
                        (channels == 1 or energies.ndim != 2 or energies.shape[0] % channels))
                    or (layout == "spin_band_k" and
                        (energies.ndim != 3 or energies.shape[0] != channels))):
                raise ValueError("inconsistent_spin_layout")
    except (ValueError, TypeError, OverflowError) as exc:
        quality.update(status="invalid", reason=str(exc), reasons=[str(exc)])
        return quality
    reasons = quality["reasons"]
    if (record.get("source"), quality["schema"]) not in (
            ("aflow", "aflow_bandsdata_v1"), ("mp", "mp_pymatgen_v1")):
        reasons.append("unknown_schema")
    if quality["layout"] not in ("bands_k", "spin_major_flat", "spin_band_k"):
        reasons.append("unknown_layout")
    channels = record.get("spin_channels")
    if isinstance(channels, (int, np.integer)) and not isinstance(channels, (bool, np.bool_)) and channels in (1, 2):
        quality.update(spin_channels=int(channels), spin_status="declared")
    else:
        reasons.append("unknown_spin")
    if not quality["semantics_evidence"]:
        reasons.append("missing_semantics_evidence")
    quality["status"] = "ambiguous" if reasons else "verified_valid"
    quality["reason"] = reasons[0] if reasons else "source_contract_verified"
    return quality


class AFLOWAdapter:
    API_ROOT = "https://aflow.org/API/aflux/"
    DATA_ROOT = "https://aflowlib.duke.edu"

    def __init__(
        self,
        cache_dir: str = "./data/raw/aflow",
        rate_limit_delay: float = 0.2,
        timeout: float = 60.0,
        max_retries: int = 3,
        source_qa_mode: str = "formal",
    ) -> None:
        if source_qa_mode not in ("formal", "diagnostic"):
            raise ValueError("source_qa_mode must be formal or diagnostic")
        self.source_qa_mode = source_qa_mode
        self.cache_dir = cache_dir
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.json_cache_dir = os.path.join(cache_dir, "json_cache", "aflow")
        self.h5_path = os.path.join(cache_dir, "aflow_bands.h5")
        self.metadata_path = os.path.join(cache_dir, "aflow_metadata.json")
        self.last_metadata_cursor: Optional[Dict[str, Any]] = None
        os.makedirs(self.json_cache_dir, exist_ok=True)

    def close(self) -> None:
        return None

    @staticmethod
    def _safe_id(auid: str) -> str:
        return "aflow-" + auid.split(":", 1)[-1]

    @staticmethod
    def _lenient_json(text: str) -> Dict[str, Any]:
        """Parse AFLOW JSON, tolerating trailing commas (server-side change).

        AFLOW recently started emitting trailing commas inside its bandsdata
        documents, which strict JSON parsers reject. Strip only commas that
        sit directly before a closing bracket/brace, outside string literals.
        """
        stripped = []
        in_string = False
        escape = False
        for i, char in enumerate(text):
            if in_string:
                stripped.append(char)
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                stripped.append(char)
                continue
            if char == ",":
                rest = text[i + 1:].lstrip()
                if rest.startswith("}") or rest.startswith("]"):
                    continue
            stripped.append(char)
        return json.loads("".join(stripped))

    @staticmethod
    def _base_data_url(aurl: str) -> str:
        path = aurl.split(":", 1)[-1].lstrip("/")
        return f"{AFLOWAdapter.DATA_ROOT}/{path}"

    def _request_bytes(self, url: str) -> bytes:
        last_error: Optional[Exception] = None
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "BandStructure-AI/phase0 (+research cache)"},
        )
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read()
                if not payload:
                    raise OSError("empty response")
                return payload
            except Exception as exc:  # network errors vary by Python version
                last_error = exc
                if attempt + 1 < self.max_retries:
                    delay = min(8.0, 0.75 * (2**attempt)) + random.uniform(0.0, 0.25)
                    time.sleep(delay)
        raise OSError(f"AFLOW request failed after {self.max_retries} attempts: {url}: {last_error}")

    def fetch_metadata(
        self,
        query_params: Optional[Dict[str, Any]] = None,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        del fields
        query = dict(query_params or {})
        min_gap, max_gap = query.get("band_gap", (0.0, 5.0))
        min_sites, max_sites = query.get("num_sites", (1, 10_000))
        start_page = max(1, int(query.get("page", 1)))
        limit = max(1, int(query.get("limit", query.get("page_size", 100))))
        page_size = max(1, min(int(query.get("page_size", min(limit, 500))), 500))
        requested_bins = max(1, int(query.get("gap_bins", 5)))
        num_bins = 1 if float(max_gap) <= float(min_gap) else min(requested_bins, limit)
        bin_edges = np.linspace(float(min_gap), float(max_gap), num_bins + 1)
        incoming_cursor = query.get("_aflow_cursor")
        cursor_is_compatible = (
            isinstance(incoming_cursor, dict)
            and incoming_cursor.get("version") == 2
            and int(incoming_cursor.get("page_size", 0)) == page_size
            and isinstance(incoming_cursor.get("next_pages"), list)
            and isinstance(incoming_cursor.get("exhausted"), list)
            and isinstance(incoming_cursor.get("pending_records"), list)
            and len(incoming_cursor["next_pages"]) == num_bins
            and len(incoming_cursor["exhausted"]) == num_bins
            and len(incoming_cursor["pending_records"]) == num_bins
            and isinstance(incoming_cursor.get("bin_edges"), list)
            and len(incoming_cursor["bin_edges"]) == num_bins + 1
            and np.allclose(incoming_cursor["bin_edges"], bin_edges)
        )
        if cursor_is_compatible:
            bin_start_pages = [
                max(1, int(page)) for page in incoming_cursor["next_pages"]
            ]
            bin_was_exhausted = [
                bool(value) for value in incoming_cursor["exhausted"]
            ]
            bin_pending_records = [
                [
                    to_jsonable(record)
                    for record in pending
                    if isinstance(record, dict) and record.get("material_id")
                ]
                if isinstance(pending, list)
                else []
                for pending in incoming_cursor["pending_records"]
            ]
        else:
            # A legacy/incompatible cursor may have advanced beyond records that
            # were silently discarded at a page tail. Re-page from 1; the
            # downloader candidate catalog deduplicates already-seen IDs.
            safe_start_page = 1 if isinstance(incoming_cursor, dict) else start_page
            bin_start_pages = [safe_start_page] * num_bins
            bin_was_exhausted = [False] * num_bins
            bin_pending_records = [[] for _ in range(num_bins)]
        base_quota, extra = divmod(limit, num_bins)
        buckets: List[List[Dict[str, Any]]] = [[] for _ in range(num_bins)]
        bin_states: List[Dict[str, Any]] = [
            {
                "bin_min": float(bin_edges[index]),
                "bin_max": float(bin_edges[index + 1]),
                "next_page": bin_start_pages[index],
                "exhausted": bin_was_exhausted[index],
                "pending_records": bin_pending_records[index],
            }
            for index in range(num_bins)
        ]

        def fetch_page(bin_index: int, phase: str) -> List[Dict[str, Any]]:
            state = bin_states[bin_index]
            page = int(state["next_page"])
            aflux_query = (
                f"Egap({state['bin_min']:g}*,*{state['bin_max']:g}),"
                f"natoms({int(min_sites)}*,*{int(max_sites)}),"
                f"Egap_type,files,paging({page},{page_size})"
            )
            url = (
                f"{self.API_ROOT}?"
                f"{urllib.parse.quote(aflux_query, safe='(),*')}"
            )
            print(
                f"[AFLUX] {phase} bin {bin_index + 1}/{num_bins} "
                f"gap=[{state['bin_min']:g},{state['bin_max']:g}) page {page}",
                flush=True,
            )
            payload = json.loads(self._request_bytes(url).decode("utf-8-sig"))
            raw_rows = list(payload.values()) if isinstance(payload, dict) else []
            if not raw_rows:
                state["exhausted"] = True
                return []
            state["next_page"] = page + 1
            if len(raw_rows) < page_size:
                state["exhausted"] = True
            return [
                record
                for record in (self._metadata_record(raw) for raw in raw_rows)
                if record is not None
            ]

        for bin_index in range(num_bins):
            quota = base_quota + int(bin_index < extra)
            state = bin_states[bin_index]
            pending = state["pending_records"]
            take = min(quota, len(pending))
            if take:
                buckets[bin_index].extend(pending[:take])
                del pending[:take]
            stalled_pages = 0
            while len(buckets[bin_index]) < quota and not state["exhausted"]:
                page_records = fetch_page(bin_index, "quota")
                if not page_records:
                    stalled_pages += 1
                    if state["exhausted"] or stalled_pages >= 3:
                        break
                    continue
                stalled_pages = 0
                needed = quota - len(buckets[bin_index])
                buckets[bin_index].extend(page_records[:needed])
                pending.extend(page_records[needed:])

        # Reallocate quota left by exhausted/invalid strata to bins that still
        # have rows. Deduplicate globally because adjacent closed intervals can
        # repeat a boundary material.
        seen_for_quota = {
            str(record["material_id"])
            for bucket in buckets
            for record in bucket
        }
        stalled_rounds = 0
        while len(seen_for_quota) < limit:
            active_bins = 0
            added_this_round = 0
            for bin_index, state in enumerate(bin_states):
                if len(seen_for_quota) >= limit:
                    break
                pending = state["pending_records"]
                if state["exhausted"] and not pending:
                    continue
                active_bins += 1
                page_records: List[Dict[str, Any]] = []
                while pending and len(seen_for_quota) < limit:
                    page_records.append(pending.pop(0))
                if len(seen_for_quota) < limit and not state["exhausted"]:
                    page_records.extend(fetch_page(bin_index, "reallocate"))
                for record_index, record in enumerate(page_records):
                    material_id = str(record["material_id"])
                    if material_id in seen_for_quota:
                        continue
                    buckets[bin_index].append(record)
                    seen_for_quota.add(material_id)
                    added_this_round += 1
                    if len(seen_for_quota) >= limit:
                        state["pending_records"][:0] = page_records[
                            record_index + 1 :
                        ]
                        break
            if not active_bins:
                break
            if added_this_round:
                stalled_rounds = 0
            else:
                stalled_rounds += 1
                if stalled_rounds >= 3:
                    break

        self.last_metadata_cursor = {
            "version": 2,
            "page_size": page_size,
            "bin_edges": [float(edge) for edge in bin_edges],
            "next_pages": [
                int(state["next_page"]) for state in bin_states
            ],
            "exhausted": [
                bool(state["exhausted"]) for state in bin_states
            ],
            "pending_records": [
                [to_jsonable(record) for record in state["pending_records"]]
                for state in bin_states
            ],
        }

        # AFLUX results tend to start at the lower interval boundary. Interleave
        # the strata so a small downloader target does not consume only bin 0.
        metadata: List[Dict[str, Any]] = []
        seen_ids = set()
        for row_index in range(max((len(bucket) for bucket in buckets), default=0)):
            for bucket in buckets:
                if row_index >= len(bucket):
                    continue
                record = bucket[row_index]
                if record["material_id"] in seen_ids:
                    continue
                seen_ids.add(record["material_id"])
                metadata.append(record)
        return metadata[:limit]

    @classmethod
    def _metadata_record(cls, raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, dict):
            return None
        auid = str(raw.get("auid") or "")
        aurl = str(raw.get("aurl") or "")
        files = raw.get("files") or []
        band_file = next(
            (name for name in files if str(name).endswith("_bandsdata.json.xz")),
            None,
        )
        if not auid or not aurl or not band_file:
            return None
        gap = float(raw.get("Egap") or 0.0)
        gap_type = str(raw.get("Egap_type") or "").lower()
        is_metal = gap <= 0.01 or "metal" in gap_type
        if is_metal or not gap_type:
            # Missing Egap_type must stay unknown; silently mapping it to
            # indirect would fabricate a provider label (constitution §4).
            is_direct = None
        else:
            is_direct = "direct" in gap_type and "indirect" not in gap_type
        return {
            "material_id": cls._safe_id(auid),
            "source": "aflow",
            "source_id": auid,
            "formula_pretty": raw.get("compound"),
            "spacegroup_number": raw.get("spacegroup_relax"),
            "pearson_symbol": raw.get("Pearson_symbol_relax"),
            "band_gap": gap,
            "num_sites": raw.get("natoms"),
            "gap_type_source": raw.get("Egap_type"),
            "is_direct": is_direct,
            "is_metal": is_metal,
            "aurl": aurl,
            "band_file": band_file,
            "download_url": f"{cls._base_data_url(aurl)}/{band_file}",
        }

    def fetch_band_structure(
        self,
        material_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cache_path = os.path.join(self.json_cache_dir, f"{material_id}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                quality = assess_source_quality(cached)
                if isinstance(cached, dict) and (cached.get("material_id") != material_id or cached.get("source") != "aflow"):
                    quality.update(status="invalid", reason="cache_identity_or_source_mismatch",
                                   reasons=["cache_identity_or_source_mismatch"])
                if (quality["status"] == "invalid" or
                        (self.source_qa_mode != "diagnostic" and quality["status"] != "verified_valid")):
                    raise SourceQualityError(quality)
                for key in ("kpoints", "k_distances", "energies"):
                    if key in cached:
                        cached[key] = np.asarray(cached[key], dtype=np.float32)
                cached["source_quality"] = quality
                cached["spin_status"] = quality["spin_status"]
                cached["is_spin_polarized"] = (quality["spin_channels"] > 1
                                               if quality["spin_channels"] is not None else None)
                return cached
            except SourceQualityError as exc:
                return {"material_id": material_id, "error": f"AFLOW cached source QA: {exc}",
                        "source_quality": exc.quality}
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                return {"material_id": material_id, "error": f"AFLOW cached read error: {type(exc).__name__}: {exc}",
                        "source_quality": {"rule_version": SOURCE_QA_RULE_VERSION,
                                           "status": "read_error", "reason": "cache_read_error",
                                           "reasons": ["cache_read_error"]}}

        metadata = dict(metadata or {})
        url = str(metadata.get("download_url") or "")
        if not url:
            return {"material_id": material_id, "error": "AFLOW download_url is missing"}
        source_sha256 = None
        raw_decoded = False
        try:
            compressed = self._request_bytes(url)
            source_sha256 = hashlib.sha256(compressed).hexdigest()
            raw = self._lenient_json(lzma.decompress(compressed).decode("utf-8-sig"))
            raw_decoded = True
            result = self._parse_band_payload(material_id, raw)
            result["source_sha256"] = source_sha256
            result["source_url"] = url
            if self.source_qa_mode != "diagnostic" and result["source_quality"]["status"] != "verified_valid":
                raise SourceQualityError(result["source_quality"])
        except SourceQualityError as exc:
            return {"material_id": material_id, "error": f"AFLOW source QA: {exc}",
                    "source_quality": exc.quality, "source_sha256": source_sha256, "source_url": url}
        except Exception as exc:
            result = {"material_id": material_id, "error": f"AFLOW parse/download error: {type(exc).__name__}: {exc}",
                      "source_sha256": source_sha256, "source_url": url}
            reason = "invalid_raw_payload" if raw_decoded else "source_response_read_error"
            result["source_quality"] = {
                "rule_version": SOURCE_QA_RULE_VERSION,
                "status": "invalid" if raw_decoded else "read_error",
                "reason": reason, "reasons": [reason], "content_sha256": None,
                "detail": f"{type(exc).__name__}: {exc}",
            }
            return result

        temporary = cache_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(result), handle, ensure_ascii=False)
        os.replace(temporary, cache_path)
        time.sleep(self.rate_limit_delay)
        return result

    @staticmethod
    def _parse_band_payload(material_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy diagnostic mapper; unknown semantics stay explicitly ambiguous."""
        if raw.get("bands_data") is None:
            raise ValueError("band data unavailable: missing bands_data")
        data = raw["bands_data"]
        raw_shape = np.asarray(data, dtype=object).shape
        if isinstance(data, dict) or len(raw_shape) > 2:
            # A possible spin/other layout is not a proven malformed material.
            # Do not guess axes, drop channels, or create a parsed cache for it.
            raise SourceQualityError({
                "rule_version": SOURCE_QA_RULE_VERSION, "status": "ambiguous",
                "reason": "unsupported_raw_layout", "reasons": ["unsupported_raw_layout"],
                "schema": "unknown", "layout": "unknown", "spin_status": "unknown",
                "spin_channels": None, "content_sha256": None,
                "fingerprint_version": SOURCE_FINGERPRINT_VERSION,
                "raw_shape": list(raw_shape),
            })
        table = _numeric_array(data)
        if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 3:
            raise ValueError(f"band data unavailable: unexpected bands_data shape {table.shape}")
        k_distances = table[:, 0]
        energies = table[:, 1:].T
        record = {
            "material_id": material_id, "source": "aflow",
            "kpoints": k_distances[:, None], "k_distances": k_distances,
            "energies": energies,
            # Never add the absolute source EF back into training energies.
            "efermi": 0.0, "source_efermi_absolute": raw.get("Efermi"),
            "energy_reference": "fermi_shifted_zero",
            "source_schema": raw.get("band_schema", "unknown"),
            "band_layout": raw.get("band_layout", "unknown"),
            "spin_channels": raw.get("spin_channels"),
            "source_semantics_evidence": raw.get("source_semantics_evidence"),
            "num_bands": int(energies.shape[0]), "num_kpoints": int(energies.shape[1]),
            **{key: raw[key] for key in ("n_bands", "n_kpoints") if key in raw},
        }
        quality = assess_source_quality(record)
        if quality["status"] == "invalid":
            raise SourceQualityError(quality)
        positions = np.asarray(raw.get("kpoint_positions", []), dtype=float)
        names = raw.get("kpoint_labels", [])
        if (positions.ndim != 1 or len(names) != len(positions)
                or not np.isfinite(positions).all()
                or np.any(positions < float(k_distances[0]) - 1e-6)
                or np.any(positions > float(k_distances[-1]) + 1e-6)):
            raise ValueError("invalid_kpoint_positions")
        record["kpath_labels"] = [
            {"index": int(np.argmin(np.abs(k_distances - position))),
             "label": str(label), "k_distance": float(position)}
            for label, position in zip(names, positions)
        ]
        for key in ("energies", "k_distances", "kpoints"):
            record[key] = record[key].astype(np.float32)
        record["source_quality"] = quality
        record["spin_status"] = quality["spin_status"]
        record["is_spin_polarized"] = (quality["spin_channels"] > 1
                                       if quality["spin_channels"] is not None else None)
        return record

    def save_to_hdf5(self, band_data: Dict[str, Any]) -> Any:
        return save_band_record(self.h5_path, band_data)

    def save_metadata_to_json(self, metadata: List[Dict[str, Any]]) -> None:
        merge_metadata_json(self.metadata_path, metadata)
