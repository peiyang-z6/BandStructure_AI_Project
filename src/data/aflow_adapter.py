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


class AFLOWAdapter:
    API_ROOT = "https://aflow.org/API/aflux/"
    DATA_ROOT = "https://aflowlib.duke.edu"

    def __init__(
        self,
        cache_dir: str = "./data/raw/aflow",
        rate_limit_delay: float = 0.2,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
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
                for key in ("kpoints", "k_distances", "energies"):
                    if key in cached:
                        cached[key] = np.asarray(cached[key], dtype=np.float32)
                return cached
            except (OSError, json.JSONDecodeError, ValueError):
                pass

        metadata = dict(metadata or {})
        url = str(metadata.get("download_url") or "")
        if not url:
            return {"material_id": material_id, "error": "AFLOW download_url is missing"}
        try:
            compressed = self._request_bytes(url)
            raw = self._lenient_json(lzma.decompress(compressed).decode("utf-8-sig"))
            result = self._parse_band_payload(material_id, raw)
            result["source_sha256"] = hashlib.sha256(compressed).hexdigest()
            result["source_url"] = url
        except Exception as exc:
            return {"material_id": material_id, "error": f"AFLOW parse/download error: {type(exc).__name__}: {exc}"}

        temporary = cache_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(result), handle, ensure_ascii=False)
        os.replace(temporary, cache_path)
        time.sleep(self.rate_limit_delay)
        return result

    @staticmethod
    def _parse_band_payload(material_id: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        table = np.asarray(raw.get("bands_data"), dtype=np.float32)
        if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 3:
            raise ValueError(f"band data unavailable: unexpected bands_data shape {table.shape}")
        k_distances = table[:, 0]
        if np.any(np.diff(k_distances) < -1.0e-6):
            raise ValueError("AFLOW k-path coordinate is not monotonic")
        energies = table[:, 1:].T
        labels = []
        for label, position in zip(raw.get("kpoint_labels", []), raw.get("kpoint_positions", [])):
            index = int(np.argmin(np.abs(k_distances - float(position))))
            labels.append({"index": index, "label": str(label), "k_distance": float(position)})
        return {
            "material_id": material_id,
            "source": "aflow",
            "kpoints": k_distances[:, None],
            "k_distances": k_distances,
            "energies": energies,
            # AFLOW bands_data energies are already shifted so E_F=0; preserve
            # the absolute source value separately for provenance only.
            "efermi": 0.0,
            "source_efermi_absolute": float(raw["Efermi"]),
            "energy_reference": "fermi_shifted_zero",
            "kpath_labels": labels,
            "is_spin_polarized": False,
            "num_bands": int(energies.shape[0]),
            "num_kpoints": int(energies.shape[1]),
        }

    def save_to_hdf5(self, band_data: Dict[str, Any]) -> Any:
        return save_band_record(self.h5_path, band_data)

    def save_metadata_to_json(self, metadata: List[Dict[str, Any]]) -> None:
        merge_metadata_json(self.metadata_path, metadata)
