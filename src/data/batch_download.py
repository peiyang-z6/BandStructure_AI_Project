"""
Robust multi-source band-structure downloader.

This script intentionally keeps the project's data split policy untouched. It only
downloads line-mode band structures and stores metadata fields required by the
spacegroup-based splitter.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.band_store import (
    atomic_write_json,
    exclusive_file_lock,
    merge_metadata_json,
    read_material_ids,
    synchronize_metadata_with_hdf5,
)


class DownloadTargetNotReached(RuntimeError):
    """Raised after checkpointing when persisted cache count misses the target."""


class RobustBandDownloader:
    """Single-threaded downloader with rate limiting and checkpoint resume."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        new_api_key: Optional[str] = None,
        legacy_api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        source: str = "mp",
        adapter: Any = None,
        workers: int = 1,
        batch_size: int = 8,
    ):
        self.new_api_key = new_api_key or (
            api_key if api_key and len(api_key.strip()) == 32 else None
        )
        self.legacy_api_key = legacy_api_key or (
            api_key if api_key and len(api_key.strip()) != 32 else None
        )
        self.api_key = self.new_api_key or self.legacy_api_key
        self.source = source.lower()
        if cache_dir is None:
            source_dir = "materials_project" if self.source == "mp" else self.source
            cache_dir = f"./data/raw/{source_dir}"
        self.cache_dir = cache_dir
        self.workers = max(1, int(workers))
        self.batch_size = max(1, int(batch_size))
        self._thread_local = threading.local()
        self._worker_adapters = []
        self._worker_adapters_lock = threading.Lock()
        self._adapter_create_lock = threading.Lock()
        os.makedirs(cache_dir, exist_ok=True)
        if adapter is not None:
            self.adapter = adapter
        elif self.source == "mp":
            from src.data.mp_adapter import MPAdapter

            self.adapter = MPAdapter(
                api_key=api_key,
                new_api_key=new_api_key,
                legacy_api_key=legacy_api_key,
                cache_dir=cache_dir,
                rate_limit_delay=0.2,
            )
        elif self.source == "aflow":
            from src.data.aflow_adapter import AFLOWAdapter

            self.adapter = AFLOWAdapter(cache_dir=cache_dir, rate_limit_delay=0.2)
        else:
            raise ValueError(f"Unsupported source: {source}. Choose 'mp' or 'aflow'.")
        self.downloaded_ids = self._load_downloaded()
        self.h5_path = os.path.join(cache_dir, f"{self.source}_bands.h5")
        self.metadata_path = os.path.join(
            cache_dir,
            f"{self.source}_metadata.json",
        )
        self.candidate_catalog_path = os.path.join(
            cache_dir,
            f"{self.source}_candidate_catalog.json",
        )
        self.candidate_cursor_path = os.path.join(
            cache_dir,
            f"{self.source}_candidate_cursor.json",
        )
        self.excluded_path = os.path.join(cache_dir, f"{self.source}_excluded.json")
        self.excluded_ids = self._load_excluded()
        self.stats = {
            "success": 0,
            "failed": 0,
            "skipped": len(self.downloaded_ids),
            "no_data": 0,
            "spacegroup_dist": defaultdict(int),
        }
        self._run_state = {
            "initial_cached": len(self.downloaded_ids),
            "requested_target": len(self.downloaded_ids),
            "attempted": 0,
            "submitted": 0,
            "completed": 0,
            "in_flight": 0,
            "peak_in_flight": 0,
            "termination_reason": "not_started",
        }

    def close(self) -> None:
        self.adapter.close()
        for adapter in self._worker_adapters:
            adapter.close()

    def _get_fetch_adapter(self):
        if self.workers == 1:
            return self.adapter
        adapter = getattr(self._thread_local, "adapter", None)
        if adapter is not None:
            return adapter
        with self._adapter_create_lock:
            adapter = getattr(self._thread_local, "adapter", None)
            if adapter is None:
                if self.source == "mp":
                    from src.data.mp_adapter import MPAdapter

                    adapter = MPAdapter(
                        new_api_key=self.new_api_key,
                        legacy_api_key=self.legacy_api_key,
                        cache_dir=self.cache_dir,
                        rate_limit_delay=0.0,
                    )
                else:
                    from src.data.aflow_adapter import AFLOWAdapter

                    adapter = AFLOWAdapter(cache_dir=self.cache_dir, rate_limit_delay=0.0)
                self._thread_local.adapter = adapter
                with self._worker_adapters_lock:
                    self._worker_adapters.append(adapter)
        return adapter

    def _fetch_candidate(self, meta: Dict[str, Any]):
        adapter = self._get_fetch_adapter()
        return meta, adapter.fetch_band_structure(meta["material_id"], metadata=meta)

    @staticmethod
    def _diverse_candidate_order(candidates):
        """Round-robin space groups, then gap types inside each group."""
        buckets = defaultdict(deque)
        for item in candidates:
            if item.get("is_metal"):
                gap_type = "metal"
            elif item.get("is_direct") is True:
                gap_type = "direct"
            elif item.get("is_direct") is False:
                gap_type = "indirect"
            else:
                gap_type = "unknown"
            buckets[item.get("spacegroup_number")].append((gap_type, item))
        for group, rows in list(buckets.items()):
            by_type = defaultdict(deque)
            for gap_type, item in rows:
                by_type[gap_type].append(item)
            interleaved = deque()
            active_types = deque(sorted(by_type))
            while active_types:
                gap_type = active_types.popleft()
                interleaved.append(by_type[gap_type].popleft())
                if by_type[gap_type]:
                    active_types.append(gap_type)
            buckets[group] = interleaved
        ordered = []
        active = deque(sorted(buckets, key=lambda group: (group is None, group or 0)))
        while active:
            group = active.popleft()
            ordered.append(buckets[group].popleft())
            if buckets[group]:
                active.append(group)
        return ordered

    def _load_downloaded(self) -> set[str]:
        h5_path = os.path.join(self.cache_dir, f"{self.source}_bands.h5")
        if not os.path.exists(h5_path):
            return set()
        with exclusive_file_lock(h5_path + ".lock"):
            return read_material_ids(h5_path)

    def _load_excluded(self) -> Dict[str, str]:
        """Persisted registry of candidates with permanently unusable band data.

        Without this, upstream empty-band records are re-downloaded on every
        resume, wasting requests and polluting the attempt log. The file is
        human-editable: delete entries to retry a material after upstream fixes.
        """
        if not os.path.exists(self.excluded_path):
            return {}
        try:
            with open(self.excluded_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                return {str(key): str(value) for key, value in loaded.items()}
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    @staticmethod
    def _candidate_matches_query(
        metadata: Dict[str, Any],
        query_params: Dict[str, Any],
    ) -> bool:
        min_gap, max_gap = query_params.get("band_gap", (None, None))
        band_gap = metadata.get("band_gap")
        if band_gap is not None:
            if min_gap is not None and float(band_gap) < float(min_gap):
                return False
            if max_gap is not None and float(band_gap) > float(max_gap):
                return False
        min_sites, max_sites = query_params.get("num_sites", (None, None))
        num_sites = metadata.get("num_sites")
        if num_sites is not None:
            if min_sites is not None and int(num_sites) < int(min_sites):
                return False
            if max_sites is not None and int(num_sites) > int(max_sites):
                return False
        if query_params.get("theoretical") is True:
            if metadata.get("theoretical") is False:
                return False
        return True

    def _load_candidate_catalog(
        self,
        query_params: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        if not os.path.exists(self.candidate_catalog_path):
            return []
        with open(self.candidate_catalog_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, list):
            raise ValueError(
                "candidate catalog must contain a JSON list: "
                f"{self.candidate_catalog_path}"
            )
        by_id: Dict[str, Dict[str, Any]] = {}
        for item in loaded:
            if not isinstance(item, dict) or not item.get("material_id"):
                continue
            if self._candidate_matches_query(item, query_params):
                by_id[str(item["material_id"])] = item
        return [by_id[material_id] for material_id in sorted(by_id)]

    @staticmethod
    def _query_fingerprint(query_params: Dict[str, Any]) -> str:
        identity = {
            key: value
            for key, value in query_params.items()
            if key not in {
                "page",
                "limit",
                "catalog_batch_size",
            }
        }
        payload = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_cursor_document(self) -> Dict[str, Any]:
        if not os.path.exists(self.candidate_cursor_path):
            return {"version": 1, "queries": {}}
        with open(self.candidate_cursor_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict) or not isinstance(
            loaded.get("queries"), dict
        ):
            raise ValueError(
                "candidate cursor must contain a queries object: "
                f"{self.candidate_cursor_path}"
            )
        return loaded

    def _fetch_next_catalog_page(
        self,
        query_params: Dict[str, Any],
    ) -> Optional[list[Dict[str, Any]]]:
        maximum_candidates = max(1, int(query_params.get("limit", 5_000)))
        current_records = self._load_candidate_catalog(query_params)
        current_ids = {
            str(item["material_id"])
            for item in current_records
            if item.get("material_id")
        }
        cursor = self._load_cursor_document()
        fingerprint = self._query_fingerprint(query_params)
        state = cursor["queries"].setdefault(
            fingerprint,
            {
                "next_page": max(1, int(query_params.get("page", 1))),
                "pages_fetched": 0,
                "exhausted": False,
                "termination_reason": None,
            },
        )
        if (
            state.get("termination_reason") == "candidate_limit_reached"
            and maximum_candidates > int(state.get("candidate_limit", 0))
        ):
            state["exhausted"] = False
            state["termination_reason"] = None
        if bool(state.get("exhausted")):
            return None
        if len(current_ids) >= maximum_candidates:
            state.update(
                {
                    "exhausted": True,
                    "termination_reason": "candidate_limit_reached",
                    "candidate_limit": maximum_candidates,
                    "candidate_count": len(current_ids),
                }
            )
            atomic_write_json(self.candidate_cursor_path, cursor)
            return None

        requested_batch = max(
            1,
            int(query_params.get("catalog_batch_size", 5_000)),
        )
        batch_limit = min(requested_batch, maximum_candidates - len(current_ids))
        page = max(1, int(state.get("next_page", 1)))
        fetch_params = dict(query_params)
        fetch_params.pop("catalog_batch_size", None)
        fetch_params["page"] = page
        gap_bins = max(1, int(fetch_params.get("gap_bins", 1)))
        fetch_limit = batch_limit
        fetch_params["limit"] = fetch_limit
        if self.source == "aflow" and isinstance(
            state.get("adapter_cursor"), dict
        ):
            fetch_params["_aflow_cursor"] = state["adapter_cursor"]
        page_size = max(1, int(fetch_params.get("page_size", 500)))
        page_span = max(
            1,
            math.ceil(math.ceil(fetch_limit / gap_bins) / page_size),
        )

        fetched = self.adapter.fetch_metadata(query_params=fetch_params)
        if fetched:
            merge_metadata_json(self.candidate_catalog_path, fetched)
        refreshed = self._load_candidate_catalog(query_params)
        new_records = [
            item
            for item in refreshed
            if str(item.get("material_id")) not in current_ids
        ]

        adapter_cursor = getattr(self.adapter, "last_metadata_cursor", None)
        if self.source == "aflow" and isinstance(adapter_cursor, dict):
            state["adapter_cursor"] = adapter_cursor
            next_pages = adapter_cursor.get("next_pages")
            if isinstance(next_pages, list) and next_pages:
                state["next_page"] = max(int(value) for value in next_pages)
            else:
                state["next_page"] = page + page_span
        else:
            state["next_page"] = page + page_span
        state["pages_fetched"] = int(state.get("pages_fetched", 0)) + page_span
        state["candidate_count"] = len(refreshed)
        state["last_new_ids"] = len(new_records)
        result: Optional[list[Dict[str, Any]]] = new_records
        if not fetched:
            state["exhausted"] = True
            state["termination_reason"] = "api_exhausted"
            result = None
        elif not new_records:
            no_new_pages = int(state.get("consecutive_no_new_pages", 0)) + 1
            state["consecutive_no_new_pages"] = no_new_pages
            state["termination_reason"] = "metadata_duplicate_page"
            if no_new_pages >= 3:
                state["exhausted"] = True
                state["termination_reason"] = "metadata_no_new_ids"
                result = None
        else:
            state["consecutive_no_new_pages"] = 0
            state["termination_reason"] = None
        atomic_write_json(self.candidate_cursor_path, cursor)
        print(
            f"[Catalog] page={page} span={page_span} fetched={len(fetched)} "
            f"new={len(new_records)} total={len(refreshed)}/{maximum_candidates}"
        )
        return result

    def _save_excluded(self) -> None:
        atomic_write_json(self.excluded_path, self.excluded_ids)

    def _start_run(self, target_count: int) -> None:
        initial_cached = len(self.downloaded_ids)
        self.stats = {
            "success": 0,
            "failed": 0,
            "skipped": initial_cached,
            "no_data": 0,
            "spacegroup_dist": defaultdict(int),
        }
        self._run_state = {
            "initial_cached": initial_cached,
            "requested_target": int(target_count),
            "attempted": 0,
            "submitted": 0,
            "completed": 0,
            "in_flight": 0,
            "peak_in_flight": 0,
            "termination_reason": "running",
        }

    def _report_payload(self, termination_reason: Optional[str] = None) -> Dict[str, Any]:
        if termination_reason is not None:
            self._run_state["termination_reason"] = termination_reason
        target_count = int(self._run_state["requested_target"])
        total_cached = len(self.downloaded_ids)
        return {
            "time": datetime.now().isoformat(),
            "source": self.source,
            "initial_cached": int(self._run_state["initial_cached"]),
            "requested_target": target_count,
            "attempted": int(self._run_state["attempted"]),
            "submitted": int(self._run_state["submitted"]),
            "completed": int(self._run_state["completed"]),
            "in_flight": int(self._run_state["in_flight"]),
            "peak_in_flight": int(self._run_state["peak_in_flight"]),
            "total_cached": total_cached,
            "target_reached": total_cached >= target_count,
            "termination_reason": str(self._run_state["termination_reason"]),
            "stats": {
                key: int(value)
                for key, value in self.stats.items()
                if key != "spacegroup_dist"
            },
            "spacegroup_dist": {
                str(key): int(value)
                for key, value in self.stats["spacegroup_dist"].items()
            },
        }

    def _checkpoint(self, termination_reason: Optional[str] = None) -> Dict[str, Any]:
        self._save_excluded()
        report = self._report_payload(termination_reason=termination_reason)
        report_path = os.path.join(
            self.cache_dir,
            f"{self.source}_download_report.json",
        )
        atomic_write_json(report_path, report)
        return report

    def query_and_download(
        self,
        target_count: int = 500,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        lock_path = os.path.join(
            self.cache_dir,
            f".{self.source}_download.lock",
        )
        with exclusive_file_lock(lock_path):
            self.downloaded_ids = self._load_downloaded()
            synchronize_metadata_with_hdf5(
                self.h5_path,
                self.metadata_path,
            )
            self._start_run(target_count)
            try:
                return self._query_and_download_locked(
                    target_count=target_count,
                    query_params=query_params,
                )
            except BaseException as exc:
                self._run_state["in_flight"] = 0
                if isinstance(exc, DownloadTargetNotReached):
                    self._checkpoint(
                        termination_reason=str(self._run_state["termination_reason"])
                    )
                else:
                    prefix = (
                        "interrupted"
                        if isinstance(exc, (KeyboardInterrupt, SystemExit))
                        else "error"
                    )
                    self._checkpoint(
                        termination_reason=f"{prefix}:{type(exc).__name__}"
                    )
                raise

    def _query_and_download_locked(
        self,
        target_count: int = 500,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        print("=" * 60)
        print(
            f"Starting robust {self.source.upper()} download | "
            f"target={target_count} | cached={len(self.downloaded_ids)}"
        )
        print("=" * 60)

        query_params = dict(query_params or {})
        query_params.setdefault("limit", metadata_candidate_limit(target_count))
        query_params.setdefault("page_size", 500)
        query_params.setdefault("gap_bins", 4 if self.source == "aflow" else 1)
        print(f"[Query] parameters: {query_params}")

        remaining = max(0, target_count - len(self.downloaded_ids))
        if remaining == 0:
            print("[Skip] cache already meets the requested total target")
            return self._print_summary()

        metadata_list = self._load_candidate_catalog(query_params)
        local_available = sum(
            item.get("material_id") not in self.downloaded_ids
            and item.get("material_id") not in self.excluded_ids
            for item in metadata_list
        )
        if local_available >= remaining:
            print(
                f"[Catalog] reusing {local_available} local candidates; "
                "metadata API skipped"
            )
        else:
            self._fetch_next_catalog_page(query_params)
            metadata_list = self._load_candidate_catalog(query_params)

        candidates = self._diverse_candidate_order([
            item
            for item in metadata_list
            if item.get("material_id")
            and item["material_id"] not in self.downloaded_ids
            and item["material_id"] not in self.excluded_ids
        ])

        print(
            f"[Download] scanning {len(candidates)} candidates until "
            f"{remaining} additional successful downloads\n"
        )

        attempted = 0
        next_index = 0
        attempted_ids: set[str] = set()
        candidate_ids = {
            str(item["material_id"])
            for item in candidates
            if item.get("material_id")
        }
        use_mp_batch = (
            self.source == "mp"
            and self.workers == 1
            and hasattr(self.adapter, "fetch_band_structures")
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            while self.stats["success"] < remaining:
                if next_index >= len(candidates):
                    new_records = self._fetch_next_catalog_page(query_params)
                    if new_records is None:
                        break
                    if not new_records:
                        continue
                    additions = self._diverse_candidate_order(
                        [
                            item
                            for item in new_records
                            if item.get("material_id")
                            and str(item["material_id"]) not in candidate_ids
                            and str(item["material_id"]) not in attempted_ids
                            and str(item["material_id"])
                            not in self.downloaded_ids
                            and str(item["material_id"]) not in self.excluded_ids
                        ]
                    )
                    candidate_ids.update(
                        str(item["material_id"]) for item in additions
                    )
                    candidates.extend(additions)
                    if not additions:
                        continue
                request_width = self.batch_size if use_mp_batch else self.workers
                request_width = min(
                    request_width,
                    len(candidates) - next_index,
                    remaining - self.stats["success"],
                )
                batch = candidates[next_index : next_index + request_width]
                next_index += request_width
                attempted_ids.update(
                    str(item["material_id"]) for item in batch
                )
                start_attempt = attempted + 1
                attempted += request_width
                self._run_state["attempted"] = attempted
                self._run_state["submitted"] += request_width
                self._run_state["in_flight"] = request_width
                self._run_state["peak_in_flight"] = max(
                    int(self._run_state["peak_in_flight"]),
                    request_width,
                )

                if use_mp_batch:
                    try:
                        fetched = list(self.adapter.fetch_band_structures(batch))
                    except Exception as exc:
                        outcomes = [
                            (
                                meta,
                                {
                                    "material_id": meta["material_id"],
                                    "error": (
                                        f"unexpected {type(exc).__name__}: {exc}"
                                    ),
                                },
                            )
                            for meta in batch
                        ]
                    else:
                        if len(fetched) != len(batch):
                            raise RuntimeError(
                                "MP batch cardinality mismatch: "
                                f"requested {len(batch)}, received {len(fetched)}"
                            )
                        outcomes = list(zip(batch, fetched))
                else:
                    futures = {
                        executor.submit(self._fetch_candidate, meta): meta for meta in batch
                    }

                    def completed_outcomes():
                        for future in concurrent.futures.as_completed(futures):
                            meta = futures[future]
                            try:
                                _meta, bs_data = future.result()
                            except Exception as exc:
                                bs_data = {
                                    "material_id": meta["material_id"],
                                    "error": (
                                        f"unexpected {type(exc).__name__}: {exc}"
                                    ),
                                }
                            yield meta, bs_data

                    outcomes = completed_outcomes()

                for offset, (meta, bs_data) in enumerate(outcomes):
                    self._run_state["completed"] += 1
                    self._run_state["in_flight"] = max(
                        0,
                        int(self._run_state["in_flight"]) - 1,
                    )
                    material_id = meta["material_id"]
                    formula = meta.get("formula_pretty") or "N/A"
                    spacegroup = meta.get("spacegroup_number")
                    position = start_attempt + offset
                    prefix = (
                        f"[{position}/{len(candidates)}] {material_id} "
                        f"({formula}) SG:{spacegroup}"
                    )
                    error = bs_data.get("error")
                    if error:
                        error_lower = str(error).lower()
                        if any(
                            token in error_lower
                            for token in (
                                "line-mode",
                                "uniform",
                                "no band",
                                "band data unavailable",
                            )
                        ):
                            self.stats["no_data"] += 1
                            self.excluded_ids[material_id] = str(error)[:200]
                            print(
                                f"{prefix} NO_USABLE_BAND_DATA: {str(error)[:90]}"
                            )
                        else:
                            self.stats["failed"] += 1
                            print(f"{prefix} FAILED: {str(error)[:120]}")
                        continue
                    bs_data["metadata"] = meta
                    save_result = self.adapter.save_to_hdf5(bs_data)
                    save_status = getattr(save_result, "status", "canonical")
                    if save_status in {"variant", "duplicate_variant"}:
                        self.stats["failed"] += 1
                        print(
                            f"{prefix} VARIANT_ISOLATED: canonical was not replaced"
                        )
                        continue
                    self.downloaded_ids.add(material_id)
                    merge_metadata_json(self.metadata_path, [meta])
                    self.stats["success"] += 1
                    if spacegroup:
                        self.stats["spacegroup_dist"][spacegroup] += 1
                    print(
                        f"{prefix} OK | "
                        f"persisted={len(self.downloaded_ids)}/{target_count} | "
                        f"submitted={self._run_state['submitted']} | "
                        f"completed={self._run_state['completed']} | "
                        f"failed={self.stats['failed']} | "
                        f"excluded={len(self.excluded_ids)} | "
                        f"in_flight={self._run_state['in_flight']}",
                        flush=True,
                    )

                self._checkpoint(termination_reason="running")

        if self.stats["success"] < remaining:
            print(
                f"[WARN] Needed {remaining} additional successes, got "
                f"{self.stats['success']} after {attempted} attempts."
            )

        if self.excluded_ids:
            self._save_excluded()

        report = self._print_summary()
        if not report["target_reached"]:
            raise DownloadTargetNotReached(
                f"requested target {target_count}, persisted "
                f"{len(self.downloaded_ids)}"
            )
        return report

    def _print_summary(self) -> Dict[str, Any]:
        target_count = int(self._run_state["requested_target"])
        termination_reason = (
            "target_reached"
            if len(self.downloaded_ids) >= target_count
            else "candidate_pool_exhausted"
        )
        report = self._checkpoint(termination_reason=termination_reason)
        print("\n" + "=" * 60)
        print("Download summary")
        print("=" * 60)
        print(f"success: {self.stats['success']}")
        print(f"line-mode unavailable: {self.stats['no_data']}")
        print(f"failed: {self.stats['failed']}")
        print(f"already cached at start: {self.stats['skipped']}")
        print(f"permanently excluded (no usable band data): {len(self.excluded_ids)}")
        print(f"total cached now: {len(self.downloaded_ids)}")
        print(f"target reached: {report['target_reached']}")
        print(f"termination reason: {report['termination_reason']}")
        print(f"new spacegroups: {len(self.stats['spacegroup_dist'])}")
        report_path = os.path.join(self.cache_dir, f"{self.source}_download_report.json")
        print(f"report saved: {report_path}")
        return report


def metadata_candidate_limit(target_count: int) -> int:
    """Bound AFLUX metadata paging while leaving headroom for no-data records."""
    return max(100, min(target_count * 2, 80_000))


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust MP/AFLOW band downloader")
    parser.add_argument("--source", choices=("mp", "aflow"), default="mp")
    parser.add_argument("--target", "-t", type=int, default=200, help="total cached target")
    parser.add_argument(
        "--cache", "-c", type=str, default=None,
        help="cache dir (default: data/raw/materials_project or data/raw/aflow)",
    )
    parser.add_argument("--min-gap", type=float, default=0.0, help="minimum band gap in eV")
    parser.add_argument("--max-gap", type=float, default=5.0, help="maximum band gap in eV")
    parser.add_argument(
        "--gap-bins",
        type=int,
        default=5,
        help="AFLOW stratified gap bins across [min-gap, max-gap]",
    )
    parser.add_argument("--max-sites", type=int, default=50, help="maximum num_sites filter")
    parser.add_argument("--workers", type=int, default=1, help="bounded concurrent fetches")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="MP task IDs per Delta query (workers must be 1)",
    )
    parser.add_argument(
        "--theoretical-only",
        action="store_true",
        help="restrict MP summaries to structures marked theoretical",
    )
    args = parser.parse_args()

    downloader = None
    try:
        kwargs: Dict[str, Any] = {
            "cache_dir": args.cache,
            "source": args.source,
            "workers": args.workers,
            "batch_size": args.batch_size,
        }
        if args.source == "mp":
            from src.data.mp_adapter import load_api_keys

            api_keys = load_api_keys()
            kwargs.update(
                new_api_key=api_keys["new"],
                # Never initialize the deprecated client when the validated
                # current key is available; this avoids misleading warnings
                # and keeps all workers on the same API generation.
                legacy_api_key=None if api_keys["new"] else api_keys["legacy"],
            )
        downloader = RobustBandDownloader(**kwargs)

        query_params = {
            "band_gap": (args.min_gap, args.max_gap),
            "num_sites": (1, args.max_sites),
        }
        if args.theoretical_only:
            query_params["theoretical"] = True
        if args.source == "aflow":
            # Fetch twice the target by default because a material may advertise
            # a bands file while the payload is missing or malformed. The 80k cap
            # keeps paging bounded while allowing a 30k usable target to query 60k.
            query_params["limit"] = metadata_candidate_limit(args.target)
            query_params["page_size"] = 500
            query_params["gap_bins"] = max(1, args.gap_bins)
        else:
            query_params["limit"] = max(3000, args.target * 5)
            query_params["page_size"] = 1000
        downloader.query_and_download(target_count=args.target, query_params=query_params)
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        if downloader is not None:
            downloader.close()


if __name__ == "__main__":
    main()
