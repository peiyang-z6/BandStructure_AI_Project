"""Harvest source-bound electronic-band figure candidates from Europe PMC OA APIs.

This module creates figure-level AI candidates. It never creates human approval or
curve-level physical annotations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zipfile


API_ROOT = "https://www.ebi.ac.uk/europepmc/webservices/rest"
USER_AGENT = "BandStructure-MCP-research/0.5 (source-bound OA figure harvest)"
DEFAULT_QUERY = (
    'OPEN_ACCESS:Y AND HAS_FT:Y AND FIG:"electronic band structure" '
    'AND LICENSE:"CC BY" AND PUB_YEAR:[2015 TO 2026]'
)
_ELECTRONIC_BAND = re.compile(r"\belectronic\s+band[\s-]+structures?\b", re.I)
_EXCLUDED_CONTEXT = re.compile(
    r"\b(?:photonic|phononic|acoustic|magnon|spin[\s-]?wave)\b|"
    r"reproduced\s+with\s+permission|adapted\s+with\s+permission|"
    r"copyright(?:ed)?\s+(?:by|material)",
    re.I,
)
_PMCID = re.compile(r"PMC[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_ASSET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_IMAGE_SUFFIXES = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif",
                   "TIFF": ".tif", "WEBP": ".webp"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    for child in element.iter():
        if _local_name(child.tag) == name:
            return " ".join("".join(child.itertext()).split())
    return ""


def normalise_license(value: Any) -> str:
    """Accept only attribution-only Creative Commons material."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing or unsupported article license")
    normalised = value.strip().lower().replace("_", "-")
    compact = re.sub(r"\s+", " ", normalised)
    if any(marker in compact for marker in ("by-nc", "by-nd", "by-sa", "/by-nc/", "/by-nd/", "/by-sa/")):
        raise ValueError("restricted Creative Commons variant is not accepted")
    if re.search(r"creativecommons\.org/licenses/by/(?:[0-9.]+/?)?$", compact):
        return "CC-BY"
    if re.fullmatch(r"cc[ -]?by(?:[ -]?[0-9.]+)?", compact):
        return "CC-BY"
    raise ValueError("missing or unsupported article license")


def parse_electronic_band_figures(xml_payload: bytes) -> list[dict[str, str]]:
    """Return reusable-looking electronic-band figures from bounded JATS XML."""
    if not isinstance(xml_payload, bytes) or not xml_payload or len(xml_payload) > 24 * 1024 * 1024:
        raise ValueError("invalid or oversized article XML")
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError as exc:
        raise ValueError("invalid article XML") from exc
    figures: list[dict[str, str]] = []
    for figure in root.iter():
        if _local_name(figure.tag) != "fig":
            continue
        caption = _child_text(figure, "caption")
        if not _ELECTRONIC_BAND.search(caption) or _EXCLUDED_CONTEXT.search(caption):
            continue
        asset_href = ""
        for graphic in figure.iter():
            if _local_name(graphic.tag) not in {"graphic", "inline-graphic"}:
                continue
            for key, value in graphic.attrib.items():
                if _local_name(key) == "href" and value:
                    asset_href = value.strip()
                    break
            if asset_href:
                break
        asset_name = PurePosixPath(asset_href.replace("\\", "/")).name
        if (not asset_name or asset_name != asset_href
                or not _SAFE_ASSET.fullmatch(asset_name)):
            continue
        figure_id = str(figure.attrib.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", figure_id):
            continue
        label = _child_text(figure, "label") or figure_id
        figures.append({
            "figure_id": figure_id,
            "figure_label": label[:256],
            "caption": caption[:20_000],
            "asset_href": asset_name,
        })
    return figures


def extract_zip_asset(zip_payload: bytes, asset_href: str) -> bytes:
    """Extract one exact, bounded image member without filesystem extraction."""
    if not isinstance(zip_payload, bytes) or not zip_payload or len(zip_payload) > 128 * 1024 * 1024:
        raise ValueError("invalid or oversized Europe PMC asset archive")
    if not isinstance(asset_href, str) or not _SAFE_ASSET.fullmatch(asset_href):
        raise ValueError("unsafe figure asset name")
    try:
        with zipfile.ZipFile(io.BytesIO(zip_payload)) as archive:
            safe_members = []
            for info in archive.infolist():
                parts = PurePosixPath(info.filename.replace("\\", "/")).parts
                if info.is_dir() or not parts or any(part in {"", ".", ".."} for part in parts):
                    continue
                if PurePosixPath(info.filename).name == asset_href:
                    safe_members.append(info)
            exact = [info for info in safe_members if info.filename == asset_href]
            candidates = exact or safe_members
            if len(candidates) != 1:
                raise ValueError("figure asset missing or ambiguous in archive")
            info = candidates[0]
            if info.file_size <= 0 or info.file_size > 32 * 1024 * 1024:
                raise ValueError("figure image exceeds decompression budget")
            if info.compress_size <= 0 or info.file_size / info.compress_size > 250:
                raise ValueError("figure image compression ratio refused")
            payload = archive.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid Europe PMC asset archive") from exc
    if len(payload) != info.file_size:
        raise ValueError("figure image size mismatch")
    return payload


def _image_metadata(payload: bytes) -> dict[str, Any]:
    from PIL import Image

    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            image_format = str(image.format or "").upper()
    except Exception as exc:
        raise ValueError("invalid figure image") from exc
    if width < 120 or height < 100 or width * height > 50_000_000:
        raise ValueError("figure image dimensions outside review bounds")
    suffix = _IMAGE_SUFFIXES.get(image_format)
    if suffix is None:
        raise ValueError("unsupported figure image format")
    return {"width": width, "height": height, "format": image_format,
            "extension": suffix, "bytes": len(payload), "sha256": _sha256(payload)}


def build_candidate_record(
    article: dict[str, Any],
    figure: dict[str, str],
    image_payload: bytes,
    article_xml_sha256: str,
    supplementary_zip_sha256: str,
) -> dict[str, Any]:
    """Build a provenance-bound figure-level candidate with no approval claims."""
    pmcid = str(article.get("pmcid") or "")
    if not _PMCID.fullmatch(pmcid):
        raise ValueError("invalid PMCID")
    if article.get("isOpenAccess") != "Y" or article.get("isRetracted") == "Y":
        raise ValueError("article is not eligible open-access source material")
    license_id = normalise_license(article.get("license"))
    if not _SHA256.fullmatch(article_xml_sha256) or not _SHA256.fullmatch(supplementary_zip_sha256):
        raise ValueError("invalid source archive hash")
    image = _image_metadata(image_payload)
    identity = {
        "pmcid": pmcid,
        "figure_id": figure["figure_id"],
        "asset_href": figure["asset_href"],
        "image_sha256": image["sha256"],
    }
    candidate_id = f"epmc-band-{_sha256(_canonical(identity))[:24]}"
    return {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "source": {
            "provider": "Europe PMC Open Access API",
            "pmcid": pmcid,
            "doi": article.get("doi"),
            "title": article.get("title"),
            "authors": article.get("authorString"),
            "journal": article.get("journalTitle"),
            "publication_year": int(article["pubYear"]) if str(article.get("pubYear", "")).isdigit() else None,
            "license": license_id,
            "figure_id": figure["figure_id"],
            "figure_label": figure["figure_label"],
            "caption": figure["caption"],
            "article_xml_url": f"{API_ROOT}/{pmcid}/fullTextXML",
            "article_xml_sha256": article_xml_sha256,
            "asset_archive_url": f"{API_ROOT}/{pmcid}/supplementaryFiles",
            "asset_archive_sha256": supplementary_zip_sha256,
            "asset_href": figure["asset_href"],
        },
        "group_id": pmcid,
        "image": image,
        "annotation": {
            "class": "electronic_band_structure_figure",
            "level": "caption_matched_ai_candidate",
            "curve_level_annotation": None,
            "physical_calibration": None,
        },
        "human_audited": False,
        "formal_approval": None,
        "operator_authentication": None,
        "eligible_for_scientific_acceptance": False,
    }


def _write_candidate(directory: Path, record: dict[str, Any], image_payload: bytes) -> dict[str, Any]:
    candidate_id = record["candidate_id"]
    if not re.fullmatch(r"epmc-band-[0-9a-f]{24}", candidate_id):
        raise ValueError("unsafe candidate ID")
    image_relpath = f"images/{candidate_id}{record['image']['extension']}"
    record_relpath = f"records/{candidate_id}.json"
    image_path = directory / image_relpath
    record_path = directory / record_relpath
    image_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    persisted = {**record, "image": {**record["image"], "path": image_relpath}}
    with image_path.open("xb") as handle:
        handle.write(image_payload)
    try:
        with record_path.open("xb") as handle:
            handle.write(json.dumps(persisted, indent=2, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8"))
    except Exception:
        image_path.unlink(missing_ok=True)
        raise
    return persisted


def _manifest(records: Iterable[dict[str, Any]], target_count: int, query: str) -> dict[str, Any]:
    items = sorted(records, key=lambda item: item["candidate_id"])
    groups = {item["group_id"] for item in items}
    return {
        "schema_version": 1,
        "status": "complete_ai_assisted_figure_candidate_snapshot",
        "scope": "figure-level electronic-band-structure candidates; not curve-level physics labels",
        "source_provider": "Europe PMC Open Access REST API",
        "source_query": query,
        "license_allowlist": ["CC-BY"],
        "target_count": target_count,
        "record_count": len(items),
        "document_group_count": len(groups),
        "group_split_rule": "split only by PMCID; never split figures from one article across partitions",
        "human_audited_count": 0,
        "formal_approved_count": 0,
        "eligible_for_scientific_acceptance": False,
        "records": [{
            "candidate_id": item["candidate_id"],
            "group_id": item["group_id"],
            "record_path": f"records/{item['candidate_id']}.json",
            "image_path": item["image"]["path"],
            "image_sha256": item["image"]["sha256"],
        } for item in items],
    }


def write_snapshot(
    output_dir: str | Path,
    candidates: Iterable[tuple[dict[str, Any], bytes]],
    target_count: int,
    query: str = DEFAULT_QUERY,
) -> dict[str, Any]:
    """Write a complete immutable snapshot; primarily used by tests and small imports."""
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError(str(directory))
    directory.mkdir(parents=True)
    records = [_write_candidate(directory, record, payload) for record, payload in candidates]
    manifest = _manifest(records, target_count, query)
    manifest_path = directory / "manifest.json"
    manifest_payload = json.dumps(manifest, indent=2, ensure_ascii=False,
                                  allow_nan=False).encode("utf-8")
    with manifest_path.open("xb") as handle:
        handle.write(manifest_payload)
    return {"manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_payload), "record_count": len(records)}


class RateLimiter:
    def __init__(self, interval_seconds: float) -> None:
        self.interval = max(float(interval_seconds), 0.0)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if delay:
            time.sleep(delay)


def _validate_api_url(url: str) -> None:
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname != "www.ebi.ac.uk"
            or not parsed.path.startswith("/europepmc/webservices/rest/")):
        raise ValueError("refused non-Europe-PMC API URL")


def fetch_bytes(url: str, max_bytes: int, limiter: RateLimiter,
                timeout: float = 60.0, retries: int = 3) -> bytes:
    _validate_api_url(url)
    last_error: Exception | None = None
    for attempt in range(retries):
        limiter.wait()
        try:
            request = Request(url, headers={"Accept": "*/*", "User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                final_url = response.geturl()
                _validate_api_url(final_url)
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > max_bytes:
                    raise ValueError("Europe PMC response exceeds declared size budget")
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise ValueError("Europe PMC response exceeds size budget")
                return payload
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                break
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f"Europe PMC request failed: {type(last_error).__name__}") from last_error


def _search_page(query: str, cursor: str, page_size: int,
                 limiter: RateLimiter, timeout: float) -> dict[str, Any]:
    params = urlencode({"query": query, "format": "json", "resultType": "core",
                        "pageSize": page_size, "cursorMark": cursor})
    payload = fetch_bytes(f"{API_ROOT}/search?{params}", 12 * 1024 * 1024,
                          limiter, timeout)
    try:
        result = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Europe PMC search response") from exc
    if not isinstance(result, dict) or not isinstance(result.get("resultList", {}).get("result"), list):
        raise ValueError("invalid Europe PMC search response")
    return result


def _article_candidates(article: dict[str, Any], limiter: RateLimiter,
                        timeout: float, max_per_article: int) -> list[tuple[dict[str, Any], bytes]]:
    pmcid = str(article.get("pmcid") or "")
    if (not _PMCID.fullmatch(pmcid) or article.get("isOpenAccess") != "Y"
            or article.get("isRetracted") == "Y"):
        return []
    try:
        normalise_license(article.get("license"))
        xml_payload = fetch_bytes(f"{API_ROOT}/{pmcid}/fullTextXML", 24 * 1024 * 1024,
                                  limiter, timeout)
        figures = parse_electronic_band_figures(xml_payload)[:max_per_article]
        if not figures:
            return []
        archive = fetch_bytes(f"{API_ROOT}/{pmcid}/supplementaryFiles", 128 * 1024 * 1024,
                              limiter, timeout)
        output = []
        for figure in figures:
            try:
                image_payload = extract_zip_asset(archive, figure["asset_href"])
                record = build_candidate_record(article, figure, image_payload,
                                                _sha256(xml_payload), _sha256(archive))
                output.append((record, image_payload))
            except ValueError:
                continue
        return output
    except (ValueError, RuntimeError):
        return []


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def harvest(
    output_dir: str | Path,
    target_count: int = 300,
    query: str = DEFAULT_QUERY,
    page_size: int = 100,
    max_articles: int = 1200,
    max_per_article: int = 2,
    workers: int = 4,
    request_interval: float = 0.2,
    timeout: float = 60.0,
    resume: bool = False,
) -> dict[str, Any]:
    """Harvest an immutable-on-completion, resumable figure candidate snapshot."""
    if not 1 <= target_count <= 5000:
        raise ValueError("target_count must be 1-5000")
    if not 1 <= page_size <= 1000 or not 1 <= max_per_article <= 5:
        raise ValueError("invalid page or per-article bound")
    if not 1 <= workers <= 8 or max_articles < target_count // max_per_article:
        raise ValueError("invalid worker or article bound")
    directory = Path(output_dir)
    manifest_path = directory / "manifest.json"
    state_path = directory / "harvest_state.json"
    if directory.exists():
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("invalid harvest output directory")
        if manifest_path.exists() or not resume:
            raise FileExistsError(str(directory))
    else:
        directory.mkdir(parents=True)
    (directory / "images").mkdir(exist_ok=True)
    (directory / "records").mkdir(exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    seen_groups: dict[str, int] = {}
    for record_path in sorted((directory / "records").glob("epmc-band-*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        image_path = directory / record["image"]["path"]
        if (not image_path.is_file() or _sha256(image_path.read_bytes()) != record["image"]["sha256"]
                or record.get("human_audited") is not False
                or record.get("eligible_for_scientific_acceptance") is not False):
            raise ValueError("resume snapshot contains invalid candidate")
        existing[record["candidate_id"]] = record
        seen_groups[record["group_id"]] = seen_groups.get(record["group_id"], 0) + 1

    state = {"cursor": "*", "articles_scanned": 0, "pages_completed": 0,
             "request_query": query, "started_utc": datetime.now(timezone.utc).isoformat()}
    if state_path.exists():
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
        if loaded.get("request_query") != query:
            raise ValueError("resume query does not match existing state")
        state.update(loaded)
    limiter = RateLimiter(request_interval)

    while len(existing) < target_count and state["articles_scanned"] < max_articles:
        page = _search_page(query, str(state["cursor"]), page_size, limiter, timeout)
        articles = page["resultList"]["result"]
        if not articles:
            break
        remaining_articles = max_articles - state["articles_scanned"]
        articles = articles[:remaining_articles]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_article_candidates, article, limiter, timeout,
                                   max_per_article) for article in articles]
            for candidates in (future.result() for future in futures):
                for record, image_payload in candidates:
                    if len(existing) >= target_count:
                        break
                    if seen_groups.get(record["group_id"], 0) >= max_per_article:
                        continue
                    if record["candidate_id"] in existing:
                        continue
                    persisted = _write_candidate(directory, record, image_payload)
                    existing[persisted["candidate_id"]] = persisted
                    seen_groups[persisted["group_id"]] = seen_groups.get(persisted["group_id"], 0) + 1
        state["articles_scanned"] += len(articles)
        state["pages_completed"] += 1
        next_cursor = page.get("nextCursorMark")
        if not isinstance(next_cursor, str) or next_cursor == state["cursor"]:
            break
        state["cursor"] = next_cursor
        state["record_count"] = len(existing)
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(state_path, state)

    manifest = _manifest(existing.values(), target_count, query)
    manifest["articles_scanned"] = state["articles_scanned"]
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    if len(existing) < target_count:
        manifest["status"] = "incomplete_ai_assisted_figure_candidate_snapshot"
        _atomic_json(state_path, {**state, "record_count": len(existing),
                                  "last_status": manifest["status"]})
        return {"status": manifest["status"], "record_count": len(existing),
                "articles_scanned": state["articles_scanned"],
                "output_dir": str(directory.resolve())}
    manifest_payload = json.dumps(manifest, indent=2, ensure_ascii=False,
                                  allow_nan=False).encode("utf-8")
    with manifest_path.open("xb") as handle:
        handle.write(manifest_payload)
    return {"status": manifest["status"], "record_count": len(existing),
            "document_group_count": manifest["document_group_count"],
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_payload),
            "output_dir": str(directory.resolve())}

