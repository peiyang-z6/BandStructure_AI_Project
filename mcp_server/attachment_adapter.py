"""Deterministic operator-side attachment preparation; never asks an LLM to encode bytes."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

from mcp_server.artifacts import put


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_adapter(request):
    from mcp_server.resource_limits import run_bounded, WorkerOutputLimit

    allowed = {
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "PATH",
        "LOCALAPPDATA",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "PROCESSOR_ARCHITECTURE",
    }
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed}
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="-1",
    )
    if os.environ.get("BAND_MCP_STORE_DIR"):
        env["BAND_MCP_STORE_DIR"] = os.environ["BAND_MCP_STORE_DIR"]
    raw = json.dumps(request, allow_nan=False).encode()
    if len(raw) > 65536:
        raise ValueError("ADAPTER_REQUEST_BUDGET")
    try:
        completed = run_bounded(
            [sys.executable, "-B", str(Path(__file__).with_name("attachment_worker.py"))],
            input=raw,
            env=env,
            timeout=120,
            max_stdout=128 * 1024,
        )
    except (subprocess.TimeoutExpired, WorkerOutputLimit) as exc:
        raise ValueError("ADAPTER_RESOURCE_BUDGET") from exc
    if completed.returncode == 5:
        raise ValueError("MEMORY_LIMIT_SETUP_FAILED")
    if completed.returncode == 4:
        raise ValueError("ADAPTER_MEMORY_LIMIT")
    if completed.returncode != 0:
        raise ValueError("ADAPTER_WORKER_FAILED")
    reply = json.loads(completed.stdout)
    if not reply.get("ok"):
        raise ValueError(reply.get("error", "ADAPTER_REFUSED"))
    return reply["result"]


def prepare(path: Path, *, page_number: int | None = None, max_dimension: int = 3000) -> dict:
    """Prepare a selected local source in a memory/output/time-bounded child."""
    return _run_adapter(
        {
            "operation": "prepare",
            "path": str(Path(path).absolute()),
            "page_number": page_number,
            "max_dimension": max_dimension,
        }
    )


def count_pdf_pages(path: Path):
    return _run_adapter({"operation": "count_pdf_pages", "path": str(Path(path).absolute())})[
        "page_count"
    ]


def _prepare_inline(
    path: Path, *, page_number: int | None = None, max_dimension: int = 3000
) -> dict:
    """Private worker implementation; public callers must use prepare()."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Choose a regular source file")
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("Source exceeds the adapter 256 MiB budget")
    if type(max_dimension) is not int or not 256 <= max_dimension <= 4096:
        raise ValueError("Invalid raster dimension")
    digest = _sha_file(path)
    metadata = {"source_filename": path.name, "original_source_sha256": digest}
    suffix = path.suffix.lower()
    if path.name.lower().endswith("vasprun.xml") or suffix == ".xml":
        from mcp_server.scientific_data import parse_vasprun

        data = parse_vasprun(path)
        data["source"] = {"sha256": digest, "filename": path.name}
        payload = json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8")
        kind = "electronic_data"
        metadata["source_format"] = "vasprun_xml"
    elif (
        path.name.upper() in {"POSCAR", "CONTCAR"}
        or path.name.upper().endswith(("_POSCAR", "_CONTCAR"))
        or suffix == ".vasp"
    ):
        from mcp_server.scientific_data import parse_poscar

        payload = json.dumps(parse_poscar(path.read_bytes()), separators=(",", ":")).encode()
        kind = "structure"
    elif suffix == ".pdf" and page_number is not None:
        import pymupdf

        with pymupdf.open(path) as document:
            if type(page_number) is not int or not 1 <= page_number <= len(document):
                raise ValueError("Invalid PDF page")
            page = document[page_number - 1]
            scale = min(2.5, max_dimension / max(page.rect.width, page.rect.height))
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale), alpha=False, colorspace=pymupdf.csRGB
            )
            payload = pix.tobytes("png")
            kind = "image"
            matrix = page.derotation_matrix
            metadata.update(
                source_format="pdf_render",
                source_page=page_number,
                total_pages=len(document),
                image_size=[pix.width, pix.height],
                source_rotation=page.rotation,
                pixel_to_unrotated_pdf_point=[
                    matrix.a / scale,
                    matrix.b / scale,
                    matrix.c / scale,
                    matrix.d / scale,
                    matrix.e + (matrix.a * pix.x + matrix.c * pix.y) / scale,
                    matrix.f + (matrix.b * pix.x + matrix.d * pix.y) / scale,
                ],
                resampling_note="Deterministic page raster; no labels or curve pixels inferred.",
            )
    elif suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            if width * height > 64 * 1024 * 1024:
                raise ValueError("Source raster exceeds adapter pixel budget")
            scale = min(1.0, max_dimension / max(width, height))
            image = image.convert("RGB")
            if scale < 1:
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )
            target = io.BytesIO()
            image.save(target, format="PNG")
            payload = target.getvalue()
            kind = "image"
            metadata.update(
                image_size=list(image.size),
                original_image_size=[width, height],
                pixel_to_original_pixel_scale=[width / image.width, height / image.height],
            )
    elif suffix == ".pdf":
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Use --page or --all-pages for a large PDF")
        payload = path.read_bytes()
        kind = "pdf"
    elif suffix == ".json":
        payload = path.read_bytes()
        data = json.loads(payload)
        if not isinstance(data, dict) or data.get("data_kind") not in {"line_mode", "uniform_mesh"}:
            raise ValueError("JSON attachment requires an explicit scientific data_kind")
        kind = "electronic_data"
    else:
        raise ValueError(
            "Supported files: raster, PDF, POSCAR/CONTCAR/.vasp, vasprun.xml, scientific JSON"
        )
    if kind in {"pdf", "image"} and len(payload) > 10 * 1024 * 1024:
        raise ValueError("Prepared image still exceeds upload budget; reduce --max-dimension")
    if _sha_file(path) != digest:
        raise ValueError("SOURCE_CHANGED_DURING_PREPARATION")
    record = put(payload, kind=kind, metadata=metadata, ttl_seconds=None, attachment=True)
    return {
        "attachment_id": record["artifact_id"],
        "kind": kind,
        "source_sha256": record["sha256"],
        "bytes": record["bytes"],
        "metadata": metadata,
    }
