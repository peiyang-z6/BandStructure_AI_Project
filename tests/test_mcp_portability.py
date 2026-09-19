"""Portable client configuration contracts for the MCP-first P2 delivery."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


def fake_install(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    server = root / "mcp_server" / "server.py"
    python = tmp_path / "runtime" / "python.exe"
    server.parent.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    server.write_text("# server\n", encoding="utf-8")
    python.write_bytes(b"runtime")
    return root, python


def test_client_config_bundle_has_parseable_standard_stdio_shapes(tmp_path):
    from mcp_server.client_configs import build_client_configs

    root, python = fake_install(tmp_path)
    bundle = build_client_configs(python, root)
    server = (root / "mcp_server" / "server.py").as_posix()
    command = python.as_posix()

    for client in ("claude_desktop", "cursor"):
        entry = bundle[f"{client}.json"]["mcpServers"]["bandstructure"]
        assert entry["type"] == "stdio"
        assert entry["command"] == command
        assert entry["args"] == ["-B", server]
        assert entry["env"]["PYTHONDONTWRITEBYTECODE"] == "1"

    vscode = bundle["vscode.mcp.json"]["servers"]["bandstructure"]
    assert vscode["type"] == "stdio" and vscode["command"] == command
    assert vscode["cwd"] == root.as_posix()

    codex = tomllib.loads(bundle["codex.config.toml"])
    entry = codex["mcp_servers"]["bandstructure"]
    assert entry["command"] == command
    assert entry["args"] == ["-B", server]
    assert entry["env"]["CUDA_VISIBLE_DEVICES"] == "-1"


def test_generated_configs_contain_no_secret_or_unverified_remote_claim(tmp_path):
    from mcp_server.client_configs import build_client_configs

    root, python = fake_install(tmp_path)
    bundle = build_client_configs(python, root)
    text = json.dumps(bundle, sort_keys=True).lower()
    for forbidden in ("password", "api_key", "bearer", "secret", "access_token"):
        assert forbidden not in text
    manifest = bundle["manifest.json"]
    assert manifest["transport"] == "stdio"
    assert manifest["openai_responses_api"] == "use_authenticated_HTTP_or_authorized_tunnel_not_this_stdio_template"
    assert manifest["generated_configs_are_installed"] is False


@pytest.mark.parametrize("missing", ["python", "root", "server"])
def test_config_builder_rejects_missing_or_nonabsolute_paths(tmp_path, missing):
    from mcp_server.client_configs import build_client_configs

    root, python = fake_install(tmp_path)
    if missing == "python":
        python = tmp_path / "missing-python.exe"
    elif missing == "root":
        root = Path("relative-project")
    else:
        (root / "mcp_server" / "server.py").unlink()
    with pytest.raises(ValueError):
        build_client_configs(python, root)


def test_generator_writes_new_hash_bound_bundle_and_refuses_overwrite(tmp_path):
    root, python = fake_install(tmp_path)
    out = tmp_path / "bundle"
    command = [
        sys.executable,
        "scripts/generate_mcp_client_configs.py",
        "--python", str(python),
        "--project-root", str(root),
        "--output-dir", str(out),
    ]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    expected = {
        "claude_desktop.json", "cursor.json", "vscode.mcp.json",
        "codex.config.toml", "manifest.json",
    }
    assert {path.name for path in out.iterdir()} == expected
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files_sha256"]) == expected - {"manifest.json"}
    assert all(len(value) == 64 for value in manifest["files_sha256"].values())

    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert "already exists" in (second.stderr + second.stdout)


def test_portability_manifest_distinguishes_templates_from_verified_clients(tmp_path):
    from mcp_server.client_configs import build_client_configs

    root, python = fake_install(tmp_path)
    manifest = build_client_configs(python, root)["manifest.json"]
    assert manifest["client_templates"] == ["claude_desktop", "cursor", "vscode", "codex"]
    assert manifest["verification_status"] == "generated_not_client_verified"
    assert manifest["windows_stdio_sandbox"] == "not_provided_by_vscode"


def test_upload_worker_environment_does_not_inherit_credentials(monkeypatch):
    from mcp_server.server import _upload_worker_environment

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross-worker-boundary")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-worker-boundary")
    env = _upload_worker_environment()
    assert "OPENAI_API_KEY" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["BAND_MCP_WORKER"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    import os
    if os.name == "nt":
        for required in ("SYSTEMROOT", "TEMP"):
            assert required in env
    else:
        assert env["LANG"]


def test_config_generation_does_not_create_artifact_store(tmp_path, monkeypatch):
    from mcp_server.client_configs import build_client_configs
    root, python = fake_install(tmp_path)
    store = tmp_path / "not-created" / "artifacts"
    monkeypatch.setenv("BAND_MCP_STORE_DIR", str(store))
    build_client_configs(python, root)
    assert not store.exists()


def test_real_upload_worker_roundtrips_document_without_parent_ocr():
    import base64
    from mcp_server.server import _run_upload_worker
    from tests.test_mcp_documents import synthetic_text_png

    result = _run_upload_worker("extract_document", {
        "payload_base64": base64.b64encode(synthetic_text_png()).decode("ascii"),
        "kind": "image",
        "max_pages": 1,
        "page_start": 1,
    })
    assert result["status"] == "ok"
    assert "Energy" in result["text"]
    assert result["worker_isolated"] is True
    assert result["confidence"] is None and result["ood_flag"] is None


def test_real_upload_worker_isolates_pdf_geometry_path():
    import base64
    from mcp_server.server import _run_upload_worker
    from tests.test_mcp_pdf_practical import sample_document

    result = _run_upload_worker("extract_band_from_image", {
        "payload_base64": base64.b64encode(sample_document()).decode("ascii"),
        "annotations": None,
        "calibration": None,
        "kind": "pdf",
        "page_number": 2,
        "pdf_result_offset": 0,
        "pdf_result_sha256": None,
    })
    assert result["status"] == "geometry_only"
    assert result["page_number"] == 2 and result["worker_isolated"] is True
    assert result["band_data"] is None


def test_upload_worker_timeout_is_fail_closed(monkeypatch):
    import subprocess
    from mcp_server import server

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr('mcp_server.resource_limits.run_bounded', timeout)
    result = server._run_upload_worker("extract_document", {
        "payload_base64": "eA==", "kind": "image", "max_pages": 1, "page_start": 1})
    assert result["status"] == "unavailable"
    assert result["error_code"] == "UPLOAD_WORKER_TIMEOUT"
    assert result["confidence"] is None and result["ood_flag"] is None
