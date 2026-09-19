"""Create a new, allowlisted source snapshot; never publish or modify Git history."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = [
    "src/utils/__init__.py",
    "src/utils/physics_validator.py",
    "src/utils/csv_safety.py",
    "src/vision/__init__.py",
    "src/vision/multi_format_parser.py",
    "src/vision/physics_reconstructor.py",
    "src/vision/panel_routing.py",
    "src/vision/axis_ocr_review.py",
    "src/vision/figure_candidate_preannotator.py",
    "src/vision/calibration_contract.py",
]
SCRIPTS = [
    "demo_mcp_client.py",
    "generate_mcp_client_configs.py",
    "run_mcp_tests.py",
    "verify_mcp_attachments.py",
    "verify_six_papers.py",
    "prepare_mcp_release.py",
    "export_oa_visual_review.py",
    "mcp_human_review_workbench.py",
    "render_paper_figures.py",
    "verify_http.py",
    "start.ps1",
    "start.sh",
]


def files(root=ROOT):
    selected = {
        root / name
        for name in [
            "README.md",
            "README.en.md",
            "CONTRIBUTING.md",
            "Dockerfile",
            "compose.yaml",
            ".dockerignore",
            ".env.example",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "pyproject.toml",
            ".gitignore",
            "MANIFEST.in",
        ]
        + RUNTIME
    }
    for pattern in [
        "mcp_server/*.py",
        "mcp_server/*.md",
        "mcp_server/config.json",
        "mcp_server/requirements.txt",
        "mcp_server/resources/*.json",
        "mcp_server/resources/*.txt",
        "mcp_server/examples/*.json",
        "mcp_server/examples/*.png",
        "docs/*.md",
        "docs/figures/*.svg",
        "docs/figures/*.png",
        "docs/figures/*.json",
        "docs/figures/*.md",
        "docker/*.txt",
        "research/historical_training/**/*.py",
        "research/historical_training/**/*.json",
        "research/historical_training/**/*.md",
        ".github/workflows/*.yml",
        "tests/test_mcp*.py",
        "tests/test_figure_candidate*.py",
        "tests/test_visual_batch_review.py",
    ]:
        selected.update(root.glob(pattern))
    selected.update(root / "scripts" / name for name in SCRIPTS)
    for path in sorted(selected):
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"Invalid release source: {path.name}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Unexpected large source: {path.name}")
        yield path


def inspect_payload(name, payload):
    """Heuristic preflight, not a complete secret or Git-history audit; never prints values."""
    if name.endswith(".png"):
        return []
    text = payload.decode("utf-8-sig")
    issues = []
    patterns = {
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "service_token": r"\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9_-]{40,})",
        "private_user_path": r"(?i)[A-Z]:[/\\]Users[/\\](?!<|example|test|fake|user(?:[/\\]|$))[^/\\\s]+",
        "private_network_address": r"\b10\.201\.\d+\.\d+\b",
    }
    for line_number, line in enumerate(text.splitlines(), 1):
        for kind, pattern in patterns.items():
            if re.search(pattern, line):
                issues.append({"file": name, "line": line_number, "kind": kind})
    return issues


def prepare(output, root=ROOT):
    output = Path(output).absolute()
    if output.exists() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Use a new output directory outside the working repository")
    if any(p.is_symlink() for p in [output, *output.parents]):
        raise ValueError("Symlink output refused")
    payloads = {p.relative_to(root).as_posix(): p.read_bytes() for p in files(root)}
    findings = [
        finding for name, payload in payloads.items() for finding in inspect_payload(name, payload)
    ]
    if findings:
        raise ValueError(json.dumps({"preflight_findings": findings}))
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    inventory = []
    for name, payload in payloads.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        inventory.append({"path": name, "bytes": len(payload), "sha256": digest})
    with zipfile.ZipFile(output / "source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            item = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, payload)
    with zipfile.ZipFile(output / "source.zip") as archive:
        assert archive.testzip() is None
        for item in inventory:
            assert hashlib.sha256(archive.read(item["path"])).hexdigest() == item["sha256"]
    manifest = {
        "status": "source_snapshot_preflight_passed",
        "file_count": len(inventory),
        "files": inventory,
        "source_zip_sha256": hashlib.sha256((output / "source.zip").read_bytes()).hexdigest(),
        "git_history_included": False,
        "published": False,
        "scientific_acceptance": False,
        "secret_scan_scope": "allowlisted source heuristic only; not a full history audit",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {k: v for k, v in manifest.items() if k != "files"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().output_dir)))
