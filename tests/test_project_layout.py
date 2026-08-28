from pathlib import Path
import hashlib
import inspect
import json
import re


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_runtime_root_uses_classified_data_and_artifact_directories():
    assert (ROOT / "data" / "raw" / "aflow").is_dir()
    assert (ROOT / "data" / "processed" / "aflow" / "ood_tensors").is_dir()
    assert (ROOT / "artifacts" / "models").is_dir()
    assert (ROOT / "artifacts" / "checkpoints").is_dir()
    assert (ROOT / "artifacts" / "reports").is_dir()
    assert (ROOT / "artifacts" / "logs").is_dir()


def test_runtime_root_contains_no_legacy_artifact_roots_or_references():
    forbidden = [
        "data_cache",
        "models",
        "checkpoints",
        "reports",
        "logs",
        "资料",
        ".sync_backup_20260813",
    ]
    assert [name for name in forbidden if (ROOT / name).exists()] == []


def test_runtime_code_contains_no_deprecated_root_paths():
    deprecated = [
        re.compile(r"data_cache"),
        re.compile(r"(?<!artifacts/)models/"),
        re.compile(r"(?<!artifacts/)reports/"),
        re.compile(r"(?<!artifacts/)checkpoints/"),
        re.compile(r"(?<!artifacts/)logs/"),
        re.compile(r"(?<!artifacts/)runs/"),
        re.compile(r'ROOT\s*/\s*"(?:models|reports|checkpoints|logs|data_cache)"'),
    ]
    violations = []
    for directory in (ROOT / "src", ROOT / "scripts"):
        for path in directory.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in deprecated:
                if pattern.search(text):
                    violations.append(f"{path.relative_to(ROOT)}: {pattern.pattern}")
    assert violations == []


def test_brain_invoker_defaults_resolve_to_latest_formal_artifacts():
    from src.vision.brain_invoker import PhysicsBrainInvoker

    expected = {
        "encoder_path": "artifacts/models/aflow_noleak_v5_30k_seed42/ssl_mbm_pretrained.keras",
        "weights_path": "artifacts/models/aflow_noleak_v5_30k_seed42/finetuned.weights.h5",
        "norm_path": "artifacts/models/aflow_noleak_v5_30k_seed42/ssl_mbm_norm_stats.json",
        "config_path": "artifacts/models/aflow_noleak_v5_30k_seed42/finetuned_config.json",
    }
    parameters = inspect.signature(PhysicsBrainInvoker.__init__).parameters
    for name, relative_path in expected.items():
        assert parameters[name].default == relative_path
        assert (ROOT / relative_path).is_file()


def test_formal_metrics_outputs_resolve_after_migration():
    summary_path = ROOT / "artifacts" / "reports" / "aflow_noleak_v4_seed42" / "metrics_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paths = []
    for value in summary["outputs"].values():
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, dict):
            paths.extend(item for item in value.values() if isinstance(item, str))
    assert paths
    assert all(path.startswith("artifacts/") for path in paths)
    assert [path for path in paths if not (ROOT / path).is_file()] == []


def test_finetuned_config_resolves_encoder_and_norm_after_migration():
    config_path = ROOT / "artifacts" / "models" / "aflow_noleak_v4_seed42" / "finetuned_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("encoder_path", "norm_path"):
        assert config[key].startswith("artifacts/")
        assert (ROOT / config[key]).is_file()


def test_tensor_manifest_records_relocation_without_rewriting_original_provenance():
    manifest_path = ROOT / "data" / "processed" / "aflow" / "ood_tensors" / "ood_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_hash = manifest["provenance"]["source_h5_sha256"]
    relocation = manifest["relocation_audit"]
    current_h5 = ROOT / relocation["current_source_h5_path"]
    current_hash = _sha256(current_h5)
    assert relocation["current_source_h5_sha256"] == current_hash
    assert relocation["original_source_h5_sha256"] == original_hash
    assert relocation["byte_identity_matches_original"] is (current_hash == original_hash)
    assert relocation["byte_identity_matches_original"] is True
    assert all(path.startswith("data/processed/aflow/ood_tensors/") for path in manifest["outputs"].values())
    assert [path for path in manifest["outputs"].values() if not (ROOT / path).is_file()] == []


def test_aflow_30000_snapshot_is_separate_from_the_immutable_v4_baseline():
    import h5py

    baseline_h5 = ROOT / "data/raw/aflow/aflow_bands.h5"
    snapshot_dir = ROOT / "data/raw/aflow/snapshots/aflow_30000_20260825"
    snapshot_h5 = snapshot_dir / "aflow_bands.h5"
    snapshot_manifest_path = snapshot_dir / "raw_snapshot_manifest.json"

    assert baseline_h5.is_file()
    assert snapshot_h5.is_file()
    assert snapshot_manifest_path.is_file()
    assert baseline_h5.resolve() != snapshot_h5.resolve()

    with h5py.File(baseline_h5, "r") as handle:
        baseline_groups = [key for key in handle if not key.startswith("__tmp__")]
        baseline_temporary = [key for key in handle if key.startswith("__tmp__")]
    with h5py.File(snapshot_h5, "r") as handle:
        snapshot_groups = [key for key in handle if not key.startswith("__tmp__")]
        snapshot_temporary = [key for key in handle if key.startswith("__tmp__")]

    manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    assert len(baseline_groups) == 6_443
    assert baseline_temporary == []
    assert len(snapshot_groups) == 30_000
    assert snapshot_temporary == []
    assert manifest["immutable"] is True
    assert manifest["h5_group_count"] == 30_000
    assert manifest["restored_v4_baseline"]["h5_group_count"] == 6_443


def test_latest_model_smoke_targets_v5_portable_paths():
    text = (ROOT / "tests" / "smoke_latest_model.py").read_text(encoding="utf-8")
    assert "aflow_noleak_v5_30k_seed42" in text
    assert "ood_tensors_v5_30000_seed42" in text
    assert "cfg['norm_path']" not in text
    assert "cfg['encoder_path']" not in text


def test_gui_tsne_path_targets_latest_formal_experiment():
    text = (ROOT / "scripts" / "gui_workbench.py").read_text(encoding="utf-8")
    expected = '"aflow_noleak_v5_30k_seed42" / "latent_tsne_spacegroups.png"'
    assert expected in text
