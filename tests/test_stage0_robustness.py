import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.parse
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np

from src.data.aflow_adapter import AFLOWAdapter
from src.data.band_store import (
    merge_metadata_json,
    save_band_record,
    synchronize_metadata_with_hdf5,
)
from src.data.batch_download import RobustBandDownloader
from src.vision.physics_reconstructor import PhysicsReconstructor
from scripts.run_full_pipeline import pipeline_layout, required_artifacts


class _FakeAdapter:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.calls = []

    def fetch_metadata(self, query_params=None):
        return [
            {
                "material_id": f"aflow-{index}",
                "spacegroup_number": index + 1,
                "band_gap": 1.0,
            }
            for index in range(5)
        ]

    def fetch_band_structure(self, material_id, metadata=None):
        self.calls.append(material_id)
        return {
            "material_id": material_id,
            "energies": np.ones((2, 4), dtype=np.float32),
            "metadata": metadata or {},
        }

    def save_to_hdf5(self, record):
        path = Path(self.cache_dir) / "aflow_bands.h5"
        with h5py.File(path, "a") as handle:
            handle.require_group(record["material_id"])

    def save_metadata_to_json(self, metadata):
        (Path(self.cache_dir) / "aflow_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

    def close(self):
        return None


def _minimal_meta(material_id, spacegroup, *, is_metal=False, is_direct=False):
    return {
        "material_id": material_id,
        "spacegroup_number": spacegroup,
        "is_metal": is_metal,
        "is_direct": is_direct,
    }


@contextmanager
def _held_exclusive_lock(lock_path):
    script = (
        "import sys; "
        "from src.data.band_store import exclusive_file_lock; "
        "\nwith exclusive_file_lock(sys.argv[1]):"
        "\n print('READY', flush=True)"
        "\n sys.stdin.readline()"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lock_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    ready = process.stdout.readline().strip()
    if ready != "READY":
        _stdout, stderr = process.communicate(timeout=5)
        raise AssertionError(f"lock holder failed before READY: {stderr}")
    try:
        yield
    finally:
        try:
            process.communicate(input="\n", timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


class Stage0RobustnessTest(unittest.TestCase):
    def test_full_pipeline_keeps_provider_artifacts_isolated(self):
        mp = pipeline_layout("mp")
        aflow = pipeline_layout("aflow")
        self.assertEqual(mp["raw_h5"], "data/raw/materials_project/mp_bands.h5")
        self.assertEqual(mp["ood_dir"], "data/processed/materials_project/ood_tensors")
        self.assertEqual(aflow["raw_h5"], "data/raw/aflow/aflow_bands.h5")
        self.assertEqual(aflow["ood_dir"], "data/processed/aflow/ood_tensors")
        self.assertEqual(aflow["model_dir"], "artifacts/models/aflow_noleak_v4_seed42")
        self.assertEqual(aflow["finetune_report_dir"], "artifacts/reports/aflow_noleak_v4_seed42")
        self.assertNotEqual(mp["ssl_encoder"], aflow["ssl_encoder"])
        self.assertNotEqual(mp["finetune_report_dir"], aflow["finetune_report_dir"])
        self.assertEqual(
            required_artifacts("aflow")["metrics"],
            "artifacts/reports/aflow_noleak_v4_seed42/metrics_summary.json",
        )

    def test_full_pipeline_accepts_explicit_versioned_experiment_layout(self):
        from scripts.run_full_pipeline import pipeline_layout, required_artifacts

        experiment_id = "aflow_noleak_v6_30k_seed42_metricfix"
        raw_h5 = "data/raw/aflow/snapshots/aflow_30000_20260825/aflow_bands.h5"
        ood_dir = "data/processed/aflow/ood_tensors_v5_30000_seed42"
        paths = pipeline_layout(
            "aflow",
            experiment_id=experiment_id,
            raw_h5=raw_h5,
            ood_dir=ood_dir,
        )
        artifacts = required_artifacts(
            "aflow",
            experiment_id=experiment_id,
            raw_h5=raw_h5,
            ood_dir=ood_dir,
        )

        self.assertEqual(paths["raw_h5"], raw_h5)
        self.assertEqual(
            paths["metadata"],
            "data/raw/aflow/snapshots/aflow_30000_20260825/aflow_metadata.json",
        )
        self.assertEqual(paths["ood_dir"], ood_dir)
        self.assertEqual(
            paths["model_dir"],
            f"artifacts/models/{experiment_id}",
        )
        self.assertIn(experiment_id, paths["vision_synthetic_dir"])
        self.assertEqual(
            artifacts["ood_split"],
            f"{ood_dir}/band_tensors_ood_split.npz",
        )
        self.assertTrue(
            artifacts["raw_snapshot_manifest"].endswith(
                "aflow_30000_20260825/raw_snapshot_manifest.json"
            )
        )
        self.assertEqual(
            artifacts["tensor_snapshot_audit"],
            f"{ood_dir}/tensor_snapshot_audit.json",
        )
        self.assertTrue(artifacts["supervised_best"].endswith("supervised/best.weights.h5"))
        self.assertTrue(artifacts["supervised_last"].endswith("supervised/last.weights.h5"))
        self.assertTrue(artifacts["inner_selection_manifest"].endswith("inner_selection_manifest.json"))
        self.assertTrue(artifacts["ssl_best_index"].endswith("ssl/ckpt-best.index"))
        self.assertTrue(artifacts["ssl_last_index"].endswith("ssl/ckpt-last.index"))
        self.assertTrue(artifacts["ssl_history"].endswith("ssl/ssl_history.json"))
        status = {
            name: {"path": path, "exists": True, "bytes": 1}
            for name, path in artifacts.items()
        }
        status["supervised_last"]["exists"] = False
        with self.assertRaisesRegex(RuntimeError, "supervised_last"):
            from scripts.run_full_pipeline import validate_pipeline_artifact_gate

            validate_pipeline_artifact_gate(
                "aflow",
                status,
                skip_vision=True,
                experiment_id=experiment_id,
                raw_h5=raw_h5,
                ood_dir=ood_dir,
            )
        self.assertNotIn("aflow_noleak_v4_seed42", json.dumps(paths))
        for unsafe_experiment in (".", "..", ".hidden", "../escape"):
            with self.assertRaisesRegex(ValueError, "Unsafe experiment"):
                pipeline_layout("aflow", experiment_id=unsafe_experiment)
        with self.assertRaisesRegex(ValueError, "Unsupported source"):
            pipeline_layout("untrusted-provider", experiment_id=experiment_id)
        with self.assertRaisesRegex(ValueError, "raw HDF5"):
            pipeline_layout(
                "aflow",
                experiment_id=experiment_id,
                raw_h5=f"artifacts/models/{experiment_id}/input.h5",
            )
        with self.assertRaisesRegex(ValueError, "OOD directory"):
            pipeline_layout(
                "aflow",
                experiment_id=experiment_id,
                ood_dir=f"artifacts/reports/{experiment_id}",
            )

    def test_fresh_cleanup_preserves_explicit_immutable_ood_input(self):
        from scripts import run_full_pipeline

        paths = run_full_pipeline.pipeline_layout(
            "aflow",
            experiment_id="aflow_noleak_v6_30k_seed42_metricfix",
            raw_h5="data/raw/aflow/snapshots/aflow_30000_20260825/aflow_bands.h5",
            ood_dir="data/processed/aflow/ood_tensors_v5_30000_seed42",
        )
        removed = []
        with patch.object(
            run_full_pipeline,
            "remove_path",
            side_effect=lambda path: removed.append(path.as_posix()),
        ):
            run_full_pipeline.clean_for_fresh_run(
                "aflow",
                clean_raw=True,
                paths=paths,
                preserve_ood_input=True,
                preserve_raw_input=True,
            )

        self.assertNotIn(paths["ood_dir"], removed)
        self.assertNotIn(paths["raw_h5"], removed)

    def test_explicit_immutable_ood_input_is_never_rebuilt(self):
        from scripts.run_full_pipeline import should_build_ood_tensors

        self.assertFalse(
            should_build_ood_tensors(
                fresh=True,
                ood_split_exists=True,
                explicit_ood_dir=True,
            )
        )
        with self.assertRaisesRegex(FileNotFoundError, "immutable OOD"):
            should_build_ood_tensors(
                fresh=False,
                ood_split_exists=False,
                explicit_ood_dir=True,
            )

    def test_explicit_raw_snapshot_never_runs_downloader(self):
        from scripts.run_full_pipeline import should_run_download

        self.assertFalse(
            should_run_download(
                force_download=False,
                raw_count=30_000,
                target=30_000,
                explicit_raw_h5=True,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "immutable raw"):
            should_run_download(
                force_download=True,
                raw_count=30_000,
                target=30_000,
                explicit_raw_h5=True,
            )
        with self.assertRaisesRegex(RuntimeError, "immutable raw"):
            should_run_download(
                force_download=False,
                raw_count=29_999,
                target=30_000,
                explicit_raw_h5=True,
            )

    def test_full_pipeline_cli_accepts_versioned_layout_arguments(self):
        from scripts import run_full_pipeline

        argv = [
            "run_full_pipeline.py",
            "--source",
            "aflow",
            "--experiment-id",
            "aflow_noleak_v6_30k_seed42_metricfix",
            "--raw-h5",
            "data/raw/snapshot.h5",
            "--ood-dir",
            "data/processed/snapshot",
            "--report-date",
            "20260827",
        ]
        with patch.object(sys, "argv", argv):
            args = run_full_pipeline.parse_args()

        self.assertEqual(
            args.experiment_id,
            "aflow_noleak_v6_30k_seed42_metricfix",
        )
        self.assertEqual(args.raw_h5, "data/raw/snapshot.h5")
        self.assertEqual(args.ood_dir, "data/processed/snapshot")
        self.assertEqual(args.report_date, "20260827")

    def test_artifact_status_requires_nonempty_regular_file(self):
        from scripts.run_full_pipeline import artifact_file_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root / "empty.bin"
            directory = root / "directory"
            filled = root / "filled.bin"
            empty.write_bytes(b"")
            directory.mkdir()
            filled.write_bytes(b"ok")

            self.assertFalse(artifact_file_status(empty)["exists"])
            self.assertFalse(artifact_file_status(directory)["exists"])
            self.assertTrue(artifact_file_status(filled)["exists"])
            self.assertEqual(artifact_file_status(filled)["bytes"], 2)

    def test_ssl_history_schema_rejects_bad_mask_and_accepts_disabled_consistency(self):
        from scripts.run_full_pipeline import validate_ssl_history_schema

        history = {
            "selection_monitor": "val_total",
            "epochs": [
                {
                    "epoch": 1,
                    "val": {"total": 1.0, "mask_fraction": 0.14},
                    "selection_weights": {"curvature": 0.5, "symmetry": 0.0},
                    "post_adaptation_weights": {"curvature": 0.5, "symmetry": 0.0},
                    "improved": True,
                }
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "mask fraction"):
            validate_ssl_history_schema(history, expected_consistency_weight=0.0)

        history["epochs"][0]["val"]["mask_fraction"] = 0.25
        result = validate_ssl_history_schema(
            history,
            expected_consistency_weight=0.0,
        )
        self.assertEqual(result["last_epoch"], 1)
        self.assertEqual(result["best_epoch"], 1)

    def test_pipeline_content_gate_uses_lightweight_selection_validator(self):
        import types

        from scripts import run_full_pipeline
        from src.utils import selection_manifest

        shadow = types.ModuleType("scripts")
        self.assertNotIn("finetune_supervised", run_full_pipeline.__dict__)
        with patch.dict(sys.modules, {"scripts": shadow}):
            with patch.object(
                selection_manifest,
                "validate_inner_selection_manifest",
                return_value={"monitor": "val_loss"},
            ) as validator:
                from scripts.run_full_pipeline import (
                    validate_versioned_artifact_content,
                )

                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    (root / "logs/ssl").mkdir(parents=True)
                    (root / "logs/ssl/ssl_history.json").write_text(
                        json.dumps(
                            {
                                "selection_monitor": "val_total",
                                "epochs": [
                                    {
                                        "epoch": 1,
                                        "val": {"total": 1.0, "mask_fraction": 0.25},
                                        "selection_weights": {
                                            "curvature": 0.5,
                                            "symmetry": 0.0,
                                        },
                                        "post_adaptation_weights": {
                                            "curvature": 0.5,
                                            "symmetry": 0.0,
                                        },
                                        "improved": True,
                                    }
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    with patch.object(
                        run_full_pipeline,
                        "rel",
                        side_effect=lambda path: root / path,
                    ), patch.object(
                        run_full_pipeline,
                        "read_checkpoint_epoch",
                        side_effect=[1, 1],
                    ):
                        validate_versioned_artifact_content(
                            types.SimpleNamespace(consistency_weight=0.0),
                            {
                                "finetune_report_dir": "reports",
                                "ssl_log_dir": "logs/ssl",
                                "ssl_checkpoint_dir": "checkpoints/ssl",
                            },
                        )

        validator.assert_called_once_with(str(root / "reports"))

    def test_versioned_artifact_content_gate_validates_selection_and_ssl_epochs(self):
        from types import SimpleNamespace

        from scripts import run_full_pipeline
        from src.utils import selection_manifest

        history = {
            "selection_monitor": "val_total",
            "epochs": [
                {
                    "epoch": 1,
                    "val": {"total": 1.0, "mask_fraction": 0.25},
                    "selection_weights": {"curvature": 0.5, "symmetry": 0.0},
                    "post_adaptation_weights": {"curvature": 0.5, "symmetry": 0.0},
                    "improved": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs/ssl").mkdir(parents=True)
            (root / "logs/ssl/ssl_history.json").write_text(
                json.dumps(history),
                encoding="utf-8",
            )
            paths = {
                "finetune_report_dir": "reports",
                "ssl_log_dir": "logs/ssl",
                "ssl_checkpoint_dir": "checkpoints/ssl",
            }
            args = SimpleNamespace(consistency_weight=0.0)
            with patch.object(
                run_full_pipeline,
                "rel",
                side_effect=lambda path: root / path,
            ), patch.object(
                selection_manifest,
                "validate_inner_selection_manifest",
                return_value={"monitor": "val_loss"},
            ) as selection_gate, patch.object(
                run_full_pipeline,
                "read_checkpoint_epoch",
                side_effect=[1, 1],
            ):
                result = run_full_pipeline.validate_versioned_artifact_content(
                    args,
                    paths,
                )

        selection_gate.assert_called_once_with(str(root / "reports"))
        self.assertEqual(result["ssl_history"]["best_epoch"], 1)
        self.assertEqual(result["ssl_history"]["last_epoch"], 1)

    def test_full_pipeline_status_only_does_not_write_manifest(self):
        from types import SimpleNamespace
        from scripts import run_full_pipeline

        args = SimpleNamespace(
            mask_ratio=0.25,
            source="aflow",
            experiment_id=None,
            raw_h5=None,
            ood_dir=None,
            fresh=False,
            clean_raw=False,
            status_only=True,
        )
        with patch.object(
            run_full_pipeline,
            "parse_args",
            return_value=args,
        ), patch.object(
            run_full_pipeline,
            "artifact_status",
            return_value={},
        ), patch.object(
            run_full_pipeline,
            "write_model_brain_manifest",
            side_effect=AssertionError("status-only must not write"),
        ):
            run_full_pipeline.main()

    def test_full_pipeline_blocks_tensor_stage_on_false_download_report(self):
        from types import SimpleNamespace
        from scripts import run_full_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stages = []

            def fake_run_command(_command, stage):
                stages.append(stage)
                if stage.startswith("Download"):
                    raw_dir = root / "data" / "raw" / "aflow"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    with h5py.File(raw_dir / "aflow_bands.h5", "w") as handle:
                        handle.create_group("aflow-a")
                        handle.create_group("aflow-b")
                    (raw_dir / "aflow_download_report.json").write_text(
                        json.dumps(
                            {
                                "requested_target": 2,
                                "total_cached": 2,
                                "target_reached": False,
                                "termination_reason": "candidate_pool_exhausted",
                            }
                        ),
                        encoding="utf-8",
                    )
                    return
                raise AssertionError(f"downstream stage must not run: {stage}")

            args = SimpleNamespace(
                mask_ratio=0.25,
                source="aflow",
                fresh=False,
                clean_raw=False,
                status_only=False,
                force_download=True,
                target=2,
                min_gap=0.0,
                max_gap=5.0,
                max_sites=50,
                download_workers=1,
                target_k=128,
                train_size=0.8,
                random_state=42,
            )
            with patch.object(run_full_pipeline, "ROOT", root), patch.object(
                run_full_pipeline, "parse_args", return_value=args
            ), patch.object(
                run_full_pipeline, "run_command", side_effect=fake_run_command
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "target_reached=false",
                ):
                    run_full_pipeline.main()

            self.assertEqual(stages, ["Download AFLOW band structures"])

    def test_full_pipeline_fails_when_successful_stage_omits_required_artifact(self):
        from types import SimpleNamespace
        from scripts import run_full_pipeline

        args = SimpleNamespace(
            mask_ratio=0.25,
            source="aflow",
            fresh=False,
            clean_raw=False,
            status_only=False,
            force_download=False,
            target=1,
            min_gap=0.0,
            max_gap=5.0,
            max_sites=50,
            download_workers=1,
            target_k=128,
            train_size=0.8,
            random_state=42,
            force_ssl=False,
            ssl_epochs=1,
            ssl_batch_size=2,
            sign_weight=2.0,
            consistency_weight=0.2,
            d_model=16,
            num_heads=2,
            num_layers=1,
            dff=32,
            projection_dim=8,
            strain_scale=0.01,
            require_gpu=False,
            disable_strain_augmentation=False,
            force_finetune=False,
            finetune_epochs=1,
            finetune_batch_size=2,
            learning_rate=1e-3,
            encoder_learning_rate=1e-5,
            type_weight=2.0,
            freeze_layers=0,
            topology_weight=0.3,
            entropy_weight=0.02,
            extremum_weight=1.0,
            skip_vision=True,
            force_vision=False,
            vision_count=1,
            vision_epochs=1,
            vision_batch=1,
            vision_imgsz=64,
            vision_val_fraction=0.2,
            vision_device="cpu",
            vision_base_model="unused.pt",
            vision_min_box_map50=0.0,
            vision_min_pose_map50=0.0,
        )
        status = {
            name: {"path": path, "exists": name != "ood_split", "bytes": 1}
            for name, path in required_artifacts("aflow").items()
        }
        status["vision_detector"]["exists"] = False
        status["vision_detector_summary"]["exists"] = False

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            run_full_pipeline,
            "parse_args",
            return_value=args,
        ), patch.object(
            run_full_pipeline,
            "rel",
            side_effect=lambda path: Path(tmp) / path,
        ), patch.object(
            run_full_pipeline,
            "count_h5_samples",
            return_value=1,
        ), patch.object(
            run_full_pipeline,
            "validate_download_gate",
            return_value=None,
        ), patch.object(
            run_full_pipeline,
            "run_command",
            return_value=None,
        ), patch.object(
            run_full_pipeline,
            "artifact_status",
            return_value=status,
        ), patch.object(
            run_full_pipeline,
            "write_model_brain_manifest",
            return_value=Path(tmp) / "manifest.json",
        ) as write_manifest:
            with self.assertRaisesRegex(
                RuntimeError,
                "Pipeline artifact gate failed.*ood_split",
            ):
                run_full_pipeline.main()

        write_manifest.assert_not_called()

    def test_full_pipeline_propagates_gpu_requirement_to_finetune(self):
        from types import SimpleNamespace
        from scripts import run_full_pipeline

        args = SimpleNamespace(
            finetune_epochs=60,
            finetune_batch_size=32,
            learning_rate=1e-3,
            encoder_learning_rate=1e-5,
            type_weight=2.0,
            freeze_layers=2,
            topology_weight=0.3,
            entropy_weight=0.02,
            extremum_weight=1.0,
            random_state=42,
            require_gpu=True,
            experiment_id="aflow_noleak_v6_30k_seed42_metricfix",
            source="aflow",
            report_date="20260827",
        )
        paths = {
            "ssl_encoder": "encoder.keras",
            "ssl_norm": "norm.json",
            "finetune_report_dir": "reports",
            "finetune_checkpoint_dir": "checkpoints",
            "finetuned_weights": "model.weights.h5",
        }
        command = run_full_pipeline.build_finetune_command(
            args, paths, "split.npz"
        )
        self.assertIn("--require-gpu", command)
        for flag, expected in (
            ("--experiment-id", args.experiment_id),
            ("--source", args.source),
            ("--report-date", args.report_date),
        ):
            index = command.index(flag)
            self.assertEqual(command[index + 1], expected)

        train_command = run_full_pipeline.build_finetune_command(
            args,
            paths,
            "split.npz",
            mode="train",
        )
        evaluation_command = run_full_pipeline.build_finetune_command(
            args,
            paths,
            "split.npz",
            mode="evaluate",
        )
        self.assertIn("--train-only", train_command)
        self.assertNotIn("--evaluation-only", train_command)
        self.assertIn("--evaluation-only", evaluation_command)
        self.assertNotIn("--train-only", evaluation_command)

    def test_full_pipeline_runs_train_only_before_outer_evaluation(self):
        from types import SimpleNamespace
        from scripts import run_full_pipeline

        args = SimpleNamespace(
            finetune_epochs=2,
            finetune_batch_size=4,
            learning_rate=1e-3,
            encoder_learning_rate=1e-5,
            type_weight=2.0,
            freeze_layers=2,
            topology_weight=0.3,
            entropy_weight=0.02,
            extremum_weight=1.0,
            random_state=42,
            require_gpu=True,
            experiment_id="aflow_noleak_v6_30k_seed42_metricfix",
            source="aflow",
            report_date="20260827",
        )
        paths = {
            "ssl_encoder": "encoder.keras",
            "ssl_norm": "norm.json",
            "finetune_report_dir": "reports",
            "finetune_checkpoint_dir": "checkpoints",
            "finetuned_weights": "model.weights.h5",
        }
        calls = []

        with patch.object(
            run_full_pipeline,
            "run_command",
            side_effect=lambda command, stage: calls.append((command, stage)),
        ):
            run_full_pipeline.run_supervised_stages(
                args,
                paths,
                "split.npz",
            )

        self.assertEqual(len(calls), 2)
        self.assertIn("--train-only", calls[0][0])
        self.assertNotIn("--evaluation-only", calls[0][0])
        self.assertIn("--evaluation-only", calls[1][0])
        self.assertNotIn("--train-only", calls[1][0])

    def test_full_pipeline_ssl_command_records_safe_physics_defaults(self):
        from types import SimpleNamespace
        from scripts import run_full_pipeline

        args = SimpleNamespace(
            ssl_epochs=60,
            ssl_batch_size=32,
            mask_ratio=0.25,
            sign_weight=2.0,
            consistency_weight=0.0,
            d_model=128,
            num_heads=4,
            num_layers=4,
            dff=256,
            projection_dim=64,
            strain_scale=0.01,
            random_state=42,
            require_gpu=True,
            disable_strain_augmentation=True,
        )
        paths = {
            "ssl_checkpoint_dir": "checkpoints/ssl",
            "ssl_log_dir": "logs/ssl",
            "model_dir": "models/v6",
        }
        command = run_full_pipeline.build_ssl_command(args, paths, "split.npz")

        consistency_index = command.index("--consistency-weight")
        self.assertEqual(command[consistency_index + 1], "0.0")
        seed_index = command.index("--random-state")
        self.assertEqual(command[seed_index + 1], "42")
        self.assertIn("--disable-strain-augmentation", command)
        self.assertIn("--require-gpu", command)

    def test_full_pipeline_defaults_disable_invalid_ssl_auxiliaries(self):
        from scripts import run_full_pipeline

        with patch.object(sys, "argv", ["run_full_pipeline.py"]):
            args = run_full_pipeline.parse_args()

        self.assertEqual(args.consistency_weight, 0.0)
        self.assertTrue(args.disable_strain_augmentation)

    def test_aflow_30000_uses_a_60000_candidate_metadata_pool(self):
        from src.data.batch_download import metadata_candidate_limit

        self.assertEqual(metadata_candidate_limit(30_000), 60_000)

    def test_latent_features_are_extracted_in_bounded_batches(self):
        import tensorflow as tf
        from types import SimpleNamespace
        from scripts.finetune_supervised import extract_encoder_features_batched

        class RecordingEncoder:
            def __init__(self):
                self.batch_sizes = []

            def __call__(self, batch, return_features=False, training=False):
                self.batch_sizes.append(int(batch.shape[0]))
                self.assertions = (return_features, training)
                return tf.reduce_mean(batch, axis=1)

        encoder = RecordingEncoder()
        model = SimpleNamespace(encoder=encoder)
        values = np.arange(10 * 4 * 2, dtype=np.float32).reshape(10, 4, 2)

        features = extract_encoder_features_batched(model, values, batch_size=3)

        self.assertEqual(encoder.batch_sizes, [3, 3, 3, 1])
        self.assertEqual(encoder.assertions, (True, False))
        np.testing.assert_allclose(features, values.mean(axis=1))

    def test_evaluation_resume_restores_best_weights(self):
        from scripts.finetune_supervised import restore_model_for_evaluation

        class RecordingModel:
            def __init__(self):
                self.loaded = None

            def load_weights(self, path):
                self.loaded = path

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.weights.h5"
            checkpoint.write_bytes(b"checkpoint")
            history_csv = Path(tmp) / "training_log.csv"
            history_csv.write_text(
                "epoch,loss,val_gap_mae\n0,1.0,0.2\n1,0.8,0.1\n",
                encoding="utf-8",
            )
            model = RecordingModel()

            history = restore_model_for_evaluation(
                model, str(checkpoint), str(history_csv)
            )

            self.assertEqual(model.loaded, str(checkpoint))
            self.assertEqual(history.epoch, [0, 1])
            self.assertEqual(history.history["loss"], [1.0, 0.8])
            self.assertEqual(history.history["val_gap_mae"], [0.2, 0.1])

    def test_evaluation_restore_is_bound_to_manifest_best_path(self):
        from scripts.finetune_supervised import restore_model_for_evaluation

        class RecordingModel:
            def __init__(self):
                self.loaded = None

            def load_weights(self, path):
                self.loaded = path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen_best = root / "frozen" / "best.weights.h5"
            alternate_best = root / "alternate" / "best.weights.h5"
            frozen_best.parent.mkdir()
            alternate_best.parent.mkdir()
            frozen_best.write_bytes(b"frozen")
            alternate_best.write_bytes(b"alternate")
            history_csv = root / "training_log.csv"
            history_csv.write_text(
                "epoch,loss,val_loss\n0,1.0,1.0\n",
                encoding="utf-8",
            )
            manifest = {
                "states": {
                    "best": {"path": str(frozen_best)},
                }
            }
            model = RecordingModel()

            with self.assertRaisesRegex(RuntimeError, "frozen best"):
                restore_model_for_evaluation(
                    model,
                    str(alternate_best),
                    str(history_csv),
                    selection_manifest=manifest,
                )

        self.assertIsNone(model.loaded)

    def test_evaluation_only_does_not_compile_a_fresh_optimizer(self):
        from scripts.finetune_supervised import should_compile_supervised_model

        self.assertFalse(should_compile_supervised_model(evaluation_only=True))
        self.assertTrue(should_compile_supervised_model(evaluation_only=False))

    def test_evaluation_only_skips_fit(self):
        from scripts.finetune_supervised import fit_or_restore_history

        class NoFitModel:
            def __init__(self):
                self.loaded = None
                self.compile_kwargs = None

            def fit(self, *args, **kwargs):
                raise AssertionError("fit must not run during evaluation-only resume")

            def load_weights(self, path):
                self.loaded = path

            def compile(self, **kwargs):
                if self.loaded is None:
                    raise AssertionError("evaluation compile must happen after best restore")
                self.compile_kwargs = kwargs

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "best.weights.h5"
            checkpoint.write_bytes(b"checkpoint")
            history_csv = Path(tmp) / "training_log.csv"
            history_csv.write_text(
                "epoch,loss,val_gap_mae\n0,1.0,0.2\n",
                encoding="utf-8",
            )
            model = NoFitModel()

            history = fit_or_restore_history(
                model=model,
                evaluation_only=True,
                checkpoint_path=str(checkpoint),
                history_csv_path=str(history_csv),
                train_ds=object(),
                val_ds=object(),
                epochs=60,
                callbacks=[],
            )

            self.assertEqual(model.loaded, str(checkpoint))
            self.assertEqual(history.history["loss"], [1.0])
            self.assertIsNotNone(model.compile_kwargs)
            self.assertFalse(model.compile_kwargs["jit_compile"])

    def test_finetune_dataset_fit_explicitly_disables_keras_shuffle(self):
        from scripts.finetune_supervised import fit_or_restore_history

        class RecordingModel:
            def fit(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                return "history"

        model = RecordingModel()
        result = fit_or_restore_history(
            model=model,
            evaluation_only=False,
            checkpoint_path="unused",
            history_csv_path="unused",
            train_ds=object(),
            val_ds=object(),
            epochs=2,
            callbacks=[],
        )

        self.assertEqual(result, "history")
        self.assertFalse(model.kwargs["shuffle"])

    def test_finetune_cli_exposes_evaluation_only(self):
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "scripts" / "finetune_supervised.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--evaluation-only", result.stdout)
        self.assertIn("--train-only", result.stdout)
        self.assertIn("--experiment-id", result.stdout)
        self.assertIn("--source", result.stdout)

    def test_supervised_whole_path_augmentation_is_disabled_by_default(self):
        from scripts.finetune_supervised import (
            DEFAULT_SUPERVISED_AUGMENTATION,
            SUPERVISED_EXECUTION_MODE_REQUIRED,
        )

        self.assertFalse(DEFAULT_SUPERVISED_AUGMENTATION)
        self.assertTrue(SUPERVISED_EXECUTION_MODE_REQUIRED)

    def test_supervised_metrics_aggregate_across_unequal_validation_batches(self):
        import tensorflow as tf

        from scripts.finetune_supervised import (
            SupervisedBandGapModel,
            compile_model,
        )

        class IdentitySequence(tf.keras.layers.Layer):
            def call(self, x, training=False):
                del training
                return x

        class TinySSLEncoder(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.encoder = IdentitySequence()
                self.pooling = tf.keras.layers.GlobalAveragePooling1D()

        tf.keras.utils.set_random_seed(123)
        model = SupervisedBandGapModel(
            TinySSLEncoder(),
            feature_mean=np.zeros(6, dtype=np.float32),
            feature_std=np.ones(6, dtype=np.float32),
            d_model=8,
            dropout=0.0,
        )
        x = np.zeros((4, 8, 6), dtype=np.float32)
        labels = {
            "gap": np.asarray([[0.0], [0.0], [0.0], [8.0]], dtype=np.float32),
            "type": tf.one_hot([0, 0, 0, 0], depth=3).numpy().astype(np.float32),
        }
        model._forward(tf.constant(x), training=False)
        compile_model(
            model,
            learning_rate=0.0,
            type_weight=1.0,
            class_weights={0: 1.0, 1: 1.0, 2: 1.0},
            encoder_learning_rate=0.0,
            topology_weight=0.0,
            entropy_weight=0.0,
            extremum_weight=0.0,
        )

        full_batch = model.test_step((tf.constant(x), labels))
        expected = float(full_batch["gap_mae"].numpy())
        model.reset_metrics()
        split_batches = tf.data.Dataset.from_tensor_slices((x, labels)).batch(3)
        measured = float(
            model.evaluate(split_batches, verbose=0, return_dict=True)["gap_mae"]
        )

        self.assertAlmostEqual(expected, 2.0, places=6)
        self.assertAlmostEqual(measured, expected, places=6)

    def test_supervised_callbacks_use_aggregate_val_loss_and_keep_best_last(self):
        from types import SimpleNamespace

        import tensorflow as tf

        from scripts.finetune_supervised import build_supervised_callbacks

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                learning_rate=1e-3,
                warmup_epochs=3,
                epochs=20,
                min_lr=1e-7,
                checkpoint_dir=str(Path(tmp) / "checkpoints"),
                output_dir=str(Path(tmp) / "reports"),
            )
            callbacks = build_supervised_callbacks(args)

        checkpoints = [
            callback
            for callback in callbacks
            if isinstance(callback, tf.keras.callbacks.ModelCheckpoint)
        ]
        early_stopping = next(
            callback
            for callback in callbacks
            if isinstance(callback, tf.keras.callbacks.EarlyStopping)
        )
        best = next(callback for callback in checkpoints if callback.save_best_only)
        last = next(callback for callback in checkpoints if not callback.save_best_only)

        self.assertEqual(best.monitor, "val_loss")
        self.assertEqual(early_stopping.monitor, "val_loss")
        self.assertFalse(early_stopping.restore_best_weights)
        self.assertTrue(str(best.filepath).endswith("best.weights.h5"))
        self.assertTrue(str(last.filepath).endswith("last.weights.h5"))
        self.assertNotEqual(str(best.filepath), str(last.filepath))

    def test_freeze_supervised_states_preserves_best_last_and_accepted(self):
        from types import SimpleNamespace

        from scripts.finetune_supervised import freeze_supervised_states

        class RecordingModel:
            def __init__(self):
                self.loaded = None

            def load_weights(self, path):
                self.loaded = str(path)

            def save_weights(self, path):
                Path(path).write_bytes(b"accepted-restored-best")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            output_dir = root / "reports"
            checkpoint_dir.mkdir()
            output_dir.mkdir()
            best = checkpoint_dir / "best.weights.h5"
            last = checkpoint_dir / "last.weights.h5"
            accepted = root / "models" / "accepted.weights.h5"
            accepted.parent.mkdir()
            best.write_bytes(b"best-state")
            last.write_bytes(b"last-state")
            history = SimpleNamespace(
                epoch=[0, 1],
                history={"val_loss": [2.0, 1.0]},
            )
            model = RecordingModel()

            manifest = freeze_supervised_states(
                model=model,
                checkpoint_dir=str(checkpoint_dir),
                accepted_model_path=str(accepted),
                output_dir=str(output_dir),
                history=history,
            )

            persisted = json.loads(
                (output_dir / "inner_selection_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            accepted_exists = accepted.exists()

        self.assertEqual(model.loaded, str(best))
        self.assertTrue(accepted_exists)
        self.assertEqual(manifest, persisted)
        self.assertEqual(manifest["monitor"], "val_loss")
        self.assertEqual(manifest["best_epoch_one_based"], 2)
        self.assertNotEqual(
            manifest["states"]["best"]["sha256"],
            manifest["states"]["last"]["sha256"],
        )
        self.assertEqual(
            manifest["states"]["accepted"]["source"],
            "restored_best",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            output_dir = root / "reports"
            checkpoint_dir.mkdir()
            output_dir.mkdir()
            (checkpoint_dir / "best.weights.h5").write_bytes(b"best")
            last = checkpoint_dir / "last.weights.h5"
            last.write_bytes(b"last")
            with self.assertRaisesRegex(ValueError, "distinct"):
                freeze_supervised_states(
                    model=RecordingModel(),
                    checkpoint_dir=str(checkpoint_dir),
                    accepted_model_path=str(last),
                    output_dir=str(output_dir),
                    history=SimpleNamespace(
                        epoch=[0],
                        history={"val_loss": [1.0]},
                    ),
                )

    def test_frozen_selection_manifest_rejects_tampered_checkpoint(self):
        from types import SimpleNamespace

        from scripts.finetune_supervised import (
            freeze_supervised_states,
            validate_inner_selection_manifest,
        )

        class FileModel:
            def load_weights(self, path):
                self.loaded = path

            def save_weights(self, path):
                Path(path).write_bytes(b"accepted")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_dir = root / "checkpoints"
            output_dir = root / "reports"
            checkpoint_dir.mkdir()
            output_dir.mkdir()
            (checkpoint_dir / "best.weights.h5").write_bytes(b"best")
            (checkpoint_dir / "last.weights.h5").write_bytes(b"last")
            accepted = root / "accepted.weights.h5"
            freeze_supervised_states(
                model=FileModel(),
                checkpoint_dir=str(checkpoint_dir),
                accepted_model_path=str(accepted),
                output_dir=str(output_dir),
                history=SimpleNamespace(epoch=[0], history={"val_loss": [1.0]}),
            )
            validate_inner_selection_manifest(str(output_dir))
            (checkpoint_dir / "best.weights.h5").write_bytes(b"tampered")

            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_inner_selection_manifest(str(output_dir))

    def test_evaluation_config_write_does_not_rewrite_accepted_weights(self):
        from scripts.finetune_supervised import save_supervised_model_artifacts

        class NoWeightSaveModel:
            def save_weights(self, path):
                raise AssertionError("evaluation must not rewrite accepted weights")

        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp) / "accepted.weights.h5"
            weights.write_bytes(b"frozen-accepted")
            before = weights.read_bytes()

            outputs = save_supervised_model_artifacts(
                NoWeightSaveModel(),
                str(weights),
                {"experiment_id": "v6-test"},
                save_weights=False,
            )

            after = weights.read_bytes()
            config = json.loads(Path(outputs["config"]).read_text(encoding="utf-8"))

        self.assertEqual(before, after)
        self.assertEqual(config["experiment_id"], "v6-test")

    def test_markdown_report_uses_runtime_provenance_and_honest_gap_semantics(self):
        from scripts.finetune_supervised import save_markdown_report

        metrics = {
            "line_mode_gap_mae": 0.1,
            "line_mode_gap_rmse": 0.2,
            "dft_gap_residual_mae": 0.6,
            "model_vs_global_dft_mae": 0.61,
            "r2": 0.9,
            "type_acc": 0.8,
            "macro_f1": 0.7,
            "direct_gap_recall": 0.6,
        }
        summary = {
            "experiment_id": "aflow_noleak_v6_30k_seed42_metricfix",
            "report_date": "20260827",
            "source": "aflow",
            "tensor_npz": "data/processed/aflow/ood_tensors_v5_30000_seed42/band_tensors_ood_split.npz",
            "train_metrics": metrics,
            "ood_test_metrics": metrics,
            "class_weights": {0: 1.0, 1: 1.0, 2: 1.0},
            "physics_violation_rates": {
                "negative_predicted_gap_rate": 0.0,
                "vbm_positive_curvature_rate": 0.1,
                "cbm_negative_curvature_rate": 0.1,
                "gap_identity_large_residual_rate": 0.0,
                "physics_score": 0.9,
                "gap_identity_mae": 0.1,
            },
            "roc_auc": {"direct": 0.8, "indirect": 0.7},
            "gap_semantics": {
                "analytic_identity_baseline_mae_ev": 0.0,
                "analytic_identity_baseline_rmse_ev": 0.0,
                "interpretation": "learned soft-extremum approximation, not independent DFT prediction",
            },
            "mc_uncertainty": {
                "raw_95_interval_coverage": 0.90,
                "tolerance_augmented_coverage": 1.0,
                "tolerance_ev": 0.5,
            },
            "classification_scope": {
                "spacegroup_macro_accuracy": 0.75,
                "feature_label_match_count": 80,
                "feature_label_mismatch_count": 20,
                "feature_label_match_accuracy": 0.95,
                "feature_label_mismatch_accuracy": 0.40,
            },
            "outputs": {
                "metrics": "artifacts/reports/aflow_noleak_v6_30k_seed42_metricfix/metrics_summary.json",
                "predictions": "artifacts/reports/aflow_noleak_v6_30k_seed42_metricfix/ood_test_predictions.json",
                "model": "artifacts/models/aflow_noleak_v6_30k_seed42_metricfix/finetuned.weights.h5",
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.md"
            save_markdown_report(summary, str(report))
            text = report.read_text(encoding="utf-8")

        self.assertIn(summary["experiment_id"], text)
        self.assertIn("20260827", text)
        self.assertIn(summary["tensor_npz"], text)
        self.assertIn("Analytic identity baseline MAE: 0.000000 eV", text)
        self.assertIn("Raw 95% interval coverage: 90.00%", text)
        self.assertIn("Tolerance-augmented coverage: 100.00%", text)
        self.assertIn("Spacegroup-macro accuracy: 75.00%", text)
        self.assertIn("Feature/label mismatch accuracy: 40.00%", text)
        self.assertNotIn("2026-06-04", text)
        self.assertNotIn("artifacts/reports/mp", text)

    def test_report_bundle_writes_dated_and_latest_alias(self):
        from scripts.finetune_supervised import save_report_bundle

        metrics = {
            "line_mode_gap_mae": 0.1,
            "line_mode_gap_rmse": 0.2,
            "dft_gap_residual_mae": 0.6,
            "model_vs_global_dft_mae": 0.61,
            "r2": 0.9,
            "type_acc": 0.8,
            "macro_f1": 0.7,
            "direct_gap_recall": 0.6,
        }
        summary = {
            "experiment_id": "v6",
            "report_date": "20260827",
            "source": "aflow",
            "tensor_npz": "split.npz",
            "train_metrics": metrics,
            "ood_test_metrics": metrics,
            "class_weights": {},
            "physics_violation_rates": {
                "negative_predicted_gap_rate": 0.0,
                "vbm_positive_curvature_rate": 0.0,
                "cbm_negative_curvature_rate": 0.0,
                "gap_identity_large_residual_rate": 0.0,
                "physics_score": 1.0,
                "gap_identity_mae": 0.0,
            },
            "roc_auc": {},
            "outputs": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_report_bundle(summary, tmp, "20260827")
            dated = Path(paths["dated"])
            latest = Path(paths["latest"])
            same_bytes = dated.read_bytes() == latest.read_bytes()

        self.assertTrue(same_bytes)
        self.assertTrue(str(dated).endswith("finetune_supervised_report_20260827.md"))
        self.assertTrue(str(latest).endswith("latest_training_report.md"))

    def test_parity_plot_labels_line_mode_function_not_dft_prediction(self):
        from scripts import finetune_supervised

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            finetune_supervised.plt,
            "xlabel",
        ) as xlabel, patch.object(
            finetune_supervised.plt,
            "ylabel",
        ) as ylabel:
            finetune_supervised.save_parity_plot(
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([0.1, 0.9], dtype=np.float32),
                str(Path(tmp) / "parity.png"),
                "test",
            )

        xlabel.assert_called_with("Analytic line-mode tensor gap (eV)")
        ylabel.assert_called_with("Learned soft-extremum gap (eV)")

    def test_error_histogram_labels_line_mode_residual_not_dft(self):
        from scripts import finetune_supervised

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            finetune_supervised.plt,
            "xlabel",
        ) as xlabel:
            finetune_supervised.save_error_histogram(
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([0.1, 0.9], dtype=np.float32),
                str(Path(tmp) / "errors.png"),
                "test",
            )

        xlabel.assert_called_with(
            "Residual: learned soft-extremum - analytic line-mode gap (eV)"
        )

    def test_mc_calibration_separates_raw_and_tolerance_coverage(self):
        from src.utils.mc_dropout import mc_calibration_check

        y_true = np.asarray([0.0, 10.0], dtype=np.float32)
        mc_result = {
            "gap_samples": np.zeros((2, 2), dtype=np.float32),
            "gap_mean": np.asarray([0.0, 0.0], dtype=np.float32),
            "gap_std": np.asarray([0.1, 0.1], dtype=np.float32),
            "gap_ci95_low": np.asarray([-0.2, -0.2], dtype=np.float32),
            "gap_ci95_high": np.asarray([0.2, 0.2], dtype=np.float32),
        }

        calibration = mc_calibration_check(
            y_true,
            mc_result,
            tolerance_ev=10.0,
        )

        self.assertEqual(calibration["raw_95_interval_coverage"], 0.5)
        self.assertEqual(calibration["tolerance_augmented_coverage"], 1.0)
        self.assertEqual(calibration["tolerance_ev"], 10.0)

    def test_mc_dropout_type_prediction_uses_bounded_batches(self):
        from src.utils.mc_dropout import mc_dropout_predict_with_type

        class RecordingModel:
            def __init__(self):
                self.batch_sizes = []

            def __call__(self, batch, training=False):
                self.batch_sizes.append(len(batch))
                n = len(batch)
                return {
                    "gap": np.zeros((n, 1), dtype=np.float32),
                    "type": np.tile(
                        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                        (n, 1),
                    ),
                }

        model = RecordingModel()
        result = mc_dropout_predict_with_type(
            model,
            np.zeros((7, 4, 6), dtype=np.float32),
            n_samples=2,
            batch_size=3,
            seed=42,
        )

        self.assertEqual(model.batch_sizes, [3, 3, 1, 3, 3, 1])
        self.assertEqual(result["gap_samples"].shape, (2, 7))

    def test_classification_scope_metrics_include_group_macro_and_mismatch_strata(self):
        from scripts.finetune_supervised import classification_scope_metrics

        type_true = np.asarray([0, 0, 1, 1], dtype=np.int32)
        type_pred = np.asarray(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.9, 0.0],
                [0.1, 0.9, 0.0],
                [0.9, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        groups = np.asarray([1, 1, 2, 2], dtype=np.int32)
        material_ids = np.asarray(["a", "b", "c", "d"])
        metrics = classification_scope_metrics(
            type_true,
            type_pred,
            groups,
            material_ids,
            mismatch_material_ids={"b", "d"},
        )

        self.assertEqual(metrics["sample_accuracy"], 0.5)
        self.assertEqual(metrics["spacegroup_macro_accuracy"], 0.5)
        self.assertEqual(metrics["feature_label_match_count"], 2)
        self.assertEqual(metrics["feature_label_mismatch_count"], 2)
        self.assertEqual(metrics["feature_label_match_accuracy"], 1.0)
        self.assertEqual(metrics["feature_label_mismatch_accuracy"], 0.0)

    def test_load_mismatch_ids_uses_split_manifest_audit(self):
        from scripts.finetune_supervised import load_mismatch_material_ids

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz_path = root / "band_tensors_ood_split.npz"
            npz_path.write_bytes(b"placeholder")
            (root / "ood_split_manifest.json").write_text(
                json.dumps(
                    {
                        "metal_feature_audit": {
                            "mismatch_material_ids": ["a", "c", "a"]
                        }
                    }
                ),
                encoding="utf-8",
            )

            ids = load_mismatch_material_ids(str(npz_path))

        self.assertEqual(ids, {"a", "c"})

    def test_finetune_loader_preserves_segment_ids(self):
        from scripts.finetune_supervised import load_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_path = root / "split.npz"
            norm_path = root / "norm.json"
            x_train = np.zeros((2, 2, 4, 3), dtype=np.float32)
            x_test = np.zeros((1, 2, 4, 3), dtype=np.float32)
            train_segments = np.asarray(
                [[0, 0, 1, 1], [2, 2, 3, 3]],
                dtype=np.int32,
            )
            test_segments = np.asarray([[4, 4, 5, 5]], dtype=np.int32)
            np.savez(
                split_path,
                X_train=x_train,
                X_test=x_test,
                y_train=np.zeros(2, dtype=np.float32),
                y_test=np.zeros(1, dtype=np.float32),
                y_type_train=np.zeros(2, dtype=np.int32),
                y_type_test=np.zeros(1, dtype=np.int32),
                groups_train=np.asarray([1, 2], dtype=np.int32),
                groups_test=np.asarray([3], dtype=np.int32),
                material_ids_train=np.asarray(["a", "b"]),
                material_ids_test=np.asarray(["c"]),
                segment_ids_train=train_segments,
                segment_ids_test=test_segments,
            )
            norm_path.write_text(
                json.dumps({"mean": [0.0] * 6, "std": [1.0] * 6}),
                encoding="utf-8",
            )

            loaded = load_dataset(str(split_path), str(norm_path))
            train_only = load_dataset(
                str(split_path),
                str(norm_path),
                include_outer_test=False,
            )

        np.testing.assert_array_equal(loaded["segment_ids_train"], train_segments)
        np.testing.assert_array_equal(loaded["segment_ids_test"], test_segments)
        for outer_key in (
            "X_test_raw",
            "X_test",
            "y_test",
            "type_test",
            "tensor_gap_test",
            "groups_test",
            "segment_ids_test",
            "material_ids_test",
        ):
            self.assertNotIn(outer_key, train_only)

    def test_downloader_default_cache_is_source_classified(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                mp = RobustBandDownloader(source="mp", adapter=_FakeAdapter(tmp))
                aflow = RobustBandDownloader(source="aflow", adapter=_FakeAdapter(tmp))
                self.assertEqual(mp.cache_dir, "./data/raw/materials_project")
                self.assertEqual(aflow.cache_dir, "./data/raw/aflow")
            finally:
                os.chdir(previous)

    def test_mp_summary_maps_current_nsites_field(self):
        from src.data.mp_adapter import MPAdapter

        class FakeSummary:
            def search(self, **kwargs):
                self.kwargs = kwargs
                return [
                    {
                        "material_id": "mp-test",
                        "formula_pretty": "Si",
                        "symmetry": {"number": 227},
                        "band_gap": 1.0,
                        "is_gap_direct": False,
                        "is_metal": False,
                        "efermi": 0.0,
                        "nsites": 2,
                        "theoretical": True,
                    }
                ]

        adapter = object.__new__(MPAdapter)
        adapter.new_rester = object()
        adapter.legacy_rester = None
        summary = FakeSummary()
        adapter._get_summary_rester = lambda: summary
        records = adapter.fetch_metadata({"material_ids": ["mp-test"]})
        self.assertIn("nsites", summary.kwargs["fields"])
        self.assertNotIn("num_sites", summary.kwargs["fields"])
        self.assertEqual(records[0]["num_sites"], 2)

    def test_mp_transient_failure_is_retried(self):
        from src.data.mp_adapter import MPAdapter

        adapter = object.__new__(MPAdapter)
        adapter.request_retries = 3
        adapter.retry_base_delay = 0.0
        attempts = []

        def flaky_operation():
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("temporary failure in name resolution")
            return "ok"

        self.assertEqual(
            adapter._call_with_retry(flaky_operation, "test operation"),
            "ok",
        )
        self.assertEqual(len(attempts), 3)

    def test_mp_batch_fetch_preserves_requested_order(self):
        from src.data.mp_adapter import MPAdapter

        adapter = object.__new__(MPAdapter)
        adapter.new_rester = object()
        adapter.json_cache_dir = "unused"
        adapter.rate_limit_delay = 0.0
        adapter.request_retries = 1
        adapter.retry_base_delay = 0.0
        adapter._load_from_json_cache = lambda material_id: (
            {"material_id": material_id, "energies": []}
            if material_id == "mp-cached"
            else None
        )
        adapter._fetch_band_structures_new_api_batch = lambda ids: {
            material_id: {"material_id": material_id, "energies": []}
            for material_id in reversed(ids)
        }
        with patch.object(adapter, "fetch_band_structure") as fallback:
            rows = adapter.fetch_band_structures(
                [
                    {"material_id": "mp-cached"},
                    {"material_id": "mp-a"},
                    {"material_id": "mp-b"},
                ]
            )
        self.assertEqual(
            [row["material_id"] for row in rows],
            ["mp-cached", "mp-a", "mp-b"],
        )
        fallback.assert_not_called()

    def test_aflow_payload_maps_real_k_distance_and_transposes_bands(self):
        raw = {
            "Efermi": 0.25,
            "bands_data": [[0.0, -1.0, 1.0], [0.4, -0.8, 1.2], [1.0, -0.5, 1.5]],
            "kpoint_labels": ["G", "X"],
            "kpoint_positions": [0.0, 1.0],
        }
        parsed = AFLOWAdapter._parse_band_payload("aflow-test", raw)
        self.assertEqual(parsed["energies"].shape, (2, 3))
        np.testing.assert_allclose(parsed["k_distances"], [0.0, 0.4, 1.0])
        self.assertEqual(parsed["kpath_labels"][1]["index"], 2)
        self.assertEqual(parsed["efermi"], 0.0)
        self.assertEqual(parsed["source_efermi_absolute"], 0.25)
        self.assertEqual(parsed["energy_reference"], "fermi_shifted_zero")

    def test_aflow_payload_rejects_declared_but_empty_band_file(self):
        with self.assertRaisesRegex(ValueError, "band data unavailable"):
            AFLOWAdapter._parse_band_payload(
                "aflow-empty",
                {"Efermi": 0.0, "bands_data": None},
            )

    def test_aflow_metadata_uses_true_closed_gap_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = AFLOWAdapter(cache_dir=tmp)
            requested_urls = []

            def fake_request(url):
                requested_urls.append(url)
                return b"{}"

            adapter._request_bytes = fake_request
            self.assertEqual(
                adapter.fetch_metadata(
                    {"band_gap": (0.2, 4.0), "num_sites": (2, 50), "limit": 1}
                ),
                [],
            )
            query = urllib.parse.unquote(requested_urls[0].split("?", 1)[1])
            self.assertIn("Egap(0.2*,*4)", query)
            self.assertIn("natoms(2*,*50)", query)

    def test_aflow_metadata_interleaves_gap_strata(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = AFLOWAdapter(cache_dir=tmp)

            def fake_request(url):
                query = urllib.parse.unquote(url.split("?", 1)[1])
                bounds = query.split("Egap(", 1)[1].split(")", 1)[0]
                lower = float(bounds.split(",", 1)[0].rstrip("*"))
                rows = {
                    str(index): {
                        "auid": f"aflow:{lower:.2f}-{index}",
                        "aurl": f"AFLOWDATA/test/{lower:.2f}/{index}",
                        "files": [f"sample-{index}_bandsdata.json.xz"],
                        "Egap": lower + 0.001 * index,
                        "Egap_type": "insulator_direct",
                        "natoms": 4,
                    }
                    for index in range(2)
                }
                return json.dumps(rows).encode("utf-8")

            adapter._request_bytes = fake_request
            records = adapter.fetch_metadata(
                {"band_gap": (0.0, 3.0), "limit": 6, "gap_bins": 3}
            )
            np.testing.assert_allclose(
                [record["band_gap"] for record in records[:3]],
                [0.0, 1.0, 2.0],
            )

    def test_aflow_metadata_cursor_preserves_unconsumed_page_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = AFLOWAdapter(cache_dir=tmp)
            raw_rows = [
                {
                    "auid": f"aflow:{index}",
                    "aurl": f"AFLOWDATA/test/{index}",
                    "files": [f"sample-{index}_bandsdata.json.xz"],
                    "Egap": 1.0,
                    "Egap_type": "insulator_direct",
                    "natoms": 2,
                }
                for index in range(12)
            ]

            def fake_request(url):
                query = urllib.parse.unquote(url.split("?", 1)[1])
                paging = query.split("paging(", 1)[1].split(")", 1)[0]
                page, size = (int(value) for value in paging.split(","))
                start = (page - 1) * size
                rows = raw_rows[start : start + size]
                return json.dumps(
                    {str(offset): row for offset, row in enumerate(rows)}
                ).encode("utf-8")

            adapter._request_bytes = fake_request
            query = {
                "band_gap": (0.0, 2.0),
                "limit": 4,
                "page_size": 3,
                "gap_bins": 1,
            }
            first = adapter.fetch_metadata(query)
            cursor = adapter.last_metadata_cursor
            second = adapter.fetch_metadata(
                {**query, "_aflow_cursor": cursor}
            )

            self.assertEqual(
                [record["material_id"] for record in first],
                ["aflow-0", "aflow-1", "aflow-2", "aflow-3"],
            )
            self.assertEqual(
                [record["material_id"] for record in second],
                ["aflow-4", "aflow-5", "aflow-6", "aflow-7"],
            )

    def test_downloader_query_fingerprint_separates_aflow_page_sizes(self):
        base = {
            "band_gap": (0.0, 2.0),
            "num_sites": (1, 20),
            "gap_bins": 2,
        }
        first = RobustBandDownloader._query_fingerprint(
            {**base, "page_size": 3}
        )
        second = RobustBandDownloader._query_fingerprint(
            {**base, "page_size": 4}
        )
        self.assertNotEqual(first, second)

    def test_downloader_candidate_catalog_never_exceeds_hard_limit(self):
        class LimitEchoAdapter(_FakeAdapter):
            limits_seen = []

            def fetch_metadata(self, query_params=None):
                limit = int((query_params or {}).get("limit", 0))
                self.limits_seen.append(limit)
                return [
                    _minimal_meta(f"aflow-hard-limit-{index}", index + 1)
                    for index in range(limit)
                ]

        with tempfile.TemporaryDirectory() as tmp:
            LimitEchoAdapter.limits_seen = []
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=LimitEchoAdapter(tmp),
            )
            query = {
                "limit": 1,
                "catalog_batch_size": 1,
                "gap_bins": 5,
                "page_size": 1,
            }

            downloader._fetch_next_catalog_page(query)

            catalog = json.loads(
                (Path(tmp) / "aflow_candidate_catalog.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(LimitEchoAdapter.limits_seen, [1])
            self.assertEqual(len(catalog), 1)

    def test_downloader_target_means_total_cache_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp) / "aflow_bands.h5", "w") as handle:
                handle.create_group("aflow-existing")
            fake = _FakeAdapter(tmp)
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=fake,
            )
            downloader.query_and_download(target_count=3)
            self.assertEqual(len(fake.calls), 2)
            self.assertEqual(len(downloader.downloaded_ids), 3)

    def test_downloader_consumes_local_candidate_catalog_before_api(self):
        class NoMetadataApiAdapter(_FakeAdapter):
            def fetch_metadata(self, query_params=None):
                raise AssertionError("metadata API must not be called")

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "aflow_candidate_catalog.json").write_text(
                json.dumps(
                    [
                        _minimal_meta("aflow-local-a", 1),
                        _minimal_meta("aflow-local-b", 2),
                    ]
                ),
                encoding="utf-8",
            )
            fake = NoMetadataApiAdapter(tmp)
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=fake,
            )

            downloader.query_and_download(target_count=2)

            self.assertEqual(fake.calls, ["aflow-local-a", "aflow-local-b"])
            self.assertEqual(len(downloader.downloaded_ids), 2)

    def test_downloader_adaptively_pages_and_resumes_persisted_cursor(self):
        class PagedAdapter(_FakeAdapter):
            pages_seen = []

            def fetch_metadata(self, query_params=None):
                page = int((query_params or {}).get("page", 1))
                self.pages_seen.append(page)
                material_id = {
                    1: "aflow-page-no-data",
                    2: "aflow-page-a",
                    3: "aflow-page-b",
                    4: "aflow-page-c",
                }.get(page)
                if material_id is None:
                    return []
                return [_minimal_meta(material_id, page)]

            def fetch_band_structure(self, material_id, metadata=None):
                if material_id == "aflow-page-no-data":
                    self.calls.append(material_id)
                    return {
                        "material_id": material_id,
                        "error": "band data unavailable: empty bands_data",
                    }
                return super().fetch_band_structure(
                    material_id,
                    metadata=metadata,
                )

        query = {
            "band_gap": (0.0, 5.0),
            "num_sites": (1, 50),
            "limit": 4,
            "page_size": 1,
            "gap_bins": 1,
            "catalog_batch_size": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            PagedAdapter.pages_seen = []
            first = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=PagedAdapter(tmp),
            )
            first.query_and_download(target_count=2, query_params=query)
            self.assertEqual(PagedAdapter.pages_seen, [1, 2, 3])

            second = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=PagedAdapter(tmp),
            )
            second.query_and_download(target_count=3, query_params=query)

            self.assertEqual(PagedAdapter.pages_seen, [1, 2, 3, 4])
            self.assertEqual(len(second.downloaded_ids), 3)
            cursor = json.loads(
                (Path(tmp) / "aflow_candidate_cursor.json").read_text(
                    encoding="utf-8"
                )
            )
            state = next(iter(cursor["queries"].values()))
            self.assertEqual(state["next_page"], 5)
            self.assertFalse(state["exhausted"])

    def test_downloader_continues_after_a_duplicate_metadata_page(self):
        class DuplicateBoundaryAdapter(_FakeAdapter):
            pages_seen = []

            def fetch_metadata(self, query_params=None):
                page = int((query_params or {}).get("page", 1))
                self.pages_seen.append(page)
                if page == 1:
                    return [_minimal_meta("aflow-duplicate", 1)]
                if page == 2:
                    return [_minimal_meta("aflow-fresh", 2)]
                return []

        query = {
            "limit": 3,
            "page_size": 1,
            "gap_bins": 1,
            "catalog_batch_size": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "aflow_candidate_catalog.json").write_text(
                json.dumps([_minimal_meta("aflow-duplicate", 1)]),
                encoding="utf-8",
            )
            (Path(tmp) / "aflow_excluded.json").write_text(
                json.dumps({"aflow-duplicate": "no usable band data"}),
                encoding="utf-8",
            )
            DuplicateBoundaryAdapter.pages_seen = []
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=DuplicateBoundaryAdapter(tmp),
            )

            downloader.query_and_download(target_count=1, query_params=query)

            self.assertEqual(DuplicateBoundaryAdapter.pages_seen, [1, 2])
            self.assertIn("aflow-fresh", downloader.downloaded_ids)

    def test_downloader_persists_and_replays_aflow_per_bin_cursor(self):
        class CursorAwareAdapter(_FakeAdapter):
            def __init__(self, cache_dir):
                super().__init__(cache_dir)
                self.metadata_calls = 0
                self.last_metadata_cursor = None

            def fetch_metadata(self, query_params=None):
                incoming = (query_params or {}).get("_aflow_cursor")
                self.metadata_calls += 1
                if self.metadata_calls == 1:
                    self.assert_cursor = incoming
                    self.last_metadata_cursor = {
                        "version": 1,
                        "next_pages": [2, 5],
                        "exhausted": [True, False],
                    }
                    return [_minimal_meta("aflow-cursor-no-data", 1)]
                if incoming != self.last_metadata_cursor:
                    raise AssertionError("per-bin AFLOW cursor was not replayed")
                self.last_metadata_cursor = {
                    "version": 1,
                    "next_pages": [2, 6],
                    "exhausted": [True, False],
                }
                return [_minimal_meta("aflow-cursor-success", 2)]

            def fetch_band_structure(self, material_id, metadata=None):
                if material_id == "aflow-cursor-no-data":
                    return {
                        "material_id": material_id,
                        "error": "band data unavailable: empty bands_data",
                    }
                return super().fetch_band_structure(
                    material_id,
                    metadata=metadata,
                )

        with tempfile.TemporaryDirectory() as tmp:
            adapter = CursorAwareAdapter(tmp)
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=adapter,
            )
            downloader.query_and_download(
                target_count=1,
                query_params={
                    "limit": 2,
                    "page_size": 1,
                    "gap_bins": 2,
                    "catalog_batch_size": 1,
                },
            )

            self.assertIsNone(adapter.assert_cursor)
            self.assertEqual(adapter.metadata_calls, 2)
            cursor = json.loads(
                (Path(tmp) / "aflow_candidate_cursor.json").read_text(
                    encoding="utf-8"
                )
            )
            state = next(iter(cursor["queries"].values()))
            self.assertEqual(state["adapter_cursor"]["next_pages"], [2, 6])

    def test_downloader_separates_candidate_catalog_from_canonical_metadata(self):
        class MixedAdapter(_FakeAdapter):
            def fetch_metadata(self, query_params=None):
                return [
                    _minimal_meta("aflow-success-a", 1),
                    _minimal_meta("aflow-no-data", 2),
                    _minimal_meta("aflow-success-b", 3),
                ]

            def fetch_band_structure(self, material_id, metadata=None):
                self.calls.append(material_id)
                if material_id == "aflow-no-data":
                    return {
                        "material_id": material_id,
                        "error": "band data unavailable: empty bands_data",
                    }
                return super().fetch_band_structure(
                    material_id,
                    metadata=metadata,
                )

        with tempfile.TemporaryDirectory() as tmp:
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=MixedAdapter(tmp),
            )
            downloader.query_and_download(target_count=2)

            catalog = json.loads(
                (Path(tmp) / "aflow_candidate_catalog.json").read_text(
                    encoding="utf-8"
                )
            )
            canonical = json.loads(
                (Path(tmp) / "aflow_metadata.json").read_text(encoding="utf-8")
            )
            with h5py.File(Path(tmp) / "aflow_bands.h5", "r") as handle:
                h5_ids = set(handle.keys())
            self.assertEqual(
                {item["material_id"] for item in catalog},
                {"aflow-success-a", "aflow-no-data", "aflow-success-b"},
            )
            self.assertEqual(
                {item["material_id"] for item in canonical},
                h5_ids,
            )
            self.assertNotIn("aflow-no-data", h5_ids)

    def test_downloader_reconciles_canonical_metadata_ids_with_hdf5(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            with h5py.File(h5_path, "w") as handle:
                for material_id, spacegroup in (("aflow-a", 1), ("aflow-b", 2)):
                    group = handle.create_group(material_id)
                    metadata = group.create_group("metadata")
                    metadata.attrs["spacegroup_number"] = spacegroup
            metadata_path = Path(tmp) / "aflow_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    [
                        {"material_id": "aflow-a", "formula_pretty": "Si"},
                        {"material_id": "aflow-stale", "formula_pretty": "X"},
                    ]
                ),
                encoding="utf-8",
            )
            fake = _FakeAdapter(tmp)
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=fake,
            )

            downloader.query_and_download(target_count=2)

            canonical = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {item["material_id"] for item in canonical},
                {"aflow-a", "aflow-b"},
            )
            by_id = {item["material_id"]: item for item in canonical}
            self.assertEqual(by_id["aflow-a"]["formula_pretty"], "Si")
            self.assertEqual(by_id["aflow-b"]["spacegroup_number"], 2)
            self.assertEqual(fake.calls, [])

    def test_metadata_reconciliation_respects_hdf5_writer_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            metadata_path = Path(tmp) / "aflow_metadata.json"
            with h5py.File(h5_path, "w") as handle:
                group = handle.create_group("aflow-existing")
                group.create_group("metadata")
            with _held_exclusive_lock(str(h5_path) + ".lock"):
                with self.assertRaisesRegex(RuntimeError, "writer lock"):
                    synchronize_metadata_with_hdf5(
                        str(h5_path),
                        str(metadata_path),
                    )

    def test_metadata_reconciliation_clears_stale_ids_when_hdf5_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            metadata_path = Path(tmp) / "aflow_metadata.json"
            metadata_path.write_text(
                json.dumps([{"material_id": "aflow-stale"}]),
                encoding="utf-8",
            )

            result = synchronize_metadata_with_hdf5(
                str(h5_path),
                str(metadata_path),
            )

            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                [],
            )
            self.assertEqual(result["h5_ids"], 0)
            self.assertEqual(result["metadata_ids"], 0)
            self.assertEqual(result["removed"], 1)

    def test_metadata_reconciliation_rich_merges_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            metadata_path = Path(tmp) / "aflow_metadata.json"
            with h5py.File(h5_path, "w") as handle:
                handle.create_group("aflow-rich-duplicate").create_group("metadata")
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "material_id": "aflow-rich-duplicate",
                            "formula_pretty": "Si",
                            "spacegroup_number": 227,
                        },
                        {
                            "material_id": "aflow-rich-duplicate",
                            "formula_pretty": None,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            synchronize_metadata_with_hdf5(
                str(h5_path),
                str(metadata_path),
            )

            reconciled = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(len(reconciled), 1)
            self.assertEqual(reconciled[0]["formula_pretty"], "Si")
            self.assertEqual(reconciled[0]["spacegroup_number"], 227)

    def test_metadata_reconciliation_audits_conflicting_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            metadata_path = Path(tmp) / "aflow_metadata.json"
            with h5py.File(h5_path, "w") as handle:
                handle.create_group("aflow-conflicting-duplicate").create_group(
                    "metadata"
                )
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "material_id": "aflow-conflicting-duplicate",
                            "band_gap": 1.0,
                        },
                        {
                            "material_id": "aflow-conflicting-duplicate",
                            "band_gap": 1.2,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            synchronize_metadata_with_hdf5(
                str(h5_path),
                str(metadata_path),
            )

            reconciled = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(reconciled[0]["band_gap"], 1.0)
            variants_path = (
                Path(tmp)
                / "provenance"
                / "metadata_variants"
                / "aflow_metadata_variants.json"
            )
            variants = json.loads(variants_path.read_text(encoding="utf-8"))
            self.assertEqual(len(variants), 1)
            self.assertIn("band_gap", variants[0]["conflicts"])

    def test_metadata_conflict_checkpoint_recovers_after_canonical_write_failure(self):
        from unittest import mock

        from src.data import band_store

        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "aflow_metadata.json"
            merge_metadata_json(
                str(metadata_path),
                [
                    {
                        "material_id": "aflow-transaction",
                        "band_gap": 1.0,
                        "num_sites": None,
                    }
                ],
            )
            original_atomic_write = band_store.atomic_write_json

            def crash_on_canonical(path, payload):
                if Path(path) == metadata_path:
                    raise OSError("simulated canonical replace failure")
                return original_atomic_write(path, payload)

            incoming = {
                "material_id": "aflow-transaction",
                "band_gap": 1.2,
                "num_sites": 2,
            }
            with mock.patch.object(
                band_store,
                "atomic_write_json",
                side_effect=crash_on_canonical,
            ):
                with self.assertRaisesRegex(OSError, "canonical replace"):
                    merge_metadata_json(str(metadata_path), [incoming])

            pending_path = (
                Path(tmp)
                / "provenance"
                / "metadata_transactions"
                / "aflow_metadata_pending.json"
            )
            variants_path = (
                Path(tmp)
                / "provenance"
                / "metadata_variants"
                / "aflow_metadata_variants.json"
            )
            self.assertTrue(pending_path.exists())
            self.assertTrue(variants_path.exists())

            merge_metadata_json(str(metadata_path), [])

            canonical = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(canonical[0]["num_sites"], 2)
            self.assertEqual(canonical[0]["band_gap"], 1.0)
            variants = json.loads(variants_path.read_text(encoding="utf-8"))
            self.assertEqual(len(variants), 1)
            self.assertFalse(pending_path.exists())

    def test_metadata_conflict_audit_is_idempotent_after_field_backfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "aflow_metadata.json"
            merge_metadata_json(
                str(metadata_path),
                [
                    {
                        "material_id": "aflow-replay",
                        "band_gap": 1.0,
                        "num_sites": None,
                    }
                ],
            )
            incoming = {
                "material_id": "aflow-replay",
                "band_gap": 1.2,
                "num_sites": 2,
            }
            merge_metadata_json(str(metadata_path), [incoming])
            merge_metadata_json(str(metadata_path), [incoming])

            variants_path = (
                Path(tmp)
                / "provenance"
                / "metadata_variants"
                / "aflow_metadata_variants.json"
            )
            variants = json.loads(variants_path.read_text(encoding="utf-8"))
            self.assertEqual(len(variants), 1)
            self.assertEqual(variants[0]["conflicts"]["band_gap"]["canonical"], 1.0)

    def test_metadata_merge_preserves_rich_fields_and_audits_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata_path = Path(tmp) / "aflow_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    [
                        {
                            "material_id": "aflow-rich",
                            "formula_pretty": "Si",
                            "spacegroup_number": 227,
                            "band_gap": 1.1,
                            "num_sites": None,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            merge_metadata_json(
                str(metadata_path),
                [
                    {
                        "material_id": "aflow-rich",
                        "formula_pretty": None,
                        "spacegroup_number": None,
                        "band_gap": 1.2,
                        "num_sites": 2,
                    }
                ],
            )

            merged = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(merged[0]["formula_pretty"], "Si")
            self.assertEqual(merged[0]["spacegroup_number"], 227)
            self.assertEqual(merged[0]["band_gap"], 1.1)
            self.assertEqual(merged[0]["num_sites"], 2)
            variants_path = (
                Path(tmp)
                / "provenance"
                / "metadata_variants"
                / "aflow_metadata_variants.json"
            )
            variants = json.loads(variants_path.read_text(encoding="utf-8"))
            self.assertEqual(variants[0]["material_id"], "aflow-rich")
            self.assertIn("band_gap", variants[0]["conflicts"])

    def test_band_store_legacy_json_attrs_hash_by_semantics_not_formatting(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            record = {
                "material_id": "aflow-legacy-json",
                "source": "aflow",
                "energies": np.asarray(
                    [[-1.0, -0.5], [0.5, 1.0]], dtype=np.float32
                ),
                "kpath_labels": ["G", "X"],
                "metadata": {
                    "material_id": "aflow-legacy-json",
                    "source": "aflow",
                    "formula_pretty": "Si",
                },
            }
            with h5py.File(h5_path, "w") as handle:
                group = handle.create_group(record["material_id"])
                group.create_dataset("energies", data=record["energies"])
                group.attrs["material_id"] = record["material_id"]
                group.attrs["source"] = "aflow"
                group.attrs["kpath_labels"] = json.dumps(
                    record["kpath_labels"],
                    ensure_ascii=False,
                )
                metadata = group.create_group("metadata")
                for key, value in record["metadata"].items():
                    metadata.attrs[key] = value
                    group.attrs[key] = value

            result = save_band_record(str(h5_path), record)

            self.assertEqual(result.status, "duplicate")
            variant_path = (
                Path(tmp)
                / "provenance"
                / "h5_variants"
                / "aflow_bands_variants.h5"
            )
            self.assertFalse(variant_path.exists())

    def test_band_store_ignores_absolute_source_efermi_in_semantic_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            record = {
                "material_id": "aflow-provenance-efermi",
                "source": "aflow",
                "energies": np.asarray([[-1.0, 1.0]], dtype=np.float32),
                "efermi": 0.0,
                "energy_reference": "aflow_bands_data_efermi_zero",
                "source_efermi_absolute": 5.0,
                "metadata": {"formula_pretty": "Si"},
            }
            first = save_band_record(str(h5_path), record)
            changed_provenance = dict(record)
            changed_provenance["source_efermi_absolute"] = 6.0
            second = save_band_record(str(h5_path), changed_provenance)

            self.assertEqual(first.status, "canonical")
            self.assertEqual(second.status, "duplicate")

    def test_band_store_treats_none_and_missing_field_as_same_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            record = {
                "material_id": "aflow-none-field",
                "source": "aflow",
                "energies": np.asarray([[-1.0, 1.0]], dtype=np.float32),
                "optional_physics": None,
                "metadata": {"formula_pretty": "Si"},
            }
            first = save_band_record(str(h5_path), record)
            missing_field = dict(record)
            missing_field.pop("optional_physics")
            second = save_band_record(str(h5_path), missing_field)

            self.assertEqual(first.status, "canonical")
            self.assertEqual(second.status, "duplicate")

    def test_band_store_recomputes_canonical_hash_instead_of_trusting_stale_attr(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            record = {
                "material_id": "aflow-stale-hash",
                "source": "aflow",
                "energies": np.asarray([[-1.0, 1.0]], dtype=np.float32),
                "metadata": {"formula_pretty": "Si"},
            }
            save_band_record(str(h5_path), record)
            with h5py.File(h5_path, "r+") as handle:
                handle["aflow-stale-hash"]["energies"][...] = np.asarray(
                    [[-2.0, 2.0]], dtype=np.float32
                )

            result = save_band_record(str(h5_path), record)

            self.assertEqual(result.status, "variant")
            with h5py.File(h5_path, "r") as handle:
                np.testing.assert_array_equal(
                    handle["aflow-stale-hash"]["energies"][...],
                    np.asarray([[-2.0, 2.0]], dtype=np.float32),
                )

    def test_band_store_identical_physics_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = str(Path(tmp) / "aflow_bands.h5")
            first = {
                "material_id": "aflow-idempotent",
                "source": "aflow",
                "energies": np.asarray([[-1.0, -0.5], [0.5, 1.0]], dtype=np.float32),
                "k_distances": np.asarray([0.0, 1.0], dtype=np.float32),
                "efermi": 0.0,
                "source_sha256": "a" * 64,
                "metadata": {"formula_pretty": "Si", "spacegroup_number": 227},
            }
            duplicate = dict(first)
            duplicate["metadata"] = {
                "formula_pretty": "SHOULD_NOT_REWRITE_CANONICAL",
                "spacegroup_number": 227,
            }

            first_result = save_band_record(h5_path, first)
            duplicate_result = save_band_record(h5_path, duplicate)

            self.assertEqual(first_result.status, "canonical")
            self.assertEqual(duplicate_result.status, "duplicate")
            with h5py.File(h5_path, "r") as handle:
                self.assertEqual(
                    handle["aflow-idempotent"]["metadata"].attrs["formula_pretty"],
                    "Si",
                )

    def test_band_store_conflict_preserves_canonical_and_writes_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = str(Path(tmp) / "aflow_bands.h5")
            canonical = {
                "material_id": "aflow-conflict",
                "source": "aflow",
                "energies": np.asarray([[-1.0, -0.5], [0.5, 1.0]], dtype=np.float32),
                "k_distances": np.asarray([0.0, 1.0], dtype=np.float32),
                "efermi": 0.0,
                "metadata": {"formula_pretty": "Si", "spacegroup_number": 227},
            }
            conflict = dict(canonical)
            conflict["energies"] = np.asarray(
                [[-2.0, -1.5], [1.5, 2.0]], dtype=np.float32
            )

            save_band_record(h5_path, canonical)
            result = save_band_record(h5_path, conflict)

            self.assertEqual(result.status, "variant")
            variant_path = (
                Path(tmp)
                / "provenance"
                / "h5_variants"
                / "aflow_bands_variants.h5"
            )
            self.assertEqual(Path(result.path), variant_path)
            with h5py.File(h5_path, "r") as handle:
                np.testing.assert_allclose(
                    handle["aflow-conflict"]["energies"][...],
                    canonical["energies"],
                )
            with h5py.File(variant_path, "r") as handle:
                self.assertEqual(len(handle), 1)
                variant = next(iter(handle.values()))
                self.assertEqual(variant.attrs["canonical_material_id"], "aflow-conflict")
                np.testing.assert_allclose(variant["energies"][...], conflict["energies"])

    def test_band_store_rejects_a_second_writer_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = str(Path(tmp) / "aflow_bands.h5")
            lock_path = h5_path + ".lock"
            record = {
                "material_id": "aflow-locked",
                "source": "aflow",
                "energies": np.asarray([[-1.0], [1.0]], dtype=np.float32),
                "efermi": 0.0,
                "metadata": {"spacegroup_number": 1},
            }
            with _held_exclusive_lock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "writer lock"):
                    save_band_record(h5_path, record)

    def test_downloader_rejects_hdf5_readback_while_writer_lock_is_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "aflow_bands.h5"
            with h5py.File(h5_path, "w") as handle:
                handle.create_group("aflow-existing")
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=_FakeAdapter(tmp),
            )
            with _held_exclusive_lock(str(h5_path) + ".lock"):
                with self.assertRaisesRegex(RuntimeError, "writer lock"):
                    downloader.query_and_download(target_count=1)

    def test_downloader_rejects_a_second_cache_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = Path(tmp) / ".aflow_download.lock"
            fake = _FakeAdapter(tmp)
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=fake,
            )
            with _held_exclusive_lock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "writer lock"):
                    downloader.query_and_download(target_count=1)
            self.assertEqual(fake.calls, [])

    def test_downloader_reports_bounded_in_flight_and_main_thread_writes(self):
        import threading
        import time

        class ConcurrentAdapter(_FakeAdapter):
            def __init__(self, cache_dir):
                super().__init__(cache_dir)
                self.guard = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.save_threads = set()

            def fetch_band_structure(self, material_id, metadata=None):
                with self.guard:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    time.sleep(0.03)
                    return super().fetch_band_structure(
                        material_id,
                        metadata=metadata,
                    )
                finally:
                    with self.guard:
                        self.active -= 1

            def save_to_hdf5(self, record):
                self.save_threads.add(threading.get_ident())
                return super().save_to_hdf5(record)

        with tempfile.TemporaryDirectory() as tmp:
            adapter = ConcurrentAdapter(tmp)
            coordinator_thread = threading.get_ident()
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=adapter,
                workers=3,
            )
            downloader._get_fetch_adapter = lambda: adapter

            downloader.query_and_download(target_count=5)

            report = json.loads(
                (Path(tmp) / "aflow_download_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(adapter.max_active, 3)
            self.assertEqual(adapter.save_threads, {coordinator_thread})
            self.assertEqual(report["peak_in_flight"], 3)
            self.assertEqual(report["in_flight"], 0)
            self.assertEqual(report["submitted"], 5)
            self.assertEqual(report["completed"], 5)

    def test_downloader_round_robins_spacegroups_and_gap_types(self):
        candidates = [
            _minimal_meta("sg1-a", 1, is_direct=True),
            _minimal_meta("sg1-b", 1, is_direct=True),
            _minimal_meta("sg2-a", 2, is_metal=True),
            _minimal_meta("sg2-b", 2, is_metal=True),
            _minimal_meta("sg3-a", 3, is_direct=False),
        ]
        ordered = RobustBandDownloader._diverse_candidate_order(candidates)
        self.assertEqual(
            [item["spacegroup_number"] for item in ordered[:3]],
            [1, 2, 3],
        )

    def test_downloader_excludes_no_data_candidates_on_resume(self):
        class NoDataAdapter(_FakeAdapter):
            def fetch_band_structure(self, material_id, metadata=None):
                self.calls.append(material_id)
                if material_id == "aflow-0":
                    return {
                        "material_id": material_id,
                        "error": "band data unavailable: unexpected bands_data shape (0,)",
                    }
                return super().fetch_band_structure(material_id, metadata=metadata)

        with tempfile.TemporaryDirectory() as tmp:
            first = RobustBandDownloader(
                cache_dir=tmp, source="aflow", adapter=NoDataAdapter(tmp)
            )
            first.query_and_download(target_count=4)
            self.assertEqual(first.stats["no_data"], 1)
            self.assertEqual(len(first.excluded_ids), 1)

            second = RobustBandDownloader(
                cache_dir=tmp, source="aflow", adapter=NoDataAdapter(tmp)
            )
            second.query_and_download(target_count=4)
            self.assertEqual(second.adapter.calls, [])
            self.assertEqual(second.stats["no_data"], 0)

    def test_downloader_same_instance_resets_run_stats_for_larger_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeAdapter(tmp)
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=fake,
            )

            downloader.query_and_download(target_count=3)
            downloader.query_and_download(target_count=5)

            self.assertEqual(len(fake.calls), 5)
            self.assertEqual(len(downloader.downloaded_ids), 5)
            self.assertEqual(downloader.stats["success"], 2)
            report = json.loads(
                (Path(tmp) / "aflow_download_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["initial_cached"], 3)
            self.assertEqual(report["requested_target"], 5)
            self.assertEqual(report["stats"]["success"], 2)
            self.assertTrue(report["target_reached"])
            self.assertEqual(report["termination_reason"], "target_reached")

    def test_downloader_raises_when_candidate_pool_cannot_reach_target(self):
        class OneCandidateAdapter(_FakeAdapter):
            def fetch_metadata(self, query_params=None):
                return [_minimal_meta("aflow-only", 1)]

        with tempfile.TemporaryDirectory() as tmp:
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=OneCandidateAdapter(tmp),
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "requested target 2.*persisted 1",
            ):
                downloader.query_and_download(target_count=2)

            report = json.loads(
                (Path(tmp) / "aflow_download_report.json").read_text(encoding="utf-8")
            )
            self.assertFalse(report["target_reached"])
            self.assertEqual(report["termination_reason"], "candidate_pool_exhausted")
            self.assertEqual(report["total_cached"], 1)

    def test_downloader_progress_uses_persisted_over_requested_target(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as tmp:
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=_FakeAdapter(tmp),
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                downloader.query_and_download(target_count=2)

            progress = [
                line
                for line in output.getvalue().splitlines()
                if "persisted=" in line
            ]
            self.assertEqual(len(progress), 2)
            self.assertIn("persisted=1/2", progress[0])
            self.assertIn("persisted=2/2", progress[1])
            self.assertTrue(all("submitted=" in line for line in progress))
            self.assertTrue(all("in_flight=" in line for line in progress))

    def test_downloader_processes_completed_future_before_peer_interrupt(self):
        import threading
        import time

        class ConcurrentInterruptingAdapter(_FakeAdapter):
            def __init__(self, cache_dir):
                super().__init__(cache_dir)
                self.no_data_ready = threading.Event()

            def fetch_metadata(self, query_params=None):
                return [
                    _minimal_meta("aflow-no-data-first", 1),
                    _minimal_meta("aflow-interrupt-second", 2),
                ]

            def fetch_band_structure(self, material_id, metadata=None):
                if material_id == "aflow-no-data-first":
                    self.no_data_ready.set()
                    return {
                        "material_id": material_id,
                        "error": "band data unavailable: empty bands_data",
                    }
                self.no_data_ready.wait(timeout=1.0)
                time.sleep(0.05)
                raise KeyboardInterrupt("simulated concurrent stop")

        with tempfile.TemporaryDirectory() as tmp:
            adapter = ConcurrentInterruptingAdapter(tmp)
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=adapter,
                workers=2,
            )
            downloader._get_fetch_adapter = lambda: adapter

            with self.assertRaisesRegex(KeyboardInterrupt, "concurrent stop"):
                downloader.query_and_download(target_count=2)

            exclusions = json.loads(
                (Path(tmp) / "aflow_excluded.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (Path(tmp) / "aflow_download_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("aflow-no-data-first", exclusions)
            self.assertEqual(report["completed"], 1)
            self.assertEqual(report["stats"]["no_data"], 1)
            self.assertEqual(
                report["termination_reason"],
                "interrupted:KeyboardInterrupt",
            )

    def test_mp_batch_rejects_short_adapter_response(self):
        class ShortMPBatchAdapter:
            def __init__(self, cache_dir):
                self.cache_dir = cache_dir
                self.saved = []

            def fetch_metadata(self, query_params=None):
                return [
                    _minimal_meta("mp-batch-a", 1),
                    _minimal_meta("mp-batch-b", 2),
                ]

            def fetch_band_structures(self, batch):
                first = batch[0]
                return [
                    {
                        "material_id": first["material_id"],
                        "energies": np.ones((2, 4), dtype=np.float32),
                    }
                ]

            def save_to_hdf5(self, record):
                self.saved.append(record["material_id"])
                with h5py.File(Path(self.cache_dir) / "mp_bands.h5", "a") as handle:
                    handle.require_group(record["material_id"])

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            adapter = ShortMPBatchAdapter(tmp)
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="mp",
                adapter=adapter,
                workers=1,
                batch_size=2,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "MP batch cardinality mismatch: requested 2, received 1",
            ):
                downloader.query_and_download(target_count=2)

            report = json.loads(
                (Path(tmp) / "mp_download_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(adapter.saved, [])
            self.assertEqual(report["submitted"], 2)
            self.assertEqual(report["completed"], 0)
            self.assertEqual(report["termination_reason"], "error:RuntimeError")

    def test_downloader_checkpoints_exclusion_and_report_before_interrupt(self):
        class InterruptingAdapter(_FakeAdapter):
            def fetch_metadata(self, query_params=None):
                return [
                    _minimal_meta("aflow-no-data", 1),
                    _minimal_meta("aflow-interrupt", 2),
                ]

            def fetch_band_structure(self, material_id, metadata=None):
                self.calls.append(material_id)
                if material_id == "aflow-no-data":
                    return {
                        "material_id": material_id,
                        "error": "band data unavailable: empty bands_data",
                    }
                raise KeyboardInterrupt("simulated stop")

        with tempfile.TemporaryDirectory() as tmp:
            downloader = RobustBandDownloader(
                cache_dir=tmp,
                source="aflow",
                adapter=InterruptingAdapter(tmp),
            )
            with self.assertRaises(KeyboardInterrupt):
                downloader.query_and_download(target_count=1)

            exclusions = json.loads(
                (Path(tmp) / "aflow_excluded.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (Path(tmp) / "aflow_download_report.json").read_text(encoding="utf-8")
            )
            self.assertIn("aflow-no-data", exclusions)
            self.assertFalse(report["target_reached"])
            self.assertEqual(
                report["termination_reason"],
                "interrupted:KeyboardInterrupt",
            )
            self.assertEqual(report["requested_target"], 1)
            self.assertEqual(report["attempted"], 2)

    def test_aflow_metadata_reallocates_quota_from_exhausted_gap_bin(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = AFLOWAdapter(cache_dir=tmp)
            requested_pages = []

            def fake_request(url):
                decoded = urllib.parse.unquote(url)
                page_match = re.search(r"paging\((\d+),(\d+)\)", decoded)
                self.assertIsNotNone(page_match)
                page = int(page_match.group(1))
                size = int(page_match.group(2))
                requested_pages.append((decoded, page, size))
                if "Egap(0*,*1)" in decoded:
                    return json.dumps({}).encode("utf-8")
                rows = {}
                for offset in range(size):
                    index = (page - 1) * size + offset
                    rows[str(index)] = {
                        "auid": f"aflow:{index:016x}",
                        "aurl": f"lib1/raw/{index}",
                        "files": [f"entry_{index}_bandsdata.json.xz"],
                        "Egap": 1.5,
                        "natoms": 2,
                    }
                return json.dumps(rows).encode("utf-8")

            adapter._request_bytes = fake_request
            records = adapter.fetch_metadata(
                {
                    "band_gap": (0.0, 2.0),
                    "num_sites": (1, 10),
                    "limit": 4,
                    "page_size": 2,
                    "gap_bins": 2,
                }
            )

            self.assertEqual(len(records), 4)
            high_bin_pages = [
                page
                for decoded, page, _size in requested_pages
                if "Egap(1*,*2)" in decoded
            ]
            self.assertEqual(high_bin_pages, [1, 2])

    def test_aflow_metadata_resumes_each_gap_bin_from_its_exact_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = AFLOWAdapter(cache_dir=tmp)
            calls = []

            def fake_request(url):
                decoded = urllib.parse.unquote(url)
                page_match = re.search(r"paging\((\d+),(\d+)\)", decoded)
                self.assertIsNotNone(page_match)
                page = int(page_match.group(1))
                is_low_bin = "Egap(0*,*1)" in decoded
                calls.append(("low" if is_low_bin else "high", page))
                if is_low_bin and page > 1:
                    return b"{}"
                prefix = 0 if is_low_bin else 100
                index = prefix + page
                return json.dumps(
                    {
                        "0": {
                            "auid": f"aflow:{index:016x}",
                            "aurl": f"lib1/raw/{index}",
                            "files": [f"entry_{index}_bandsdata.json.xz"],
                            "Egap": 0.5 if is_low_bin else 1.5,
                            "natoms": 2,
                        }
                    }
                ).encode("utf-8")

            adapter._request_bytes = fake_request
            query = {
                "band_gap": (0.0, 2.0),
                "num_sites": (1, 10),
                "limit": 4,
                "page_size": 1,
                "gap_bins": 2,
            }
            first = adapter.fetch_metadata(query)
            first_call_count = len(calls)
            cursor = getattr(adapter, "last_metadata_cursor", None)
            second = adapter.fetch_metadata({**query, "_aflow_cursor": cursor})
            second_calls = calls[first_call_count:]

            self.assertEqual(len(first), 4)
            self.assertEqual(len(second), 4)
            self.assertNotIn(("low", 1), second_calls)
            self.assertNotIn(("low", 2), second_calls)
            high_pages = [page for gap_bin, page in second_calls if gap_bin == "high"]
            self.assertEqual(high_pages[0], 4)

    def test_aflow_adapter_propagates_band_store_commit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            adapter = AFLOWAdapter(cache_dir=tmp)
            result = adapter.save_to_hdf5(
                {
                    "material_id": "aflow-status",
                    "source": "aflow",
                    "energies": np.asarray([[-1.0], [1.0]], dtype=np.float32),
                    "efermi": 0.0,
                    "metadata": {"spacegroup_number": 1},
                }
            )
            self.assertEqual(result.status, "canonical")

    def test_aflow_missing_gap_type_stays_unknown(self):
        record = AFLOWAdapter._metadata_record(
            {
                "auid": "aflow:unknown-type",
                "aurl": "AFLOWDATA/test/unknown",
                "files": ["x_bandsdata.json.xz"],
                "Egap": 1.2,
                "Egap_type": "",
                "natoms": 4,
            }
        )
        self.assertIsNone(record["is_direct"])
        self.assertFalse(record["is_metal"])

    def test_finetune_gpu_requirement_fails_fast_without_gpu(self):
        try:
            from scripts import finetune_supervised
        except ImportError:
            self.skipTest("TensorFlow stack is not installed in the lightweight host environment")
        with patch.object(
            finetune_supervised.tf.config,
            "list_physical_devices",
            return_value=[],
        ):
            with self.assertRaisesRegex(RuntimeError, "No TensorFlow GPU"):
                finetune_supervised.configure_tensorflow_runtime(require_gpu=True)

    def test_required_gpu_tensor_gate_rejects_cpu_fallback(self):
        from types import SimpleNamespace
        from src.utils import assert_tensor_on_gpu

        cpu_tensor = SimpleNamespace(device="/job:localhost/device:CPU:0")
        gpu_tensor = SimpleNamespace(device="/job:localhost/device:GPU:0")

        with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
            assert_tensor_on_gpu(cpu_tensor, "probe", require_gpu=True)
        self.assertIn(
            "/DEVICE:GPU:",
            assert_tensor_on_gpu(gpu_tensor, "probe", require_gpu=True).upper(),
        )
        assert_tensor_on_gpu(cpu_tensor, "portable smoke", require_gpu=False)

    def test_finetune_cleanup_does_not_delete_a_sibling_experiment_report(self):
        from scripts.finetune_supervised import clean_finetune_artifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "exp"
            output_dir.mkdir()
            sibling = root / "finetune_supervised_report_20260604.md"
            sibling.write_text("legacy", encoding="utf-8")
            clean_finetune_artifacts(
                str(output_dir),
                str(root / "checkpoints"),
                str(root / "model.weights.h5"),
            )
            self.assertTrue(sibling.is_file())

    def test_supervised_report_path_is_dated_inside_the_experiment_directory(self):
        from scripts.finetune_supervised import supervised_report_path

        path = supervised_report_path("artifacts/reports/exp", "20260825")
        self.assertEqual(path, os.path.join("artifacts", "reports", "exp", "finetune_supervised_report_20260825.md"))
        for unsafe in ("../escape", "2026-08-27", "", "123456789"):
            with self.assertRaisesRegex(ValueError, "YYYYMMDD"):
                supervised_report_path("artifacts/reports/exp", unsafe)

    def test_finetune_freezes_ssl_only_mask_token(self):
        try:
            import tensorflow as tf
            from src.engine.finetune_trainer import freeze_encoder_layers
            from src.models import SSLEncoder
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        encoder = SSLEncoder(
            num_features=6,
            seq_len=8,
            d_model=8,
            num_heads=2,
            num_layers=1,
            dff=16,
            projection_dim=4,
        )
        sample = tf.zeros((1, 8, 6), dtype=tf.float32)
        encoder(sample, training=False)
        encoder.reconstruct(sample, training=False)
        mask_token_id = id(encoder.mask_token)

        info = freeze_encoder_layers(encoder, freeze_layers=0)

        self.assertNotIn(mask_token_id, {id(var) for var in encoder.trainable_variables})
        self.assertIn("mask_token", info["frozen_heads"])

    def test_supervised_compile_explicitly_disables_xla_jit(self):
        from scripts.finetune_supervised import compile_model

        class RecordingModel:
            def compile(self, **kwargs):
                self.compile_kwargs = kwargs

        model = RecordingModel()
        compile_model(
            model,
            learning_rate=1e-3,
            type_weight=2.0,
            class_weights={0: 1.0, 1: 1.0, 2: 1.0},
        )
        self.assertIn("jit_compile", model.compile_kwargs)
        self.assertFalse(model.compile_kwargs["jit_compile"])

    def test_gpu_requirements_pin_the_v100_compatible_cuda_stack(self):
        requirements = (
            Path(__file__).resolve().parents[1] / "requirements-gpu.txt"
        ).read_text(encoding="utf-8")
        expected = {
            "nvidia-cublas-cu12==12.5.3.2",
            "nvidia-cuda-cupti-cu12==12.5.82",
            "nvidia-cuda-nvcc-cu12==12.5.82",
            "nvidia-cuda-nvrtc-cu12==12.5.82",
            "nvidia-cuda-runtime-cu12==12.5.82",
            "nvidia-cudnn-cu12==9.3.0.75",
            "nvidia-cufft-cu12==11.2.3.61",
            "nvidia-curand-cu12==10.3.6.82",
            "nvidia-cusolver-cu12==11.6.3.83",
            "nvidia-cusparse-cu12==12.5.1.3",
            "nvidia-nvjitlink-cu12==12.5.82",
        }
        configured = {
            line.strip()
            for line in requirements.splitlines()
            if line.strip().startswith("nvidia-")
        }
        self.assertEqual(configured, expected)

    def test_finetune_cli_exposes_report_date(self):
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "scripts" / "finetune_supervised.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--report-date", result.stdout)

    def test_finetune_cli_exposes_require_gpu(self):
        import subprocess
        import sys

        script = Path(__file__).resolve().parents[1] / "scripts" / "finetune_supervised.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--require-gpu", result.stdout)

    def test_manual_calibration_supports_reversed_axes_and_fermi_shift(self):
        reconstructor = PhysicsReconstructor(target_k_points=32)
        result = reconstructor.reconstruct_from_manual(
            source_path="synthetic.png",
            panel_bbox=[0, 0, 100, 100],
            y_calibration=[{"y": 90, "value": -4}, {"y": 10, "value": 4}],
            x_calibration=[{"x": 90, "value": 1}, {"x": 10, "value": 0}],
            vbm_pixel={"x": 50, "y": 55},
            cbm_pixel={"x": 50, "y": 35},
            valence_points=[{"x": 10, "y": 65}, {"x": 90, "y": 65}],
            conduction_points=[{"x": 10, "y": 25}, {"x": 90, "y": 25}],
            fermi_y_pixel=50,
        )
        self.assertGreater(result.tensor_gap, 0.0)
        self.assertAlmostEqual(result.metadata["raw_fermi_energy_ev"], 0.0, places=6)
        self.assertEqual(result.raw_tensor.shape, (1, 2, 32, 3))

    def test_manual_calibration_rejects_coincident_points(self):
        reconstructor = PhysicsReconstructor(target_k_points=16)
        with self.assertRaisesRegex(ValueError, "Y-axis calibration"):
            reconstructor.reconstruct_from_manual(
                source_path="bad.png",
                panel_bbox=[0, 0, 10, 10],
                y_calibration=[{"y": 2, "value": 0}, {"y": 2, "value": 1}],
                x_calibration=[{"x": 0, "value": 0}, {"x": 10, "value": 1}],
                vbm_pixel={"x": 5, "y": 6},
                cbm_pixel={"x": 5, "y": 4},
                valence_points=[{"x": 1, "y": 7}, {"x": 9, "y": 7}],
                conduction_points=[{"x": 1, "y": 3}, {"x": 9, "y": 3}],
            )

    def test_physics_loss_second_derivative_respects_segments(self):
        try:
            import tensorflow as tf
            from src.models.losses import CurvatureConsistencyLoss
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")
        segments = tf.constant([[0, 0, 0, 1, 1, 1]], dtype=tf.int32)
        base = tf.constant([[0.0, 1.0, 4.0, 0.0, 1.0, 4.0]], dtype=tf.float32)
        shifted = tf.constant([[0.0, 1.0, 4.0, 100.0, 101.0, 104.0]], dtype=tf.float32)
        loss = CurvatureConsistencyLoss()
        base_curv = loss.second_derivative(base, segment_ids=segments)
        shifted_curv = loss.second_derivative(shifted, segment_ids=segments)
        np.testing.assert_allclose(base_curv.numpy(), shifted_curv.numpy(), atol=1e-6)

    def test_posthoc_local_curvature_respects_segments(self):
        from src.utils.physics_validator import local_curvature

        segments = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        base = np.asarray([0.0, 1.0, 4.0, 0.0, 1.0, 4.0], dtype=np.float32)
        shifted = np.asarray(
            [0.0, 1.0, 4.0, 100.0, 101.0, 104.0],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            local_curvature(base, segment_ids=segments),
            local_curvature(shifted, segment_ids=segments),
            atol=1e-6,
        )

    def test_physics_validator_consumes_segment_ids(self):
        from src.utils.physics_validator import PhysicsValidator

        segment_ids = np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.int32)
        base = np.zeros((1, 2, 6, 3), dtype=np.float32)
        shifted = np.zeros_like(base)
        base[0, 0, :, 0] = [0.0, 1.0, 4.0, -4.0, -3.0, 0.0]
        base[0, 1, :, 0] = [4.0, 1.0, 0.0, 8.0, 5.0, 4.0]
        shifted[:] = base
        shifted[0, 0, 3:, 0] -= 100.0
        shifted[0, 1, 3:, 0] += 100.0
        predicted_gap = np.asarray([-4.0], dtype=np.float32)
        validator = PhysicsValidator()

        base_stats = validator.validate(
            predicted_gap,
            base,
            segment_ids=segment_ids,
        )
        shifted_stats = validator.validate(
            predicted_gap,
            shifted,
            segment_ids=segment_ids,
        )

        self.assertEqual(
            base_stats["vbm_positive_curvature_rate"],
            shifted_stats["vbm_positive_curvature_rate"],
        )
        self.assertEqual(
            base_stats["cbm_negative_curvature_rate"],
            shifted_stats["cbm_negative_curvature_rate"],
        )

    def test_finetune_physics_stats_consume_segment_ids(self):
        from scripts.finetune_supervised import physics_violation_stats

        segment_ids = np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.int32)
        base = np.zeros((1, 2, 6, 3), dtype=np.float32)
        shifted = np.zeros_like(base)
        base[0, 0, :, 0] = [0.0, 1.0, 4.0, -4.0, -3.0, 0.0]
        base[0, 1, :, 0] = [4.0, 1.0, 0.0, 8.0, 5.0, 4.0]
        shifted[:] = base
        shifted[0, 0, 3:, 0] -= 100.0
        shifted[0, 1, 3:, 0] += 100.0
        predicted_gap = np.asarray([-4.0], dtype=np.float32)

        base_stats = physics_violation_stats(
            predicted_gap,
            base,
            segment_ids=segment_ids,
        )
        shifted_stats = physics_violation_stats(
            predicted_gap,
            shifted,
            segment_ids=segment_ids,
        )

        self.assertEqual(
            base_stats["vbm_positive_curvature_rate"],
            shifted_stats["vbm_positive_curvature_rate"],
        )
        self.assertEqual(
            base_stats["cbm_negative_curvature_rate"],
            shifted_stats["cbm_negative_curvature_rate"],
        )

    def test_curvature_zoom_examples_pass_segment_ids_to_curvature_helper(self):
        from scripts import finetune_supervised

        tensors = np.zeros((1, 2, 6, 3), dtype=np.float32)
        tensors[0, 0, :, 0] = [0.0, 1.0, 4.0, 0.0, 1.0, 4.0]
        tensors[0, 1, :, 0] = [4.0, 1.0, 0.0, 4.0, 1.0, 0.0]
        segments = np.asarray([[0, 0, 0, 1, 1, 1]], dtype=np.int32)
        seen = []

        def fake_curvature(band, segment_ids=None):
            seen.append(np.asarray(segment_ids).copy())
            return np.zeros_like(band, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            finetune_supervised,
            "_local_curvature",
            side_effect=fake_curvature,
        ):
            finetune_supervised.save_curvature_zoom_examples(
                tensors,
                tensors,
                np.asarray(["sample"]),
                str(Path(tmp) / "curvature.png"),
                segment_ids=segments,
                max_examples=1,
            )

        self.assertEqual(len(seen), 4)
        for observed in seen:
            np.testing.assert_array_equal(observed, segments[0])

    def test_zero_ssl_consistency_weight_stays_disabled_after_adaptation(self):
        try:
            import tensorflow as tf
            from src.engine.ssl_trainer import MBMTrainer
            from src.models import SSLEncoder
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        with tempfile.TemporaryDirectory() as tmp:
            model = SSLEncoder(
                num_features=6,
                seq_len=4,
                d_model=8,
                num_heads=2,
                num_layers=1,
                dff=16,
                projection_dim=4,
            )
            model(tf.zeros((1, 4, 6)), training=False)
            trainer = MBMTrainer(
                model=model,
                learning_rate=1e-3,
                mask_ratio=0.25,
                sign_weight=0.1,
                consistency_weight=0.0,
                checkpoint_dir=tmp,
                log_dir=tmp,
            )
            trainer._adapt_physics_weights(
                {
                    "curvature_loss": 0.0,
                    "symmetry_loss": 0.0,
                    "mse_loss": 1.0,
                }
            )

        self.assertEqual(float(trainer.symmetry_weight.numpy()), 0.0)

    def test_ssl_best_checkpoint_records_the_improved_epoch(self):
        try:
            import tensorflow as tf
            from src.engine.ssl_trainer import MBMTrainer
            from src.models import SSLEncoder
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        class ConstantMetricTrainer(MBMTrainer):
            def _run_epoch(self, dataset, training):
                del dataset, training
                return {
                    "total": 1.0,
                    "mse_loss": 1.0,
                    "masked_mae": 1.0,
                    "mask_fraction": 0.25,
                    "curvature_loss": 0.0,
                    "symmetry_loss": 0.0,
                }

            def _adapt_physics_weights(self, val_metrics):
                del val_metrics

        with tempfile.TemporaryDirectory() as tmp:
            model = SSLEncoder(
                num_features=6,
                seq_len=4,
                d_model=8,
                num_heads=2,
                num_layers=1,
                dff=16,
                projection_dim=4,
            )
            model(tf.zeros((1, 4, 6)), training=False)
            trainer = ConstantMetricTrainer(
                model=model,
                learning_rate=1e-3,
                mask_ratio=0.25,
                sign_weight=0.1,
                consistency_weight=0.0,
                checkpoint_dir=tmp,
                log_dir=tmp,
                early_stopping_patience=0,
            )
            dataset = tf.data.Dataset.from_tensors(
                (
                    tf.zeros((1, 4, 6)),
                    tf.zeros((1,), tf.int32),
                    tf.zeros((1, 4), tf.int32),
                )
            )
            trainer.train(dataset, dataset, epochs=1)
            reader = tf.train.load_checkpoint(str(Path(tmp) / "ckpt-best"))
            saved_epoch = int(reader.get_tensor("epoch/.ATTRIBUTES/VARIABLE_VALUE"))

        self.assertEqual(saved_epoch, 1)

    def test_ssl_trainer_early_stops_after_patience(self):
        try:
            import tensorflow as tf
            from src.engine.ssl_trainer import MBMTrainer
            from src.models import SSLEncoder
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        class ConstantMetricTrainer(MBMTrainer):
            def _run_epoch(self, dataset, training):
                return {
                    "total": 1.0, "mse_loss": 1.0, "masked_mae": 1.0,
                    "mask_fraction": 0.25, "curvature_loss": 0.0, "symmetry_loss": 0.0,
                }
            def _adapt_physics_weights(self, val_metrics):
                del val_metrics
                self.curvature_weight.assign(9.0)
                self.symmetry_weight.assign(8.0)
            def save_checkpoint(self, name):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            model = SSLEncoder(
                num_features=6, seq_len=4, d_model=8, num_heads=2,
                num_layers=1, dff=16, projection_dim=4,
            )
            model(tf.zeros((1, 4, 6)), training=False)
            trainer = ConstantMetricTrainer(
                model=model, learning_rate=1e-3, mask_ratio=0.25,
                sign_weight=0.1, consistency_weight=0.1,
                checkpoint_dir=tmp, log_dir=tmp,
                warmup_epochs=0, early_stopping_patience=2,
                early_stopping_min_delta=0.0,
            )
            dataset = tf.data.Dataset.from_tensors(
                (tf.zeros((1, 4, 6)), tf.zeros((1,), tf.int32), tf.zeros((1, 4), tf.int32))
            )
            trainer.train(dataset, dataset, epochs=10)
            self.assertEqual(int(trainer.epoch_var.numpy()), 3)
            history = json.loads(
                (Path(tmp) / "ssl_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(history["epochs"]), 3)
            self.assertEqual(history["epochs"][-1]["epoch"], 3)
            self.assertEqual(history["selection_monitor"], "val_total")
            self.assertAlmostEqual(
                history["epochs"][0]["selection_weights"]["curvature"],
                0.1,
                places=6,
            )
            self.assertEqual(
                history["epochs"][0]["post_adaptation_weights"]["curvature"],
                9.0,
            )

    def test_warmup_cosine_learning_rate_boundaries(self):
        try:
            from src.engine import ssl_trainer
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")
        self.assertTrue(hasattr(ssl_trainer, "warmup_cosine_learning_rate"))
        schedule = ssl_trainer.warmup_cosine_learning_rate
        self.assertAlmostEqual(float(schedule(0, 10, 2, 1e-3, 1e-5)), 0.0, places=10)
        self.assertAlmostEqual(float(schedule(2, 10, 2, 1e-3, 1e-5)), 1e-3, places=10)
        self.assertAlmostEqual(float(schedule(9, 10, 2, 1e-3, 1e-5)), 1e-5, places=10)

    def test_ssl_epoch_metrics_weight_unequal_batches(self):
        try:
            import tensorflow as tf
            from src.engine.ssl_trainer import MBMTrainer
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        class SyntheticMetricTrainer(MBMTrainer):
            def _val_step(self, batch, segment_ids):
                del segment_ids
                sample_count = tf.cast(tf.shape(batch)[0], tf.float32)
                value = tf.where(sample_count > 1.0, 1.0, 9.0)
                return {
                    "total": value,
                    "mse_loss": value,
                    "masked_mae": value,
                    "mask_fraction": tf.where(sample_count > 1.0, 0.20, 0.30),
                    "curvature_loss": value,
                    "symmetry_loss": value,
                    "sample_count": sample_count,
                    "masked_elements": sample_count * 10.0,
                    "position_count": sample_count * 50.0,
                }

        trainer = object.__new__(SyntheticMetricTrainer)
        x = tf.zeros((4, 8, 6), dtype=tf.float32)
        groups = tf.range(4, dtype=tf.int32)
        segments = tf.zeros((4, 8), dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((x, groups, segments)).batch(3)
        metrics = trainer._run_epoch(dataset, training=False)

        self.assertAlmostEqual(metrics["mse_loss"], 3.0, places=6)
        self.assertAlmostEqual(metrics["masked_mae"], 3.0, places=6)
        self.assertAlmostEqual(metrics["mask_fraction"], 0.225, places=6)

    def test_ssl_validation_uses_fixed_corruption(self):
        try:
            import tensorflow as tf
            from src.engine.ssl_trainer import MBMTrainer
            from src.models import SSLEncoder
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        with tempfile.TemporaryDirectory() as tmp:
            tf.keras.utils.set_random_seed(321)
            model = SSLEncoder(
                num_features=6,
                seq_len=16,
                d_model=8,
                num_heads=2,
                num_layers=1,
                dff=16,
                projection_dim=4,
                dropout_rate=0.0,
            )
            model(tf.zeros((1, 16, 6)), training=False)
            model.reconstruct(tf.zeros((1, 16, 6)), training=False)
            trainer = MBMTrainer(
                model=model,
                learning_rate=1e-3,
                mask_ratio=0.25,
                sign_weight=0.0,
                consistency_weight=0.0,
                checkpoint_dir=tmp,
                log_dir=tmp,
                min_span=3,
                max_span=5,
            )
            x = tf.random.stateless_normal((4, 16, 6), seed=[7, 13])
            groups = tf.range(4, dtype=tf.int32)
            segment_row = tf.constant([0] * 8 + [1] * 8, dtype=tf.int32)
            segments = tf.tile(segment_row[None, :], [4, 1])
            dataset = tf.data.Dataset.from_tensor_slices((x, groups, segments)).batch(2)

            first = trainer._run_epoch(dataset, training=False)
            second = trainer._run_epoch(dataset, training=False)

        for key in first:
            self.assertAlmostEqual(first[key], second[key], places=7, msg=key)

    def test_ssl_cli_defaults_disable_invalid_consistency_and_path_warp(self):
        from scripts import train_ssl

        with patch.object(sys, "argv", ["train_ssl.py"]):
            args = train_ssl.parse_args()

        self.assertEqual(args.consistency_weight, 0.0)
        self.assertTrue(args.disable_strain_augmentation)

    def test_ssl_loader_keeps_segment_ids_aligned_with_groups(self):
        from scripts.train_ssl import load_ood_tensor_data
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.npz"
            groups = np.asarray([10, 11, 12, 13], dtype=np.int32)
            X = np.zeros((4, 2, 8, 3), dtype=np.float32)
            segment_ids = np.repeat(groups[:, None], 8, axis=1)
            np.savez(
                path,
                X_train=X,
                groups_train=groups,
                y_type_train=np.asarray([0, 1, 2, 1], dtype=np.int32),
                segment_ids_train=segment_ids,
            )
            X_fit, X_val, g_fit, g_val, s_fit, s_val = load_ood_tensor_data(
                str(path), validation_size=0.25, random_state=42
            )
            self.assertEqual(X_fit.shape[-1], 6)
            self.assertEqual(X_val.shape[-1], 6)
            for group, segments in zip(g_fit, s_fit):
                self.assertTrue(np.all(segments == group))
            for group, segments in zip(g_val, s_val):
                self.assertTrue(np.all(segments == group))

    def test_span_mask_stops_at_segment_boundary(self):
        try:
            import tensorflow as tf
            from src.engine import ssl_trainer
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")
        self.assertTrue(hasattr(ssl_trainer, "span_mask_from_starts"))
        starts = tf.constant([[6]], dtype=tf.int32)
        segments = tf.constant([[0] * 8 + [1] * 8], dtype=tf.int32)
        mask = ssl_trainer.span_mask_from_starts(
            starts, span_length=4, seq_len=16, segment_ids=segments
        )
        expected = np.zeros((1, 16, 1), dtype=np.float32)
        expected[0, 6:8, 0] = 1.0
        np.testing.assert_array_equal(mask.numpy(), expected)

    def test_random_span_mask_is_seeded_and_respects_actual_ratio_contract(self):
        try:
            import tensorflow as tf
            from src.engine.ssl_trainer import random_span_mask
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        batch_size = 16
        seq_len = 128
        segments = tf.constant(
            np.tile(
                np.repeat(np.arange(4, dtype=np.int32), 32)[None, :],
                (batch_size, 1),
            )
        )
        kwargs = dict(
            batch_size=tf.constant(batch_size),
            seq_len=tf.constant(seq_len),
            mask_ratio=0.25,
            min_span=5,
            max_span=15,
            segment_ids=segments,
            seed=tf.constant([17, 29], dtype=tf.int32),
        )
        first = random_span_mask(**kwargs).numpy()
        second = random_span_mask(**kwargs).numpy()
        fractions = first.mean(axis=(1, 2))

        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(fractions >= 0.15), fractions)
        self.assertTrue(np.all(fractions <= 0.30), fractions)
        self.assertGreaterEqual(float(fractions.mean()), 0.20)

    def test_ssl_encoder_uses_learnable_mask_token(self):
        try:
            import tensorflow as tf
            from src.models import SSLEncoder
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")
        model = SSLEncoder(
            num_features=6, seq_len=4, d_model=8, num_heads=2,
            num_layers=1, dff=16, projection_dim=4,
        )
        sample = tf.zeros((1, 4, 6), dtype=tf.float32)
        model(sample, training=False)
        model.mask_token.assign(tf.ones_like(model.mask_token))
        mask = tf.constant([[[0.0], [1.0], [0.0], [1.0]]], dtype=tf.float32)
        masked = model.apply_mask_token(sample, mask)
        np.testing.assert_allclose(masked.numpy()[0, 0], 0.0)
        np.testing.assert_allclose(masked.numpy()[0, 1], 1.0)
        np.testing.assert_allclose(masked.numpy()[0, 2], 0.0)
        np.testing.assert_allclose(masked.numpy()[0, 3], 1.0)

    def test_ssl_encoder_round_trip_in_project_loader(self):
        try:
            import tensorflow as tf
            from src.models import SSLEncoder, load_ssl_encoder
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "encoder.keras"
            model = SSLEncoder(
                num_features=6,
                seq_len=8,
                d_model=8,
                num_heads=2,
                num_layers=1,
                dff=16,
                projection_dim=4,
            )
            sample = tf.zeros((1, 8, 6), dtype=tf.float32)
            model(sample, training=False)
            model.reconstruct(sample, training=False)
            model.save(model_path)

            loaded = load_ssl_encoder(model_path)
            self.assertEqual(tuple(loaded(sample, training=False).shape), (1, 4))
            self.assertEqual(tuple(loaded.reconstruct(sample, training=False).shape), (1, 8, 6))

    def test_extremum_gap_head_enforces_fermi_anchored_zero_gap(self):
        try:
            import tensorflow as tf
            from scripts.finetune_supervised import ExtremumExpectedGapHead
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")

        # The server's V100/CUDA13 stack has a known cuDNN Conv1D ABI issue;
        # this unit test validates gate semantics, not GPU kernels, so run it on CPU.
        with tf.device("/CPU:0"):
            head = ExtremumExpectedGapHead(
                feature_mean=np.zeros(6, dtype=np.float32),
                feature_std=np.ones(6, dtype=np.float32),
                hidden_dim=8,
                dropout=0.0,
                metal_anchor_gate_enabled=True,
            )
            # sample 0: max(VBM)=min(CBM)=0 -> explicit Fermi-anchored metal
            # sample 1: max(VBM)=0, min(CBM)=1 -> ordinary semiconductor
            x = np.zeros((2, 4, 6), dtype=np.float32)
            x[0, :, 0] = [-1.0, -0.4, 0.0, -0.2]
            x[0, :, 3] = [0.0, 0.2, 0.7, 0.3]
            x[1, :, 0] = [-1.0, -0.4, 0.0, -0.2]
            x[1, :, 3] = [1.0, 1.2, 1.7, 1.3]
            encoded = tf.zeros((2, 4, 8), dtype=tf.float32)
            gap, _topology, _probs = head(encoded, tf.constant(x), training=False)
        self.assertAlmostEqual(float(gap[0, 0]), 0.0, places=7)
        self.assertGreater(float(gap[1, 0]), 0.0)

    def test_extremum_gap_head_metal_gate_is_disabled_by_default(self):
        try:
            import tensorflow as tf
            from scripts.finetune_supervised import ExtremumExpectedGapHead
        except ImportError:
            self.skipTest("TensorFlow is not installed in the lightweight host environment")
        with tf.device("/CPU:0"):
            head = ExtremumExpectedGapHead(
                feature_mean=np.zeros(6, dtype=np.float32),
                feature_std=np.ones(6, dtype=np.float32),
                hidden_dim=8,
                dropout=0.0,
            )
            x = np.zeros((1, 4, 6), dtype=np.float32)
            x[0, :, 0] = [-1.0, -0.4, 0.0, -0.2]
            x[0, :, 3] = [0.0, 0.2, 0.7, 0.3]
            encoded = tf.zeros((1, 4, 8), dtype=tf.float32)
            head(encoded, tf.constant(x), training=False)
            for layer in (head.local_conv, head.logit_layer):
                layer.set_weights([np.zeros_like(weight) for weight in layer.get_weights()])
            gap, _topology, _probs = head(encoded, tf.constant(x), training=False)
        self.assertAlmostEqual(float(gap[0, 0]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
