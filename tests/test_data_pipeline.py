import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.ood_tensor_builder import (
    _curvature,
    _fermi_crossing_band_index,
    build_group_ood_split,
    build_group_validation_split,
    process_band_data,
    save_ood_tensors,
)


class DataPipelineContractTest(unittest.TestCase):
    def _write_synthetic_h5(self, path: Path, count: int = 10) -> None:
        with h5py.File(path, "w") as f:
            k = np.linspace(-1.0, 1.0, 24, dtype=np.float32)
            for idx in range(count):
                grp = f.create_group(f"mp-test-{idx}")
                vbm = -k**2 - 0.02 * idx
                cbm = k**2 + 1.0 + 0.01 * idx
                lower = vbm - 1.0
                upper = cbm + 1.0
                energies = np.stack([lower, vbm, cbm, upper], axis=0).astype(np.float32)
                grp.create_dataset("energies", data=energies)
                grp.attrs["spacegroup_number"] = 10 + idx
                grp.attrs["band_gap"] = 1.0 + 0.01 * idx
                grp.attrs["efermi"] = 0.5
                grp.attrs["num_kpoints"] = len(k)
                grp.attrs["kpath_labels"] = json.dumps(
                    [
                        {"index": 0, "label": "\\Gamma"},
                        {"index": len(k) - 1, "label": "X"},
                    ]
                )

    def test_ood_split_uses_spacegroup_groups_without_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "bands.h5"
            self._write_synthetic_h5(h5_path, count=12)

            X, y, groups, _material_ids, gap_types, _metadata = process_band_data(
                str(h5_path),
                target_k_points=128,
            )
            train_idx, test_idx = build_group_ood_split(
                X,
                y,
                groups,
                train_size=0.75,
                random_state=42,
                class_labels=gap_types,
            )

            intersection = set(groups[train_idx].tolist()).intersection(set(groups[test_idx].tolist()))
            print(f"train/test spacegroup intersection: {sorted(intersection)}")
            self.assertEqual(len(intersection), 0)

    def test_6d_physical_feature_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "bands.h5"
            self._write_synthetic_h5(h5_path, count=4)

            X, _y, _groups, _material_ids, gap_types, _metadata = process_band_data(
                str(h5_path),
                target_k_points=128,
            )

            self.assertEqual(X.shape, (4, 2, 128, 3))
            flattened = np.transpose(X, (0, 2, 1, 3)).reshape(X.shape[0], X.shape[2], -1)
            self.assertEqual(flattened.shape[-1], 6)
            self.assertTrue(np.all(gap_types == 1))

            vbm_e = flattened[0, :, 0]
            vbm_curv = flattened[0, :, 1]
            vbm_k_dist = flattened[0, :, 2]
            cbm_e = flattened[0, :, 3]
            cbm_curv = flattened[0, :, 4]
            cbm_k_dist = flattened[0, :, 5]

            self.assertTrue(np.all(np.isfinite(vbm_curv)))
            self.assertTrue(np.all(np.isfinite(cbm_curv)))
            self.assertLess(float(vbm_curv[np.argmax(vbm_e)]), 0.0)
            self.assertGreater(float(cbm_curv[np.argmin(cbm_e)]), 0.0)
            self.assertLessEqual(float(np.max(np.abs(vbm_k_dist))), 1.0)
            self.assertLessEqual(float(np.max(np.abs(cbm_k_dist))), 1.0)
            self.assertAlmostEqual(float(vbm_k_dist[np.argmax(vbm_e)]), 0.0, places=6)
            self.assertAlmostEqual(float(cbm_k_dist[np.argmin(cbm_e)]), 0.0, places=6)

    def test_fermi_crossing_is_detected_within_a_single_segment_only(self):
        segment_ids = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        false_global_crossing = np.asarray([
            [-1.0, -0.8, -0.6, 0.6, 0.8, 1.0],
            [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0],
        ], dtype=np.float32)
        self.assertIsNone(
            _fermi_crossing_band_index(
                false_global_crossing, 0.0, segment_ids=segment_ids
            )
        )
        true_segment_crossing = false_global_crossing.copy()
        true_segment_crossing[0, :3] = [-0.5, 0.0, 0.5]
        self.assertEqual(
            _fermi_crossing_band_index(
                true_segment_crossing, 0.0, segment_ids=segment_ids
            ),
            0,
        )

    def test_curvature_does_not_cross_kpath_segment_boundaries(self):
        k_axis = np.linspace(0.0, 1.0, 6, dtype=np.float32)
        segment_ids = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)
        base = np.asarray([0.0, 1.0, 4.0, 0.0, 1.0, 4.0], dtype=np.float32)
        shifted = np.asarray([0.0, 1.0, 4.0, 100.0, 101.0, 104.0], dtype=np.float32)
        base_curv = _curvature(base, k_axis, segment_ids=segment_ids)
        shifted_curv = _curvature(shifted, k_axis, segment_ids=segment_ids)
        np.testing.assert_allclose(base_curv, shifted_curv, rtol=0.0, atol=1e-5)

    def test_process_band_data_records_resampled_kpath_segment_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "segments.h5"
            k = np.linspace(0.0, 1.0, 16, dtype=np.float32)
            vbm = -k**2
            cbm = 1.0 + k**2
            with h5py.File(h5_path, "w") as handle:
                grp = handle.create_group("segmented")
                grp.create_dataset("energies", data=np.stack([vbm - 1.0, vbm, cbm, cbm + 1.0]))
                grp.create_dataset("k_distances", data=k)
                grp.attrs["spacegroup_number"] = 10
                grp.attrs["band_gap"] = 1.0
                grp.attrs["efermi"] = 0.5
                grp.attrs["is_metal"] = False
                grp.attrs["is_direct"] = True
                grp.attrs["num_kpoints"] = len(k)
                grp.attrs["kpath_labels"] = json.dumps([
                    {"index": 0, "label": "G"},
                    {"index": 8, "label": "X"},
                    {"index": 15, "label": "M"},
                ])

            _X, _y, _groups, _ids, _types, metadata_bundle = process_band_data(
                str(h5_path), target_k_points=8
            )
            sample = metadata_bundle[0]["samples"][0]
            self.assertEqual(sample["segment_ids"], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_tensor_features_do_not_depend_on_provider_metal_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "bands.h5"
            k = np.linspace(0.0, 1.0, 16, dtype=np.float32)
            crossing = np.linspace(-0.5, 0.5, 16, dtype=np.float32)
            lower = -1.5 - k**2
            upper = 1.5 + k**2
            energies = np.stack([lower, crossing, upper], axis=0).astype(np.float32)
            with h5py.File(h5_path, "w") as handle:
                for material_id, is_metal in (("a-metal", True), ("b-nonmetal", False)):
                    grp = handle.create_group(material_id)
                    grp.create_dataset("energies", data=energies)
                    grp.create_dataset("k_distances", data=k)
                    grp.attrs["spacegroup_number"] = 10 if is_metal else 11
                    grp.attrs["band_gap"] = 0.0 if is_metal else 1.0
                    grp.attrs["efermi"] = 5.0
                    grp.attrs["source"] = "aflow"
                    grp.attrs["is_metal"] = is_metal
                    grp.attrs["is_direct"] = not is_metal
                    grp.attrs["num_kpoints"] = len(k)

            X, _y, _groups, material_ids, gap_types, metadata_bundle = process_band_data(
                str(h5_path), target_k_points=128
            )
            self.assertEqual(material_ids.tolist(), ["a-metal", "b-nonmetal"])
            np.testing.assert_allclose(X[0], X[1], rtol=0.0, atol=1e-6)
            tensor_gap = float(np.min(X[0, 1, :, 0]) - np.max(X[0, 0, :, 0]))
            self.assertGreater(tensor_gap, 0.0)
            self.assertLess(tensor_gap, 0.1)
            self.assertEqual(gap_types.tolist(), [0, 1])
            samples = metadata_bundle[0]["samples"]
            self.assertTrue(samples[0]["metal_feature_inferred"])
            self.assertTrue(samples[0]["metal_label_matches_feature"])
            self.assertTrue(samples[1]["metal_feature_inferred"])
            self.assertFalse(samples[1]["metal_label_matches_feature"])

    def test_saved_split_contains_segments_and_provenance_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.h5"
            source_path.write_bytes(b"trusted synthetic source")
            output_dir = Path(tmp) / "out"
            X = np.zeros((2, 2, 8, 3), dtype=np.float32)
            y = np.asarray([0.0, 1.0], dtype=np.float32)
            groups = np.asarray([1, 2], dtype=np.int32)
            material_ids = np.asarray(["metal", "nonmetal"])
            gap_types = np.asarray([0, 1], dtype=np.int32)
            metadata_bundle = [{
                "samples": [
                    {
                        "material_id": "metal", "spacegroup_number": 1,
                        "formula_pretty": "Fe", "pearson_symbol": "cI2", "num_sites": 2,
                        "source": "aflow", "segment_ids": [0] * 8,
                        "metal_feature_inferred": True, "metal_label_matches_feature": True,
                    },
                    {
                        "material_id": "nonmetal", "spacegroup_number": 2,
                        "formula_pretty": "Si", "pearson_symbol": "cF8", "num_sites": 8,
                        "source": "aflow", "segment_ids": [0, 0, 0, 0, 1, 1, 1, 1],
                        "metal_feature_inferred": True, "metal_label_matches_feature": False,
                    },
                ],
                "skipped": [],
            }]
            manifest = save_ood_tensors(
                output_dir=str(output_dir), X=X, y=y, groups=groups,
                material_ids=material_ids, gap_types=gap_types,
                train_idx=np.asarray([0]), test_idx=np.asarray([1]),
                metadata_bundle=metadata_bundle, target_k_points=8,
                train_size=0.5, random_state=42,
                source_h5_path=str(source_path), metadata_path=None,
            )
            split = np.load(output_dir / "band_tensors_ood_split.npz")
            self.assertEqual(split["segment_ids_train"].shape, (1, 8))
            self.assertEqual(split["segment_ids_test"].tolist(), [[0, 0, 0, 0, 1, 1, 1, 1]])
            self.assertEqual(manifest["metal_feature_audit"]["mismatch_count"], 1)
            self.assertEqual(manifest["distribution_audit"]["composition_overlap_count"], 0)
            self.assertEqual(manifest["distribution_audit"]["prototype_overlap_count"], 0)
            self.assertEqual(len(manifest["provenance"]["source_h5_sha256"]), 64)
            self.assertEqual(len(manifest["provenance"]["builder_sha256"]), 64)

    def test_inner_validation_is_group_disjoint_and_not_outer_test(self):
        groups = np.repeat(np.arange(12), 2)
        labels = np.tile([1, 2], 12)
        fit_idx, val_idx = build_group_validation_split(
            groups,
            validation_size=0.25,
            random_state=7,
            class_labels=labels,
        )
        self.assertTrue(len(fit_idx) > len(val_idx) > 0)
        self.assertFalse(set(groups[fit_idx]).intersection(set(groups[val_idx])))


if __name__ == "__main__":
    unittest.main()
