import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.ood_tensor_builder import build_group_ood_split, process_band_data


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

            X, y, groups, _material_ids, _metadata = process_band_data(
                str(h5_path),
                target_k_points=128,
            )
            train_idx, test_idx = build_group_ood_split(X, y, groups, train_size=0.75, random_state=42)

            intersection = set(groups[train_idx].tolist()).intersection(set(groups[test_idx].tolist()))
            print(f"train/test spacegroup intersection: {sorted(intersection)}")
            self.assertEqual(len(intersection), 0)

    def test_6d_physical_feature_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "bands.h5"
            self._write_synthetic_h5(h5_path, count=4)

            X, _y, _groups, _material_ids, _metadata = process_band_data(
                str(h5_path),
                target_k_points=128,
            )

            self.assertEqual(X.shape, (4, 2, 128, 3))
            flattened = np.transpose(X, (0, 2, 1, 3)).reshape(X.shape[0], X.shape[2], -1)
            self.assertEqual(flattened.shape[-1], 6)

            vbm_e = flattened[0, :, 0]
            vbm_curv = flattened[0, :, 1]
            vbm_k_dist = flattened[0, :, 2]
            cbm_e = flattened[0, :, 3]
            cbm_curv = flattened[0, :, 4]
            cbm_k_dist = flattened[0, :, 5]

            np.testing.assert_allclose(vbm_curv, np.gradient(np.gradient(vbm_e)), atol=1e-6)
            np.testing.assert_allclose(cbm_curv, np.gradient(np.gradient(cbm_e)), atol=1e-6)
            self.assertLessEqual(float(np.max(np.abs(vbm_k_dist))), 1.0)
            self.assertLessEqual(float(np.max(np.abs(cbm_k_dist))), 1.0)
            self.assertAlmostEqual(float(vbm_k_dist[np.argmax(vbm_e)]), 0.0, places=6)
            self.assertAlmostEqual(float(cbm_k_dist[np.argmin(cbm_e)]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
