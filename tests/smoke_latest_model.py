from pathlib import Path
import json
import sys

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.finetune_supervised import SupervisedBandGapModel, load_dataset
from src.models import load_ssl_encoder

EXPERIMENT_ID = "aflow_noleak_v5_30k_seed42"
TENSOR_NPZ = (
    ROOT
    / "data"
    / "processed"
    / "aflow"
    / "ood_tensors_v5_30000_seed42"
    / "band_tensors_ood_split.npz"
)
MODEL_DIR = ROOT / "artifacts" / "models" / EXPERIMENT_ID
CONFIG = json.loads((MODEL_DIR / "finetuned_config.json").read_text(encoding="utf-8"))

data = load_dataset(str(TENSOR_NPZ), str(MODEL_DIR / "ssl_mbm_norm_stats.json"))
encoder = load_ssl_encoder(str(MODEL_DIR / "ssl_mbm_pretrained.keras"), compile=False)
model = SupervisedBandGapModel(
    encoder,
    feature_mean=data["feature_mean"],
    feature_std=data["feature_std"],
    metal_anchor_gate_enabled=CONFIG.get("metal_anchor_gap_gate_enabled", False),
)
model(tf.zeros([1, data["X_test"].shape[1], data["X_test"].shape[2]], dtype=tf.float32))
model.load_weights(str(MODEL_DIR / "finetuned.weights.h5"))

outputs = model(data["X_test"][:2], training=False)
gap = outputs["gap"].numpy().reshape(-1)
type_probabilities = outputs["type"].numpy()
assert np.isfinite(gap).all() and np.isfinite(type_probabilities).all()
assert np.allclose(type_probabilities.sum(axis=1), 1.0, atol=1e-5)
print(
    json.dumps(
        {
            "experiment_id": EXPERIMENT_ID,
            "input_shape": list(data["X_test"][:2].shape),
            "gap_ev": gap.tolist(),
            "type_probabilities": type_probabilities.tolist(),
            "type_probability_sums": type_probabilities.sum(axis=1).tolist(),
            "gap_device": outputs["gap"].device,
            "type_device": outputs["type"].device,
        },
        indent=2,
    )
)
