"""Extract ONE split of frozen normalized MBM pooled features.

--pairs is a single credentialed NPZ, not a directory containing both splits.
The cache keeps the original row IDs/mask plus valid-only embeddings, exactly
once in valid order. Formal execution is GPU-only; --test-scope is bounded CPU.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_p2_pairs as contracts
from src.data.band_structure_dataset import normalize_mbm_inputs
from src.models import load_ssl_encoder


def configure_runtime(test_scope=False, samples=None, epochs=1):
    if epochs < 1 or (samples is not None and samples < 1):
        raise ValueError("positive samples and epochs required")
    if test_scope and (epochs > 2 or (samples is not None and samples > 32)):
        raise ValueError("CPU test scope: at most 32 raw samples and 2 epochs")
    gpus = tf.config.list_physical_devices("GPU")
    if not test_scope and not gpus:
        raise RuntimeError("GPU required; formal execution has no CPU fallback")
    tf.config.set_soft_device_placement(False)
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return "/CPU:0" if test_scope else "/GPU:0"


def require_gpu_tensor(tensor, formal):
    if formal and "GPU:" not in tensor.device.upper():
        raise RuntimeError("GPU required at forward/loss/gradient seam")
    tf.debugging.assert_all_finite(tensor, "nonfinite P2 tensor")


def tf_state_sha(model):
    h = hashlib.sha256()
    for v in model.weights:
        a = np.asarray(v.numpy())
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


inspect_anchor = contracts.inspect_anchor


def load_anchor(encoder_path, norm_path, expected=None, *, anchor_contract=None, test_scope=False):
    stats, identity = inspect_anchor(encoder_path,norm_path,expected,
                                     anchor_contract=anchor_contract,test_scope=test_scope)
    with contracts.snapshot(encoder_path, identity["encoder"]) as (copy, _):
        model = load_ssl_encoder(str(copy), compile=False)
    if model.get_config() != identity["config"]:
        raise ValueError("deserialized anchor config differs from validated snapshot")
    model.trainable = False
    contracts.verify_sources(contracts.source_refs(identity))
    return model, stats, identity


def encode_bands(model, raw, stats, batch_size=128, formal=False):
    if batch_size < 1:
        raise ValueError("positive batch_size required")
    before = tf_state_sha(model)
    results = []
    for start in range(0, len(raw), batch_size):
        x = normalize_mbm_inputs(raw[start:start+batch_size], stats)
        embedding = model(tf.constant(x), return_features=True, training=False)
        require_gpu_tensor(embedding, formal)
        results.append(embedding.numpy())
    if before != tf_state_sha(model):
        raise ValueError("frozen band state changed during inference")
    return np.concatenate(results)


def extract_split(pairs, band_encoder, norm, out_dir, split, test_scope=False, batch_size=128, *, anchor_contract=None, run_dir=None):
    device = configure_runtime(test_scope)
    if split not in ("train","test") or (split == "test" and run_dir is None) or (split == "train" and run_dir is not None):
        raise ValueError("test extraction requires a frozen P2 run; train extraction forbids a run")
    frozen = contracts.verify_selection(run_dir) if split == "test" else None
    if frozen is not None and frozen["test_scope"] and not test_scope:
        raise ValueError("test selection cannot authorize formal extraction")
    out = contracts.fresh_directory(out_dir)
    with tf.device(device):
        model, stats, anchor = load_anchor(band_encoder, norm, expected=None if frozen is None else frozen["anchor"],
                                           anchor_contract=anchor_contract, test_scope=test_scope)
    a, parent = contracts.load_pairs(pairs, split, test_scope)
    configure_runtime(test_scope, len(a["material_ids"]))
    if parent["feature_schema"] != anchor["feature_schema"]:
        raise ValueError("raw pair/anchor feature schema mismatch")
    if frozen is not None:
        contracts.reject_outer_overlap(a["material_ids"], a["groups"], frozen)
    else:
        contracts.inner_partition(a["material_ids"], a["groups"], a["valid_idx"],
                                  anchor=anchor, raw_contract=parent["raw_contract"])
    with tf.device(device):
        embs = encode_bands(model, a["band_input"][a["valid"]], stats, batch_size, not test_scope)
    if not np.isfinite(embs).all():
        raise ValueError("nonfinite band embeddings")
    arrays = {k: a[k] for k in ("material_ids", "valid", "valid_idx", "valid_material_ids", "groups")}
    arrays["band_embeddings"] = embs
    sources = contracts.source_refs(parent, anchor, frozen)
    code = contracts.code_refs(Path(__file__), ROOT/"src/data/band_structure_dataset.py",
                               ROOT/"src/models/band_structure_encoder.py", ROOT/"scripts/prepare_p2_pairs.py")
    receipt = {"kind":"band_embeddings", "split":split, "test_scope":bool(test_scope),
               "frozen_run":None if frozen is None else contracts.frozen_run_ref(run_dir, frozen),
               "anchor":anchor, "feature_semantics":contracts.FEATURE_SEMANTICS,
               "input_space":"frozen_mbm_pooled", "parent":parent,
               "sources":{"pairs":sources[Path(pairs)], "pairs_completion":sources[contracts.completion_path(pairs)]},
               "code":code, "runtime":{"python":platform.python_version(), "tensorflow":tf.__version__,
                                         "numpy":np.__version__, "device":device,
                                         "scope":"cpu_test" if test_scope else "formal_gpu"}, "batch_size":batch_size}
    return contracts.save_completed_npz(out/f"band_embeddings_{split}.npz", arrays, receipt, sources)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True)
    p.add_argument("--band-encoder", required=True)
    p.add_argument("--norm", required=True)
    p.add_argument("--anchor-contract", required=True)
    p.add_argument("--run-dir", help="required frozen P2 run for independent test extraction only")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-scope", action="store_true")
    a = p.parse_args(argv)
    print(extract_split(a.pairs, a.band_encoder, a.norm, a.out_dir, a.split, a.test_scope, a.batch_size,
                        anchor_contract=a.anchor_contract, run_dir=a.run_dir))


if __name__ == "__main__":
    main()
