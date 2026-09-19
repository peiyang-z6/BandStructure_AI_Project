"""Derive the three P0 task labels WITHOUT rebuilding tensors (P0A).

v4-v7 tensors are immutable. The three reframed labels are derived post-hoc:
- line_mode_topology: from the frozen tensor's own edge envelopes
  (metal = Fermi crossing on path per manifest audit; direct = argmax(VBM)
  == argmin(CBM) k-index; else indirect).
- provider_global_electronic_type: from provider metadata (is_metal/is_direct).
- line_global_disagreement: binary disagreement between the two.

Output: three_task_labels.npz + three_task_labels_report.json in the
tensor output dir. The original band_tensors_*.npz files are NOT modified.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils.selection_manifest import sha256_file, validate_material_ids, validate_task_labels

from src.data.ood_tensor_builder import (
    derive_line_global_disagreement,
    derive_line_mode_topology,
    derive_provider_global_type,
)


def load_metadata_by_id(metadata_path):
    data = json.load(open(metadata_path, encoding="utf-8"))
    if isinstance(data, dict):
        # support {material_id: record} and {"records": [...]}
        if "records" in data:
            return {str(r["material_id"]): r for r in data["records"]}
        return {str(k): v for k, v in data.items()}
    return {str(r.get("material_id")): r for r in data}


def derive_labels(
    X: np.ndarray,
    material_ids: np.ndarray,
    manifest: dict,
    metadata_by_id: dict,
    *,
    strict: bool = False,
) -> dict:
    if strict:
        validate_material_ids(np.asarray(material_ids).tolist())
        validate_material_ids([s['material_id'] for s in manifest.get('samples', [])], name='manifest IDs')
        if X.ndim != 4 or X.shape[0] != len(material_ids) or X.shape[1] != 2 or X.shape[3] != 3 or not np.all(np.isfinite(X)):
            raise ValueError('Expected finite ID-aligned (N,2,K,3) tensors')
    samples = {s["material_id"]: s for s in manifest.get("samples", [])}
    n = len(X)
    line_topo = np.full(n, -1, dtype=np.int32)
    provider_type = np.full(n, -1, dtype=np.int32)
    disagreement = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        mid = str(material_ids[i])
        smp = samples.get(mid)
        if strict and (smp is None or type(smp.get('metal_feature_inferred')) is not bool):
            raise ValueError(f'{mid}: missing explicit line-mode crossing evidence')
        crossing = bool(smp.get("metal_feature_inferred", False)) if smp else False
        vbm = X[i, 0, :, 0]
        cbm = X[i, 1, :, 0]
        line_topo[i] = derive_line_mode_topology(vbm, cbm, crossing=crossing)

        meta = metadata_by_id.get(mid, {})
        if smp and not meta:
            # fallback: provider type encoded in legacy y_type within manifest
            legacy = int(smp.get("gap_type", -1))
            if legacy >= 0 and smp.get("gap_type_source", "").startswith("provider"):
                provider_type[i] = legacy
        else:
            is_metal = meta.get("is_metal")
            is_direct = meta.get("is_direct")
            if strict and any(v is not None and type(v) is not bool for v in (is_metal, is_direct)):
                raise ValueError(f'{mid}: provider flags must be explicit booleans, not truthy strings')
            provider_type[i] = derive_provider_global_type(
                is_metal=None if is_metal is None else bool(is_metal),
                is_direct=None if is_direct is None else bool(is_direct),
            )
        disagreement[i] = derive_line_global_disagreement(
            int(line_topo[i]), int(provider_type[i])
        )
    labels = {
        "line_mode_topology": line_topo,
        "provider_global_electronic_type": provider_type,
        "line_global_disagreement": disagreement,
    }
    if strict:
        validate_task_labels({k: v.tolist() for k, v in labels.items()}, np.asarray(material_ids).tolist())
    return labels


def derive_split_sidecars(npz_path, manifest_path, metadata_path, output_dir):
    """Preparation only: derive both frozen partitions, never used by train-only.

    Full metadata/audit are read here, before training, not by the trainer. Output
    is a new immutable directory; no existing sidecar or tensor is rewritten.
    """
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing nonempty label output: {out}')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    metadata = load_metadata_by_id(metadata_path)
    partitions = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for split in ('train', 'test'):
            ids = data[f'material_ids_{split}']
            labels = derive_labels(data[f'X_{split}'], ids, manifest, metadata, strict=True)
            partitions[split] = dict(material_ids=ids, schema_version=np.array(2), partition=np.array(split), **labels)
    if set(partitions['train']['material_ids']) & set(partitions['test']['material_ids']):
        raise ValueError('Frozen train/test IDs overlap')
    out.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 2, 'partitions': {}, 'inputs': {}}
    for role, path in [('source_tensor', npz_path), ('manifest', manifest_path), ('metadata', metadata_path), ('derivation_code', __file__)]:
        p = Path(path).resolve()
        report['inputs'][role] = {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    for split, payload in partitions.items():
        path = out / f'three_task_labels_{split}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, **payload)
        report['partitions'][split] = {'count': len(payload['material_ids']), 'path': str(path.resolve()),
                                      'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
    report['provider_fallback_policy'] = 'Explicit provider-sourced manifest gap_type only when metadata absent; never line-mode substitution'
    with (out / 'three_task_labels_report.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive P0 three-task labels")
    parser.add_argument("--npz", required=True, help="frozen band_tensors_ood_split.npz (default) or legacy full NPZ")
    parser.add_argument("--manifest", required=True, help="ood_split_manifest.json path")
    parser.add_argument("--metadata", required=True, help="aflow_metadata.json path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--legacy-full", action="store_true", help="explicit diagnostic-only full-sidecar derivation; output must be new")
    args = parser.parse_args()

    if not args.legacy_full:
        print(json.dumps(derive_split_sidecars(args.npz, args.manifest, args.metadata, args.output_dir), indent=2))
        return
    out = Path(args.output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing nonempty label output: {out}')
    out.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz)
    X = data["X"]
    material_ids = data["material_ids"].astype(str)
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    metadata_by_id = load_metadata_by_id(args.metadata)

    labels = derive_labels(X, material_ids, manifest, metadata_by_id)
    out_npz = os.path.join(args.output_dir, "three_task_labels.npz")
    np.savez_compressed(
        out_npz,
        material_ids=material_ids,
        line_mode_topology=labels["line_mode_topology"],
        provider_global_electronic_type=labels["provider_global_electronic_type"],
        line_global_disagreement=labels["line_global_disagreement"],
    )

    report = {}
    for key, arr in labels.items():
        valid = arr[arr >= 0]
        report[key] = {
            "coverage": int(np.sum(arr >= 0)),
            "total": int(len(arr)),
            "distribution": {
                str(k): int(v) for k, v in zip(*np.unique(valid, return_counts=True))
            },
        }
    report["agreement_note"] = (
        "line_global_disagreement == 1 iff line_mode_topology != "
        "provider_global_electronic_type (samples where both labels exist)"
    )
    report["inputs"] = {
        "npz": args.npz,
        "manifest": args.manifest,
        "metadata": args.metadata,
    }
    with open(os.path.join(args.output_dir, "three_task_labels_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("three-task label derivation complete")
    for key, sub in report.items():
        if isinstance(sub, dict) and "distribution" in sub:
            print(f"  {key}: coverage {sub['coverage']}/{sub['total']} dist {sub['distribution']}")
    print(f"  wrote {out_npz}")


if __name__ == "__main__":
    main()
