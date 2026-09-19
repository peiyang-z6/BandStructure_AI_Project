"""Selection-manifest validation shared by the pipeline and the finetune CLI.

This module intentionally has no heavy dependencies (no TensorFlow, no
matplotlib). The full pipeline imports it directly so that a `scripts`
package name collision elsewhere on the import path cannot break the final
artifact content gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import math
from pathlib import Path
from typing import Dict


TASK_DIMENSIONS = {
    "line_mode_topology": 3,
    "provider_global_electronic_type": 3,
    "line_global_disagreement": 2,
}
TASK_HEADS = dict(zip(TASK_DIMENSIONS, ("topology", "type", "disagreement")))


def validate_new_run_outputs(experiment_id, seed, output_dir, checkpoint_dir, model_path):
    """Read-only preflight; never clean or repair historical run directories."""
    if (not isinstance(experiment_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', experiment_id)
            or re.search(r'(^|[_-])v[4-7]([_-]|$)', experiment_id)
            or type(seed) is not int or seed not in (42, 2024, 7)
            or re.findall(r'(?:^|[_-])seed(\d+)(?=$|[_-])', experiment_id) != [str(seed)]):
        raise ValueError('A new nonlegacy experiment ID with the matching seed42/seed2024/seed7 token is required')
    directories = [Path(output_dir).resolve(), Path(checkpoint_dir).resolve(), Path(model_path).resolve().parent]
    for directory in directories:
        if experiment_id not in directory.parts:
            raise ValueError(f'Output path belongs to a different run: {directory}')
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError(f'Refusing nonempty run output: {directory}')
    if len(set(directories)) != 3 or any(a in b.parents for a in directories for b in directories if a != b):
        raise ValueError('Report, checkpoint and model directories must be distinct and nonnested')


def validate_material_ids(ids, *, name="material_ids"):
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x.strip() for x in ids):
        raise ValueError(f"{name} must be a nonempty list of material IDs")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} has duplicate IDs")
    return ids


def validate_task_labels(labels, material_ids):
    """Validate complete integer supervision, without importing array/ML libraries."""
    validate_material_ids(material_ids)
    for task, size in TASK_DIMENSIONS.items():
        values = labels.get(task)
        if not isinstance(values, list) or len(values) != len(material_ids):
            raise ValueError(f"{task} labels missing or not ID-aligned")
        if any(type(value) is not int or not 0 <= value < size for value in values):
            raise ValueError(f"{task} requires complete integer labels in [0, {size})")
    topo, provider, disagreement = (labels[task] for task in TASK_DIMENSIONS)
    if any(d != int(t != p) for t, p, d in zip(topo, provider, disagreement)):
        raise ValueError("line_global_disagreement is inconsistent with source task labels")


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_state_path(manifest: Dict[str, object], label: str) -> Path:
    states = manifest.get("states")
    if not isinstance(states, dict) or not isinstance(states.get(label), dict):
        raise RuntimeError(f"Inner selection manifest missing {label} state")
    resolved = manifest.get('_validated_state_paths', {})
    path = Path(str(resolved.get(label, states[label].get("path", ""))))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def validate_artifact_record(item, role):
    if not isinstance(item, dict) or not isinstance(item.get('path'), str) or not item['path']:
        raise RuntimeError(f'Missing artifact path for {role}')
    path = Path(item['path']).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f'Missing or empty artifact {role}: {path}')
    if type(item.get('bytes')) is not int or path.stat().st_size != item['bytes']:
        raise RuntimeError(f'Artifact {role} size mismatch')
    if sha256_file(path) != item.get('sha256'):
        raise RuntimeError(f'Artifact {role} hash mismatch')
    return path


def validate_training_contract(contract):
    if not isinstance(contract, dict):
        raise RuntimeError('Formal schema 2 requires training_contract')
    fit = validate_material_ids(contract.get('fit_material_ids'), name='fit_material_ids')
    selection = validate_material_ids(contract.get('selection_material_ids'), name='selection_material_ids')
    if set(fit) & set(selection):
        raise RuntimeError('Fit and selection IDs overlap')
    for scope, ids in [('fit', fit), ('selection', selection)]:
        groups = contract.get(f'{scope}_groups')
        if not isinstance(groups, list) or len(groups) != len(ids) or any(type(g) is not int or not 1 <= g <= 230 for g in groups):
            raise RuntimeError(f'{scope}_groups must be sample-aligned spacegroups')
    if set(contract['fit_groups']) & set(contract['selection_groups']):
        raise RuntimeError('Fit and selection groups overlap')
    if contract.get('task_names') != list(TASK_DIMENSIONS) or contract.get('task_dimensions') != TASK_DIMENSIONS:
        raise RuntimeError('Formal three-task names/dimensions mismatch')
    shape = contract.get('input_shape')
    if not isinstance(shape, list) or len(shape) != 2 or type(shape[0]) is not int or shape[0] < 2 or shape[1] != 6:
        raise RuntimeError('Missing 6D sequence input_shape')
    if contract.get('normalization_source') != 'inner_fit' or contract.get('normalization_fit_material_ids') != fit:
        raise RuntimeError('Normalization must be bound to actual inner-fit IDs')
    artifacts = contract.get('artifacts')
    required = ('source_tensor', 'train_labels', 'encoder', 'normalization', 'trainer_code')
    if not isinstance(artifacts, dict) or any(role not in artifacts for role in required):
        raise RuntimeError('Missing required training artifacts')
    for role, item in artifacts.items():
        validate_artifact_record(item, role)
    norm = json.loads(Path(artifacts['normalization']['path']).read_text(encoding='utf-8'))
    if (norm.get('source') != 'inner_fit' or norm.get('fit_material_ids') != fit
            or norm.get('fit_groups') != contract['fit_groups']):
        raise RuntimeError('Actual normalization artifact is not bound to inner-fit IDs/groups')
    for field in ('mean', 'std'):
        values = norm.get(field)
        if (not isinstance(values, list) or len(values) != 6
                or any(type(v) not in (int, float) or not math.isfinite(v) or (field == 'std' and v <= 0) for v in values)):
            raise RuntimeError(f'Invalid actual normalization {field}')
    return contract


def validate_inner_selection_manifest(output_dir: str, *, mode: str = 'formal', state_path_map: dict | None = None) -> Dict[str, object]:
    """Fail closed before outer evaluation unless all frozen states still match."""
    manifest_path = Path(output_dir) / "inner_selection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing inner selection manifest before outer evaluation: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if '_validated_state_paths' in manifest:
        raise RuntimeError('Runtime-resolved paths cannot be supplied by a stored manifest')
    if state_path_map is not None:
        if mode != 'legacy' or manifest.get('schema_version') != 1:
            raise ValueError('Explicit state relocation is only available in legacy schema1 evidence mode')
        if not isinstance(state_path_map, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in state_path_map.items()):
            raise ValueError('Legacy state_path_map must map exact original paths to recovered paths')
        states = manifest.get('states', {})
        manifest['_validated_state_paths'] = {}
        for label in ('best', 'last', 'accepted'):
            original = states.get(label, {}).get('path')
            if original not in state_path_map:
                raise ValueError(f'Legacy recovery map missing {label} original path')
            manifest['_validated_state_paths'][label] = state_path_map[original]
    if mode not in ('formal', 'legacy'):
        raise ValueError('Selection validation mode must be formal or legacy')
    schema = manifest.get('schema_version')
    if schema == 1:
        if mode != 'legacy':
            raise RuntimeError('Schema 1 requires explicit legacy validation; not a formal three-task run')
    elif schema == 2:
        validate_training_contract(manifest.get('training_contract'))
        seed, experiment_id = manifest.get('seed'), manifest.get('experiment_id')
        if (type(seed) is not int or seed not in (42, 2024, 7) or not isinstance(experiment_id, str)
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', experiment_id)
                or re.findall(r'(?:^|[_-])seed(\d+)(?=$|[_-])', experiment_id) != [str(seed)]):
            raise RuntimeError('Formal experiment ID/seed mismatch')
        epoch, count = manifest.get('best_epoch_one_based'), manifest.get('epochs_recorded')
        if type(epoch) is not int or type(count) is not int or not 1 <= epoch <= count:
            raise RuntimeError('Missing valid aggregate inner monitor selection epoch')
    else:
        raise RuntimeError('Unsupported selection schema_version')
    if manifest.get("outer_test_accessed") is not False:
        raise RuntimeError("Inner selection manifest is not blind to outer test")
    if manifest.get("monitor") != "val_loss":
        raise RuntimeError("Inner selection manifest does not use aggregate val_loss")
    states = manifest.get("states")
    if not isinstance(states, dict):
        raise RuntimeError("Inner selection manifest has no frozen states")
    resolved_state_paths = []
    for label in ("best", "last", "accepted"):
        item = states.get(label)
        if not isinstance(item, dict):
            raise RuntimeError(f"Inner selection manifest missing {label} state")
        path = manifest_state_path(manifest, label)
        resolved_state_paths.append(path)
        if not path.is_file():
            raise RuntimeError(f"Frozen {label} state is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != item.get("sha256"):
            raise RuntimeError(
                f"Frozen {label} state hash mismatch: {actual_hash} != {item.get('sha256')}"
            )
        if path.stat().st_size != int(item.get("bytes", -1)):
            raise RuntimeError(f"Frozen {label} state size mismatch: {path}")
    if len(set(resolved_state_paths)) != 3:
        raise RuntimeError(
            "Frozen supervised best, last, and accepted paths are not distinct"
        )
    if schema == 2 and (states['accepted'].get('source') != 'restored_best'
                        or states['accepted']['sha256'] != states['best']['sha256']):
        raise RuntimeError('Formal accepted state is not an exact frozen best copy')
    return manifest


# P2 upstream namespace: intentionally independent of P0/P3 selection schemas.
P2_RAW_CODE = ('src/data/ood_tensor_builder.py', 'scripts/build_ood_tensors.py',
               'src/utils/selection_manifest.py')
P2_SSL_CODE = ('scripts/train_ssl.py', 'src/engine/ssl_trainer.py',
               'src/models/band_structure_encoder.py', 'src/models/losses.py',
               'src/data/ood_tensor_builder.py', 'src/data/band_structure_dataset.py',
               'src/utils/selection_manifest.py', 'src/utils/__init__.py')
P2_FEATURE_ORDER = ['VBM_E', 'VBM_curv', 'VBM_k_dist', 'CBM_E', 'CBM_curv', 'CBM_k_dist']
P2_RAW_SEMANTICS = {'energy': 'E-E_F', 'normalization': 'none',
                    'bands': 'label_free_occupied_empty_edge_envelopes',
                    'interpolation': 'pchip', 'curvature': 'relative_proxy_segment_local',
                    'extremum_distance': 'signed_index_distance_over_K_minus_1'}


def p2_file_ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def p2_code_refs(required):
    root = Path(__file__).resolve().parents[2]
    return {name: p2_file_ref(root / name) for name in required}


def p2_feature_schema(seq_len):
    return {'name': 'mbm_flat6d', 'seq_len': seq_len, 'channels': P2_FEATURE_ORDER,
            'energy_reference': 'fermi_zero', 'k_axis': 'source_cumulative_path_coordinate',
            'segment_policy': 'stored_segment_ids'}


def _p2_contract(path, kind, required, allow_test_scope):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(c, dict) or type(c.get('schema_version')) is not int
            or c['schema_version'] != 1 or c.get('kind') != kind
            or c.get('scope') not in ('formal', 'test') or any(k not in c for k in required)):
        raise ValueError(f'Incomplete {kind} schema; legacy evidence cannot be promoted')
    if c['scope'] == 'test' and not allow_test_scope:
        raise ValueError('Synthetic test scope is not formal')
    return c


def _p2_bound_ref(ref, path, role):
    _p2_ref_shape(ref)
    path = _p2_no_link_path(path)
    if ref['path'] != str(path):
        raise ValueError(f'{role} artifact path mismatch')
    try:
        actual = p2_file_ref(path)
    except OSError as exc:
        raise ValueError(f'Missing {role} artifact') from exc
    if ref != actual:
        raise ValueError(f'{role} artifact bytes/path/SHA mismatch')
    return ref


def _p2_no_link_path(path):
    """Check lexical/resolved paths without opening any artifact bytes."""
    path = Path(path).absolute()
    if ('..' in path.parts or path.resolve() != path
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError('Upstream artifact path escape/symlink')
    return path


def validate_raw_tensor_contract(contract_path, *, tensor_path, split,
                                 expected_ids=None, expected_groups=None, allow_test_scope=False,
                                 required_scope=None):
    contract_path = _p2_no_link_path(contract_path)
    tensor_path = _p2_no_link_path(tensor_path)
    c = _p2_contract(contract_path, 'raw_tensor', (
        'input_space', 'feature_schema', 'raw_semantics', 'material_ids', 'groups',
        'tensor_ref', 'source', 'source_qualifications', 'code', 'construction_id'), allow_test_scope)
    if required_scope is not None and (required_scope not in ('test', 'formal') or c['scope'] != required_scope):
        raise ValueError('CPU test scope requires a bounded synthetic tensor contract; required scope mismatch')
    # Admission only: check the untrusted header before NPZ hashing/decoding,
    # then still require the full code/bytes/embedded-observation binding below.
    _p2_rows(c['material_ids'], c['groups'])
    source = c['source']
    if (not isinstance(source, dict) or source.get('kind') != 'new_hdf5_construction'
            or type(source.get('raw_rows')) is not int or source['raw_rows'] < len(c['material_ids'])
            or (c['scope'] == 'test' and source['raw_rows'] > 32)):
        raise ValueError('Invalid new construction source/scope')
    import numpy as np  # lazy; never imports a training framework
    _p2_check_code(c['code'], P2_RAW_CODE)
    _p2_bound_ref(c['tensor_ref'], tensor_path, 'raw tensor')
    if split not in ('train', 'test') or c.get('split') != split:
        raise ValueError('Raw contract split mismatch')
    schema = c['feature_schema']
    if (not isinstance(schema, dict) or type(schema.get('seq_len')) is not int
            or schema['seq_len'] < 2 or schema != p2_feature_schema(schema['seq_len'])
            or c['input_space'] != 'raw_canonical_6d' or c['raw_semantics'] != P2_RAW_SEMANTICS):
        raise ValueError('Ambiguous raw feature semantics or input space')
    _p2_rows(c['material_ids'], c['groups'])
    quals = c['source_qualifications']
    if not isinstance(quals, list) or len(quals) != len(c['material_ids']):
        raise ValueError('Missing row-aligned source qualifications')
    for mid, q in zip(c['material_ids'], quals):
        if (not isinstance(q, dict) or q.get('material_id') != mid
                or q.get('source') not in ('materials_project', 'aflow')
                or q.get('energy_reference') not in ('absolute', 'fermi_shifted_zero')
                or type(q.get('efermi')) not in (int, float) or not math.isfinite(q['efermi'])
                or (q['energy_reference'] == 'fermi_shifted_zero' and q['efermi'] != 0.)
                or q.get('energy_quantity') != 'band_energy_eV'
                or not isinstance(q.get('protocol'), str) or not q['protocol']):
            raise ValueError('Ambiguous energy/Fermi/source/protocol qualification')
    _p2_ref_shape(source.get('h5_ref'))  # identity only: never follow a raw HDF5 provenance path
    with np.load(tensor_path, allow_pickle=False) as z:
        required = ('construction_id', f'input_space_{split}', f'scope_{split}',
                    f'feature_schema_{split}', f'X_{split}', f'segment_ids_{split}',
                    f'material_ids_{split}', f'groups_{split}', f'observation_sha256_{split}')
        if any(key not in z for key in required):
            raise ValueError('Missing actual NPZ construction metadata/arrays')
        if (z['construction_id'].item() != c['construction_id']
                or z[f'scope_{split}'].item() != c['scope']
                or json.loads(z[f'feature_schema_{split}'].item()) != schema):
            raise ValueError('NPZ construction/schema/scope binding mismatch')
        if not np.issubdtype(z[f'groups_{split}'].dtype, np.integer):
            raise ValueError('NPZ spacegroups must be integers, not inferred casts')
        x, segments = z[f'X_{split}'], z[f'segment_ids_{split}']
        if (x.shape != (len(c['material_ids']), 2, schema['seq_len'], 3)
                or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all()
                or np.any(x[:, 0, :, 0] > 1.e-6) or np.any(x[:, 1, :, 0] < -1.e-6)
                or segments.shape != (len(x), schema['seq_len'])
                or not np.issubdtype(segments.dtype, np.integer) or np.any(segments < 0)):
            raise ValueError('Invalid raw canonical tensor/segment semantics')
        for band, extremum in ((0, np.argmax), (1, np.argmin)):
            centers = extremum(x[:, band, :, 0], axis=1)
            distances = (np.arange(schema['seq_len'])[None, :] - centers[:, None]) / (schema['seq_len']-1)
            if not np.allclose(x[:, band, :, 2], distances, rtol=0, atol=1.e-6):
                raise ValueError('Extremum distance is not raw canonical feature space')
        for key in ('input_space', f'input_space_{split}'):
            if key in z and z[key].item() != 'raw_canonical_6d':
                raise ValueError('Normalized NPZ cannot be relabelled raw')
        if z[f'observation_sha256_{split}'].item() != p2_raw_observation_sha(source, quals):
            raise ValueError('Source qualifications differ from actual builder observations')
        if (z[f'material_ids_{split}'].tolist() != c['material_ids']
                or z[f'groups_{split}'].tolist() != c['groups']):
            raise ValueError('Raw IDs/groups do not match actual split rows')
    if expected_ids is not None and list(expected_ids) != c['material_ids']:
        raise ValueError('Expected material IDs/order mismatch')
    if expected_groups is not None and list(expected_groups) != c['groups']:
        raise ValueError('Expected row-aligned groups mismatch')
    c['contract_ref'] = p2_file_ref(contract_path)
    return c


def _p2_rows(ids, groups):
    validate_material_ids(ids)
    if (not isinstance(groups, list) or len(groups) != len(ids)
            or any(type(g) is not int or not 1 <= g <= 230 for g in groups)):
        raise ValueError('Groups must be row-aligned integer spacegroups')


def _p2_ref_shape(ref):
    if (not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not ref['path']
            or type(ref.get('bytes')) is not int or ref['bytes'] <= 0
            or not isinstance(ref.get('sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', ref['sha256']) is None):
        raise ValueError('Missing artifact identity declaration')


def _p2_check_code(refs, required):
    # Reject keys/paths BEFORE any open. No traversal, absolute paths, arbitrary
    # upstream provenance, or a different stage's smaller code set is accepted.
    if not isinstance(refs, dict) or set(refs) != set(required):
        raise ValueError('Required stage-specific code set mismatch')
    root = Path(__file__).resolve().parents[2]
    for name in required:
        target = root / name
        ref = refs[name]
        _p2_ref_shape(ref)
        if target.resolve() != target or ref['path'] != str(target):
            raise ValueError('Code path escape/mismatch')
        _p2_bound_ref(ref, target, 'code')


def validate_ssl_anchor_contract(contract_path, *, encoder_path, norm_path, allow_test_scope=False):
    c = _p2_contract(contract_path, 'ssl_anchor', (
        'feature_schema', 'encoder_ref', 'norm_ref', 'run_id', 'material_ids', 'groups',
        'fit_material_ids', 'fit_groups', 'selection_material_ids', 'selection_groups',
        'source_tensor_ref', 'source_tensor_contract_ref', 'history', 'history_ref', 'best_epoch',
        'completed_epochs', 'optimizer_iterations', 'states', 'checkpoints', 'code',
        'config_ref', 'runtime', 'outer_test_accessed', 'selection_monitor', 'accepted_source'), allow_test_scope)
    _p2_rows(c['material_ids'], c['groups'])
    _p2_rows(c['fit_material_ids'], c['fit_groups'])
    _p2_rows(c['selection_material_ids'], c['selection_groups'])
    _p2_check_code(c['code'], P2_SSL_CODE)
    _p2_bound_ref(c['encoder_ref'], encoder_path, 'encoder')
    _p2_bound_ref(c['norm_ref'], norm_path, 'SSL anchor normalization')
    _p2_ssl_training_evidence(c, Path(encoder_path).resolve().parent)
    import numpy as np
    if not isinstance(c['run_id'], str) or re.fullmatch('[0-9a-f]{32}', c['run_id']) is None:
        raise ValueError('Invalid producer run ID')
    source_ref, source_contract_ref = c['source_tensor_ref'], c['source_tensor_contract_ref']
    _p2_ref_shape(source_ref)
    _p2_ref_shape(source_contract_ref)
    source_path = Path(source_ref['path'])
    # Only the canonical train receipt adjacent to a constructed NPZ is followed.
    # Raw HDF5 and arbitrary/outer provenance paths are never opened by this API.
    if (not source_path.is_absolute() or source_path.name != 'band_tensors_ood_split.npz'
            or Path(source_contract_ref['path']) != source_path.with_name('band_tensors_ood_split.train.tensor_contract.json')):
        raise ValueError('Invalid upstream train tensor/contract path')
    # Check BOTH paths before following either receipt or NPZ (also parent links).
    _p2_no_link_path(source_path)
    _p2_no_link_path(source_contract_ref['path'])
    _p2_bound_ref(source_contract_ref, source_contract_ref['path'], 'source tensor contract')
    raw = validate_raw_tensor_contract(source_contract_ref['path'], tensor_path=source_path,
                                      split='train', expected_ids=c['material_ids'],
                                      expected_groups=c['groups'], allow_test_scope=allow_test_scope)
    if c['scope'] != raw['scope'] or c['feature_schema'] != raw['feature_schema']:
        raise ValueError('Anchor/raw schema or scope mismatch')
    _p2_bound_ref(source_ref, source_path, 'source tensor')
    for pool in ('fit', 'selection'):
        ids, groups = c[f'{pool}_material_ids'], c[f'{pool}_groups']
        _p2_rows(ids, groups)
        mapping = dict(zip(c['material_ids'], c['groups']))
        if any(mapping.get(mid) != group for mid, group in zip(ids, groups)):
            raise ValueError('Allocation groups differ from actual raw rows')
        if [mid for mid in c['material_ids'] if mid in set(ids)] != ids:
            raise ValueError('Allocation order differs from actual raw rows')
    if (set(c['fit_material_ids']) & set(c['selection_material_ids'])
            or set(c['fit_groups']) & set(c['selection_groups'])
            or set(c['fit_material_ids'] + c['selection_material_ids']) != set(c['material_ids'])):
        raise ValueError('SSL allocation has overlap or incomplete coverage')
    norm = json.loads(Path(norm_path).read_text(encoding='utf-8'))
    if (norm.get('source') != 'inner_fit' or norm.get('run_id') != c['run_id']
            or norm.get('encoder_ref') != c['encoder_ref']
            or norm.get('fit_material_ids') != c['fit_material_ids']
            or norm.get('fit_groups') != c['fit_groups']
            or norm.get('source_tensor_ref') != source_ref
            or norm.get('source_tensor_contract_ref') != source_contract_ref):
        raise ValueError('Normalization is not bound to this encoder/run/actual fit pool')
    for field in ('mean', 'std'):
        values = norm.get(field)
        if (not isinstance(values, list) or len(values) != 6
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       or (field == 'std' and v <= 0) for v in values)):
            raise ValueError('Invalid normalization statistics')
    with np.load(source_path, allow_pickle=False) as z:
        ids = z['material_ids_train'].tolist()
        row_index = {mid: i for i, mid in enumerate(ids)}
        fit = [row_index[mid] for mid in c['fit_material_ids']]
        x = z['X_train'][fit].astype(np.float32).transpose(0, 2, 1, 3)
        x = x.reshape(len(fit), c['feature_schema']['seq_len'], 6)
        mean, std = x.mean(axis=(0, 1)), np.maximum(x.std(axis=(0, 1)), 1.e-6)
    if not np.array_equal(mean, norm['mean']) or not np.array_equal(std, norm['std']):
        raise ValueError('Norm was not fitted on the recorded actual inner-fit tensors')
    _p2_encoder_origin(c, norm)
    c['contract_ref'] = p2_file_ref(contract_path)
    return c


def _p2_local_ref(ref, directory, name):
    _p2_ref_shape(ref)
    target = directory / name
    if target.resolve() != target or ref['path'] != str(target):
        raise ValueError('Run artifact path escape/mismatch')
    return _p2_bound_ref(ref, target, name)


def validate_p2_mask_configuration(seq_len, min_span, max_span, mask_ratio):
    """Sufficient prefix-search guarantee, NOT a claim of necessity/feasibility.

    Every position is a start, so union coverage eventually reaches K. Each
    prefix adds <= max_span positions (even after segment truncation/overlap).
    If that increment fits the inclusive budget window it cannot jump over it.
    Short K may have valid non-prefix masks yet be unsupported by this proof.
    """
    import numpy as np
    if (type(seq_len) is not int or seq_len < 2
            or type(min_span) is not int or type(max_span) is not int
            or not 5 <= min_span <= max_span <= 15
            or type(mask_ratio) not in (int, float) or not .15 <= mask_ratio <= .30):
        raise ValueError('Unsupported mask/span configuration; require 5-15 spans and 15-30% ratio')
    # Match the unchanged trainer's float32 ceil/floor, not float64 arithmetic.
    minimum = math.ceil(float(np.float32(seq_len)*np.float32(.15)))
    maximum = math.floor(float(np.float32(seq_len)*np.float32(.30)))
    if minimum > maximum:
        raise ValueError('Infeasible integer mask budget for this K')
    if max_span > maximum-minimum+1:
        raise ValueError('Unsupported mask prefix-search configuration; no per-sample budget guarantee')


def validate_p2_ssl_history(history, *, expected_consistency_weight):
    """Post-run safety gate on actual history; never infer per-sample coverage.

    The configuration proof supplies that guarantee. Epoch averages must still
    pass, and the original pipeline's disabled-consistency gate is retained.
    """
    if (type(expected_consistency_weight) not in (int, float)
            or not math.isfinite(expected_consistency_weight) or expected_consistency_weight < 0):
        raise ValueError('Missing finite configured consistency weight')
    if not isinstance(history, dict) or history.get('selection_monitor') != 'val_total':
        raise ValueError('Missing actual SSL selection history')
    epochs = history.get('epochs')
    if not isinstance(epochs, list) or not epochs:
        raise ValueError('Missing actual SSL epochs')
    for epoch in epochs:
        for pool in ('train', 'val'):
            fraction = epoch.get(pool, {}).get('mask_fraction')
            if type(fraction) not in (int, float) or not .15 <= fraction <= .30:
                raise ValueError(f'SSL {pool} mask fraction outside 15-30%: {fraction}')
        for key in ('selection_weights', 'post_adaptation_weights'):
            weights = epoch.get(key)
            if (not isinstance(weights, dict)
                    or any(type(weights.get(k)) not in (int, float) or not math.isfinite(weights[k])
                           or weights[k] < 0 for k in ('curvature', 'symmetry'))):
                raise ValueError('Missing finite pre/post adaptation weights')
            if expected_consistency_weight == 0. and abs(weights['symmetry']) > 1e-12:
                raise ValueError('SSL consistency weight revived from zero')


def _p2_ssl_training_evidence(c, directory):
    import numpy as np
    if c['outer_test_accessed'] is not False or c['selection_monitor'] != 'val_total':
        raise ValueError('SSL selection is not blind inner val_total')
    if (type(c['completed_epochs']) is not int or c['completed_epochs'] < 1
            or type(c['best_epoch']) is not int or not 1 <= c['best_epoch'] <= c['completed_epochs']
            or type(c['optimizer_iterations']) is not int or c['optimizer_iterations'] <= 0):
        raise ValueError('Missing actual epochs/optimizer updates')
    _p2_local_ref(c['history_ref'], directory, 'ssl_history.json')
    _p2_local_ref(c['config_ref'], directory, 'ssl_run_config.json')
    history = json.loads((directory / 'ssl_history.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'ssl_run_config.json').read_text(encoding='utf-8'))
    if history != c['history'] or history.get('schema_version') != 1 or history.get('selection_monitor') != 'val_total':
        raise ValueError('Contract lacks the actual trainer history')
    epochs = history.get('epochs')
    if not isinstance(epochs, list) or len(epochs) != c['completed_epochs']:
        raise ValueError('Missing complete trainer history')
    validate_p2_ssl_history(history, expected_consistency_weight=config.get('consistency_weight'))
    schema = c['feature_schema']
    validate_p2_mask_configuration(schema.get('seq_len') if isinstance(schema, dict) else None,
                                   config.get('min_span'), config.get('max_span'), config.get('mask_ratio'))
    if (type(config.get('epochs')) is not int or config['epochs'] < len(epochs)
            or type(config.get('batch_size')) is not int or config['batch_size'] < 1
            or type(config.get('early_stopping_min_delta')) not in (int, float)
            or not math.isfinite(config['early_stopping_min_delta']) or config['early_stopping_min_delta'] < 0):
        raise ValueError('Missing actual run configuration')
    runtime = c['runtime']
    if (not isinstance(runtime, dict) or type(config.get('test_scope')) is not bool
            or config['test_scope'] != (c['scope'] == 'test')
            or (c['scope'] == 'test' and config['epochs'] > 2)
            or type(runtime.get('required_gpu')) is not bool
            or config.get('require_gpu') != runtime['required_gpu']
            or not isinstance(runtime.get('backward_devices'), list) or not runtime['backward_devices']):
        raise ValueError('Missing or contradictory scope/runtime evidence')
    devices = [runtime.get('embedding_device'), runtime.get('reconstruction_device'), *runtime['backward_devices']]
    if any(not isinstance(d, str) or '/DEVICE:' not in d.upper() for d in devices):
        raise ValueError('Missing actual forward/backward devices')
    if c['scope'] == 'formal' and (not runtime['required_gpu'] or any('/DEVICE:GPU:' not in d.upper() for d in devices)):
        raise ValueError('CPU execution cannot be promoted to formal')
    previous, selected = float('inf'), None
    for index, epoch in enumerate(epochs, 1):
        if type(epoch.get('epoch')) is not int or epoch['epoch'] != index or type(epoch.get('improved')) is not bool:
            raise ValueError('Noncontiguous or ambiguous history')
        weights = epoch.get('selection_weights', {})
        if any(type(weights.get(k)) not in (int, float) or not math.isfinite(weights[k]) or weights[k] < 0
               for k in ('curvature', 'symmetry')):
            raise ValueError('Missing real selection loss weights')
        for pool in ('train', 'val'):
            metrics = epoch.get(pool, {})
            keys = ('total', 'mse_loss', 'masked_mae', 'mask_fraction', 'curvature_loss', 'symmetry_loss')
            if any(type(metrics.get(k)) not in (int, float) or not math.isfinite(metrics[k]) or metrics[k] < 0 for k in keys):
                raise ValueError('Missing finite actual training/selection metrics')
            total = metrics['mse_loss'] + weights['curvature']*metrics['curvature_loss'] + weights['symmetry']*metrics['symmetry_loss']
            if not np.isclose(total, metrics['total'], rtol=1.e-6, atol=1.e-7):
                raise ValueError('History monitor differs from actual loss components')
        improved = epoch['val']['total'] < previous - config['early_stopping_min_delta']
        if epoch['improved'] != improved:
            raise ValueError('History selection decision is inconsistent')
        if improved:
            previous, selected = float(np.float32(epoch['val']['total'])), index
    if selected != c['best_epoch']:
        raise ValueError('Best epoch does not match actual inner history')
    steps = (len(c['fit_material_ids']) + config['batch_size'] - 1)//config['batch_size']
    if c['optimizer_iterations'] != len(epochs)*steps:
        raise ValueError('Optimizer updates do not cover actual fit batches')
    states = c['states']
    if not isinstance(states, dict) or set(states) != {'best', 'last', 'accepted'}:
        raise ValueError('Missing frozen SSL states')
    for key, name in (('best', 'ssl_mbm_best.keras'), ('last', f"ssl_mbm_final_epoch{len(epochs)}.keras"),
                      ('accepted', 'ssl_mbm_pretrained.keras')):
        _p2_local_ref(states[key], directory, name)
    if (c['accepted_source'] != 'restored_best_exact_copy' or states['accepted'] != c['encoder_ref']
            or states['accepted']['sha256'] != states['best']['sha256']):
        raise ValueError('Accepted encoder is not the exact restored best copy')
    checkpoints = c['checkpoints']
    if not isinstance(checkpoints, dict) or set(checkpoints) != {'best', 'last'}:
        raise ValueError('Missing actual TF checkpoint evidence')
    for state, refs in checkpoints.items():
        names = {f'ckpt-{state}.index', f'ckpt-{state}.data-00000-of-00001'}
        if not isinstance(refs, list) or len(refs) != 2:
            raise ValueError('Incomplete actual TF checkpoints')
        if {Path(ref.get('path', '')).name for ref in refs} != names:
            raise ValueError('Unexpected TF checkpoint set')
        for ref in refs:
            _p2_local_ref(ref, directory, Path(ref['path']).name)


def _p2_encoder_origin(c, norm):
    import zipfile
    keys = ('scope', 'run_id', 'feature_schema', 'material_ids', 'groups',
            'fit_material_ids', 'fit_groups', 'selection_material_ids', 'selection_groups',
            'source_tensor_ref', 'source_tensor_contract_ref', 'history_ref', 'config_ref',
            'optimizer_iterations', 'checkpoints', 'code', 'runtime')
    for state in ('best', 'last'):
        try:
            with zipfile.ZipFile(c['states'][state]['path']) as archive:
                for member in ('assets/ssl_origin.json', 'model.weights.h5'):
                    if archive.namelist().count(member) != 1:
                        raise ValueError('Missing unique native encoder training origin/weights')
                origin = json.loads(archive.read('assets/ssl_origin.json'))
                weight_sha = hashlib.sha256(archive.read('model.weights.h5')).hexdigest()
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError('Encoder lacks actual native training origin') from exc
        if (origin.get('kind') != 'observed_ssl_training' or type(origin.get('schema_version')) is not int
                or origin['schema_version'] != 1 or origin.get('state') != state
                or origin.get('keras_weights_sha256') != weight_sha
                or any(origin.get(key) != c[key] for key in keys)
                or origin.get('normalization') != {key: norm[key] for key in ('mean', 'std')}):
            raise ValueError('Encoder weights do not belong to this observed SSL run/norm/history')
        for key in ('initial_state_sha256', 'state_sha256'):
            if not isinstance(origin.get(key), str) or re.fullmatch('[0-9a-f]{64}', origin[key]) is None:
                raise ValueError('Missing observed optimization state')
        if origin['initial_state_sha256'] == origin['state_sha256']:
            raise ValueError('Encoder has no observed parameter update')


def p2_raw_observation_sha(source, qualifications):
    payload = json.dumps({'source': source, 'source_qualifications': qualifications},
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
