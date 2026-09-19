"""Strict P0 entry contracts; all data are synthetic and CPU bounded."""
import json
from pathlib import Path

import numpy as np
import pytest

TASKS = ('line_mode_topology', 'provider_global_electronic_type', 'line_global_disagreement')


def tiny_split(tmp_path):
    rng = np.random.default_rng(42)
    payload = {}
    for split, n in [('train', 12), ('test', 4)]:
        x = rng.normal(size=(n, 2, 8, 3)).astype('float32')
        x[:, 0, :, 0] -= 2
        x[:, 1, :, 0] += 2
        payload.update({f'X_{split}': x, f'y_{split}': np.ones(n, 'float32'),
                        f'y_type_{split}': np.arange(n, dtype='int32') % 3,
                        f'material_ids_{split}': np.array([f'{split}-{i}' for i in range(n)]),
                        f'groups_{split}': np.repeat(np.arange(n // 2), 2) + (1 if split == 'train' else 101),
                        f'segment_ids_{split}': np.zeros((n, 8), 'int32')})
    npz = tmp_path / 'split.npz'
    np.savez(npz, **payload)
    norm = tmp_path / 'norm.json'
    norm.write_text(json.dumps({'mean': [0.] * 6, 'std': [1.] * 6}))
    return npz, norm, payload


def test_strict_loader_refuses_missing_labels_instead_of_zero_loss(tmp_path):
    from scripts.finetune_supervised import load_dataset
    npz, norm, _ = tiny_split(tmp_path)
    with pytest.raises(ValueError, match='three_task_labels_train'):
        load_dataset(str(npz), str(norm), include_outer_test=False, strict=True)


def label_payload(ids, partition='train'):
    n = len(ids)
    topo = np.arange(n, dtype='int32') % 3
    provider = (topo + 1) % 3
    return dict(material_ids=ids, schema_version=np.array(2), partition=np.array(partition),
                line_mode_topology=topo, provider_global_electronic_type=provider,
                line_global_disagreement=(topo != provider).astype('int32'))


@pytest.mark.parametrize('defect', ['duplicate', 'extra', 'missing_task', 'invalid', 'float', 'shape', 'inconsistent', 'partition'])
def test_strict_sidecar_requires_total_bijective_valid_labels(tmp_path, defect):
    from scripts.finetune_supervised import load_three_task_labels
    ids = np.array(['a', 'b', 'c'])
    labels = label_payload(ids.copy())
    if defect == 'duplicate': labels['material_ids'][1] = 'a'
    if defect == 'extra': labels = label_payload(np.array(['a', 'b', 'c', 'd']))
    if defect == 'missing_task': del labels[TASKS[0]]
    if defect == 'invalid': labels[TASKS[0]][0] = -1
    if defect == 'float': labels[TASKS[0]] = np.array([0., .5, 2.])
    if defect == 'shape': labels[TASKS[0]] = labels[TASKS[0]][:, None]
    if defect == 'inconsistent': labels[TASKS[2]][0] = 0
    if defect == 'partition': labels['partition'] = np.array('test')
    np.savez(tmp_path / 'three_task_labels_train.npz', **labels)
    with pytest.raises(ValueError):
        load_three_task_labels(str(tmp_path / 'split.npz'), ids, partition='train')


def test_train_only_never_opens_full_sidecar_or_outer_arrays(tmp_path, monkeypatch):
    from scripts.finetune_supervised import load_dataset
    npz, norm, payload = tiny_split(tmp_path)
    labels = label_payload(payload['material_ids_train'])
    np.savez(tmp_path / 'three_task_labels_train.npz', **labels)
    original = np.load
    reads = []
    class Guard:
        def __init__(self, path, *a, **kw):
            assert Path(path).name not in ('three_task_labels.npz', 'three_task_labels_test.npz')
            self.data = original(path, *a, **kw)
        def __enter__(self): return self
        def __exit__(self, *a): self.data.close()
        @property
        def files(self): return self.data.files
        def __contains__(self, k): return k in self.data
        def __getitem__(self, k):
            assert not k.endswith('_test'), f'outer member accessed: {k}'
            reads.append(k)
            return self.data[k]
    monkeypatch.setattr(np, 'load', Guard)
    result = load_dataset(str(npz), str(norm), include_outer_test=False, strict=True)
    np.testing.assert_array_equal(result['type_train'], labels[TASKS[1]])
    assert reads and not any(k.endswith('_test') for k in result)


def test_frozen_split_derivation_writes_separate_immutable_sidecars(tmp_path):
    from scripts import derive_three_task_labels as derive
    npz, _, payload = tiny_split(tmp_path)
    ids = np.concatenate([payload['material_ids_train'], payload['material_ids_test']])
    manifest = tmp_path / 'manifest.json'
    metadata = tmp_path / 'metadata.json'
    manifest.write_text(json.dumps({'samples': [{'material_id': str(mid), 'metal_feature_inferred': False} for mid in ids]}))
    metadata.write_text(json.dumps({str(mid): {'is_metal': False, 'is_direct': True} for mid in ids}))
    out = tmp_path / 'new-label-version'
    before = npz.read_bytes()
    assert hasattr(derive, 'derive_split_sidecars'), 'partitioned label derivation is missing'
    report = derive.derive_split_sidecars(npz, manifest, metadata, out)
    for split in ('train', 'test'):
        with np.load(out / f'three_task_labels_{split}.npz') as data:
            np.testing.assert_array_equal(data['material_ids'], payload[f'material_ids_{split}'])
            assert set(TASKS).issubset(data.files)
        assert report['partitions'][split]['count'] == len(payload[f'X_{split}'])
    assert not (out / 'three_task_labels.npz').exists()
    with pytest.raises(FileExistsError): derive.derive_split_sidecars(npz, manifest, metadata, out)
    assert npz.read_bytes() == before


@pytest.mark.parametrize('bad', ['missing_crossing', 'unknown_provider', 'string_bool', 'duplicate_id'])
def test_strict_derivation_does_not_invent_physical_labels(tmp_path, bad):
    from scripts.derive_three_task_labels import derive_labels
    ids = np.array(['a', 'b'])
    x = np.zeros((2, 2, 8, 3), 'float32')
    manifest = {'samples': [{'material_id': m, 'metal_feature_inferred': False} for m in ids]}
    metadata = {m: {'is_metal': False, 'is_direct': True} for m in ids}
    if bad == 'missing_crossing': del manifest['samples'][0]['metal_feature_inferred']
    if bad == 'unknown_provider': metadata['a'] = {}
    if bad == 'string_bool': metadata['a']['is_metal'] = 'false'
    if bad == 'duplicate_id': ids[1] = 'a'
    with pytest.raises(ValueError): derive_labels(x, ids, manifest, metadata, strict=True)


def tiny_model():
    import tensorflow as tf
    from src.models import SSLEncoder
    from src.engine.finetune_trainer import freeze_encoder_layers
    from scripts.finetune_supervised import SupervisedBandGapModel, compile_model
    tf.keras.utils.set_random_seed(42)
    enc = SSLEncoder(num_features=6, seq_len=8, d_model=8, num_heads=2,
                     num_layers=1, dff=16, projection_dim=4)
    x = tf.constant(np.random.default_rng(1).normal(size=(6, 8, 6)).astype('float32'))
    enc(x, training=False)
    enc.reconstruct(x, training=False)
    freeze_encoder_layers(enc, 0)
    model = SupervisedBandGapModel(enc, np.zeros(6), np.ones(6), d_model=8, dropout=0.)
    model(x, training=False)
    compile_model(model, 1e-3, 1., {0: 1., 1: 1., 2: 1.})
    labels = {'gap': tf.ones((6, 1)), 'type': tf.one_hot([0, 1, 2, 0, 1, 2], 3),
              'topology': tf.one_hot([1, 1, 0, 0, 2, 2], 3),
              'disagreement': tf.one_hot([1, 0, 1, 0, 1, 0], 2)}
    return model, x, labels


@pytest.mark.parametrize('missing', ['topology', 'disagreement'])
def test_actual_train_step_rejects_unsupervised_heads(missing):
    model, x, labels = tiny_model()
    del labels[missing]
    with pytest.raises(ValueError, match='supervision'):
        model.train_step((x, labels))


def test_actual_three_head_training_has_nonzero_losses_and_updates():
    model, x, labels = tiny_model()
    heads = (model.type_head, model.topology_head, model.disagreement_head)
    before = [[v.numpy().copy() for v in h.trainable_variables] for h in heads]
    metrics = model.train_step((x, labels))
    for k in ('type_loss', 'topology_head_loss', 'disagreement_loss'):
        assert np.isfinite(float(metrics[k])) and float(metrics[k]) > 0
    for h, prior in zip(heads, before):
        assert any(not np.array_equal(v.numpy(), b) for v, b in zip(h.trainable_variables, prior))


def test_dataset_entry_rejects_missing_or_invalid_supervision():
    from scripts.finetune_supervised import make_tf_dataset
    x = np.zeros((3, 8, 6), 'float32')
    with pytest.raises(ValueError, match='supervision'):
        make_tf_dataset(x, np.ones(3), np.arange(3), 2, False)
    with pytest.raises(ValueError, match='supervision'):
        make_tf_dataset(x, np.ones(3), np.arange(3), 2, False,
                        y_topology=np.array([-1, 1, 2]), y_disagreement=np.zeros(3, 'int32'))


def manifest_fixture(tmp_path, schema=2):
    from src.utils.selection_manifest import sha256_file
    def artifact(name, content=b'file'):
        p = tmp_path / name
        p.write_bytes(content)
        return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
    manifest = {'schema_version': schema, 'monitor': 'val_loss', 'outer_test_accessed': False,
                'experiment_id': 'p0-contract-seed42', 'seed': 42, 'best_epoch_one_based': 1,
                'epochs_recorded': 1, 'states': {k: artifact(k, b'best' if k != 'last' else b'last') for k in ('best', 'last', 'accepted')}}
    manifest['states']['accepted']['source'] = 'restored_best'
    if schema == 2:
        manifest['training_contract'] = {
            'fit_material_ids': ['a', 'b'], 'selection_material_ids': ['c'],
            'fit_groups': [1, 1], 'selection_groups': [2], 'task_names': list(TASKS),
            'task_dimensions': dict(zip(TASKS, [3, 3, 2])), 'input_shape': [8, 6],
            'normalization_source': 'inner_fit', 'normalization_fit_material_ids': ['a', 'b'],
            'artifacts': {k: artifact(k) for k in ('source_tensor', 'train_labels', 'encoder', 'normalization', 'trainer_code')}}
        manifest['training_contract']['artifacts']['normalization'] = artifact('normalization', json.dumps({
            'schema_version': 2, 'source': 'inner_fit', 'fit_material_ids': ['a', 'b'], 'fit_groups': [1, 1],
            'mean': [0.] * 6, 'std': [1.] * 6}).encode())
    (tmp_path / 'inner_selection_manifest.json').write_text(json.dumps(manifest))
    return manifest


def test_shared_gate_requires_explicit_legacy_mode(tmp_path):
    from src.utils.selection_manifest import validate_inner_selection_manifest
    manifest_fixture(tmp_path, 1)
    with pytest.raises(RuntimeError, match='legacy'):
        validate_inner_selection_manifest(str(tmp_path))
    assert validate_inner_selection_manifest(str(tmp_path), mode='legacy')['schema_version'] == 1


@pytest.mark.parametrize('defect', ['missing_contract', 'overlap_id', 'overlap_group', 'bad_dimension', 'missing_artifact', 'tamper', 'wrong_norm_ids'])
def test_formal_gate_rejects_incomplete_or_mixed_contracts(tmp_path, defect):
    from src.utils.selection_manifest import validate_inner_selection_manifest
    m = manifest_fixture(tmp_path)
    c = m['training_contract']
    if defect == 'missing_contract': del m['training_contract']
    if defect == 'overlap_id': c['selection_material_ids'] = ['a']
    if defect == 'overlap_group': c['selection_groups'] = [1]
    if defect == 'bad_dimension': c['task_dimensions'][TASKS[2]] = 3
    if defect == 'missing_artifact': del c['artifacts']['encoder']
    if defect == 'tamper': (tmp_path / 'normalization').write_bytes(b'other-run')
    if defect == 'wrong_norm_ids': c['normalization_fit_material_ids'] = ['c']
    (tmp_path / 'inner_selection_manifest.json').write_text(json.dumps(m))
    with pytest.raises((RuntimeError, ValueError)):
        validate_inner_selection_manifest(str(tmp_path))


def test_shared_gate_is_lightweight_and_validates_schema2(tmp_path):
    import subprocess, sys
    manifest_fixture(tmp_path)
    code = ('import sys; from src.utils.selection_manifest import validate_inner_selection_manifest; '
            f'assert validate_inner_selection_manifest({str(tmp_path)!r})["schema_version"] == 2; '
            'assert "tensorflow" not in sys.modules; assert "numpy" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('occupied', ['reports', 'checkpoints', 'models'])
def test_formal_training_refuses_nonempty_run_outputs(tmp_path, occupied):
    from src.utils import selection_manifest as sm
    run = tmp_path / 'new-p0-seed42'
    target = run / occupied
    target.mkdir(parents=True)
    (target / 'old-artifact').write_bytes(b'immutable')
    assert hasattr(sm, 'validate_new_run_outputs'), 'immutable run reservation is missing'
    with pytest.raises(FileExistsError):
        sm.validate_new_run_outputs('new-p0-seed42', 42, run / 'reports', run / 'checkpoints', run / 'models' / 'accepted.weights.h5')
    assert (target / 'old-artifact').read_bytes() == b'immutable'


def test_formal_seed_id_and_paths_cannot_mix_runs(tmp_path):
    from src.utils import selection_manifest as sm
    assert hasattr(sm, 'validate_new_run_outputs')
    for name, seed in [('old-v7-seed42', 42), ('new-p0-seed42', 7), ('../new-seed42', 42)]:
        with pytest.raises(ValueError):
            sm.validate_new_run_outputs(name, seed, tmp_path / name / 'r', tmp_path / name / 'c', tmp_path / name / 'm' / 'a.weights.h5')
    for seed in (42, 2024, 7):
        name = f'new-p0-seed{seed}'
        with pytest.raises(ValueError):
            sm.validate_new_run_outputs(name, seed, tmp_path / name / 'r', tmp_path / 'other' / 'c', tmp_path / name / 'm' / 'a.weights.h5')


def tiny_training_args(tmp_path):
    from types import SimpleNamespace
    npz, norm, payload = tiny_split(tmp_path)
    np.savez(tmp_path / 'three_task_labels_train.npz', **label_payload(payload['material_ids_train']))
    model, _, _ = tiny_model()
    encoder = tmp_path / 'tiny-encoder.keras'
    model.encoder.save(encoder)
    run = tmp_path / 'synthetic-p0-seed42'
    return SimpleNamespace(experiment_id=run.name, random_state=42, tensor_npz=str(npz), norm=str(norm),
                           encoder=str(encoder), output_dir=str(run / 'reports'), checkpoint_dir=str(run / 'checkpoints'),
                           model_path=str(run / 'models' / 'accepted.weights.h5'), train_labels=None,
                           test_labels=None, validation_size=.33, batch_size=3, head_d_model=8,
                           freeze_layers=0, epochs=2, warmup_epochs=1, min_lr=1e-6,
                           learning_rate=1e-3, encoder_learning_rate=1e-4, type_weight=1.,
                           topology_weight=.3, entropy_weight=.01, extremum_weight=.5,
                           augment=False, strain_scale=.01, enable_metal_anchor_gate=False,
                           require_gpu=False, fresh=False), payload


def test_real_cpu_training_freezes_actual_inner_contract_without_outer(tmp_path, monkeypatch):
    from scripts import finetune_supervised as ft
    from src.utils.selection_manifest import validate_inner_selection_manifest
    args, payload = tiny_training_args(tmp_path)
    original_get = np.lib.npyio.NpzFile.__getitem__
    reads = []
    def guarded_get(self, key):
        assert not key.endswith('_test'), f'outer access during training: {key}'
        reads.append(key)
        return original_get(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', guarded_get)
    original_open = Path.open
    def guarded_open(self, *a, **kw):
        assert self != Path(args.norm), 'old normalization is not actual inner-fit provenance'
        return original_open(self, *a, **kw)
    monkeypatch.setattr(Path, 'open', guarded_open)
    assert hasattr(ft, 'run_strict_training'), 'formal CPU training seam is missing'
    ft.run_strict_training(args)
    m = validate_inner_selection_manifest(args.output_dir)
    c = m['training_contract']
    ids = payload['material_ids_train'].tolist()
    fit_idx = [ids.index(mid) for mid in c['fit_material_ids']]
    norm = json.loads(Path(c['artifacts']['normalization']['path']).read_text())
    raw_fit = ft.flatten_tensor(payload['X_train'][fit_idx]).astype('float64')
    np.testing.assert_allclose(norm['mean'], raw_fit.mean(axis=(0, 1)), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(norm['std'], np.maximum(raw_fit.std(axis=(0, 1)), 1e-6), rtol=1e-6)
    assert c['fit_groups'] == payload['groups_train'][fit_idx].tolist()
    assert set(c['fit_material_ids']) | set(c['selection_material_ids']) == set(ids)
    assert m['experiment_id'] == args.experiment_id and m['seed'] == 42
    assert m['states']['accepted']['sha256'] == m['states']['best']['sha256']
    history = json.loads((Path(args.output_dir) / 'training_history.json').read_text())
    assert all(history[k][0] > 0 for k in ('type_loss', 'topology_head_loss', 'disagreement_loss'))
    assert reads
    with pytest.raises(FileExistsError): ft.run_strict_training(args)


def test_real_frozen_evaluation_saves_all_probabilities_and_ids(tmp_path):
    from scripts import finetune_supervised as ft
    args, payload = tiny_training_args(tmp_path)
    frozen = ft.run_strict_training(args)
    before = {k: Path(v['path']).read_bytes() for k, v in frozen['states'].items()}
    np.savez(tmp_path / 'three_task_labels_test.npz', **label_payload(payload['material_ids_test'], 'test'))
    args.evaluation_output_dir = str(Path(args.output_dir).parent / 'evaluation')
    args.norm = frozen['training_contract']['artifacts']['normalization']['path']
    args.n_boot = 5
    assert hasattr(ft, 'run_strict_evaluation'), 'frozen evaluation seam is missing'
    result = ft.run_strict_evaluation(args)
    with np.load(result['predictions']['path']) as saved:
        np.testing.assert_array_equal(saved['material_ids'], payload['material_ids_test'])
        assert str(saved['experiment_id']) == args.experiment_id
        for task, n in zip(TASKS, (3, 3, 2)):
            p = saved[f'probabilities_{task}']
            assert p.shape == (4, n) and np.all(np.isfinite(p))
            np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
            assert saved[f'labels_{task}'].shape == (4,)
    assert set(result['three_task_benchmark']) == set(TASKS)
    for k, item in frozen['states'].items(): assert Path(item['path']).read_bytes() == before[k]
    with pytest.raises(FileExistsError): ft.run_strict_evaluation(args)


def test_formal_eval_never_opens_arrays_before_frozen_gate(tmp_path, monkeypatch):
    from scripts import finetune_supervised as ft
    from types import SimpleNamespace
    def prohibited(*a, **kw): raise AssertionError('array I/O occurred before frozen gate')
    monkeypatch.setattr(np, 'load', prohibited)
    assert hasattr(ft, 'run_strict_evaluation')
    with pytest.raises(FileNotFoundError, match='selection manifest'):
        ft.run_strict_evaluation(SimpleNamespace(output_dir=str(tmp_path)))


def test_formal_cli_runs_real_tiny_training(tmp_path):
    import subprocess, sys
    args, _ = tiny_training_args(tmp_path)
    cmd = [sys.executable, 'scripts/finetune_supervised.py', '--train-only', '--head-d-model', '8']
    for name in ('experiment_id', 'tensor_npz', 'encoder', 'output_dir', 'checkpoint_dir', 'model_path',
                 'epochs', 'batch_size', 'warmup_epochs', 'freeze_layers', 'validation_size'):
        cmd += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    manifest = json.loads((Path(args.output_dir) / 'inner_selection_manifest.json').read_text())
    assert manifest['schema_version'] == 2 and manifest['outer_test_accessed'] is False


@pytest.mark.parametrize('defect', ['all_zero', 'nan', 'soft_label', 'zero_weight', 'disabled_head', 'provider_rule'])
def test_training_rejects_fake_or_disabled_head_supervision(defect):
    import tensorflow as tf
    model, x, labels = tiny_model()
    weights = None
    if defect == 'all_zero': labels['topology'] = tf.zeros_like(labels['topology'])
    if defect == 'nan': labels['disagreement'] = tf.fill([6, 2], float('nan'))
    if defect == 'soft_label': labels['topology'] = tf.fill([6, 3], 1 / 3)
    if defect == 'zero_weight': weights = {'type': tf.zeros(6)}
    if defect == 'disabled_head': model.topology_head_weight = 0.
    if defect == 'provider_rule': model.topology_rule_weight = .5
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        model.train_step((x, labels, weights))


def test_physics_auxiliary_uses_line_labels_not_global_provider(monkeypatch):
    model, x, labels = tiny_model()
    seen = []
    actual = model._auxiliary_losses
    def instrument(x, y, *args):
        seen.append(np.asarray(y))
        return actual(x, y, *args)
    monkeypatch.setattr(model, '_auxiliary_losses', instrument)
    model.train_step((x, labels))
    np.testing.assert_array_equal(seen[0], labels['topology'])


def test_partition_schema_checked_before_full_label_record_io(tmp_path, monkeypatch):
    from scripts.finetune_supervised import load_three_task_labels
    p = tmp_path / 'legacy-full.npz'
    np.savez(p, material_ids=np.array(['a', 'b', 'outer']), **{k: np.zeros(3, 'int32') for k in TASKS})
    actual = np.lib.npyio.NpzFile.__getitem__
    def prohibit_records(self, key):
        assert key not in ('material_ids', *TASKS), 'legacy full record was read before schema gate'
        return actual(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', prohibit_records)
    with pytest.raises(ValueError, match='schema'):
        load_three_task_labels(str(p), np.array(['a', 'b']), partition='train', sidecar_path=str(p))


@pytest.mark.parametrize('defect', ['norm_source', 'norm_std', 'norm_ids', 'accepted_not_best', 'seed', 'empty_monitor'])
def test_gate_checks_actual_normalizer_and_selection_identity(tmp_path, defect):
    from src.utils.selection_manifest import validate_inner_selection_manifest, sha256_file
    m = manifest_fixture(tmp_path)
    if defect.startswith('norm_'):
        p = tmp_path / 'normalization'
        n = json.loads(p.read_text())
        if defect == 'norm_source': n['source'] = 'all_data'
        if defect == 'norm_std': n['std'][0] = -1.
        if defect == 'norm_ids': n['fit_material_ids'] = ['c']
        p.write_text(json.dumps(n))
        m['training_contract']['artifacts']['normalization'].update(bytes=p.stat().st_size, sha256=sha256_file(p))
    if defect == 'accepted_not_best':
        p = tmp_path / 'accepted'
        p.write_bytes(b'foreign-run')
        m['states']['accepted'].update(bytes=p.stat().st_size, sha256=sha256_file(p))
    if defect == 'seed': m['seed'] = 7
    if defect == 'empty_monitor': m['best_epoch_one_based'] = None
    (tmp_path / 'inner_selection_manifest.json').write_text(json.dumps(m))
    with pytest.raises((RuntimeError, ValueError)):
        validate_inner_selection_manifest(str(tmp_path))


def test_evaluation_rejects_missing_predictions_without_silent_filtering():
    from scripts.finetune_supervised import evaluate_three_tasks
    labels = {k: np.array([0, 1, 0]) for k in TASKS}
    with pytest.raises(ValueError, match='probabilit'):
        evaluate_three_tasks(labels, {}, np.array([1, 1, 2]), n_boot=2)


def test_legacy_evaluation_config_copy_never_rewrites_frozen_config(tmp_path):
    from scripts.finetune_supervised import save_supervised_model_artifacts
    model, _, _ = tiny_model()
    weights = tmp_path / 'old.weights.h5'
    model.save_weights(weights)
    config = tmp_path / 'old_config.json'
    config.write_text('{"experiment_id":"historical"}')
    original = config.read_bytes()
    copy_dir = tmp_path / 'new-evaluation'
    copy_dir.mkdir()
    result = save_supervised_model_artifacts(model, str(weights), {'evaluation_mode': 'legacy_diagnostic'},
                                            save_weights=False, config_output_dir=str(copy_dir))
    assert config.read_bytes() == original
    assert Path(result['config']).parent == copy_dir
    with pytest.raises(FileExistsError):
        save_supervised_model_artifacts(model, str(weights), {}, save_weights=False, config_output_dir=str(copy_dir))


def test_legacy_cli_requires_new_output_before_any_array_io(tmp_path, monkeypatch):
    from scripts import finetune_supervised as ft
    from types import SimpleNamespace
    manifest_fixture(tmp_path, 1)
    args = SimpleNamespace(output_dir=str(tmp_path), evaluation_output_dir=str(tmp_path))
    assert hasattr(ft, 'prepare_legacy_evaluation'), 'legacy output isolation is missing'
    with pytest.raises(FileExistsError): ft.prepare_legacy_evaluation(args)


def test_legacy_train_only_helper_does_not_open_full_sidecar(tmp_path, monkeypatch):
    from scripts.finetune_supervised import load_dataset
    npz, norm, payload = tiny_split(tmp_path)
    np.savez(tmp_path / 'three_task_labels.npz', material_ids=payload['material_ids_train'])
    actual = np.load
    def guard(path, *a, **kw):
        assert Path(path).name != 'three_task_labels.npz', 'legacy train-only opened full labels'
        return actual(path, *a, **kw)
    monkeypatch.setattr(np, 'load', guard)
    result = load_dataset(str(npz), str(norm), include_outer_test=False)
    assert result['topology_train'] is None  # historical helper is not a training authorization


def test_legacy_evaluation_does_not_swallow_corrupt_label_alignment(tmp_path):
    from scripts.finetune_supervised import load_dataset
    npz, norm, payload = tiny_split(tmp_path)
    np.savez(tmp_path / 'three_task_labels.npz', material_ids=payload['material_ids_train'][:1],
             **{k: np.zeros(1, 'int32') for k in TASKS})
    with pytest.raises(ValueError, match='misses'):
        load_dataset(str(npz), str(norm), include_outer_test=True)


@pytest.mark.parametrize('bad', ['ids', 'groups_length', 'groups_fractional', 'targets', 'segments'])
def test_strict_tensor_partition_cannot_silently_drop_or_misalign_rows(tmp_path, bad):
    from scripts.finetune_supervised import load_dataset
    npz, norm, payload = tiny_split(tmp_path)
    np.savez(tmp_path / 'three_task_labels_train.npz', **label_payload(payload['material_ids_train']))
    if bad == 'ids': payload['material_ids_train'] = payload['material_ids_train'][:-1]
    if bad == 'groups_length': payload['groups_train'] = payload['groups_train'][:-1]
    if bad == 'groups_fractional': payload['groups_train'] = payload['groups_train'].astype(float) + .5
    if bad == 'targets': payload['y_train'] = payload['y_train'][:-1]
    if bad == 'segments': payload['segment_ids_train'] = np.zeros((12, 7), 'int32')
    np.savez(npz, **payload)
    with pytest.raises(ValueError): load_dataset(str(npz), str(norm), include_outer_test=False, strict=True)


def test_training_refuses_input_changed_after_fit(tmp_path, monkeypatch):
    from scripts import finetune_supervised as ft
    args, _ = tiny_training_args(tmp_path)
    write = ft._new_json
    def instrument(path, payload):
        write(path, payload)
        if Path(path).name == 'training_history.json':
            with (tmp_path / 'three_task_labels_train.npz').open('ab') as h:
                h.write(b'changed after model consumed labels')
    monkeypatch.setattr(ft, '_new_json', instrument)
    with pytest.raises(RuntimeError, match='(hash|size) mismatch'):
        ft.run_strict_training(args)
    assert not (Path(args.output_dir) / 'inner_selection_manifest.json').exists()


def test_inner_selection_probabilities_are_preserved_before_outer(tmp_path):
    from scripts import finetune_supervised as ft
    args, _ = tiny_training_args(tmp_path)
    manifest = ft.run_strict_training(args)
    p = Path(args.output_dir) / 'inner_selection_predictions.npz'
    assert p.exists(), 'inner-selected per-ID probabilities are missing'
    with np.load(p) as data:
        assert data['material_ids'].tolist() == manifest['training_contract']['selection_material_ids']
        assert str(data['scope']) == 'inner_selection_not_calibration'
        for task in TASKS: assert f'probabilities_{task}' in data


def test_freeze_rejects_nonfinite_monitor_before_creating_accepted(tmp_path):
    from scripts import finetune_supervised as ft
    from types import SimpleNamespace
    c = manifest_fixture(tmp_path)['training_contract']
    model, _, _ = tiny_model()
    ckpt, out = tmp_path / 'ckpt', tmp_path / 'out'
    ckpt.mkdir(); out.mkdir()
    model.save_weights(ckpt / 'best.weights.h5'); model.save_weights(ckpt / 'last.weights.h5')
    target = tmp_path / 'accepted.weights.h5'
    with pytest.raises(ValueError, match='monitor'):
        ft.freeze_supervised_states(model, str(ckpt), str(target), str(out),
                                    SimpleNamespace(epoch=[0], history={'val_loss': [float('nan')]}),
                                    training_contract=c, experiment_id='p0-seed42', seed=42)
    assert not target.exists()


def test_new_model_config_names_actual_heads_labels_and_probabilities(tmp_path):
    from scripts import finetune_supervised as ft
    args, _ = tiny_training_args(tmp_path)
    m = ft.run_strict_training(args)
    cfg = json.loads(Path(m['training_contract']['artifacts']['model_config']['path']).read_text())
    assert 'heads' in cfg, 'new config must not repeat the historical gap/type-only omission'
    assert set(cfg['heads']) == {'gap', 'type', 'topology', 'disagreement'}
    for task, head, size in zip(TASKS, ('topology', 'type', 'disagreement'), (3, 3, 2)):
        assert cfg['heads'][head] == {'label_name': task, 'probability_name': f'probabilities_{task}', 'dimension': size}


@pytest.mark.parametrize('experiment_id,seed', [('p0_seed42_seed2024', 42), ('p0_seed42_seed2024', 2024), ('p0_seed7_seed7', 7)])
@pytest.mark.parametrize('gate', ['preflight', 'selection'])
def test_experiment_identity_has_one_unambiguous_seed_token(tmp_path, experiment_id, seed, gate):
    from src.utils import selection_manifest as sm
    run = tmp_path / experiment_id
    if gate == 'preflight':
        with pytest.raises(ValueError):
            sm.validate_new_run_outputs(experiment_id, seed, run / 'reports', run / 'checkpoints', run / 'models/accepted.weights.h5')
        return
    manifest = manifest_fixture(tmp_path)
    manifest['experiment_id'] = experiment_id
    manifest['seed'] = seed
    (tmp_path / 'inner_selection_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        sm.validate_inner_selection_manifest(str(tmp_path))


def test_explicit_legacy_recovery_map_preserves_manifest_bytes(tmp_path):
    from src.utils.selection_manifest import validate_inner_selection_manifest, manifest_state_path
    m = manifest_fixture(tmp_path, 1)
    source_manifest = tmp_path / 'inner_selection_manifest.json'
    original = source_manifest.read_bytes()
    mapping = {}
    for name, state in m['states'].items():
        old = Path(state['path'])
        recovered = tmp_path / (name + '-recovered')
        old.rename(recovered)
        mapping[state['path']] = str(recovered)
    validated = validate_inner_selection_manifest(str(tmp_path), mode='legacy', state_path_map=mapping)
    assert validated['schema_version'] == 1
    assert manifest_state_path(validated, 'best') == Path(mapping[m['states']['best']['path']])
    assert source_manifest.read_bytes() == original
    with pytest.raises(ValueError, match='legacy'):
        validate_inner_selection_manifest(str(tmp_path), state_path_map=mapping)
