"""P2 contracts exercised only on generated, bounded synthetic fixtures."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_normalize_canonical_order_actual_statistics_no_refit():
    from src.data import band_structure_dataset as m
    assert hasattr(m, 'normalize_mbm_inputs'), 'shared MBM transform is missing'
    raw = np.arange(72, dtype=np.float32).reshape(3, 2, 4, 3)
    stats = {'mean': [2, 4, 6, 8, 10, 12], 'std': [1, 2, 3, 4, 5, 6]}
    old = raw.copy()
    flat = np.concatenate([raw[:, 0], raw[:, 1]], axis=-1)
    expected = (flat - np.asarray(stats['mean'], np.float32)) / np.asarray(stats['std'], np.float32)
    np.testing.assert_array_equal(m.normalize_mbm_inputs(raw, stats), expected)
    np.testing.assert_array_equal(m.normalize_mbm_inputs(flat, stats), expected)
    np.testing.assert_array_equal(raw, old)
    np.testing.assert_array_equal(m.normalize_mbm_inputs(raw[:1], stats), expected[:1])


@pytest.mark.parametrize('case', ['nan', 'inf', 'wrong_order', 'zero_std', 'negative_std', 'bad_mean', 'broadcast_stats'])
def test_normalize_rejects_corrupt_contract(case):
    from src.data.band_structure_dataset import normalize_mbm_inputs
    x = np.ones((2, 2, 4, 3), np.float32)
    stats = {'mean': [0.] * 6, 'std': [1.] * 6}
    if case == 'nan': x[0, 0, 0, 0] = np.nan
    if case == 'inf': x[0, 0, 0, 0] = np.inf
    if case == 'wrong_order': x = np.ones((2, 3, 4, 2), np.float32)
    if case == 'zero_std': stats['std'][0] = 0
    if case == 'negative_std': stats['std'][0] = -1
    if case == 'bad_mean': stats['mean'][0] = np.nan
    if case == 'broadcast_stats': stats['mean'] = [[0, 0, 0], [0, 0, 0]]
    with pytest.raises(ValueError):
        normalize_mbm_inputs(x, stats)


def tensor_contract_path(source, split='train'):
    return Path(source).with_name(f'band_tensors_ood_split.{split}.tensor_contract.json')


def synthetic_sources(tmp_path, n=8, split='train', prefix='s', group_start=1, target_k=128):
    # Actual upstream HDF5 -> canonical builder CLI, never a re-signed NPZ.
    import h5py
    import subprocess
    import sys
    total = 2 * int(round((n//2) / (.8 if split == 'train' else .2)))
    assert total <= 32
    raw = tmp_path/f'raw-{split}.h5'
    k = np.linspace(0.,1.,24,dtype=np.float32)
    with h5py.File(raw, 'w') as f:
        f.attrs['scope'] = 'test'
        for i in range(total):
            g = f.create_group(f'{prefix}{i:03d}')
            val = -.2-(k-.3)**2-i*.01; cond = .5+(k-.6)**2+i*.01
            g.create_dataset('energies', data=np.stack([val-1,val,cond,cond+1])+5.)
            g.create_dataset('k_distances',data=k)
            g.attrs.update(spacegroup_number=group_start+i//2, band_gap=.7+i*.02,
                source='materials_project', efermi=5., energy_reference='absolute',
                energy_quantity='band_energy_eV', num_kpoints=len(k), is_metal=False, is_direct=False,
                kpath_labels=json.dumps([{'index':0,'label':'G'},{'index':23,'label':'X'}]))
    out = tmp_path/f'tensors-{split}'
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/build_ood_tensors.py'),
           '--h5',str(raw),'--metadata','','--output',str(out),'--target-k',str(target_k),'--test-scope']
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
    (tmp_path/f'builder-{split}.log').write_text(result.stdout+result.stderr)
    (tmp_path/f'builder-{split}-command.json').write_text(json.dumps(cmd))
    assert result.returncode == 0, result.stdout+result.stderr
    source = out/'band_tensors_ood_split.npz'
    with np.load(source,allow_pickle=False) as f:
        ids, groups = f[f'material_ids_{split}'], f[f'groups_{split}']
    assert len(ids) == n
    side = tmp_path/f'side-{split}.json'
    side.write_text(json.dumps([{'material_id':str(mid), 'lattice':(np.eye(3)*3).tolist(),
        'species_per_atom':['Si'], 'fractional_coordinates':[[0,0,0]], 'spacegroup_number':int(group)}
        for mid,group in zip(ids,groups)]))
    return source, side, ids

def test_prepare_split_completes_without_outer_access(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p
    assert hasattr(p, 'prepare_split'), 'split-local preparation is missing'
    source, side, ids = synthetic_sources(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__
    def no_outer(self, key):
        assert 'test' not in key, f'outer array opened: {key}'
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', no_outer)
    path = p.prepare_split(source, side, tmp_path/'new_pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    arrays, contract = p.load_pairs(path, 'train', test_scope=True)
    np.testing.assert_array_equal(arrays['material_ids'], ids)
    np.testing.assert_array_equal(arrays['valid_idx'], np.arange(8))
    assert contract['input_space'] == 'raw_canonical_6d'
    assert contract['status'] == 'complete'
    assert contract['sources']['tensor']['sha256'] == p.file_ref(source)['sha256']
    assert not (tmp_path/'new_pairs'/'p2_test.npz').exists()
    with pytest.raises((FileExistsError, ValueError)):
        p.prepare_split(source, side, tmp_path/'new_pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))


@pytest.mark.parametrize('damage', ['duplicate_ids', 'reordered_valid_ids', 'bad_valid_idx', 'short_groups',
                                    'nan_graph', 'bad_neighbor', 'normalized_cache', 'wrong_semantics'])
def test_pair_loader_rejects_semantic_corruption_even_with_payload_sha(tmp_path, damage):
    from scripts import prepare_p2_pairs as p
    source, side, ids = synthetic_sources(tmp_path)
    path = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    a, c = p.load_pairs(path, 'train', test_scope=True)
    if damage == 'duplicate_ids': a['material_ids'][1] = a['material_ids'][0]
    if damage == 'reordered_valid_ids': a['valid_material_ids'] = a['valid_material_ids'][::-1]
    if damage == 'bad_valid_idx': a['valid_idx'] = a['valid_idx'][::-1]
    if damage == 'short_groups': a['groups'] = a['groups'][:-1]
    if damage == 'nan_graph': a['neighbor_dist'][0, 0, 0] = np.nan
    if damage == 'bad_neighbor': a['neighbor_list'][0, 0, 0] = 99
    if damage == 'normalized_cache': c['input_space'] = 'mbm_normalized'
    if damage == 'wrong_semantics': c['feature_semantics'] = {'order':['wrong']}
    np.savez(path, **a)
    c['artifact'] = p.file_ref(path)
    p.completion_path(path).write_text(json.dumps(c))
    with pytest.raises(ValueError):
        p.load_pairs(path, 'train', test_scope=True)


def synthetic_anchor(tmp_path, source=None):
    import subprocess
    import sys
    from src.models import load_ssl_encoder
    source = source or tmp_path/'tensors-train'/'band_tensors_ood_split.npz'
    out = tmp_path/'ssl'
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/train_ssl.py'),
        '--tensor-npz',str(source),'--tensor-contract',str(tensor_contract_path(source)),
        '--test-scope','--epochs','1','--batch-size','4','--warmup-epochs','0',
        '--d-model','128','--num-heads','4','--num-layers','1','--dff','128','--projection-dim','128',
        '--model-dir',str(out),'--checkpoint-dir',str(tmp_path/'ssl-checkpoints'),'--log-dir',str(tmp_path/'ssl-logs')]
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    (tmp_path/'ssl-native.log').write_text(result.stdout+result.stderr)
    (tmp_path/'ssl-native-command.json').write_text(json.dumps(cmd))
    assert result.returncode == 0, result.stdout+result.stderr
    encoder, norm = out/'ssl_mbm_pretrained.keras', out/'ssl_mbm_norm_stats.json'
    return load_ssl_encoder(encoder, compile=False), encoder, norm, json.loads(norm.read_text())


def anchor_contract_path(encoder):
    return Path(encoder).with_name('ssl_anchor_contract.json')

def test_embedding_split_real_encoder_matches_normalized_forward(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p, extract_band_embeddings as e
    assert hasattr(e, 'extract_split'), 'split-local normalized extraction is missing'
    from src.data.band_structure_dataset import normalize_mbm_inputs
    source, side, ids = synthetic_sources(tmp_path)
    records = json.loads(side.read_text()); side.write_text(json.dumps(records[1:]))
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    band, encoder, norm, stats = synthetic_anchor(tmp_path)
    a, c = p.load_pairs(pair, 'train', test_scope=True)
    before = p.file_ref(encoder)
    original = np.lib.npyio.NpzFile.__getitem__
    def no_outer(self, key):
        assert 'test' not in key, f'outer read {key}'
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', no_outer)
    out = e.extract_split(pair, encoder, norm, tmp_path/'emb', 'train', test_scope=True, batch_size=3, anchor_contract=anchor_contract_path(encoder))
    emb, receipt = p.read_completed_npz(out, 'band_embeddings', 'train', test_scope=True)
    expected = band(normalize_mbm_inputs(a['band_input'][a['valid']], stats),
                    training=False, return_features=True).numpy()
    np.testing.assert_allclose(emb['band_embeddings'], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(emb['valid_idx'], np.arange(1, 8))
    np.testing.assert_array_equal(emb['valid_material_ids'], ids[1:])
    assert receipt['anchor']['encoder'] == before == p.file_ref(encoder)
    assert receipt['anchor']['norm'] == p.file_ref(norm)
    assert receipt['anchor']['norm_stats'] == stats
    assert receipt['anchor']['feature_mode'] == 'SSLEncoder.return_features.pooled'


def test_tf_formal_fails_closed_before_data_io(tmp_path, monkeypatch):
    from scripts import extract_band_embeddings as e
    def deny(*args, **kwargs): raise AssertionError('data opened before GPU gate')
    monkeypatch.setattr(np, 'load', deny)
    with pytest.raises(RuntimeError, match='GPU'):
        e.extract_split('missing.npz', 'missing.keras', 'missing.json', tmp_path/'out', 'train')


@pytest.mark.parametrize('samples,epochs', [(33, 1), (8, 3), (0, 1), (8, 0)])
def test_cpu_test_scope_is_hard_bounded(samples, epochs):
    from scripts import extract_band_embeddings as e
    assert hasattr(e, 'configure_runtime'), 'explicit bounded runtime policy is missing'
    with pytest.raises(ValueError, match='scope|samples|epochs'):
        e.configure_runtime(test_scope=True, samples=samples, epochs=epochs)


def test_tf_train_only_inner_selection_freezes_three_real_checkpoints(tmp_path, monkeypatch):
    from scripts import train_p2_contrastive as t, prepare_p2_pairs as p
    assert hasattr(t, 'train_only'), 'train-only selection/freeze boundary is missing'
    source, side, ids = synthetic_sources(tmp_path, n=14)
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    _, encoder, norm, _ = synthetic_anchor(tmp_path)
    original = np.lib.npyio.NpzFile.__getitem__
    def no_outer(self, key):
        assert 'test' not in key, f'outer array opened: {key}'
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', no_outer)
    run = tmp_path/'run'
    manifest = t.train_only(pair, encoder, norm, run, test_scope=True, epochs=2,
                            batch_size=3, validation_size=.5, anchor_contract=anchor_contract_path(encoder))
    assert manifest['status'] == 'frozen'
    assert manifest['config']['hidden_dim'] == 128
    assert manifest['config']['conv_layers'] == 3
    assert manifest['config']['temperature'] == .07
    fit, val = manifest['inner']['fit_indices'], manifest['inner']['validation_indices']
    assert sorted(fit + val) == list(range(14))
    groups = np.asarray(manifest['inner']['groups'])
    assert not set(groups[fit]) & set(groups[val])
    assert manifest['inner']['material_ids'] == ids.tolist()
    assert all(h['validation_n'] == len(val) and h['fit_n'] == len(fit) for h in manifest['history'])
    assert manifest['best_epoch'] == min(manifest['history'], key=lambda h:h['val_loss'])['epoch']
    verified = p.verify_selection(run, 'tensorflow')
    assert verified == manifest
    for state in ('best', 'last', 'accepted'):
        assert p.file_ref(run/manifest['checkpoints'][state]['file']) == manifest['checkpoints'][state]['artifact']
    assert manifest['checkpoints']['best']['artifact'] == manifest['checkpoints']['accepted']['artifact']
    model, frozen = t.load_frozen(run, test_scope=True, band_encoder=encoder, norm=norm, anchor_contract=anchor_contract_path(encoder))
    assert frozen == manifest
    assert t.tf_state_sha(model) == manifest['checkpoints']['accepted']['state_sha256']
    pairs, _ = p.load_pairs(pair, 'train', test_scope=True)
    s, b = t.encode_pairs(model, encoder, norm, pairs, np.asarray(val), 3, test_scope=True, anchor_contract=anchor_contract_path(encoder))
    from src.evaluation.retrieval import info_nce_temperature
    assert info_nce_temperature(s, b, .07) == pytest.approx(min(h['val_loss'] for h in manifest['history']), abs=1e-5)


def trained_tf_fixture(tmp_path):
    from scripts import train_p2_contrastive as t, prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    _, encoder, norm, _ = synthetic_anchor(tmp_path)
    run = tmp_path/'run'
    m = t.train_only(pair, encoder, norm, run, test_scope=True, epochs=1, batch_size=3, anchor_contract=anchor_contract_path(encoder))
    source, side, ids = synthetic_sources(tmp_path, n=4, split='test', prefix='outer', group_start=100)
    outer = p.prepare_split(source, side, tmp_path/'outer', 'test', test_scope=True, tensor_contract=tensor_contract_path(source, 'test'))
    return run, m, encoder, norm, outer, ids


def test_tf_evaluation_only_persists_complete_ranks_without_state_mutation(tmp_path):
    from scripts import train_p2_contrastive as t, prepare_p2_pairs as p
    assert hasattr(t, 'evaluation_only'), 'frozen evaluation-only boundary is missing'
    run, m, encoder, norm, outer, ids = trained_tf_fixture(tmp_path)
    before = {f.name:p.file_ref(f) for f in run.iterdir() if f.is_file()}
    report = t.evaluation_only(outer, encoder, norm, run, tmp_path/'evaluation', test_scope=True, batch_size=3, anchor_contract=anchor_contract_path(encoder))
    after = {f.name:p.file_ref(f) for f in run.iterdir() if f.is_file()}
    assert before == after
    assert report['state_before'] == report['state_after'] == m['checkpoints']['accepted']['state_sha256']
    assert report['selection'] == p.file_ref(run/'selection.json')
    for direction in ('structure_to_band', 'band_to_structure'):
        d = report['retrieval'][direction]
        assert d['query_ids'] == d['gallery_ids'] == ids.tolist()
        assert d['query_n'] == d['gallery_n'] == 4
        assert len(d['ranks']) == 4
        ranks = np.asarray(d['ranks'])
        assert d['metrics']['mrr'] == pytest.approx(np.mean(1/ranks))
        assert d['metrics']['map'] == d['metrics']['mrr']
        assert d['metrics']['median_rank'] == np.median(ranks)
        for k in (1, 5, 10): assert d['metrics'][f'recall@{k}'] == np.mean(ranks <= k)
        assert 'single positive' in d['map_interpretation']
    saved = json.loads((tmp_path/'evaluation'/'report.json').read_text())
    assert saved == report
    with np.load(tmp_path/'evaluation'/'predictions.npz') as z:
        assert z['structure_embeddings'].shape == z['band_embeddings'].shape == (4, 128)
        np.testing.assert_array_equal(z['material_ids'], ids)


@pytest.mark.parametrize('damage', ['outer_id', 'outer_group', 'norm_changed'])
def test_eval_refuses_seen_pool_or_different_anchor(tmp_path, damage):
    from scripts import train_p2_contrastive as t, prepare_p2_pairs as p
    run, m, encoder, norm, outer, _ = trained_tf_fixture(tmp_path)
    if damage == 'norm_changed':
        stats = json.loads(norm.read_text()); stats['mean'][0] += .1
        norm.write_text(json.dumps(stats))
    else:
        a, c = p.load_pairs(outer, 'test', test_scope=True)
        if damage == 'outer_id':
            a['material_ids'][0] = m['inner']['material_ids'][0]; a['valid_material_ids'][0] = m['inner']['material_ids'][0]
        if damage == 'outer_group': a['groups'][0] = m['inner']['groups'][0]
        np.savez(outer, **a); c['artifact'] = p.file_ref(outer)
        p.completion_path(outer).write_text(json.dumps(c))
    with pytest.raises(ValueError, match='overlap|anchor'):
        t.evaluation_only(outer, encoder, norm, run, tmp_path/'reject', test_scope=True, anchor_contract=anchor_contract_path(encoder))
    assert not (tmp_path/'reject'/'stage_completion.json').exists()


def test_equivariant_prepare_split_binds_pair_order_and_completion(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p
    assert hasattr(p, 'prepare_equivariant_split'), 'credentialed single-split graph preparation missing'
    source, side, ids = synthetic_sources(tmp_path)
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    original = np.lib.npyio.NpzFile.__getitem__
    def no_outer(self, key):
        assert 'test' not in key
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', no_outer)
    graph = p.prepare_equivariant_split(pair, side, tmp_path/'graphs', 'train', max_atoms=2, test_scope=True)
    a, c = p.read_completed_npz(graph, 'equivariant_graphs', 'train', test_scope=True)
    assert c['sources']['pairs'] == p.file_ref(pair)
    assert c['sources']['sidecar'] == p.file_ref(side)
    np.testing.assert_array_equal(a['material_ids'], ids)
    np.testing.assert_array_equal(a['valid_idx'], np.arange(8))
    assert np.all(a['edge_len'] > 0)
    assert np.allclose(np.linalg.norm(a['edge_vec'], axis=1), a['edge_len'])


def torch_anchor_kwargs(tmp_path):
    directory = tmp_path/'anchor'/'ssl'
    return {'band_encoder':directory/'ssl_mbm_pretrained.keras', 'norm':directory/'ssl_mbm_norm_stats.json',
            'anchor_contract':directory/'ssl_anchor_contract.json'}


def synthetic_torch_inputs(tmp_path, split='train', n=8, prefix='s', group_start=1, sparse=False, run_dir=None):
    from scripts import prepare_p2_pairs as p, extract_band_embeddings as e
    source, side, ids = synthetic_sources(tmp_path, n=n, split=split, prefix=prefix, group_start=group_start)
    if sparse:
        side.write_text(json.dumps(json.loads(side.read_text())[1:]))
    pair = p.prepare_split(source, side, tmp_path/f'pairs-{split}', split, test_scope=True, tensor_contract=tensor_contract_path(source, split))
    anchor_dir = tmp_path/'anchor'; anchor_dir.mkdir(exist_ok=True)
    if not (anchor_dir/'ssl'/'ssl_mbm_pretrained.keras').exists(): synthetic_anchor(anchor_dir, source)
    emb = e.extract_split(pair, anchor_dir/'ssl'/'ssl_mbm_pretrained.keras', anchor_dir/'ssl'/'ssl_mbm_norm_stats.json', tmp_path/f'emb-{split}',
                          split, test_scope=True, batch_size=3, anchor_contract=anchor_contract_path(anchor_dir / 'ssl' / 'ssl_mbm_pretrained.keras'), run_dir=run_dir)
    graph = p.prepare_equivariant_split(pair, side, tmp_path/f'graphs-{split}', split, max_atoms=2, test_scope=True)
    return graph, emb, ids


def test_torch_load_split_aligns_sparse_valid_rows_once(tmp_path):
    from scripts import train_p2_equivariant as t
    assert hasattr(t, 'load_split'), 'cross-framework identity loader is missing'
    graph, emb, ids = synthetic_torch_inputs(tmp_path, sparse=True)
    ds, b, c = t.load_split(graph, emb, 'train', 'cpu', test_scope=True)
    assert len(ds) == len(b) == 7
    np.testing.assert_array_equal(ds.idx, np.arange(1, 8))
    np.testing.assert_array_equal(ds.ids, ids[1:])
    g = ds.collate(np.asarray([0, 6]))
    assert g['descriptors'].shape == (2, 22)
    model = t.EquivariantStructureEncoder()
    model.eval()
    import torch
    with torch.no_grad(): result = model(g, g['descriptors'])
    assert tuple(result.shape) == (2, 128)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize('damage', ['id_order', 'valid_order', 'embedding_length', 'raw_length', 'nan_edges', 'offsets', 'source_pair'])
def test_torch_loader_rejects_corrupt_or_misaligned_caches(tmp_path, damage):
    from scripts import train_p2_equivariant as t, prepare_p2_pairs as p
    graph, emb, ids = synthetic_torch_inputs(tmp_path, sparse=True)
    path, kind = (emb, 'band_embeddings') if damage in ('id_order', 'embedding_length', 'source_pair') else (graph, 'equivariant_graphs')
    a, c = p.read_completed_npz(path, kind, 'train', test_scope=True)
    if damage == 'id_order':
        a['material_ids'][[1, 2]] = a['material_ids'][[2, 1]]
        a['valid_material_ids'] = a['material_ids'][a['valid']]
    if damage == 'valid_order': a['valid_idx'] = a['valid_idx'][::-1]
    if damage == 'embedding_length': a['band_embeddings'] = a['band_embeddings'][:-1]
    if damage == 'raw_length': a['descriptors'] = np.vstack((a['descriptors'], a['descriptors'][:1]))
    if damage == 'nan_edges': a['edge_vec'][0, 0] = np.nan
    if damage == 'offsets': a['edge_offsets'][0] = [0, 999999]
    if damage == 'source_pair': c['sources']['pairs']['sha256'] = '0'*64
    np.savez(path, **a); c['artifact'] = p.file_ref(path)
    p.completion_path(path).write_text(json.dumps(c))
    with pytest.raises(ValueError):
        t.load_split(graph, emb, 'train', 'cpu', test_scope=True)


def test_torch_inference_freezes_real_e3nn_batchnorm_and_disables_grad(tmp_path):
    from scripts import train_p2_equivariant as t
    import torch
    assert hasattr(t, 'encode_evaluation'), 'eval/no_grad inference boundary is missing'
    graph, emb, _ = synthetic_torch_inputs(tmp_path)
    ds, _, _ = t.load_split(graph, emb, 'train', 'cpu', test_scope=True)
    model = t.EquivariantStructureEncoder()
    model.train()
    initial = t.torch_state_sha(model)
    g = ds.collate(np.asarray([0, 1]))
    model(g, g['descriptors'])
    before = t.torch_state_sha(model)
    assert initial != before, 'fixture must witness actual training-mode BN mutation'
    buffers = {k:v.clone() for k,v in model.named_buffers()}
    observed = []
    hook = model.register_forward_hook(lambda m,args,out: observed.append((m.training, torch.is_grad_enabled())))
    result = t.encode_evaluation(model, ds, np.arange(len(ds)), 3, formal=False)
    hook.remove()
    assert result.shape == (8, 128)
    assert observed and all(x == (False, False) for x in observed)
    assert not model.training
    assert t.torch_state_sha(model) == before
    for k,v in model.named_buffers(): assert torch.equal(v, buffers[k])
    again = t.encode_evaluation(model, ds, np.arange(len(ds)), 3, formal=False)
    np.testing.assert_array_equal(result, again)


def test_torch_train_only_full_inner_selection_freeze_reload(tmp_path, monkeypatch):
    from scripts import train_p2_equivariant as t, prepare_p2_pairs as p
    assert hasattr(t, 'train_only'), 'torch train-only selection/freeze boundary missing'
    graph, emb, ids = synthetic_torch_inputs(tmp_path, n=14)
    original = np.lib.npyio.NpzFile.__getitem__
    def no_outer(self, key):
        assert 'test' not in key, 'outer data opened while training'
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', no_outer)
    m = t.train_only(graph, emb, tmp_path/'run', epochs=2, batch_size=3, validation_size=.5, test_scope=True, **torch_anchor_kwargs(tmp_path))
    assert m['config']['num_layers'] == 3 and m['config']['multiplicity'] == 32
    assert m['config']['embedding_dim'] == 128 and m['config']['temperature'] == .07
    assert m['inner']['material_ids'] == ids.tolist()
    assert m == p.verify_selection(tmp_path/'run', 'torch')
    assert m['checkpoints']['best']['artifact'] == m['checkpoints']['accepted']['artifact']
    model, loaded = t.load_frozen(tmp_path/'run', test_scope=True, **torch_anchor_kwargs(tmp_path))
    assert loaded == m and not model.training
    assert t.torch_state_sha(model) == m['checkpoints']['accepted']['state_sha256']
    ds, b, _ = t.load_split(graph, emb, 'train', 'cpu', test_scope=True)
    val = np.asarray(m['inner']['validation_indices'])
    s = t.encode_evaluation(model, ds, val, 3)
    from src.evaluation.retrieval import info_nce_temperature
    assert min(h['val_loss'] for h in m['history']) == pytest.approx(info_nce_temperature(s, b[val], .07), abs=1e-5)
    assert all(h['validation_n'] == len(val) for h in m['history'])
    with pytest.raises(RuntimeError, match='GPU'):
        t.train_only('missing', 'missing', tmp_path/'formal')
    with pytest.raises(ValueError, match='scope'):
        t.train_only(graph, emb, tmp_path/'overscope', epochs=3, test_scope=True)


def test_torch_evaluation_only_preserves_bn_and_rejects_sha_before_outer_io(tmp_path, monkeypatch):
    from scripts import train_p2_equivariant as t, prepare_p2_pairs as p
    assert hasattr(t, 'evaluation_only'), 'torch frozen evaluation-only boundary missing'
    graph, emb, _ = synthetic_torch_inputs(tmp_path)
    run = tmp_path/'run'
    m = t.train_only(graph, emb, run, epochs=1, batch_size=3, test_scope=True, **torch_anchor_kwargs(tmp_path))
    og, ob, ids = synthetic_torch_inputs(tmp_path, split='test', n=4, prefix='outer', group_start=100, run_dir=run)
    before = {f.name:p.file_ref(f) for f in run.iterdir()}
    report = t.evaluation_only(og, ob, run, tmp_path/'eval', batch_size=3, test_scope=True, **torch_anchor_kwargs(tmp_path))
    assert before == {f.name:p.file_ref(f) for f in run.iterdir()}
    assert report['state_before'] == report['state_after'] == m['checkpoints']['accepted']['state_sha256']
    for d in report['retrieval'].values():
        assert d['query_n'] == d['gallery_n'] == len(ids) == len(d['ranks'])
        assert d['query_ids'] == d['gallery_ids'] == ids.tolist()
    reloaded, _ = t.load_frozen(run, test_scope=True, **torch_anchor_kwargs(tmp_path))
    ds, b, _ = t.load_split(og, ob, 'test', 'cpu', test_scope=True)
    s = t.encode_evaluation(reloaded, ds, np.arange(len(ds)), 3)
    with np.load(tmp_path/'eval'/'predictions.npz') as z:
        np.testing.assert_array_equal(s, z['structure_embeddings'])
        np.testing.assert_array_equal(b, z['band_embeddings'])
    with (run/'accepted.pt').open('ab') as f: f.write(b'corruption')
    def deny(*a, **kw): raise AssertionError('outer input read before frozen SHA gate')
    monkeypatch.setattr(np, 'load', deny)
    with pytest.raises(ValueError, match='SHA'):
        t.evaluation_only(og, ob, run, tmp_path/'rejected', test_scope=True, **torch_anchor_kwargs(tmp_path))


@pytest.mark.parametrize('framework', ['tf', 'torch'])
def test_native_cli_train_then_separate_evaluation(tmp_path, framework):
    import subprocess
    import sys
    from scripts import prepare_p2_pairs as p
    root = Path(__file__).resolve().parents[1]
    def cli(script, *args):
        args = list(args)
        if script == 'train_p2_equivariant.py':
            args += ['--band-encoder',encoder,'--norm',norm]
        if '--tensor-npz' in args:
            args += ['--tensor-contract', tensor_contract_path(args[args.index('--tensor-npz')+1], args[args.index('--split')+1])]
        if '--band-encoder' in args:
            args += ['--anchor-contract', anchor_contract_path(args[args.index('--band-encoder')+1])]
        command = [sys.executable, str(root/'scripts'/script), *map(str,args)]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, timeout=90)
        step = f'{script}-{len(list(tmp_path.glob("*.log")))}'
        (tmp_path/f'{step}.log').write_text(result.stdout+result.stderr)
        (tmp_path/f'{step}-command.json').write_text(json.dumps(command))
        assert result.returncode == 0, f'{command}\n{result.stdout}\n{result.stderr}'
    source, side, _ = synthetic_sources(tmp_path)
    _, encoder, norm, _ = synthetic_anchor(tmp_path)
    cli('prepare_p2_pairs.py', '--tensor-npz',source,'--sidecar',side,'--split','train','--out-dir',tmp_path/'pairs', '--test-scope')
    pair = tmp_path/'pairs'/'p2_train.npz'
    run = tmp_path/'run'
    if framework == 'tf':
        cli('train_p2_contrastive.py', '--train-only','--train',pair,'--band-encoder',encoder,'--norm',norm,
            '--output-dir',run,'--epochs',1,'--batch-size',3,'--test-scope')
    else:
        cli('extract_band_embeddings.py','--pairs',pair,'--band-encoder',encoder,'--norm',norm,
            '--split','train','--out-dir',tmp_path/'emb','--test-scope')
        cli('prepare_p2_pairs.py','--equivariant','--pairs',pair,'--sidecar',side,'--split','train',
            '--out-dir',tmp_path/'graphs','--max-atoms',2,'--test-scope')
        cli('train_p2_equivariant.py','--train-only','--graphs',tmp_path/'graphs'/'graphs_train.npz',
            '--band-emb',tmp_path/'emb'/'band_embeddings_train.npz','--output-dir',run,'--epochs',1,
            '--batch-size',3,'--test-scope')
    assert (run/'selection.json').is_file(), 'native trainer did not produce frozen selection'
    p.verify_selection(run, 'tensorflow' if framework == 'tf' else 'torch')
    # Outer preparation starts only AFTER the independent training process exits.
    source, side, ids = synthetic_sources(tmp_path, n=4, split='test', prefix='out', group_start=100)
    cli('prepare_p2_pairs.py','--tensor-npz',source,'--sidecar',side,'--split','test','--out-dir',tmp_path/'outer','--test-scope')
    pair = tmp_path/'outer'/'p2_test.npz'
    if framework == 'tf':
        cli('train_p2_contrastive.py','--evaluation-only','--test',pair,'--band-encoder',encoder,'--norm',norm,
            '--run-dir',run,'--output-dir',tmp_path/'eval','--batch-size',3,'--test-scope')
    else:
        cli('extract_band_embeddings.py','--pairs',pair,'--band-encoder',encoder,'--norm',norm,
            '--split','test','--out-dir',tmp_path/'outer-emb','--run-dir',run,'--test-scope')
        cli('prepare_p2_pairs.py','--equivariant','--pairs',pair,'--sidecar',side,'--split','test',
            '--out-dir',tmp_path/'outer-graphs','--max-atoms',2,'--test-scope')
        cli('train_p2_equivariant.py','--evaluation-only','--graphs',tmp_path/'outer-graphs'/'graphs_test.npz',
            '--band-emb',tmp_path/'outer-emb'/'band_embeddings_test.npz','--run-dir',run,
            '--output-dir',tmp_path/'eval','--batch-size',3,'--test-scope')
    report = json.loads((tmp_path/'eval'/'report.json').read_text())
    assert report['retrieval']['structure_to_band']['query_ids'] == ids.tolist()
    assert report['state_before'] == report['state_after']


@pytest.mark.parametrize('damage', ['short_groups', 'duplicate_ids', 'float_segments'])
def test_prepare_cannot_authorize_invalid_sample_contract(tmp_path, damage):
    from scripts import prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    with np.load(source) as f:
        a = {k:f[k] for k in ('X_train','material_ids_train','groups_train','segment_ids_train')}
    if damage == 'short_groups': a['groups_train'] = a['groups_train'][:-1]
    if damage == 'duplicate_ids': a['material_ids_train'][1] = a['material_ids_train'][0]
    if damage == 'float_segments': a['segment_ids_train'] = a['segment_ids_train'].astype(float)
    np.savez(source, **a)
    with pytest.raises(ValueError):
        p.prepare_split(source, side, tmp_path/'out', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    assert not (tmp_path/'out'/'p2_train.npz.completion.json').exists()


def test_frozen_selection_verifies_nested_upstream_code_before_outer(tmp_path, monkeypatch):
    from scripts import train_p2_contrastive as t, prepare_p2_pairs as p
    run, m, encoder, norm, outer, _ = trained_tf_fixture(tmp_path)
    m['data']['pairs']['code']['src/data/crystal_graph.py']['sha256'] = '0'*64
    (run/'selection.json').write_text(json.dumps(m))
    c = json.loads((run/'stage_completion.json').read_text())
    c['selection'] = p.file_ref(run/'selection.json'); c['upstream'] = m['data']
    (run/'stage_completion.json').write_text(json.dumps(c))
    def deny(*a, **kw): raise AssertionError('outer I/O before complete upstream code verification')
    monkeypatch.setattr(np, 'load', deny)
    with pytest.raises(ValueError, match='code SHA'):
        t.evaluation_only(outer, encoder, norm, run, tmp_path/'rejected', test_scope=True, anchor_contract=anchor_contract_path(encoder))


@pytest.mark.parametrize('field', ['code', 'sources', 'test_scope'])
def test_incomplete_new_receipt_cannot_masquerade_as_formal_contract(tmp_path, field):
    from scripts import prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    c = json.loads(p.completion_path(pair).read_text()); del c[field]
    p.completion_path(pair).write_text(json.dumps(c))
    with pytest.raises(ValueError, match='contract|completion'):
        p.load_pairs(pair, 'train', test_scope=True)


def test_split_loader_refuses_outer_arrays_before_decoding(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    a, c = p.load_pairs(pair, 'train', test_scope=True)
    a['X_test'] = np.asarray([object()], dtype=object)
    np.savez(pair, **a); c['artifact'] = p.file_ref(pair)
    p.completion_path(pair).write_text(json.dumps(c))
    old = np.lib.npyio.NpzFile.__getitem__
    def no_outer(self, key):
        assert key != 'X_test', 'split loader opened an outer member'
        return old(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, '__getitem__', no_outer)
    with pytest.raises(ValueError, match='arrays|schema'):
        p.load_pairs(pair, 'train', test_scope=True)


# Existing guards: independent regression, not additional TDD RED claims.
def test_legacy_inspection_never_authorizes_or_modifies_cache(tmp_path, capsys):
    from scripts import prepare_p2_pairs as p
    legacy = tmp_path/'legacy.npz'
    np.savez(legacy, band_input=np.zeros((1,4,6)), valid=np.ones(1,bool))
    before = p.file_ref(legacy)
    with pytest.raises(ValueError, match='legacy'):
        p.load_pairs(legacy, 'train', test_scope=True)
    p.main(['--legacy-diagnostic', str(legacy)])
    assert json.loads(capsys.readouterr().out)['status'] == 'legacy_diagnostic_not_accepted'
    assert p.file_ref(legacy) == before


def test_interrupted_completion_and_changed_upstream_leave_no_authorization(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    original = p.os.replace
    def deny_completion(src, dst):
        if str(dst).endswith('.completion.json'):
            raise OSError('synthetic interrupted completion publish')
        return original(src, dst)
    monkeypatch.setattr(p.os, 'replace', deny_completion)
    with pytest.raises(OSError, match='interrupted'):
        p.prepare_split(source, side, tmp_path/'interrupted', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    with pytest.raises(ValueError, match='completion'):
        p.load_pairs(tmp_path/'interrupted'/'p2_train.npz', 'train', test_scope=True)
    monkeypatch.setattr(p.os, 'replace', original)
    snapshot = p.file_ref(side)
    side.write_text('[]')
    out = p.fresh_directory(tmp_path/'changed')
    with pytest.raises(ValueError, match='source changed'):
        p.save_completed_npz(out/'sample.npz', {'x':np.arange(3)}, {'code':{}}, {side:snapshot})
    assert not p.completion_path(out/'sample.npz').exists()


def test_tf_checkpoint_corruption_rejected_before_any_outer_array(tmp_path, monkeypatch):
    from scripts import train_p2_contrastive as t
    run, m, encoder, norm, outer, _ = trained_tf_fixture(tmp_path)
    with (run/'last.weights.h5').open('ab') as f: f.write(b'corrupt-last')
    def deny(*a, **kw): raise AssertionError('outer I/O before SHA validation')
    monkeypatch.setattr(np, 'load', deny)
    with pytest.raises(ValueError, match='SHA'):
        t.evaluation_only(outer, encoder, norm, run, tmp_path/'rejected', test_scope=True, anchor_contract=anchor_contract_path(encoder))


# Cycle-1 independent review counterexamples (S1, L1-L7).
def test_S1_anchor_dependency_code_escape_rejected_before_payloads(tmp_path,monkeypatch):
    from scripts import prepare_p2_pairs as p
    source,_,_ = synthetic_sources(tmp_path)
    _,encoder,norm,_ = synthetic_anchor(tmp_path)
    ac,rc = anchor_contract_path(encoder),tensor_contract_path(source)
    raw = json.loads(rc.read_text()); raw['code']['../synthetic-secret.py'] = {'path':'../synthetic-secret.py','bytes':1,'sha256':'0'*64}
    rc.write_text(json.dumps(raw))
    anchor = json.loads(ac.read_text()); anchor['source_tensor_contract_ref'].update(p.file_ref(rc))
    ac.write_text(json.dumps(anchor))
    original = p.file_ref
    def deny_payloads(path):
        if Path(path) not in (ac,rc): raise AssertionError('payload/code read before dependency path preflight')
        return original(path)
    monkeypatch.setattr(p,'file_ref',deny_payloads)
    with pytest.raises(ValueError,match='code|path'):
        p.inspect_anchor(encoder,norm,anchor_contract=ac,test_scope=True)

@pytest.mark.parametrize('damage',['graph_schema','embedding_space','embedding_anchor'])
def test_L4_torch_complete_data_semantics_gate_before_deserialization(tmp_path,monkeypatch,damage):
    from scripts import train_p2_equivariant as t, prepare_p2_pairs as p
    import torch
    graph,emb,_ = synthetic_torch_inputs(tmp_path)
    run = tmp_path/'run'
    m = t.train_only(graph,emb,run,epochs=1,batch_size=3,test_scope=True,**torch_anchor_kwargs(tmp_path))
    m = json.loads(json.dumps(m))
    if damage == 'graph_schema': m['data']['graphs']['graph_schema'] = 'legacy-v2'
    if damage == 'embedding_space': m['data']['embeddings']['input_space'] = 'unverified'
    if damage == 'embedding_anchor': m['data']['embeddings']['anchor']['norm_stats']['mean'][0] += 7.
    # JSON copies avoid relying on in-memory shared object identity.
    m = json.loads(json.dumps(m))
    (run/'selection.json').write_text(json.dumps(m))
    c = json.loads((run/'stage_completion.json').read_text()); c['upstream'] = m['data']; c['selection'] = p.file_ref(run/'selection.json')
    (run/'stage_completion.json').write_text(json.dumps(c))
    def deny(*a,**kw): raise AssertionError('deserialization before complete frozen data gate')
    monkeypatch.setattr(torch,'load',deny)
    with pytest.raises(ValueError): t.load_frozen(run,test_scope=True,**torch_anchor_kwargs(tmp_path))


def test_L6_upstream_evidence_mutation_cannot_complete_evaluation(tmp_path,monkeypatch):
    from scripts import train_p2_contrastive as t
    run,m,encoder,norm,outer,_ = trained_tf_fixture(tmp_path)
    original = t.encode_pairs
    def mutate(*a,**kw):
        values = original(*a,**kw)
        history = encoder.with_name('ssl_history.json')
        history.write_bytes(history.read_bytes()+b'\n')
        return values
    monkeypatch.setattr(t,'encode_pairs',mutate)
    with pytest.raises(ValueError,match='changed|snapshot|SHA'):
        t.evaluation_only(outer,encoder,norm,run,tmp_path/'raced',test_scope=True,anchor_contract=anchor_contract_path(encoder))
    assert not (tmp_path/'raced/stage_completion.json').exists()

def test_S1_loader_preflights_nested_paths_before_payload_or_code_io(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    pair = p.prepare_split(source,side,tmp_path/'pairs','train',test_scope=True,tensor_contract=tensor_contract_path(source))
    c = json.loads(p.completion_path(pair).read_text())
    c['parent'] = {'kind':'pairs','code':{'../synthetic.py':{'bytes':1,'sha256':'0'*64}}}
    p.completion_path(pair).write_text(json.dumps(c))
    reads = []
    original = p.file_ref
    def observe(path):
        reads.append(str(path)); return original(path)
    monkeypatch.setattr(p,'file_ref',observe)
    with pytest.raises(ValueError): p.load_pairs(pair,'train',test_scope=True)
    assert reads == [str(p.completion_path(pair))], 'only completion snapshot can be read before full preflight'


def test_L2_torch_training_requires_explicit_verified_anchor_before_cache_io(tmp_path,monkeypatch):
    from scripts import train_p2_equivariant as t
    def deny(*a,**kw): raise AssertionError('cache opened without explicit upstream anchor')
    monkeypatch.setattr(t,'load_split',deny)
    with pytest.raises(ValueError,match='anchor|contract'):
        t.train_only('untrusted-graphs','untrusted-embeddings',tmp_path/'out',epochs=1,test_scope=True)


@pytest.mark.parametrize('framework',['tf','torch'])
def test_L4_direct_reload_requires_actual_anchor_before_deserialization(tmp_path,monkeypatch,framework):
    from scripts import train_p2_contrastive as tf_trainer, train_p2_equivariant as torch_trainer, prepare_p2_pairs as p
    if framework == 'tf':
        run,m,encoder,norm,_,_ = trained_tf_fixture(tmp_path)
        trainer = tf_trainer
    else:
        graph,emb,_ = synthetic_torch_inputs(tmp_path)
        run = tmp_path/'run'
        m = torch_trainer.train_only(graph,emb,run,epochs=1,batch_size=3,test_scope=True, **torch_anchor_kwargs(tmp_path))
        trainer = torch_trainer
    def deny(*a,**kw): raise AssertionError('model deserialized before anchor preflight')
    monkeypatch.setattr(tf_trainer.CGCNNEncoder,'load_weights',deny)
    import torch
    monkeypatch.setattr(torch,'load',deny)
    with pytest.raises(ValueError,match='anchor|contract'):
        trainer.load_frozen(run,test_scope=True)

def test_L5_standalone_outer_extraction_requires_frozen_p2_before_outer_io(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p, extract_band_embeddings as e
    source, side, _ = synthetic_sources(tmp_path)
    _, encoder, norm, _ = synthetic_anchor(tmp_path)
    source, side, _ = synthetic_sources(tmp_path,n=4,split='test',prefix='outer',group_start=100)
    outer = p.prepare_split(source,side,tmp_path/'outer','test',test_scope=True,tensor_contract=tensor_contract_path(source,'test'))
    def deny(*a, **kw): raise AssertionError('standalone extraction opened outer without P2 freeze')
    monkeypatch.setattr(p, 'load_pairs', deny)
    with pytest.raises(ValueError, match='frozen|selection|run'):
        e.extract_split(outer,encoder,norm,tmp_path/'forbidden','test',test_scope=True,
                        anchor_contract=anchor_contract_path(encoder))
    assert not p.completion_path(tmp_path/'forbidden'/'band_embeddings_test.npz').exists()

def test_L2_unknown_random_encoder_and_unbound_norm_are_blocked(tmp_path):
    from scripts import extract_band_embeddings as e
    from src.models.band_structure_encoder import SSLEncoder
    import tensorflow as tf
    model = SSLEncoder(num_features=6,seq_len=16,d_model=128,num_heads=4,num_layers=1,dff=128)
    model(tf.zeros((1,16,6)),training=False)
    encoder = tmp_path/'random.keras'; model.save(encoder)
    norm = tmp_path/'unbound.json'; norm.write_text(json.dumps({'mean':[0.]*6,'std':[1.]*6}))
    with pytest.raises(ValueError, match='anchor|contract|history'):
        e.load_anchor(encoder, norm)


def test_L2_downstream_selection_uses_actual_upstream_allocation(tmp_path):
    from scripts import prepare_p2_pairs as p, train_p2_contrastive as t
    source, side, _ = synthetic_sources(tmp_path, n=14)
    _, encoder, norm, _ = synthetic_anchor(tmp_path)
    anchor = json.loads(anchor_contract_path(encoder).read_text())
    pair = p.prepare_split(source,side,tmp_path/'pairs','train',test_scope=True,tensor_contract=tensor_contract_path(source))
    m = t.train_only(pair,encoder,norm,tmp_path/'run',test_scope=True,epochs=1,batch_size=3, anchor_contract=anchor_contract_path(encoder))
    assert set(m['inner']['validation_ids']) == set(anchor['selection_material_ids'])
    assert not set(m['inner']['validation_ids']) & set(anchor['fit_material_ids'])
    assert set(m['inner']['fit_ids']) == set(anchor['fit_material_ids'])

def test_L1_shape_or_declared_normalized_npz_cannot_be_resigned_raw(tmp_path):
    from scripts import prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    with np.load(source, allow_pickle=False) as f:
        arrays = {k:f[k] for k in ('X_train','material_ids_train','groups_train','segment_ids_train')}
    arrays['input_space'] = np.asarray('mbm_normalized')
    unbound = tmp_path/'normalized-unbound.npz'; np.savez(unbound, **arrays)
    with pytest.raises(ValueError, match='contract|raw|normalized'):
        p.prepare_split(unbound, side, tmp_path/'forbidden', 'train', test_scope=True)
    assert not p.completion_path(tmp_path/'forbidden'/'p2_train.npz').exists()

def test_L6_extraction_cannot_relabel_old_arrays_after_source_replacement(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p, extract_band_embeddings as e
    source, side, _ = synthetic_sources(tmp_path)
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    _, encoder, norm, _ = synthetic_anchor(tmp_path)
    original = e.encode_bands
    def mutate(*a, **kw):
        values = original(*a, **kw)
        data, c = p.load_pairs(pair, 'train', test_scope=True)
        data['band_input'] += .5
        np.savez(pair, **data); c['artifact'] = p.file_ref(pair)
        p.completion_path(pair).write_text(json.dumps(c))
        return values
    monkeypatch.setattr(e, 'encode_bands', mutate)
    with pytest.raises(ValueError, match='changed|snapshot|SHA'):
        e.extract_split(pair, encoder, norm, tmp_path/'raced', 'train', test_scope=True, anchor_contract=anchor_contract_path(encoder))
    assert not p.completion_path(tmp_path/'raced'/'band_embeddings_train.npz').exists()


def test_L6_eval_reopened_norm_must_match_initial_snapshot(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p, train_p2_contrastive as t
    run, m, encoder, norm, outer, _ = trained_tf_fixture(tmp_path)
    original = p.load_pairs
    def mutate(*a, **kw):
        value = original(*a, **kw)
        if a[1] == 'test':
            stats = json.loads(norm.read_text()); stats['mean'][0] += 10.
            norm.write_text(json.dumps(stats))
        return value
    monkeypatch.setattr(p, 'load_pairs', mutate)
    with pytest.raises(ValueError, match='changed|snapshot|SHA|anchor'):
        t.evaluation_only(outer, encoder, norm, run, tmp_path/'raced', test_scope=True, anchor_contract=anchor_contract_path(encoder))
    assert not (tmp_path/'raced'/'stage_completion.json').exists()


def test_L6_npz_decode_cannot_consume_bytes_other_than_hashed_snapshot(tmp_path, monkeypatch):
    from scripts import prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    data, _ = p.load_pairs(pair, 'train', test_scope=True)
    original = np.load
    def replace_then_decode(*a, **kw):
        data['band_input'] += 123
        np.savez(pair, **data)
        return original(*a, **kw)
    monkeypatch.setattr(np, 'load', replace_then_decode)
    with pytest.raises(ValueError, match='changed|snapshot|SHA'):
        p.load_pairs(pair, 'train', test_scope=True)

@pytest.mark.parametrize('damage', ['code', 'data', 'anchor', 'test_scope', 'inner', 'history', 'best_epoch',
                                    'empty_history', 'bad_best_epoch', 'inner_overlap', 'config', 'runtime',
                                    'anchor_stats', 'anchor_config', 'anchor_history','raw_space','raw_schema'])
def test_L4_full_selection_gate_precedes_model_deserialization_and_outer(tmp_path, monkeypatch, damage):
    from scripts import prepare_p2_pairs as p, train_p2_contrastive as t
    run, m, encoder, norm, outer, _ = trained_tf_fixture(tmp_path)
    if damage == 'empty_history': m['history'] = []; m['best_epoch'] = None
    elif damage == 'bad_best_epoch': m['best_epoch'] = 999
    elif damage == 'inner_overlap': m['inner']['validation_indices'] = m['inner']['fit_indices']
    elif damage == 'anchor_stats': m['anchor']['norm_stats']['mean'][0] += 10.
    elif damage == 'anchor_config': m['anchor']['config']['d_model'] = 32
    elif damage == 'anchor_history': m['anchor']['upstream']['best_epoch'] = 999
    elif damage == 'raw_space': m['data']['pairs']['raw_contract']['input_space'] = 'mbm_normalized'
    elif damage == 'raw_schema': m['data']['pairs']['feature_schema']['energy_reference'] = 'absolute'
    else: m.pop(damage)
    (run/'selection.json').write_text(json.dumps(m))
    c = json.loads((run/'stage_completion.json').read_text())
    c['selection'] = p.file_ref(run/'selection.json'); c['upstream'] = m.get('data')
    (run/'stage_completion.json').write_text(json.dumps(c))
    def deny(*a, **kw): raise AssertionError('model or outer read before complete prior validation')
    monkeypatch.setattr(t.CGCNNEncoder, 'load_weights', deny)
    from scripts import extract_band_embeddings as e
    monkeypatch.setattr(e, 'load_ssl_encoder', deny)
    monkeypatch.setattr(p, 'load_pairs', deny)
    with pytest.raises(ValueError):
        t.evaluation_only(outer, encoder, norm, run, tmp_path/'invalid', test_scope=True, anchor_contract=anchor_contract_path(encoder))
    assert not (tmp_path/'invalid'/'stage_completion.json').exists()


def test_L4_empty_history_random_checkpoint_cannot_freeze(tmp_path):
    from scripts import prepare_p2_pairs as p, train_p2_contrastive as t
    # Real checkpoint bytes, but no optimizer has ever touched this model.
    import tensorflow as tf
    import shutil
    model = t.CGCNNEncoder()
    model({'atom_features':tf.zeros((1,2,110)), 'neighbor_list':tf.fill((1,2,12),-1),
           'neighbor_dist':tf.zeros((1,2,12))}, training=False)
    run = tmp_path/'random'; run.mkdir(); model.save_weights(run/'best.weights.h5')
    checkpoints = {}
    for key in ('best','last','accepted'):
        if key != 'best': shutil.copyfile(run/'best.weights.h5', run/f'{key}.weights.h5')
        checkpoints[key] = {'file':f'{key}.weights.h5', 'artifact':p.file_ref(run/f'{key}.weights.h5'),
                            'state_sha256':t.tf_state_sha(model)}
    with pytest.raises(ValueError):
        p.freeze_selection(run, {'framework':'tensorflow','test_scope':False, 'data':{},
                                'history':[], 'best_epoch':None, 'checkpoints':checkpoints})
    assert not (run/'stage_completion.json').exists()

@pytest.mark.parametrize('damage', ['missing_schema', 'legacy_schema', 'bad_builder_sha', 'bad_max_neighbors',
                                    'bad_cutoff', 'fractional_images', 'wrong_pair_valid'])
def test_L3_real_loader_rejects_v3_semantic_bypass(tmp_path, damage):
    from scripts import train_p2_equivariant as t, prepare_p2_pairs as p
    graph, emb, _ = synthetic_torch_inputs(tmp_path)
    a, c = p.read_completed_npz(graph, 'equivariant_graphs', 'train', test_scope=True)
    if damage == 'missing_schema': del a['schema_version']; c.pop('graph_schema')
    if damage == 'legacy_schema': a['schema_version'] = np.asarray('legacy-v2'); c['graph_schema'] = 'legacy-v2'
    if damage == 'bad_builder_sha': a['builder_sha256'] = np.asarray('0'*64)
    if damage == 'bad_max_neighbors': a['max_neighbors'] = np.asarray(999)
    if damage == 'bad_cutoff': a['cutoff_angstrom'] = np.asarray(999.)
    if damage == 'fractional_images': a['edge_image'] = a['edge_image'].astype(float) + .5
    if damage == 'wrong_pair_valid': a['pair_valid'] = np.zeros_like(a['pair_valid'])
    np.savez(graph, **a); c['artifact'] = p.file_ref(graph)
    p.completion_path(graph).write_text(json.dumps(c))
    with pytest.raises(ValueError):
        t.load_split(graph, emb, 'train', 'cpu', test_scope=True)


def test_L3_adapter_preserves_and_loader_checks_neighbor_diagnostics(tmp_path):
    from scripts import train_p2_equivariant as t, prepare_p2_pairs as p
    graph, emb, _ = synthetic_torch_inputs(tmp_path)
    a, c = p.read_completed_npz(graph, 'equivariant_graphs', 'train', test_scope=True)
    required = {'neighbor_count','truncated_neighbors','distance_12_13','boundary_tied'}
    assert required <= set(a), 'periodic adapter discarded builder diagnostics'
    assert a['boundary_tied'].any(), 'synthetic cubic cell must witness tied shell cutoff'
    a['truncated_neighbors'][:] = -1
    np.savez(graph, **a); c['artifact'] = p.file_ref(graph)
    p.completion_path(graph).write_text(json.dumps(c))
    with pytest.raises(ValueError, match='diagnostic|neighbor'):
        t.load_split(graph, emb, 'train', 'cpu', test_scope=True)

@pytest.mark.parametrize('scale,dtype', [(1e30, np.float32), (1e300, np.float64), (1e-300, np.float64)])
def test_L7_extreme_finite_diagonal_has_exact_ranks(scale, dtype):
    from scripts.prepare_p2_pairs import retrieval_evidence
    x = np.eye(3, dtype=dtype) * dtype(scale)
    with np.errstate(all='raise'):
        report = retrieval_evidence(x, x, np.asarray(['a','b','c']), chunk_size=2)
    for d in report.values():
        assert d['ranks'] == [1,1,1]
        np.testing.assert_allclose(d['positive_cosine'], 1.)
        assert d['metrics']['recall@1'] == 1.


def test_L7_zero_vectors_cannot_produce_rank_evidence():
    from scripts.prepare_p2_pairs import retrieval_evidence
    with pytest.raises(ValueError, match='zero|norm'):
        retrieval_evidence(np.zeros((2,3)), np.zeros((2,3)), np.asarray(['a','b']))

@pytest.mark.parametrize('form', ['absolute', 'traversal', 'windows', 'symlink'])
def test_S1_code_escape_rejected_before_file_read(tmp_path, monkeypatch, form):
    from scripts import prepare_p2_pairs as p
    root = tmp_path/'root'; (root/'scripts').mkdir(parents=True)
    sentinel = tmp_path/'sentinel.py'; sentinel.write_text('synthetic only')
    monkeypatch.setattr(p, 'ROOT', root)
    rel = {'absolute':str(sentinel), 'traversal':'../sentinel.py',
           'windows':'C:\\synthetic\\sentinel.py', 'symlink':'scripts/prepare_p2_pairs.py'}[form]
    if form == 'symlink': (root/rel).symlink_to(sentinel)
    reads = []
    def spy(path):
        reads.append(str(path)); return {'sha256':'0'*64, 'bytes':14}
    monkeypatch.setattr(p, 'file_ref', spy)
    with pytest.raises(ValueError, match='code|path'):
        p.check_code({rel:{'sha256':'0'*64, 'bytes':14}})
    assert reads == [], 'rejected source must never be opened'


def test_S1_stage_cannot_omit_required_code_bindings(tmp_path):
    from scripts import prepare_p2_pairs as p
    source, side, _ = synthetic_sources(tmp_path)
    pair = p.prepare_split(source, side, tmp_path/'pairs', 'train', test_scope=True, tensor_contract=tensor_contract_path(source, 'train'))
    c = json.loads(p.completion_path(pair).read_text())
    del c['code']['src/data/crystal_graph.py']
    p.completion_path(pair).write_text(json.dumps(c))
    with pytest.raises(ValueError, match='code'):
        p.load_pairs(pair, 'train', test_scope=True)


def test_S1_entire_nested_tree_is_path_checked_before_any_code_read(monkeypatch):
    from scripts import prepare_p2_pairs as p
    reads = []
    good = {k:{'bytes':1,'sha256':'0'*64} for k in p.CODE_STAGES['pairs']}
    raw = {'kind':'raw_tensor', 'code':{k:{'bytes':1,'sha256':'0'*64} for k in p.CODE_STAGES['raw_tensor']}}
    value = {'kind':'pairs','code':good,'raw_contract':raw,
             'parent':{'code':{'../synthetic.py':{'bytes':1,'sha256':'0'*64}}}}
    monkeypatch.setattr(p,'file_ref',lambda path: reads.append(path) or {'bytes':1,'sha256':'0'*64})
    with pytest.raises(ValueError,match='code|path'):
        p.check_code_tree(value)
    assert not reads, 'all nested code paths must pass preflight before any file opens'


@pytest.fixture(scope='module')
def known_stage_producer(tmp_path_factory):
    """Real K128 producer, validated at its original source ROOT before negatives."""
    from scripts import prepare_p2_pairs as p
    directory = tmp_path_factory.mktemp('known-stage-producer')
    source, side, ids = synthetic_sources(directory)
    _, encoder, norm, _ = synthetic_anchor(directory, source)
    stats, anchor = p.inspect_anchor(encoder, norm, anchor_contract=anchor_contract_path(encoder), test_scope=True)
    pair = p.prepare_split(source, side, directory/'pairs', 'train', test_scope=True,
                           tensor_contract=tensor_contract_path(source))
    arrays, receipt = p.load_pairs(pair, 'train', test_scope=True)
    assert arrays['material_ids'].tolist() == ids.tolist()
    assert anchor['upstream']['feature_schema']['seq_len'] == 128
    for epoch in anchor['upstream']['history']['epochs']:
        for pool in ('train', 'val'):
            assert .15 <= epoch[pool]['mask_fraction'] <= .30
    return dict(directory=directory, source=source, side=side, encoder=encoder, norm=norm,
                anchor=anchor, pair=pair, receipt=receipt)


def test_S1_R1_real_producer_positive(known_stage_producer):
    from scripts import prepare_p2_pairs as p
    f = known_stage_producer
    _, actual = p.inspect_anchor(f['encoder'], f['norm'], test_scope=True,
                                 anchor_contract=anchor_contract_path(f['encoder']))
    assert actual == f['anchor']
    _, receipt = p.load_pairs(f['pair'], 'train', test_scope=True)
    assert receipt == f['receipt']


def damage_known_stage(node, damage, expected_stage):
    from scripts import prepare_p2_pairs as p
    if damage == 'kind_and_code':
        del node['kind']; del node['code']
    elif damage == 'unknown_kind_no_code':
        node['kind'] = 'unrecognized_stage'; del node['code']
    elif damage == 'code_only':
        del node['code']
    else:
        other = 'raw_tensor' if expected_stage != 'raw_tensor' else 'ssl_anchor'
        node['code'] = p.code_refs(*(p.ROOT/rel for rel in sorted(p.CODE_STAGES[other])))
        if damage == 'wrong_known_stage_wrong_fieldset':
            node['kind'] = other


def assert_known_stage_preflight(call, allowed_json, monkeypatch, trap, record_property, label):
    """Trap real file_ref or open; receipt self-read/private JSON is not payload I/O."""
    import builtins
    import io
    from scripts import prepare_p2_pairs as p
    allowed_json = {Path(path).resolve() for path in allowed_json}
    names = {path.name for path in allowed_json}
    events, forbidden = [], []
    actual_ref, actual_open, actual_io_open = p.file_ref, builtins.open, io.open
    def observe(path, api):
        if not isinstance(path, (str, bytes, Path)):
            return
        path = Path(path)
        allowed = (path.resolve() in allowed_json or
                   (path.parent.name.startswith('p2-read-') and path.name in names))
        row = {'api':api, 'path':str(path), 'input_json':allowed}
        events.append(row)
        if not allowed:
            forbidden.append(row)
            if trap == 'open' or api == 'file_ref':
                raise AssertionError('dependency/code/payload I/O before known-stage rejection: ' + str(path))
    def guarded_ref(path):
        observe(path, 'file_ref')
        return actual_ref(path)
    def guarded_open(path, *args, **kwargs):
        if args[:1] != ('xb',) and kwargs.get('mode') != 'xb':
            observe(path, 'builtins.open')
        return actual_open(path, *args, **kwargs)
    def guarded_io_open(path, *args, **kwargs):
        # Private-copy creation is not a read of a producer/dependency file.
        if args[:1] != ('xb',) and kwargs.get('mode') != 'xb':
            observe(path, 'io.open')
        return actual_io_open(path, *args, **kwargs)
    error = None
    with monkeypatch.context() as m:
        if trap == 'file_ref':
            m.setattr(p, 'file_ref', guarded_ref)
        else:
            m.setattr(builtins, 'open', guarded_open)
            m.setattr(io, 'open', guarded_io_open)
        try:
            call()
        except Exception as exc:
            error = {'type':type(exc).__name__, 'message':str(exc)}
    row = {'name':label, 'trap':trap, 'outcome':error, 'forbidden_io':forbidden,
           'input_json_reads':[e for e in events if e['input_json']]}
    record_property('S1_R1', json.dumps(row))
    print('S1_R1 ' + json.dumps(row))
    assert not forbidden, row
    assert error and error['type'] == 'ValueError', row
    assert any(word in error['message'] for word in ('kind', 'stage', 'code')), row


@pytest.mark.parametrize('trap', ['file_ref', 'open'])
@pytest.mark.parametrize('damage', ['kind_and_code', 'unknown_kind_no_code',
    'wrong_known_stage_wrong_fieldset', 'code_only', 'wrong_fieldset_only'])
@pytest.mark.parametrize('entry', ['inspect_anchor', 'inspect_anchor_encoder', 'prepare_split', 'anchor_raw_dependency'])
def test_S1_R1_known_entry_preflight(known_stage_producer, tmp_path, monkeypatch, record_property, trap, damage, entry):
    from scripts import prepare_p2_pairs as p
    f = known_stage_producer
    anchor_path = anchor_contract_path(f['encoder'])
    raw_path = tensor_contract_path(f['source'])
    if entry in ('inspect_anchor', 'inspect_anchor_encoder'):
        value = json.loads(anchor_path.read_text())
        damage_known_stage(value, damage, 'ssl_anchor')
        bad = tmp_path/'anchor.json'; bad.write_text(json.dumps(value))
        # Second variant reproduces the original encoder-only sentinel report.
        allowed = [bad, raw_path] if entry == 'inspect_anchor_encoder' else [bad]
        call = lambda: p.inspect_anchor(f['encoder'], f['norm'], anchor_contract=bad, test_scope=True)
    elif entry == 'prepare_split':
        value = json.loads(raw_path.read_text())
        damage_known_stage(value, damage, 'raw_tensor')
        bad = tmp_path/'raw.json'; bad.write_text(json.dumps(value))
        allowed = [bad]
        call = lambda: p.prepare_split(f['source'], f['side'], tmp_path/'rejected', 'train',
                                       test_scope=True, tensor_contract=bad)
    else:
        # The adversarial dependency is a new local copy, never an edited producer receipt.
        import shutil
        source = tmp_path/f['source'].name; shutil.copyfile(f['source'], source)
        value = json.loads(raw_path.read_text())
        damage_known_stage(value, damage, 'raw_tensor')
        bad_raw = tensor_contract_path(source); bad_raw.write_text(json.dumps(value))
        anchor = json.loads(anchor_path.read_text())
        anchor['source_tensor_ref'] = {'path':str(source), **p.file_ref(source)}
        anchor['source_tensor_contract_ref'] = {'path':str(bad_raw), **p.file_ref(bad_raw)}
        bad = tmp_path/'anchor.json'; bad.write_text(json.dumps(anchor))
        allowed = [bad, bad_raw]
        call = lambda: p.inspect_anchor(f['encoder'], f['norm'], anchor_contract=bad, test_scope=True)
    assert_known_stage_preflight(call, allowed, monkeypatch, trap, record_property, f'{entry}.{damage}')
    assert not (tmp_path/'rejected').exists()


@pytest.fixture(scope='module')
def known_role_producer(known_stage_producer):
    from scripts import prepare_p2_pairs as p, extract_band_embeddings as e
    from scripts import train_p2_contrastive as tf_trainer, train_p2_equivariant as torch_trainer
    f = known_stage_producer
    d = f['directory']
    kwargs = dict(test_scope=True, anchor_contract=anchor_contract_path(f['encoder']))
    embeddings = e.extract_split(f['pair'], f['encoder'], f['norm'], d/'embeddings', 'train', **kwargs)
    graphs = p.prepare_equivariant_split(f['pair'], f['side'], d/'graphs', 'train', max_atoms=2, test_scope=True)
    tf_run, torch_run = d/'tf-run', d/'torch-run'
    tf_trainer.train_only(f['pair'], f['encoder'], f['norm'], tf_run, epochs=1, batch_size=3, **kwargs)
    torch_trainer.train_only(graphs, embeddings, torch_run, epochs=1, batch_size=3,
                            band_encoder=f['encoder'], norm=f['norm'], **kwargs)
    # Both genuine frozen priors must pass, before any adversarial copy is made.
    p.verify_selection(tf_run, 'tensorflow')
    p.verify_selection(torch_run, 'torch')
    return {**f, 'equivariant_graphs':graphs, 'band_embeddings':embeddings,
            'pairs':f['pair'], 'tensorflow':tf_run, 'torch':torch_run}


@pytest.mark.parametrize('trap', ['file_ref', 'open'])
@pytest.mark.parametrize('damage', ['kind_and_code', 'unknown_kind_no_code',
    'wrong_known_stage_wrong_fieldset', 'code_only', 'wrong_fieldset_only'])
@pytest.mark.parametrize('stage,keys,child_stage', [
    ('pairs', ('raw_contract',), 'raw_tensor'),
    ('equivariant_graphs', ('parent',), 'pairs'),
    ('band_embeddings', ('parent',), 'pairs'),
    ('band_embeddings', ('anchor', 'upstream'), 'ssl_anchor'),
    ('tensorflow', ('anchor', 'upstream'), 'ssl_anchor'),
    ('tensorflow', ('data', 'pairs'), 'pairs'),
    ('torch', ('data', 'graphs'), 'equivariant_graphs'),
    ('torch', ('data', 'embeddings'), 'band_embeddings'),
])
def test_S1_R1_known_child_preflight(known_role_producer, tmp_path, monkeypatch, record_property,
                                    trap, damage, stage, keys, child_stage):
    import shutil
    from scripts import prepare_p2_pairs as p
    path = known_role_producer[stage]
    frozen = stage in ('tensorflow', 'torch')
    value = json.loads((path/'selection.json' if frozen else p.completion_path(path)).read_text())
    child = value
    for key in keys: child = child[key]
    damage_known_stage(child, damage, child_stage)
    if frozen:
        allowed = []
        call = lambda: p.validate_selection_metadata(value, path, stage)
    else:
        bad = tmp_path/path.name; shutil.copyfile(path, bad)
        receipt = p.completion_path(bad); receipt.write_text(json.dumps(value))
        allowed = [receipt]
        call = lambda: p.read_completed_npz(bad, stage, 'train', test_scope=True)
    assert_known_stage_preflight(call, allowed, monkeypatch, trap, record_property,
                                  f'{stage}.{".".join(keys)}.{damage}')


@pytest.mark.parametrize('trap', ['file_ref', 'open'])
@pytest.mark.parametrize('damage', ['kind_and_code', 'unknown_kind_no_code',
    'wrong_known_stage_wrong_fieldset', 'code_only', 'wrong_fieldset_only'])
@pytest.mark.parametrize('api', ['receipt_metadata', 'anchor_metadata'])
def test_S1_R1_direct_known_metadata_preflight(known_stage_producer, monkeypatch, record_property, trap, damage, api):
    from scripts import prepare_p2_pairs as p
    f = known_stage_producer
    if api == 'receipt_metadata':
        value = json.loads(json.dumps(f['receipt']))
        damage_known_stage(value['raw_contract'], damage, 'raw_tensor')
        call = lambda: p.validate_receipt_metadata(value, 'pairs', 'train', True)
    else:
        value = json.loads(json.dumps(f['anchor']))
        damage_known_stage(value['upstream'], damage, 'ssl_anchor')
        call = lambda: p.validate_anchor_metadata(value, True)
    assert_known_stage_preflight(call, [], monkeypatch, trap, record_property, f'{api}.{damage}')


def test_short_K16_native_SSL_cannot_publish_anchor(tmp_path):
    """K128 is the legal positive; keep the original short-mask budget rejection."""
    import subprocess
    import sys
    source, _, _ = synthetic_sources(tmp_path, target_k=16)
    out = tmp_path/'ssl-short'
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1]/'scripts/train_ssl.py'),
        '--tensor-npz',str(source),'--tensor-contract',str(tensor_contract_path(source)),
        '--test-scope','--epochs','1','--batch-size','4','--warmup-epochs','0',
        '--d-model','128','--num-heads','4','--num-layers','1','--dff','128','--projection-dim','128',
        '--model-dir',str(out),'--checkpoint-dir',str(tmp_path/'checkpoints'),'--log-dir',str(tmp_path/'logs')]
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=120)
    (tmp_path/'ssl-short-command.json').write_text(json.dumps(cmd))
    (tmp_path/'ssl-short.log').write_text(result.stdout+result.stderr)
    assert result.returncode != 0, 'out-of-budget short K must not produce a valid SSL anchor'
    assert 'mask' in (result.stdout+result.stderr).lower()
    assert not (out/'ssl_anchor_contract.json').exists()
